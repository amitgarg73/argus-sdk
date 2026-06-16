"""
Tests for argus.engine — generic eval engine + pattern detectors + RCA.
All L3 evals are now config-driven via ag_eval_configs rows (metric type + agent_pattern).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from argus.engine import (
    EvalResult, Incident,
    run_evals_from_config, run_evals_and_persist,
    run_all_detectors, run_detectors_and_persist,
    register_eval, get_registry,
    build_annotated_call_stack, generate_fix_suggestion, summarize_incident,
)
from argus.engine.eval_engine import (
    _metric_agent_presence,
    _metric_tool_success_rate,
    _metric_tool_diversity,
    _metric_token_budget,
    _metric_data_freshness,
    _metric_exit_quality,
    _metric_cost_anomaly,
    _metric_pipeline_coverage,
    _metric_agent_completion,
    _metric_session_field_ratio,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _session(**kw) -> dict:
    base = {
        "id":                  "sess-test",
        "total_cost_usd":      0.10,
        "total_tokens_input":  3000,
        "total_tokens_output": 1500,
        "trades_executed":     2,
        "trades_proposed":     2,
        "terminal_reason":     "trade_executed",
        "started_at":          "2026-06-08T10:00:00",
        "completed_at":        "2026-06-08T10:05:00",
        "tenant_id":           "tenant-abc",
        "workflow_id":         "workflow-xyz",
    }
    return {**base, **kw}


def _healthy_traces() -> list[dict]:
    return [
        {"agent": "market",       "step_type": "tool_call",    "tool_name": "get_market",
         "outcome": "success",   "error": None, "latency_ms": 200,
         "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100},
        {"agent": "research",     "step_type": "llm_call",     "tool_name": None,
         "outcome": "success",   "error": None, "latency_ms": 800,
         "created_at": "2026-06-08T10:01:00",
         "tokens_input": 300, "tokens_output": 150},
        {"agent": "research",     "step_type": "tool_call",    "tool_name": "get_news",
         "outcome": "success",   "error": None, "latency_ms": 250,
         "created_at": "2026-06-08T10:01:10",
         "tokens_input": 0, "tokens_output": 0},
        {"agent": "research",     "step_type": "tool_call",    "tool_name": "get_atr",
         "outcome": "success",   "error": None, "latency_ms": 180,
         "created_at": "2026-06-08T10:01:20",
         "tokens_input": 0, "tokens_output": 0},
        {"agent": "risk",         "step_type": "tool_call",    "tool_name": "check_risk",
         "outcome": "success",   "error": None, "latency_ms": 150,
         "created_at": "2026-06-08T10:02:00",
         "tokens_input": 50, "tokens_output": 30},
        {"agent": "risk",         "step_type": "decision",     "tool_name": None,
         "outcome": "approved",  "error": None, "latency_ms": 10,
         "created_at": "2026-06-08T10:02:10",
         "tokens_input": 0, "tokens_output": 0},
        {"agent": "orchestrator", "step_type": "decision",     "tool_name": None,
         "outcome": "execute",   "error": None, "latency_ms": 20,
         "created_at": "2026-06-08T10:03:00",
         "tokens_input": 150, "tokens_output": 80},
    ]


def _timeout_traces() -> list[dict]:
    return [
        {"agent": "market",   "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success", "error": None, "latency_ms": 200,
         "created_at": "2026-06-08T10:00:30", "tokens_input": 200, "tokens_output": 100},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out", "latency_ms": 30000,
         "created_at": "2026-06-08T10:01:00", "tokens_input": 0, "tokens_output": 0},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out", "latency_ms": 30000,
         "created_at": "2026-06-08T10:01:30", "tokens_input": 0, "tokens_output": 0},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out", "latency_ms": 30000,
         "created_at": "2026-06-08T10:02:00", "tokens_input": 0, "tokens_output": 0},
    ]


def _mk_ec(eval_name: str, metric: str, agent: str = "session",
           threshold: float = 0.5, **extra_config) -> dict:
    """Build a minimal ag_eval_configs row for testing."""
    return {
        "eval_name":  eval_name,
        "agent":      agent,
        "threshold":  threshold,
        "layer":      3,
        "eval_type":  "rule",
        "enabled":    True,
        "config":     {"metric": metric, **extra_config},
    }


def _tool_trace(agent: str, tool: str, outcome: str = "success",
                error=None, latency_ms: int = 200) -> dict:
    return {
        "agent": agent, "step_type": "tool_call", "tool_name": tool,
        "outcome": outcome, "error": error, "latency_ms": latency_ms,
        "created_at": "2026-06-08T10:01:00",
        "tokens_input": 10, "tokens_output": 5,
    }


def _llm_trace(agent: str, outcome: str = "success") -> dict:
    return {
        "agent": agent, "step_type": "llm_call", "tool_name": None,
        "outcome": outcome, "error": None, "latency_ms": 500,
        "created_at": "2026-06-08T10:00:30",
        "tokens_input": 100, "tokens_output": 50,
    }


# ── Metric function unit tests ────────────────────────────────────────────────

class TestMetricAgentPresence:
    def test_no_traces_returns_zero(self):
        score, detail = _metric_agent_presence([])
        assert score == 0.0
        assert "no agent traces" in detail["reason"]

    def test_all_success_returns_one(self):
        traces = [_tool_trace("a", "t1"), _tool_trace("a", "t2")]
        score, detail = _metric_agent_presence(traces)
        assert score == 1.0

    def test_all_errors_returns_zero(self):
        traces = [_tool_trace("a", "t1", outcome="error")]
        score, detail = _metric_agent_presence(traces)
        assert score == 0.0

    def test_partial_errors_proportional(self):
        traces = [_tool_trace("a", "t1"), _tool_trace("a", "t2", outcome="error")]
        score, _ = _metric_agent_presence(traces)
        assert score == 0.5


class TestMetricToolSuccessRate:
    def test_no_tool_calls_returns_one(self):
        score, detail = _metric_tool_success_rate([_llm_trace("a")])
        assert score == 1.0

    def test_all_success(self):
        traces = [_tool_trace("a", "t"), _tool_trace("a", "t2")]
        score, _ = _metric_tool_success_rate(traces)
        assert score == 1.0

    def test_partial_failure(self):
        traces = [_tool_trace("a", "t"), _tool_trace("a", "t2", outcome="error")]
        score, detail = _metric_tool_success_rate(traces)
        assert score == 0.5
        assert detail["failed"] == 1


class TestMetricToolDiversity:
    def test_no_tools_zero(self):
        score, _ = _metric_tool_diversity([], diversity_min=2)
        assert score == 0.0

    def test_exceeds_min_returns_one(self):
        traces = [_tool_trace("a", "t1"), _tool_trace("a", "t2"), _tool_trace("a", "t3")]
        score, detail = _metric_tool_diversity(traces, diversity_min=2)
        assert score == 1.0
        assert detail["distinct_tools"] == 3

    def test_exactly_min_returns_0_7(self):
        traces = [_tool_trace("a", "t1"), _tool_trace("a", "t2")]
        score, _ = _metric_tool_diversity(traces, diversity_min=2)
        assert score == 0.7


class TestMetricTokenBudget:
    def test_low_tokens_high_score(self):
        traces = [{"tokens_input": 100, "tokens_output": 50, "step_type": "llm_call",
                   "agent": "a", "outcome": "success"}]
        score, _ = _metric_token_budget(traces, {}, threshold=10_000)
        assert score > 0.9

    def test_over_budget_returns_low_score(self):
        traces = [{"tokens_input": 15_000, "tokens_output": 10_000, "step_type": "llm_call",
                   "agent": "a", "outcome": "success"}]
        score, _ = _metric_token_budget(traces, {}, threshold=10_000)
        assert score < 0.5


class TestMetricDataFreshness:
    def test_no_traces_returns_zero(self):
        score, _ = _metric_data_freshness([], {}, freshness_minutes=10)
        assert score == 0.0

    def test_no_timestamps_returns_one(self):
        traces = [{"agent": "a", "step_type": "tool_call", "created_at": None}]
        score, _ = _metric_data_freshness(traces, {}, freshness_minutes=10)
        assert score == 1.0

    def test_small_lag_high_score(self):
        traces = [{"created_at": "2026-06-08T10:00:30+00:00"}]
        session = {"started_at": "2026-06-08T10:00:00+00:00"}
        score, detail = _metric_data_freshness(traces, session, freshness_minutes=10)
        assert score > 0.9
        assert detail["lag_minutes"] == pytest.approx(0.5, abs=0.1)


class TestMetricExitQuality:
    def test_good_exit(self):
        s = _session(terminal_reason="trade_executed", trades_executed=1)
        score, _ = _metric_exit_quality(s, {"trade_executed"}, {"partial"}, {"error"})
        assert score == 1.0

    def test_bad_exit(self):
        s = _session(terminal_reason="error", trades_executed=0)
        score, _ = _metric_exit_quality(s, {"trade_executed"}, {"partial"}, {"error"})
        assert score == 0.2

    def test_partial_exit(self):
        s = _session(terminal_reason="partial", trades_executed=0)
        score, _ = _metric_exit_quality(s, {"trade_executed"}, {"partial"}, {"error"})
        assert score == 0.5

    def test_silent_exit(self):
        s = _session(terminal_reason="", trades_executed=0)
        score, _ = _metric_exit_quality(s, {"trade_executed"}, {"partial"}, {"error"})
        assert score == 0.0


class TestMetricCostAnomaly:
    def test_insufficient_history_returns_one(self):
        score, detail = _metric_cost_anomaly(_session(), [0.1, 0.1], anomaly_sigma=2.0)
        assert score == 1.0

    def test_normal_cost_passes(self):
        costs = [0.10] * 10
        score, _ = _metric_cost_anomaly(_session(total_cost_usd=0.10), costs, anomaly_sigma=2.0)
        assert score == 1.0

    def test_outlier_cost_fails(self):
        # use costs with variance so stdev > 0; session cost is 50x the mean
        costs = [0.08, 0.09, 0.10, 0.11, 0.12, 0.10, 0.09, 0.11, 0.08, 0.10]
        score, detail = _metric_cost_anomaly(_session(total_cost_usd=5.0), costs, anomaly_sigma=2.0)
        assert score < 1.0
        assert detail["z_score"] > 2.0


class TestMetricPipelineCoverage:
    def test_no_required_agents_returns_one(self):
        score, _ = _metric_pipeline_coverage([], required_agents=[])
        assert score == 1.0

    def test_all_present(self):
        traces = [_tool_trace("market", "t"), _tool_trace("research", "t")]
        score, detail = _metric_pipeline_coverage(traces, ["market", "research"])
        assert score == 1.0
        assert detail["missing"] == []

    def test_partial_coverage(self):
        traces = [_tool_trace("market", "t")]
        score, detail = _metric_pipeline_coverage(traces, ["market", "research"])
        assert score == 0.5
        assert "research" in detail["missing"]

    def test_shadow_agent_counts(self):
        traces = [_tool_trace("market_shadow", "t"), _tool_trace("research", "t")]
        score, _ = _metric_pipeline_coverage(traces, ["market", "research"])
        assert score == 1.0


class TestMetricAgentCompletion:
    def test_no_llm_returns_zero(self):
        traces = [_tool_trace("a", "t", outcome="error")]
        score, _ = _metric_agent_completion(traces)
        assert score == 0.0

    def test_llm_and_tool_success_returns_one(self):
        traces = [_llm_trace("a"), _tool_trace("a", "t")]
        score, _ = _metric_agent_completion(traces)
        assert score == 1.0

    def test_llm_only_tool_failed_returns_0_3(self):
        traces = [_llm_trace("a"), _tool_trace("a", "t", outcome="error")]
        score, _ = _metric_agent_completion(traces)
        assert score == 0.3

    def test_decision_trace_counts_as_llm(self):
        traces = [{"agent": "a", "step_type": "decision", "outcome": "approved",
                   "tool_name": None, "latency_ms": 10, "created_at": "2026-06-08T10:00:00",
                   "tokens_input": 0, "tokens_output": 0}]
        score, _ = _metric_agent_completion(traces)
        assert score == 1.0


class TestMetricSessionFieldRatio:
    def test_denominator_zero(self):
        # both zero → neutral (1.0); non-zero numerator with zero denominator → 0.0
        s_both_zero = _session(trades_proposed=0, trades_executed=0)
        score, _ = _metric_session_field_ratio(s_both_zero, "trades_executed", "trades_proposed", 0.5)
        assert score == 1.0

    def test_ratio_at_threshold_returns_one(self):
        s = _session(trades_executed=1, trades_proposed=2)
        score, _ = _metric_session_field_ratio(s, "trades_executed", "trades_proposed", 0.5)
        assert score == 1.0

    def test_below_threshold(self):
        s = _session(trades_executed=1, trades_proposed=10)
        score, _ = _metric_session_field_ratio(s, "trades_executed", "trades_proposed", 0.5)
        assert score < 1.0


# ── run_evals_from_config integration tests ───────────────────────────────────

class TestRunEvalsFromConfig:
    def test_returns_empty_for_empty_configs(self):
        results = run_evals_from_config(_session(), _healthy_traces(), [])
        assert results == []

    def test_skips_non_l3_rows(self):
        ec = _mk_ec("my_eval", "agent_presence", agent="market")
        ec["layer"] = 4  # L4 — should be skipped
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert results == []

    def test_skips_non_rule_rows(self):
        ec = _mk_ec("my_eval", "agent_presence", agent="market")
        ec["eval_type"] = "llm_judge"  # not rule — should be skipped
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert results == []

    def test_skips_disabled_rows(self):
        ec = _mk_ec("my_eval", "agent_presence", agent="market")
        ec["enabled"] = False
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert results == []

    def test_scores_in_range(self):
        ecs = [
            _mk_ec("market_presence", "agent_presence",
                   agent="market", agent_pattern="market", agent_match="exact"),
            _mk_ec("research_tools",  "tool_success_rate",
                   agent="research", agent_pattern="research", agent_match="exact"),
        ]
        results = run_evals_from_config(_session(), _healthy_traces(), ecs)
        assert len(results) == 2
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_db_row_has_correct_fields(self):
        ec = _mk_ec("market_presence", "agent_presence",
                    agent="market", agent_pattern="market", agent_match="exact")
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert len(results) == 1
        expected = {"id", "session_id", "agent", "eval_name", "score", "passed", "threshold", "detail"}
        assert set(results[0].to_db_row("sess-test").keys()) == expected

    def test_agent_pattern_any_matches_all(self):
        ec = _mk_ec("all_agents_presence", "agent_presence", agent_match="any")
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert len(results) == 1
        assert results[0].score > 0.0

    def test_agent_pattern_prefix_match(self):
        ec = _mk_ec("research_completion", "agent_completion",
                    agent_pattern="research", agent_match="prefix")
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert len(results) == 1

    def test_threshold_key_override(self):
        ec = _mk_ec("token_check", "token_budget",
                    agent_pattern="research", agent_match="prefix",
                    threshold_key="token_spiral_threshold")
        pcfg = {"token_spiral_threshold": 500_000}
        results = run_evals_from_config(_session(), _healthy_traces(), [ec], pcfg)
        assert len(results) == 1
        assert results[0].passed  # threshold now very high → passes easily

    def test_exit_quality_reads_pipeline_config(self):
        ec = _mk_ec("exit_quality", "exit_quality", threshold=0.5)
        pcfg = {
            "good_exits": ["trade_executed"],
            "partial_exits": ["no_candidates"],
            "bad_exits": ["error"],
        }
        sess = _session(terminal_reason="trade_executed", trades_executed=1)
        results = run_evals_from_config(sess, [], [ec], pcfg)
        assert results[0].passed

    def test_pipeline_coverage_reads_required_agents(self):
        ec = _mk_ec("coverage", "pipeline_coverage", threshold=0.8)
        pcfg = {"required_agents": ["market", "research", "risk", "orchestrator"]}
        results = run_evals_from_config(_session(), _healthy_traces(), [ec], pcfg)
        assert len(results) == 1
        assert results[0].score == 1.0  # all 4 present in healthy traces

    def test_unknown_metric_type_skipped(self):
        ec = _mk_ec("mystery", "nonexistent_metric_type")
        results = run_evals_from_config(_session(), _healthy_traces(), [ec])
        assert results == []

    def test_multiple_evals_run_all(self):
        ecs = [
            _mk_ec("presence",  "agent_presence",    agent_match="any"),
            _mk_ec("tools",     "tool_success_rate", agent_match="any"),
            _mk_ec("diversity", "tool_diversity",    agent_match="any"),
            _mk_ec("exit",      "exit_quality"),
            _mk_ec("coverage",  "pipeline_coverage"),
        ]
        results = run_evals_from_config(_session(), _healthy_traces(), ecs)
        assert len(results) == 5


# ── @register_eval tests ──────────────────────────────────────────────────────

class TestRegisterEval:
    def test_global_registration_callable(self):
        @register_eval("test_custom_global")
        def my_eval(traces, session, ec, cfg):
            return EvalResult("test_custom_global", "session", 0.9, True, 0.5, {})

        reg = get_registry()
        assert "test_custom_global" in reg

    def test_workflow_scoped_registration(self):
        @register_eval("scoped_eval", workflow_id="wf-abc")
        def my_eval(traces, session, ec, cfg):
            return EvalResult("scoped_eval", "session", 0.8, True, 0.5, {})

        # visible to wf-abc
        reg_wf = get_registry(workflow_id="wf-abc")
        assert "scoped_eval" in reg_wf
        # not visible globally (no workflow prefix)
        reg_global = get_registry(workflow_id=None)
        assert "scoped_eval" not in reg_global

    def test_custom_metric_dispatched(self):
        @register_eval("dispatch_test_eval")
        def my_eval(traces, session, ec, cfg):
            return EvalResult("dispatch_test_eval", "session", 0.75, True, 0.5,
                              {"source": "custom"})

        ec = _mk_ec("dispatch_test_eval", "custom")
        reg = get_registry()
        results = run_evals_from_config(_session(), [], [ec], registry=reg)
        assert len(results) == 1
        assert results[0].score == 0.75
        assert results[0].detail["source"] == "custom"

    def test_unregistered_custom_metric_skipped(self):
        ec = _mk_ec("not_registered_xyz", "custom")
        results = run_evals_from_config(_session(), [], [ec])
        assert results == []


# ── Pattern detector tests ────────────────────────────────────────────────────

class TestRunAllDetectors:
    def test_no_incidents_for_healthy_session(self):
        sess   = _session()
        traces = _healthy_traces()
        assert run_all_detectors(sess, traces, []) == []

    def test_tool_timeout_fires(self):
        sess      = _session(trades_executed=0, terminal_reason="error", total_cost_usd=0.30)
        traces    = _timeout_traces()
        incidents = run_all_detectors(sess, traces, [])
        names = [i.pattern_name for i in incidents]
        assert "Tool Timeout Loop" in names

    def test_incident_db_row_has_required_fields(self):
        sess      = _session(trades_executed=0, terminal_reason="error")
        incidents = run_all_detectors(sess, _timeout_traces(), [])
        assert incidents
        required = {"id", "session_id", "pattern_name", "severity", "root_cause",
                    "call_stack", "failed_evals", "cost_wasted", "tokens_wasted",
                    "fix_suggestion", "is_simulated"}
        assert required.issubset(incidents[0].to_db_row().keys())

    def test_failed_evals_include_role_cause(self):
        # Every failed_eval entry must have role='cause' so the Argus UI can render
        # the call chain without needing quality attribution data.
        sess      = _session(trades_executed=0, terminal_reason="error")
        incidents = run_all_detectors(sess, _timeout_traces(), [])
        for inc in incidents:
            for entry in inc.failed_evals:
                if isinstance(entry, dict) and "eval_name" in entry:
                    assert entry.get("role") == "cause", (
                        f"failed_eval entry missing role='cause' in {inc.pattern_name}: {entry}"
                    )

    def test_context_spiral_uses_analysis_agent_config(self):
        spiral_traces = [
            {"agent": "analyser", "step_type": "llm_call", "tool_name": None,
             "outcome": "success", "tokens_input": 25_000, "tokens_output": 20_000,
             "latency_ms": 2000, "created_at": "2026-06-08T10:00:30"},
        ]
        sess = _session(trades_executed=0, total_cost_usd=0.50)
        pcfg = {"analysis_agent": "analyser", "token_spiral_threshold": 40_000}
        incidents = run_all_detectors(sess, spiral_traces, [], pipeline_config=pcfg)
        names = [i.pattern_name for i in incidents]
        assert "Context Spiral" in names

    def test_pipeline_break_uses_pipeline_order_config(self):
        traces = [
            _llm_trace("ingestion"),  # first in order
            # "processor" (last in order) never ran
        ]
        sess = _session(trades_executed=0)
        pcfg = {"pipeline_order": ["ingestion", "processor"]}
        incidents = run_all_detectors(sess, traces, [], pipeline_config=pcfg)
        names = [i.pattern_name for i in incidents]
        assert "Pipeline Break" in names

    def test_handoff_schema_break_uses_handoff_pairs_config(self):
        traces = [
            _llm_trace("ingest"),  # src completes
            {"agent": "transform", "step_type": "tool_call", "tool_name": "parse",
             "outcome": "error", "error": "schema mismatch", "latency_ms": 50,
             "created_at": "2026-06-08T10:01:00", "tokens_input": 0, "tokens_output": 0},
        ]
        sess = _session(trades_executed=0)
        pcfg = {"handoff_pairs": [["ingest", "transform"]]}
        incidents = run_all_detectors(sess, traces, [], pipeline_config=pcfg)
        names = [i.pattern_name for i in incidents]
        assert "Handoff Schema Break" in names


# ── RCA engine tests ──────────────────────────────────────────────────────────

class TestRcaEngine:
    def _timeout_incident(self) -> Incident:
        sess      = _session(trades_executed=0, terminal_reason="error")
        incidents = run_all_detectors(sess, _timeout_traces(), [])
        return next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")

    def test_annotated_call_stack_marks_root(self):
        inc       = self._timeout_incident()
        annotated = build_annotated_call_stack(_timeout_traces(), inc)
        roots     = [a for a in annotated if a["is_root"]]
        assert len(roots) == 1

    def test_annotated_call_stack_passes_through_extra_fields(self):
        traces = [{**t, "tenant_id": "t"} for t in _timeout_traces()]
        inc = Incident(
            session_id="s", pattern_name="Pipeline Break", severity="warning",
            root_cause="", call_stack=[], failed_evals=[],
        )
        result = build_annotated_call_stack(traces, inc)
        assert all("tenant_id" in r for r in result)

    def test_fix_suggestion_mentions_tool(self):
        inc = self._timeout_incident()
        fix = generate_fix_suggestion(inc, _timeout_traces())
        assert "get_stock_data" in fix
        assert "timeout" in fix.lower()

    def test_summarize_incident_duration(self):
        inc     = self._timeout_incident()
        summary = summarize_incident(inc, _session())
        assert summary["duration_s"] == 300


# ── run_and_persist tests ─────────────────────────────────────────────────────

class TestRunAndPersist:
    def test_run_evals_and_persist_without_db_returns_results(self):
        ecs = [
            _mk_ec("presence", "agent_presence",
                   agent_pattern="market", agent_match="exact"),
        ]
        results = run_evals_and_persist(_session(), _healthy_traces(), ecs)
        assert len(results) == 1

    def test_run_evals_and_persist_multiple_evals(self):
        ecs = [
            _mk_ec("presence",  "agent_presence",    agent_match="any"),
            _mk_ec("tools",     "tool_success_rate", agent_match="any"),
            _mk_ec("exit",      "exit_quality"),
        ]
        results = run_evals_and_persist(_session(), _healthy_traces(), ecs)
        assert len(results) == 3

    def test_run_detectors_and_persist_without_db_returns_list(self):
        sess      = _session()
        incidents = run_detectors_and_persist(sess, _healthy_traces(), [])
        assert isinstance(incidents, list)
        assert incidents == []  # healthy session

    def test_eval_result_eval_name_matches_config(self):
        ec      = _mk_ec("my_tool_rate", "tool_success_rate", agent_match="any")
        results = run_evals_and_persist(_session(), _healthy_traces(), [ec])
        assert results[0].eval_name == "my_tool_rate"
