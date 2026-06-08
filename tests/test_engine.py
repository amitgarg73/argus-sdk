"""
Tests for argus.engine — verifies the migrated engine modules work correctly
under the argus.* import path and accept ag_* table field shapes.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from argus.engine import (
    EvalResult, Incident,
    run_all_evals, run_all_detectors,
    build_annotated_call_stack, generate_fix_suggestion, summarize_incident,
)


def _session(**kw) -> dict:
    base = {
        "id":                  "sess-test",
        "total_cost_usd":      0.10,
        "total_tokens_input":  3000,
        "total_tokens_output": 1500,
        "trades_executed":     2,
        "trades_proposed":     2,
        "terminal_reason":     "converged",
        "started_at":          "2026-06-08T10:00:00",
        "completed_at":        "2026-06-08T10:05:00",
        # ag_* extras — must be silently ignored
        "tenant_id":           "tenant-abc",
        "workflow_id":         "workflow-xyz",
    }
    return {**base, **kw}


def _healthy_traces() -> list[dict]:
    return [
        {"agent": "market",       "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None, "tenant_id": "t"},
        {"agent": "research",     "step_type": "llm_call",  "tool_name": None,
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:00",
         "tokens_input": 300, "tokens_output": 150, "entity_id": None},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_news",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_atr",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:20",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "risk",         "step_type": "tool_call", "tool_name": "check_risk",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:02:00",
         "tokens_input": 50, "tokens_output": 30, "entity_id": None},
        {"agent": "risk",         "step_type": "decision",  "tool_name": None,
         "outcome": "approved",  "error": None, "created_at": "2026-06-08T10:02:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": None},
        {"agent": "orchestrator", "step_type": "decision",  "tool_name": None,
         "outcome": "execute",   "error": None, "created_at": "2026-06-08T10:03:00",
         "tokens_input": 150, "tokens_output": 80, "entity_id": None},
    ]


def _timeout_traces() -> list[dict]:
    return [
        {"agent": "market",   "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out",
         "created_at": "2026-06-08T10:01:00", "tokens_input": 0, "tokens_output": 0},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out",
         "created_at": "2026-06-08T10:01:30", "tokens_input": 0, "tokens_output": 0},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out",
         "created_at": "2026-06-08T10:02:00", "tokens_input": 0, "tokens_output": 0},
    ]


class TestRunAllEvals:
    def test_returns_17_evals(self):
        evals = run_all_evals(_session(), _healthy_traces())
        assert len(evals) == 17

    def test_all_pass_for_healthy_session(self):
        evals = run_all_evals(_session(), _healthy_traces())
        failed = [e for e in evals if not e.passed]
        assert not failed, [e.eval_name for e in failed]

    def test_scores_in_range(self):
        for e in run_all_evals(_session(), _healthy_traces()):
            assert 0.0 <= e.score <= 1.0

    def test_accepts_ag_extra_fields(self):
        sess = {**_session(), "pipeline_id": "pipe-xyz", "extra": "ignored"}
        assert len(run_all_evals(sess, _healthy_traces())) == 17

    def test_db_row_has_correct_fields(self):
        expected = {"id", "session_id", "agent", "eval_name", "score", "passed", "threshold", "detail"}
        for e in run_all_evals(_session(), _healthy_traces()):
            assert set(e.to_db_row("s").keys()) == expected


class TestRunAllDetectors:
    def test_no_incidents_for_healthy_session(self):
        sess   = _session()
        traces = _healthy_traces()
        evals  = run_all_evals(sess, traces)
        assert run_all_detectors(sess, traces, evals) == []

    def test_tool_timeout_fires(self):
        sess   = _session(trades_executed=0, terminal_reason="error", total_cost_usd=0.30)
        traces = _timeout_traces()
        evals  = run_all_evals(sess, traces)
        incidents = run_all_detectors(sess, traces, evals)
        names = [i.pattern_name for i in incidents]
        assert "Tool Timeout Loop" in names

    def test_incident_db_row_has_required_fields(self):
        sess   = _session(trades_executed=0, terminal_reason="error")
        traces = _timeout_traces()
        evals  = run_all_evals(sess, traces)
        incidents = run_all_detectors(sess, traces, evals)
        assert incidents
        required = {"id", "session_id", "pattern_name", "severity", "root_cause",
                    "call_stack", "failed_evals", "cost_wasted", "tokens_wasted",
                    "fix_suggestion", "is_simulated"}
        assert required.issubset(incidents[0].to_db_row().keys())


class TestRcaEngine:
    def _timeout_incident(self) -> Incident:
        sess   = _session(trades_executed=0, terminal_reason="error")
        traces = _timeout_traces()
        evals  = run_all_evals(sess, traces)
        incidents = run_all_detectors(sess, traces, evals)
        return next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")

    def test_annotated_call_stack_marks_root(self):
        inc = self._timeout_incident()
        annotated = build_annotated_call_stack(_timeout_traces(), inc)
        roots = [a for a in annotated if a["is_root"]]
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
        inc = self._timeout_incident()
        summary = summarize_incident(inc, _session())
        assert summary["duration_s"] == 300


class TestRunAndPersist:
    def test_run_evals_and_persist_without_db_returns_results(self):
        from argus.engine import run_evals_and_persist
        results = run_evals_and_persist(_session(), _healthy_traces())
        assert len(results) == 17

    def test_run_detectors_and_persist_without_db_returns_list(self):
        from argus.engine import run_detectors_and_persist
        sess   = _session()
        traces = _healthy_traces()
        evals  = run_all_evals(sess, traces)
        incidents = run_detectors_and_persist(sess, traces, evals)
        assert isinstance(incidents, list)
