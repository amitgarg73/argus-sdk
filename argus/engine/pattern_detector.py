"""
Pattern detector — rule-based detection over traces + eval results.
Returns Incident objects. Each detector fires on one named pattern or None.
"""
from __future__ import annotations

import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter

from argus.engine.eval_engine import EvalResult


def _real_error(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None

TOOL_RETRY_THRESHOLD       = 3
CONTEXT_SPIRAL_TOKENS      = 40_000
COST_ANOMALY_SIGMA         = 2.0
COST_ANOMALY_SIGMA_INC     = COST_ANOMALY_SIGMA
HYPERACTIVE_POLL_THRESHOLD = 6
FABRICATION_LATENCY_MS     = 50
FABRICATION_MIN_COUNT      = 2
_HTTP_ERROR_CODES          = {"400", "401", "403", "429", "500", "502", "503"}

CB_CONFIG: dict[str, list[tuple[str, float]]] = {
    "research": [("tool_success_rate", 0.80), ("completion", 0.70)],
    "risk":     [("assessment_complete", 1.0)],
}

_DETECTOR_DEFAULTS: dict = {
    "tool_retry_threshold":       TOOL_RETRY_THRESHOLD,
    "token_spiral_threshold":     CONTEXT_SPIRAL_TOKENS,
    "hyperactive_poll_threshold": HYPERACTIVE_POLL_THRESHOLD,
    "fabrication_latency_ms":     FABRICATION_LATENCY_MS,
    "fabrication_min_count":      FABRICATION_MIN_COUNT,
    "cost_anomaly_sigma":         COST_ANOMALY_SIGMA_INC,
    "cb_config":                  CB_CONFIG,
}


def _build_detector_config(pipeline_config: dict | None) -> dict:
    cfg = dict(_DETECTOR_DEFAULTS)
    if pipeline_config:
        cfg.update(pipeline_config)
    return cfg


@dataclass
class Incident:
    session_id:     str
    pattern_name:   str
    severity:       str
    root_cause:     str
    call_stack:     list[dict]
    failed_evals:   list[dict]
    cost_wasted:    float = 0.0
    tokens_wasted:  int   = 0
    fix_suggestion: str   = ""
    is_simulated:   bool  = False

    def to_db_row(self) -> dict:
        return {
            "id":             str(uuid.uuid4()),
            "session_id":     self.session_id,
            "pattern_name":   self.pattern_name,
            "severity":       self.severity,
            "root_cause":     self.root_cause,
            "call_stack":     self.call_stack,
            "failed_evals":   self.failed_evals,
            "cost_wasted":    round(self.cost_wasted, 4),
            "tokens_wasted":  self.tokens_wasted,
            "fix_suggestion": self.fix_suggestion,
            "is_simulated":   self.is_simulated,
        }


def _failed_eval_rows(evals: list[EvalResult]) -> list[dict]:
    return [
        {"agent": e.agent, "eval_name": e.eval_name,
         "score": e.score, "threshold": e.threshold}
        for e in evals if not e.passed
    ]


def _trace_to_stack_frame(t: dict) -> dict:
    return {
        "agent":       t.get("agent", ""),
        "step_type":   t.get("step_type", ""),
        "tool_name":   t.get("tool_name", ""),
        "outcome":     t.get("outcome", ""),
        "error":       t.get("error", ""),
        "latency_ms":  t.get("latency_ms", 0),
        "tokens":      (t.get("tokens_input") or 0) + (t.get("tokens_output") or 0),
        "created_at":  t.get("created_at", ""),
    }


# ── Individual pattern detectors ──────────────────────────────────────────────

def detect_tool_timeout_loop(
    session: dict, traces: list[dict], evals: list[EvalResult], *,
    retry_threshold: int = TOOL_RETRY_THRESHOLD,
) -> Incident | None:
    """Fires when the same tool_name has retry_threshold+ error traces."""
    tool_errors: dict[str, list[dict]] = {}
    for t in sorted(traces, key=lambda x: x.get("created_at", "")):
        if (t.get("step_type") or "") == "tool_call" and (
            t.get("outcome") == "error" or bool(_real_error(t.get("error")))
        ):
            tool = t.get("tool_name") or "unknown_tool"
            tool_errors.setdefault(tool, []).append(t)

    for tool_name, errors in tool_errors.items():
        if len(errors) >= retry_threshold:
            session_id = session.get("id", "")
            cost       = float(session.get("total_cost_usd") or 0)
            tokens     = int((session.get("total_tokens_input") or 0) +
                             (session.get("total_tokens_output") or 0))
            agent      = errors[0].get("agent", "unknown")
            err_msg    = errors[0].get("error") or "unknown error"
            return Incident(
                session_id   = session_id,
                pattern_name = "Tool Timeout Loop",
                severity     = "critical",
                root_cause   = (
                    f"{tool_name} failed {len(errors)} times in {agent} agent. "
                    f"Error: {err_msg[:120]}"
                ),
                call_stack   = [_trace_to_stack_frame(t) for t in errors],
                failed_evals = _failed_eval_rows(evals),
                cost_wasted  = cost,
                tokens_wasted= tokens,
                fix_suggestion = (
                    f"Add max_retries=2 and timeout=10s to {tool_name} "
                    f"in the {agent} agent tool configuration. "
                    f"Consider exponential backoff between retries."
                ),
            )
    return None


def detect_context_spiral(
    session: dict, traces: list[dict], evals: list[EvalResult], *,
    spiral_tokens: int = CONTEXT_SPIRAL_TOKENS,
    analysis_agent: str = "research",
) -> Incident | None:
    """Fires when analysis_agent burns > spiral_tokens tokens with 0 trades executed."""
    agent_lower = analysis_agent.lower()
    agent_tokens = sum(
        (t.get("tokens_input") or 0) + (t.get("tokens_output") or 0)
        for t in traces
        if (t.get("agent") or "").lower() == agent_lower
    )
    trades = int(session.get("trades_executed") or 0)

    if agent_tokens > spiral_tokens and trades == 0:
        session_id   = session.get("id", "")
        cost         = float(session.get("total_cost_usd") or 0)
        agent_traces = sorted(
            [t for t in traces if (t.get("agent") or "").lower() == agent_lower],
            key=lambda x: x.get("created_at", "")
        )
        return Incident(
            session_id   = session_id,
            pattern_name = "Context Spiral",
            severity     = "warning",
            root_cause   = (
                f"{analysis_agent.capitalize()} agent consumed {agent_tokens:,} tokens "
                f"with 0 trades produced. Agent gathered context "
                f"without reaching a decision."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in agent_traces[-20:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = cost,
            tokens_wasted= agent_tokens,
            fix_suggestion = (
                f"Add a hard token budget to the {analysis_agent} agent "
                f"(e.g. max_tokens={spiral_tokens // 2}). "
                "Force a decision step if budget is reached without a recommendation."
            ),
        )
    return None


def detect_pipeline_break(
    session: dict, traces: list[dict], evals: list[EvalResult],
    pipeline_order: list[str] | None = None,
) -> Incident | None:
    """Fires when the first agent in pipeline_order ran but the last agent did not."""
    order = [a.lower() for a in (pipeline_order or ["research", "orchestrator"])]
    if len(order) < 2:
        return None
    first_agent = order[0]
    last_agent  = order[-1]

    agents_seen: set[str] = {(t.get("agent") or "").lower() for t in traces}
    # expand shadow/prefixed agent names (e.g. market_shadow → market)
    expanded = set(agents_seen)
    for name in agents_seen:
        for agent in order:
            if name.startswith(agent + "_") or name == agent + "_shadow":
                expanded.add(agent)

    first_ran = first_agent in expanded
    last_ran  = last_agent in expanded

    if first_ran and not last_ran:
        session_id    = session.get("id", "")
        cost          = float(session.get("total_cost_usd") or 0)
        first_traces  = sorted(
            [t for t in traces if (t.get("agent") or "").lower() == first_agent],
            key=lambda x: x.get("created_at", "")
        )
        last_first = first_traces[-1] if first_traces else {}
        return Incident(
            session_id   = session_id,
            pattern_name = "Pipeline Break",
            severity     = "critical",
            root_cause   = (
                f"{first_agent.capitalize()} agent completed but {last_agent} never started. "
                f"Last {first_agent} step: {last_first.get('step_type', 'unknown')} "
                f"(outcome: {last_first.get('outcome', 'unknown')})"
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in first_traces[-10:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = cost,
            tokens_wasted= int((session.get("total_tokens_input") or 0) +
                               (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                f"Check for uncaught exceptions between {first_agent} and {last_agent} steps. "
                f"Add try/except around the {first_agent} -> {last_agent} handoff. "
                f"Ensure {first_agent} agent errors are propagated, not silently swallowed."
            ),
        )
    return None


def detect_cost_anomaly(
    session: dict, traces: list[dict], evals: list[EvalResult],
    recent_costs: list[float] | None = None, *,
    anomaly_sigma: float = COST_ANOMALY_SIGMA_INC,
) -> Incident | None:
    """Fires when session cost > mean + anomaly_sigma * stdev of recent sessions."""
    if not recent_costs or len(recent_costs) < 5:
        return None

    cost  = float(session.get("total_cost_usd") or 0)
    mean  = statistics.mean(recent_costs)
    stdev = statistics.stdev(recent_costs) if len(recent_costs) > 1 else 0
    z     = (cost - mean) / stdev if stdev > 0 else 0

    if z > anomaly_sigma:
        session_id = session.get("id", "")
        sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
        return Incident(
            session_id   = session_id,
            pattern_name = "Cost Anomaly",
            severity     = "warning",
            root_cause   = (
                f"Session cost USD{cost:.4f} is {z:.1f} standard deviations above "
                f"the mean (USD{mean:.4f}). Significantly higher than typical sessions."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in sorted_traces[-15:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = max(0.0, cost - mean),
            tokens_wasted= 0,
            fix_suggestion = (
                f"Investigate which agent consumed the excess cost. "
                f"Mean session cost is USD{mean:.4f}. "
                f"Check token usage by agent in Session Deep Dive."
            ),
        )
    return None


def detect_silent_exit(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """Fires when session ends with 0 trades and no terminal reason (and no tool errors)."""
    trades = int(session.get("trades_executed") or 0)
    reason = session.get("terminal_reason") or ""

    has_tool_errors = any(
        (t.get("step_type") or "") == "tool_call" and (
            t.get("outcome") == "error" or bool(_real_error(t.get("error")))
        )
        for t in traces
    )
    if trades == 0 and not reason and not has_tool_errors and traces:
        session_id = session.get("id", "")
        sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
        return Incident(
            session_id   = session_id,
            pattern_name = "Silent Exit",
            severity     = "info",
            root_cause   = (
                "Session ended with 0 trades and no terminal reason logged. "
                "No tool errors detected. Likely a logic path that skips "
                "both trade entry and explicit rejection."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in sorted_traces[-10:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = float(session.get("total_cost_usd") or 0),
            tokens_wasted= int((session.get("total_tokens_input") or 0) +
                               (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                "Add explicit terminal_reason logging for all exit paths "
                "in the orchestrator (e.g. 'no_opportunity', 'risk_rejected', "
                "'market_closed'). Every session should record why it ended."
            ),
        )
    return None


def detect_empty_result_loop(
    session: dict, traces: list[dict], evals: list[EvalResult], *,
    retry_threshold: int = TOOL_RETRY_THRESHOLD,
) -> Incident | None:
    """Fires when same tool is called retry_threshold+ times successfully but session fails."""
    tool_calls = [
        t for t in traces
        if (t.get("step_type") or "") == "tool_call"
        and (t.get("outcome") or "") == "success"
    ]
    tool_counts = Counter(t.get("tool_name") or "unknown" for t in tool_calls)
    trades      = int(session.get("trades_executed") or 0)

    for tool_name, count in tool_counts.items():
        if count >= retry_threshold and trades == 0:
            session_id = session.get("id", "")
            relevant   = [t for t in tool_calls if t.get("tool_name") == tool_name]
            return Incident(
                session_id   = session_id,
                pattern_name = "Empty Result Loop",
                severity     = "warning",
                root_cause   = (
                    f"{tool_name} called {count} times successfully but "
                    f"produced no actionable output (0 trades). "
                    f"Tool may be returning empty or low-quality results."
                ),
                call_stack   = [_trace_to_stack_frame(t) for t in relevant],
                failed_evals = _failed_eval_rows(evals),
                cost_wasted  = float(session.get("total_cost_usd") or 0),
                tokens_wasted= int((session.get("total_tokens_input") or 0) +
                                   (session.get("total_tokens_output") or 0)),
                fix_suggestion = (
                    f"Add result quality validation after {tool_name} calls. "
                    f"If result is empty or below quality threshold, "
                    f"fail fast rather than retrying with same parameters."
                ),
            )
    return None


def detect_hyperactive_polling(
    session: dict, traces: list[dict], evals: list[EvalResult], *,
    poll_threshold: int = HYPERACTIVE_POLL_THRESHOLD,
) -> Incident | None:
    """Fires when the same tool is called poll_threshold+ times successfully in one session."""
    tool_success: dict[str, list[dict]] = {}
    for t in sorted(traces, key=lambda x: x.get("created_at", "")):
        if (t.get("step_type") or "") == "tool_call" and t.get("outcome") != "error":
            name = t.get("tool_name") or "unknown"
            tool_success.setdefault(name, []).append(t)

    for tool_name, calls in tool_success.items():
        if len(calls) >= poll_threshold:
            session_id = session.get("id", "")
            agent = calls[0].get("agent", "unknown")
            total_latency = sum(c.get("latency_ms") or 0 for c in calls)
            return Incident(
                session_id    = session_id,
                pattern_name  = "Hyperactive Polling Loop",
                severity      = "warning",
                root_cause    = (
                    f"{tool_name} called {len(calls)} times successfully by {agent} agent. "
                    f"Agent is polling without reaching a decision threshold. "
                    f"Total time in tool calls: {total_latency:,}ms."
                ),
                call_stack    = [_trace_to_stack_frame(t) for t in calls],
                failed_evals  = _failed_eval_rows(evals),
                cost_wasted   = float(session.get("total_cost_usd") or 0),
                tokens_wasted = int((session.get("total_tokens_input") or 0) +
                                    (session.get("total_tokens_output") or 0)),
                fix_suggestion = (
                    f"Add a max_iterations cap to {agent} agent's {tool_name} loop "
                    f"(e.g. max_calls=3). Force a decision step once the cap is hit. "
                    f"Consider caching results between calls to avoid redundant fetches."
                ),
            )
    return None


def detect_tool_fabrication(
    session: dict, traces: list[dict], evals: list[EvalResult], *,
    fab_latency_ms: float = FABRICATION_LATENCY_MS,
    fab_min_count: int = FABRICATION_MIN_COUNT,
) -> Incident | None:
    """Fires when fab_min_count+ tool_call traces complete in under fab_latency_ms."""
    suspects = [
        t for t in traces
        if (t.get("step_type") or "") == "tool_call"
        and t.get("outcome") != "error"
        and 0 < (t.get("latency_ms") or 0) < fab_latency_ms
    ]
    if len(suspects) >= fab_min_count:
        session_id = session.get("id", "")
        agents = list({t.get("agent", "unknown") for t in suspects})
        tools  = list({t.get("tool_name", "unknown") for t in suspects})
        return Incident(
            session_id    = session_id,
            pattern_name  = "Tool Call Fabrication",
            severity      = "critical",
            root_cause    = (
                f"{len(suspects)} tool call(s) completed in under {fab_latency_ms}ms "
                f"with success status. External API calls take 100ms+. "
                f"Affected agents: {agents}. Tools: {tools}."
            ),
            call_stack    = [_trace_to_stack_frame(t) for t in suspects],
            failed_evals  = _failed_eval_rows(evals),
            cost_wasted   = float(session.get("total_cost_usd") or 0),
            tokens_wasted = 0,
            fix_suggestion = (
                "Verify that tool calls are making real external requests. "
                "Add response validation: check that tool output is consistent with "
                "what the actual API returns (schema, latency, value ranges). "
                "Enable request logging in tool wrappers to confirm HTTP calls are made."
            ),
        )
    return None


def detect_handoff_schema_break(
    session: dict, traces: list[dict], evals: list[EvalResult],
    handoff_pairs: list[list[str]] | None = None,
) -> Incident | None:
    """Fires when source agent completes but target agent's first trace is an error."""
    pairs = handoff_pairs or [["research", "risk"]]
    sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))

    for pair in pairs:
        if len(pair) < 2:
            continue
        src, tgt = pair[0].lower(), pair[1].lower()

        src_traces = [t for t in sorted_traces
                      if (t.get("agent") or "").lower().startswith(src)]
        tgt_traces = [t for t in sorted_traces
                      if (t.get("agent") or "").lower() == tgt]

        if not src_traces or not tgt_traces:
            continue

        src_ok = any(
            (t.get("step_type") or "") in ("llm_call", "agent_message", "decision")
            and t.get("outcome") != "error"
            for t in src_traces
        )
        if not src_ok:
            continue

        first_tgt   = tgt_traces[0]
        tgt_errored = (
            first_tgt.get("outcome") == "error"
            or bool(_real_error(first_tgt.get("error")))
        )
        if not tgt_errored:
            continue

        session_id = session.get("id", "")
        err_msg    = first_tgt.get("error") or "unknown schema error"
        return Incident(
            session_id    = session_id,
            pattern_name  = "Handoff Schema Break",
            severity      = "critical",
            root_cause    = (
                f"{src.capitalize()} agent completed successfully but {tgt} agent errored on its "
                f"first trace. {src.capitalize()} output did not match {tgt} agent's expected schema. "
                f"Error at handoff: {str(err_msg)[:120]}"
            ),
            call_stack    = (
                [_trace_to_stack_frame(t) for t in src_traces[-5:]]
                + [_trace_to_stack_frame(first_tgt)]
            ),
            failed_evals  = _failed_eval_rows(evals),
            cost_wasted   = float(session.get("total_cost_usd") or 0),
            tokens_wasted = int((session.get("total_tokens_input") or 0) +
                                (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                f"Add schema validation at the {src} -> {tgt} handoff. "
                f"Define a typed contract (Pydantic model or TypedDict) for what {src} "
                f"must return before passing to {tgt}. Fail fast with a clear error if the "
                f"contract is violated, rather than letting {tgt} agent crash on bad input."
            ),
        )
    return None


def detect_error_misinterpretation(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """Fires when tool calls return HTTP error codes but the agent continues with LLM calls."""
    sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
    mishandled: list[dict] = []

    for i, t in enumerate(sorted_traces):
        if (t.get("step_type") or "") != "tool_call":
            continue
        err = str(t.get("error") or "")
        if not any(code in err for code in _HTTP_ERROR_CODES):
            continue
        subsequent = sorted_traces[i + 1:]
        continued = any(
            s.get("agent") == t.get("agent")
            and (s.get("step_type") or "") == "llm_call"
            for s in subsequent
        )
        if continued:
            mishandled.append(t)

    if len(mishandled) >= 2:
        session_id   = session.get("id", "")
        agents_hit   = list({t.get("agent", "unknown") for t in mishandled})
        error_codes  = [str(t.get("error") or "")[:60] for t in mishandled]
        return Incident(
            session_id    = session_id,
            pattern_name  = "Error Misinterpretation",
            severity      = "warning",
            root_cause    = (
                f"{len(mishandled)} HTTP error(s) received but agent(s) {agents_hit} "
                f"continued processing instead of aborting or retrying correctly. "
                f"Errors: {error_codes[:3]}"
            ),
            call_stack    = [_trace_to_stack_frame(t) for t in mishandled],
            failed_evals  = _failed_eval_rows(evals),
            cost_wasted   = float(session.get("total_cost_usd") or 0),
            tokens_wasted = int((session.get("total_tokens_input") or 0) +
                                (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                "Add explicit HTTP error code handlers in tool wrappers: "
                "429 = exponential backoff + retry; "
                "401 = surface credential error to operator, do not continue; "
                "400 = log malformed request and abort the tool call chain; "
                "503 = circuit breaker open, fail fast."
            ),
        )
    return None


# ── Quality Pattern Detectors (Q3) ────────────────────────────────────────────

GROUNDING_FAILURE_WINDOW    = 3
GROUNDING_FAILURE_THRESHOLD = 0.40
COHERENCE_BREAK_THRESHOLD   = 0.50
CASCADE_WINDOW              = 5
CASCADE_MIN_DIMS            = 3
CASCADE_DROP                = 0.20
DEGRADATION_WINDOW          = 5
DEGRADATION_SLOPE_THRESHOLD = -0.015
DEGRADATION_OP_FLOOR        = 0.80

_CASCADE_DIMS = [
    ("research_quality",     "data_grounding"),
    ("research_quality",     "thesis_coherence"),
    ("research_quality",     "actionability"),
    ("research_quality",     "catalyst_specificity"),
    ("research_quality",     "volatility_accounting"),
    ("risk_quality",         "research_consistency"),
    ("risk_quality",         "parameter_completeness"),
    ("risk_quality",         "position_sizing_rationale"),
    ("orchestrator_quality", "decision_consistency"),
    ("orchestrator_quality", "resolution_completeness"),
    ("orchestrator_quality", "reasoning_transparency"),
    ("session_quality",      "upstream_integration"),
    ("session_quality",      "pipeline_coherence"),
]


def _qscore(evals: list[dict], agent: str, eval_name: str) -> float | None:
    for e in evals:
        if e.get("agent") == agent and e.get("eval_name") == eval_name:
            v = e.get("score")
            return float(v) if v is not None else None
    return None


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def detect_grounding_failure(
    session: dict,
    sessions_history: list[dict],
    evals_by_session: dict[str, list],
) -> Incident | None:
    """Fires when research.data_grounding < 0.40 for 3 consecutive sessions."""
    window = sessions_history[-GROUNDING_FAILURE_WINDOW:]
    if len(window) < GROUNDING_FAILURE_WINDOW:
        return None

    scores = []
    for s in window:
        sc = _qscore(evals_by_session.get(s["id"], []), "research_quality", "data_grounding")
        if sc is None:
            return None
        scores.append(sc)

    if not all(sc < GROUNDING_FAILURE_THRESHOLD for sc in scores):
        return None

    return Incident(
        session_id    = session["id"],
        pattern_name  = "Proactive: Grounding Failure",
        severity      = "warning",
        root_cause    = (
            f"research.data_grounding scored below {GROUNDING_FAILURE_THRESHOLD} for "
            f"{GROUNDING_FAILURE_WINDOW} consecutive sessions "
            f"(scores: {', '.join(f'{sc:.2f}' for sc in scores)}). "
            f"Research is consistently making decisions without sufficient data sources."
        ),
        call_stack    = [],
        failed_evals  = [
            {"agent": "research_quality", "eval_name": "data_grounding",
             "score": round(scores[-1], 3), "threshold": GROUNDING_FAILURE_THRESHOLD}
        ],
        cost_wasted   = 0.0,
        tokens_wasted = 0,
        fix_suggestion = (
            "Require at least 3 distinct tool types per ticker investigation "
            "(price data, news, ATR). Add an explicit output requirement: "
            "cite one price level and one quantitative indicator before proposing a trade."
        ),
    )


def detect_coherence_break(
    session: dict,
    sessions_history: list[dict],
    evals_by_session: dict[str, list],
) -> Incident | None:
    """Fires when orchestrator.decision_consistency < 0.50 in the current session."""
    sid   = session["id"]
    score = _qscore(evals_by_session.get(sid, []), "orchestrator_quality", "decision_consistency")
    if score is None or score >= COHERENCE_BREAK_THRESHOLD:
        return None

    return Incident(
        session_id    = sid,
        pattern_name  = "Proactive: Coherence Break",
        severity      = "critical",
        root_cause    = (
            f"orchestrator.decision_consistency scored {score:.2f} "
            f"(threshold {COHERENCE_BREAK_THRESHOLD}). "
            f"The final trade decision does not align with research and risk agent outputs."
        ),
        call_stack    = [],
        failed_evals  = [
            {"agent": "orchestrator_quality", "eval_name": "decision_consistency",
             "score": round(score, 3), "threshold": COHERENCE_BREAK_THRESHOLD}
        ],
        cost_wasted   = float(session.get("total_cost_usd") or 0),
        tokens_wasted = 0,
        fix_suggestion = (
            "Add a consistency check in the orchestrator prompt: the final decision must "
            "reference specific research findings and risk verdicts. Use a structured "
            "handoff format so the orchestrator cannot ignore upstream outputs."
        ),
    )


def detect_quality_cascade(
    session: dict,
    sessions_history: list[dict],
    evals_by_session: dict[str, list],
) -> Incident | None:
    """Fires when 3+ quality dimensions decline > 0.20 over the last 5 sessions."""
    window = sessions_history[-CASCADE_WINDOW:]
    if len(window) < 3:
        return None

    declining: list[dict] = []
    for agent, dim in _CASCADE_DIMS:
        vals = [
            sc for s in window
            for sc in [_qscore(evals_by_session.get(s["id"], []), agent, dim)]
            if sc is not None
        ]
        if len(vals) < 3:
            continue
        drop = vals[0] - vals[-1]
        if drop >= CASCADE_DROP:
            declining.append({
                "agent": agent, "dim": dim,
                "drop": round(drop, 3), "first": round(vals[0], 3), "last": round(vals[-1], 3),
            })

    if len(declining) < CASCADE_MIN_DIMS:
        return None

    severity    = "critical" if len(declining) >= 5 else "warning"
    dim_summary = "; ".join(
        f"{d['agent'].replace('_quality','')}.{d['dim']} "
        f"({d['first']:.2f}→{d['last']:.2f})"
        for d in declining[:4]
    )
    return Incident(
        session_id    = session["id"],
        pattern_name  = "Proactive: Quality Cascade",
        severity      = severity,
        root_cause    = (
            f"{len(declining)} quality dimensions declined >{CASCADE_DROP} over the last "
            f"{len(window)} sessions. Degrading: {dim_summary}"
            f"{'...' if len(declining) > 4 else ''}. "
            f"Systemic degradation across agents — not an isolated bad session."
        ),
        call_stack    = [],
        failed_evals  = [
            {"agent": d["agent"], "eval_name": d["dim"],
             "score": d["last"], "threshold": round(d["first"] - CASCADE_DROP, 3)}
            for d in declining
        ],
        cost_wasted   = 0.0,
        tokens_wasted = 0,
        fix_suggestion = (
            "Review agent prompts for each degrading dimension. "
            "Check for recent changes to market data quality, model version, or tool schemas. "
            "Run a manual quality review of the last 3 sessions before the next trading session."
        ),
    )


def detect_silent_degradation(
    session: dict,
    sessions_history: list[dict],
    evals_by_session: dict[str, list],
) -> Incident | None:
    """Fires when composite quality is declining while operational evals stay stable."""
    window = sessions_history[-DEGRADATION_WINDOW:]
    if len(window) < 3:
        return None

    composites: list[float] = []
    op_rates:   list[float] = []
    for s in window:
        evals = evals_by_session.get(s["id"], [])
        comp_scores = [
            float(e["score"]) for e in evals
            if e.get("eval_name") == "composite_score"
            and (e.get("agent") or "").endswith("_quality")
            and e.get("score") is not None
        ]
        op_evals = [
            e for e in evals
            if not (e.get("agent") or "").endswith("_quality")
            and e.get("agent") not in ("business",)
        ]
        if not comp_scores:
            continue
        composites.append(statistics.mean(comp_scores))
        if op_evals:
            op_rates.append(
                sum(1 for e in op_evals if e.get("passed", False)) / len(op_evals)
            )

    if len(composites) < 3:
        return None

    sl = _slope(composites)
    if sl >= DEGRADATION_SLOPE_THRESHOLD:
        return None

    op_stable = (statistics.mean(op_rates) >= DEGRADATION_OP_FLOOR) if op_rates else True
    if not op_stable:
        return None

    severity = "warning" if composites[-1] < 0.60 else "info"
    return Incident(
        session_id    = session["id"],
        pattern_name  = "Proactive: Silent Degradation",
        severity      = severity,
        root_cause    = (
            f"Composite quality declining at {sl:.3f}/session over the last {len(composites)} "
            f"sessions (latest: {composites[-1]:.2f}). Operational evals are stable. "
            f"Quality is eroding quietly before any structural failure."
        ),
        call_stack    = [],
        failed_evals  = [
            {"agent": a, "eval_name": "composite_score",
             "score": round(composites[-1], 3), "threshold": 0.60}
            for a in ["research_quality", "risk_quality",
                      "orchestrator_quality", "session_quality"]
        ],
        cost_wasted   = 0.0,
        tokens_wasted = 0,
        fix_suggestion = (
            "Manual review of recent session outputs is needed now. "
            "Check for prompt drift, model version changes, or degrading market data quality. "
            "If trend continues 2 more sessions, escalate to critical."
        ),
    )


def run_quality_detectors(
    session: dict,
    sessions_history: list[dict],
    evals_by_session: dict[str, list],
) -> list[Incident]:
    """Run all Q3 proactive quality pattern detectors."""
    incidents: list[Incident] = []
    for detector in [
        detect_grounding_failure,
        detect_coherence_break,
        detect_quality_cascade,
        detect_silent_degradation,
    ]:
        try:
            result = detector(session, sessions_history, evals_by_session)
            if result:
                incidents.append(result)
        except Exception:
            pass
    return incidents


def compute_shadow_cb_fires(
    evals: list[EvalResult],
    cb_config: dict | None = None,
) -> list[dict]:
    """Return list of shadow CB fire events based on cb_config thresholds."""
    config = cb_config if cb_config is not None else CB_CONFIG
    fires = []
    for e in evals:
        if e.agent not in config:
            continue
        for eval_name, threshold in config[e.agent]:
            if e.eval_name == eval_name and e.score < threshold:
                fires.append({
                    "agent":      e.agent,
                    "eval_name":  e.eval_name,
                    "score":      round(e.score, 3),
                    "threshold":  threshold,
                })
    return fires


# ── Main runners ──────────────────────────────────────────────────────────────

DETECTORS = [
    detect_tool_timeout_loop,
    detect_context_spiral,
    detect_pipeline_break,
    detect_empty_result_loop,
    detect_silent_exit,
    detect_hyperactive_polling,
    detect_tool_fabrication,
    detect_handoff_schema_break,
    detect_error_misinterpretation,
]


def run_all_detectors(
    session: dict,
    traces: list[dict],
    evals: list[EvalResult],
    pipeline_config: dict | None = None,
    recent_costs: list[float] | None = None,
) -> list[Incident]:
    """
    Run all pattern detectors. Pass pipeline_config (from load_pipeline_config()) to use
    workflow-specific thresholds; omit to use module defaults.
    """
    cfg            = _build_detector_config(pipeline_config)
    retry_thresh   = int(cfg["tool_retry_threshold"])
    spiral_tok     = int(cfg["token_spiral_threshold"])
    poll_thresh    = int(cfg["hyperactive_poll_threshold"])
    fab_lat        = float(cfg["fabrication_latency_ms"])
    fab_min        = int(cfg["fabrication_min_count"])
    anomaly_sig    = float(cfg["cost_anomaly_sigma"])
    analysis_agent = str(cfg.get("analysis_agent", "research"))
    pipeline_order = list(cfg.get("pipeline_order", ["research", "orchestrator"]))
    handoff_pairs  = list(cfg.get("handoff_pairs",  [["research", "risk"]]))

    incidents: list[Incident] = []
    for detector, kwargs in [
        (detect_tool_timeout_loop,   {"retry_threshold": retry_thresh}),
        (detect_context_spiral,      {"spiral_tokens": spiral_tok, "analysis_agent": analysis_agent}),
        (detect_pipeline_break,      {"pipeline_order": pipeline_order}),
        (detect_empty_result_loop,   {"retry_threshold": retry_thresh}),
        (detect_silent_exit,         {}),
        (detect_hyperactive_polling, {"poll_threshold": poll_thresh}),
        (detect_tool_fabrication,    {"fab_latency_ms": fab_lat, "fab_min_count": fab_min}),
        (detect_handoff_schema_break,    {"handoff_pairs": handoff_pairs}),
        (detect_error_misinterpretation, {}),
    ]:
        try:
            result = detector(session, traces, evals, **kwargs)
            if result:
                incidents.append(result)
        except Exception:
            pass
    try:
        result = detect_cost_anomaly(session, traces, evals, recent_costs,
                                     anomaly_sigma=anomaly_sig)
        if result:
            incidents.append(result)
    except Exception:
        pass
    return incidents


def run_and_persist(
    session: dict,
    traces: list[dict],
    evals: list[EvalResult],
    pipeline_config: dict | None = None,
    recent_costs: list[float] | None = None,
    db=None,
    tenant_id: str = "",
) -> list[Incident]:
    """Run all detectors and write incidents to ag_incidents if db is provided."""
    incidents = run_all_detectors(session, traces, evals, pipeline_config, recent_costs)
    if db and incidents:
        rows = [i.to_db_row() for i in incidents]
        if tenant_id:
            rows = [{**row, "tenant_id": tenant_id, "status": "open"} for row in rows]
        db.table("ag_incidents").insert(rows).execute()
    return incidents
