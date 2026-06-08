"""
RCA engine — enriches an Incident with a structured call stack and fix suggestion.
Identifies the failure chain from ag_traces for the incident's session.
"""
from __future__ import annotations
import math

from argus.engine.pattern_detector import Incident


def _real_error(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None

SEVERITY_COLORS = {
    "critical": "#EF4444",
    "warning":  "#F59E0B",
    "info":     "#3B82F6",
}

STEP_TYPE_ICONS = {
    "tool_call": "T",
    "llm_call":  "L",
    "decision":  "D",
    "error":     "!",
}


def build_annotated_call_stack(
    traces: list[dict],
    incident: Incident,
) -> list[dict]:
    """
    Return the full session trace list annotated with is_root and is_relevant flags.
    is_root    = the specific step identified as the failure origin
    is_relevant = part of the failure chain (shown expanded in UI)
    """
    root_tool = None
    if incident.pattern_name == "Tool Timeout Loop" and incident.call_stack:
        root_tool = incident.call_stack[0].get("tool_name")

    sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
    annotated = []
    first_error_seen = False

    for t in sorted_traces:
        is_error   = bool(_real_error(t.get("error"))) or t.get("outcome") == "error"
        is_root    = False
        is_relevant = False

        if incident.pattern_name == "Tool Timeout Loop":
            if root_tool and t.get("tool_name") == root_tool:
                is_relevant = True
                if not first_error_seen and is_error:
                    is_root         = True
                    first_error_seen = True

        elif incident.pattern_name == "Context Spiral":
            if (t.get("agent") or "").lower() == "research":
                is_relevant = True

        elif incident.pattern_name == "Pipeline Break":
            if (t.get("agent") or "").lower() in ("research", "orchestrator"):
                is_relevant = True

        elif incident.pattern_name in ("Cost Anomaly", "Silent Exit", "Empty Result Loop"):
            is_relevant = is_error

        annotated.append({
            **t,
            "is_root":     is_root,
            "is_relevant": is_relevant,
            "tokens":      (t.get("tokens_input") or 0) + (t.get("tokens_output") or 0),
        })

    return annotated


def generate_fix_suggestion(incident: Incident, traces: list[dict]) -> str:
    """Return a detailed fix suggestion based on pattern type and trace evidence."""
    if incident.pattern_name == "Tool Timeout Loop":
        tools = {f.get("tool_name") for f in incident.call_stack if f.get("tool_name")}
        agents = {f.get("agent") for f in incident.call_stack if f.get("agent")}
        tool  = next(iter(tools), "the tool")
        agent = next(iter(agents), "the agent")
        return (
            f"1. Add timeout=10 to {tool} calls in the {agent} agent.\n"
            f"2. Add max_retries=2 with exponential backoff (2s, 4s).\n"
            f"3. If all retries fail, return a structured error — do not loop.\n"
            f"4. Consider a fallback data source if {tool} is unavailable."
        )
    if incident.pattern_name == "Context Spiral":
        return (
            f"1. Set a hard token budget on the research agent "
            f"(recommended: {incident.tokens_wasted // 2:,} tokens max).\n"
            f"2. Add a 'decide or abort' checkpoint after each information-gathering step.\n"
            f"3. Log the reason if the agent exits without a recommendation."
        )
    if incident.pattern_name == "Pipeline Break":
        return (
            "1. Wrap the research -> orchestrator handoff in try/except.\n"
            "2. Log a terminal_reason on any uncaught exception.\n"
            "3. Add a health check after research completes before calling orchestrator."
        )
    if incident.pattern_name == "Cost Anomaly":
        return (
            f"1. Review Session Deep Dive for this session to find the high-cost agent.\n"
            f"2. Add per-session cost budgets with early abort if exceeded.\n"
            f"3. Set alerts at mean + 1.5 sigma for proactive notification."
        )
    if incident.pattern_name == "Silent Exit":
        return (
            "1. Add terminal_reason logging to every orchestrator exit path.\n"
            "2. Ensure the orchestrator always records 'no_opportunity', "
            "'risk_rejected', or another explicit reason.\n"
            "3. Treat a session with 0 trades and no reason as a bug, not normal operation."
        )
    if incident.pattern_name == "Empty Result Loop":
        tools = {f.get("tool_name") for f in incident.call_stack if f.get("tool_name")}
        tool  = next(iter(tools), "the tool")
        return (
            f"1. Add result quality validation after {tool} returns.\n"
            f"2. If result is empty, below quality threshold, or missing required fields: "
            f"fail fast, do not retry with same parameters.\n"
            f"3. Log what {tool} returned when quality check fails."
        )
    return incident.fix_suggestion


def summarize_incident(incident: Incident, session: dict) -> dict:
    """Return a human-readable summary dict for display."""
    duration_s = 0
    if session.get("started_at") and session.get("completed_at"):
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(session["started_at"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(session["completed_at"].replace("Z", "+00:00"))
            duration_s = int(abs((t1 - t0).total_seconds()))
        except Exception:
            pass

    return {
        "pattern":       incident.pattern_name,
        "severity":      incident.severity,
        "severity_color":SEVERITY_COLORS.get(incident.severity, "#94A3B8"),
        "root_cause":    incident.root_cause,
        "fix_suggestion":incident.fix_suggestion,
        "cost_wasted":   incident.cost_wasted,
        "tokens_wasted": incident.tokens_wasted,
        "duration_s":    duration_s,
        "trades":        int(session.get("trades_executed") or 0),
        "agents":        list({f.get("agent") for f in incident.call_stack if f.get("agent")}),
        "failed_eval_count": len(incident.failed_evals),
    }
