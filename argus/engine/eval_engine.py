"""
Operational eval engine — 17 rule-based evals across 4 agents + session holistic.

Inputs: ag_sessions dict + ag_traces list (plain dicts — no ORM coupling).
All field names match ag_sessions / ag_traces schema.

Pass pipeline_config (from load_pipeline_config()) to run_all_evals() to use
workflow-specific thresholds from ag_pipeline_config. Without it, module-level
defaults apply — suitable for tests and generic pipelines.
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Module-level defaults — used when no pipeline_config is supplied.
REQUIRED_AGENTS = ["market", "research", "risk", "orchestrator"]
TOKEN_SPIRAL_THRESHOLD     = 40_000
TOKEN_EFFICIENCY_THRESHOLD = 30_000
TOOL_SUCCESS_MIN_RATE      = 0.80
COST_ANOMALY_SIGMA         = 2.0
DATA_FRESHNESS_MINUTES     = 10
TOOL_DIVERSITY_MIN         = 2

COST_PER_TRADE_THRESHOLD   = 0.50
RESEARCH_CONVERSION_MIN    = 0.30
PROPOSAL_ACCEPTANCE_MIN    = 0.40

_GOOD_EXITS    = {"eod_complete", "converged", "no_opportunity", "risk_rejected",
                  "market_closed", "position_limit", "daily_limit"}
_PARTIAL_EXITS = {"structural_block"}
_BAD_EXITS     = {"in_progress", "error"}

_DEFAULTS: dict = {
    "required_agents":            REQUIRED_AGENTS,
    "token_spiral_threshold":     TOKEN_SPIRAL_THRESHOLD,
    "token_efficiency_threshold": TOKEN_EFFICIENCY_THRESHOLD,
    "tool_success_min_rate":      TOOL_SUCCESS_MIN_RATE,
    "cost_anomaly_sigma":         COST_ANOMALY_SIGMA,
    "data_freshness_minutes":     DATA_FRESHNESS_MINUTES,
    "tool_diversity_min":         TOOL_DIVERSITY_MIN,
    "cost_per_trade_threshold":   COST_PER_TRADE_THRESHOLD,
    "research_conversion_min":    RESEARCH_CONVERSION_MIN,
    "proposal_acceptance_min":    PROPOSAL_ACCEPTANCE_MIN,
    "good_exits":                 list(_GOOD_EXITS),
    "partial_exits":              list(_PARTIAL_EXITS),
    "bad_exits":                  list(_BAD_EXITS),
}


def _build_config(pipeline_config: dict | None) -> dict:
    cfg = dict(_DEFAULTS)
    if pipeline_config:
        cfg.update(pipeline_config)
    return cfg


@dataclass
class EvalResult:
    eval_name:  str
    agent:      str
    score:      float
    passed:     bool
    threshold:  float
    detail:     dict = field(default_factory=dict)

    def to_db_row(self, session_id: str) -> dict:
        return {
            "id":         str(uuid.uuid4()),
            "session_id": session_id,
            "agent":      self.agent,
            "eval_name":  self.eval_name,
            "score":      round(self.score, 3),
            "passed":     self.passed,
            "threshold":  self.threshold,
            "detail":     self.detail,
        }


# ── Market agent evals ────────────────────────────────────────────────────────

def eval_market_data_completeness(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if market agent produced at least one successful trace."""
    market = [t for t in traces if (t.get("agent") or "").lower() in ("market", "market_shadow")]
    if not market:
        return EvalResult("data_completeness", "market", 0.0, False, 0.7,
                          {"reason": "no market agent traces found"})
    success = [t for t in market if t.get("outcome") != "error"
               and t.get("step_type") != "error"]
    score   = len(success) / len(market)
    return EvalResult("data_completeness", "market", round(score, 2), score >= 0.7, 0.7,
                      {"total": len(market), "success": len(success)})


def eval_market_data_freshness(
    traces: list[dict], session: dict, *,
    freshness_minutes: int = DATA_FRESHNESS_MINUTES,
) -> EvalResult:
    """Score proportional to how quickly market data arrived after session start."""
    market = sorted(
        [t for t in traces if (t.get("agent") or "").lower() in ("market", "market_shadow")],
        key=lambda x: x.get("created_at", "")
    )
    if not market:
        return EvalResult("data_freshness", "market", 0.0, False, 0.5,
                          {"reason": "no market agent traces"})
    started = session.get("started_at") or ""
    first   = market[0].get("created_at") or ""
    if not started or not first:
        return EvalResult("data_freshness", "market", 1.0, True, 0.5,
                          {"reason": "timestamps unavailable — assuming fresh"})
    try:
        t0    = datetime.fromisoformat(started.replace("Z", "+00:00"))
        t1    = datetime.fromisoformat(first.replace("Z", "+00:00"))
        delta = abs((t1 - t0).total_seconds()) / 60
        score = max(0.0, 1.0 - delta / (freshness_minutes * 2))
        return EvalResult("data_freshness", "market", round(score, 2),
                          delta <= freshness_minutes, 0.5,
                          {"lag_minutes": round(delta, 1),
                           "threshold_minutes": freshness_minutes})
    except Exception as e:
        return EvalResult("data_freshness", "market", 1.0, True, 0.5,
                          {"reason": f"parse error: {e}"})


# ── Research agent evals ──────────────────────────────────────────────────────

def eval_research_completion(traces: list[dict], session: dict) -> EvalResult:
    """
    Score 1.0 if research agent ran its LLM AND at least one tool call succeeded.
    Score 0.3 if LLM ran but all tool calls failed.
    Score 0.0 if no research traces or no successful LLM call.
    """
    research = [t for t in traces if (t.get("agent") or "").lower().startswith("research")]
    if not research:
        return EvalResult("completion", "research", 0.0, False, 0.9,
                          {"reason": "no research agent traces"})
    llm_ok = any(
        (t.get("step_type") or "") == "llm_call" and (t.get("outcome") or "") == "success"
        or (t.get("step_type") or "") in ("agent_message", "decision")
        and (t.get("outcome") or "") not in ("", "error")
        for t in research
    )
    if not llm_ok:
        return EvalResult("completion", "research", 0.0, False, 0.9,
                          {"reason": "no successful llm_call or agent_message trace"})
    tool_calls = [t for t in research if (t.get("step_type") or "") == "tool_call"]
    tool_ok    = any(t.get("outcome") != "error" for t in tool_calls)
    if tool_calls and not tool_ok:
        return EvalResult("completion", "research", 0.3, False, 0.9,
                          {"reason": "llm ran but all tool calls failed",
                           "tool_calls": len(tool_calls)})
    return EvalResult("completion", "research", 1.0, True, 0.9,
                      {"llm_ok": True, "tool_calls_ok": True})


def eval_research_token_efficiency(
    traces: list[dict], session: dict, *,
    efficiency_threshold: int = TOKEN_EFFICIENCY_THRESHOLD,
) -> EvalResult:
    """Score proportional to research token usage vs threshold."""
    research = [t for t in traces if (t.get("agent") or "").lower().startswith("research")]
    tokens   = sum((t.get("tokens_input") or 0) + (t.get("tokens_output") or 0) for t in research)
    score    = max(0.0, 1.0 - tokens / (efficiency_threshold * 2))
    return EvalResult("token_efficiency", "research", round(score, 2),
                      tokens < efficiency_threshold, 0.5,
                      {"tokens_used": tokens, "threshold": efficiency_threshold})


def eval_research_tool_success_rate(
    traces: list[dict], session: dict, *,
    success_min_rate: float = TOOL_SUCCESS_MIN_RATE,
) -> EvalResult:
    """Score 1.0 if >= success_min_rate of research agent tool calls succeeded."""
    tools = [
        t for t in traces
        if (t.get("agent") or "").lower().startswith("research")
        and (t.get("step_type") or "") == "tool_call"
    ]
    if not tools:
        return EvalResult("tool_success_rate", "research", 1.0, True, success_min_rate,
                          {"reason": "no tool calls made"})
    success = sum(1 for t in tools if t.get("outcome") != "error")
    rate    = success / len(tools)
    return EvalResult("tool_success_rate", "research", round(rate, 2),
                      rate >= success_min_rate, success_min_rate,
                      {"total": len(tools), "success": success,
                       "failed": len(tools) - success, "rate": round(rate, 2)})


def eval_research_tool_diversity(
    traces: list[dict], session: dict, *,
    diversity_min: int = TOOL_DIVERSITY_MIN,
) -> EvalResult:
    """
    Score based on distinct tool names the research agent called.
    3+ distinct tools = 1.0, 2 = 0.7, 1 = 0.3, 0 = 0.0.
    """
    tools = [
        t for t in traces
        if (t.get("agent") or "").lower().startswith("research")
        and (t.get("step_type") or "") == "tool_call"
    ]
    distinct = len({(t.get("tool_name") or "").strip() for t in tools
                    if (t.get("tool_name") or "").strip()})
    if distinct >= 3:
        score = 1.0
    elif distinct == 2:
        score = 0.7
    elif distinct == 1:
        score = 0.3
    else:
        score = 0.0
    return EvalResult("tool_diversity", "research", score, distinct >= diversity_min,
                      diversity_min,
                      {"distinct_tools": distinct, "threshold": diversity_min,
                       "tool_names": list({(t.get("tool_name") or "").strip() for t in tools
                                           if (t.get("tool_name") or "").strip()})})


# ── Risk agent evals ──────────────────────────────────────────────────────────

def eval_risk_assessment_complete(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if risk agent has at least one successful trace."""
    risk = [t for t in traces if (t.get("agent") or "").lower() == "risk"]
    if not risk:
        return EvalResult("assessment_complete", "risk", 0.0, False, 1.0,
                          {"reason": "no risk agent traces"})
    success = [t for t in risk if t.get("outcome") != "error"
               and t.get("step_type") != "error"]
    score   = 1.0 if success else 0.0
    return EvalResult("assessment_complete", "risk", score, bool(success), 1.0,
                      {"total": len(risk), "success": len(success)})


def eval_risk_within_parameters(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if risk agent has no error traces."""
    risk   = [t for t in traces if (t.get("agent") or "").lower() == "risk"]
    errors = [t for t in risk if t.get("error")]
    score  = 1.0 if not errors else max(0.0, 1.0 - len(errors) / max(1, len(risk)))
    return EvalResult("within_parameters", "risk", round(score, 2),
                      not errors, 0.9,
                      {"risk_traces": len(risk), "error_traces": len(errors),
                       "errors": [t.get("error") for t in errors[:3]]})


# ── Orchestrator evals ────────────────────────────────────────────────────────

def eval_orchestrator_decision_made(
    traces: list[dict], session: dict, *,
    good_exits: set = _GOOD_EXITS,
    partial_exits: set = _PARTIAL_EXITS,
    bad_exits: set = _BAD_EXITS,
) -> EvalResult:
    """
    Score 1.0 if trades executed or named good exit.
    Score 0.5 if partial exit (e.g. circuit_breaker).
    Score 0.2 if bad exit or orchestrator ran without recording a decision.
    Score 0.0 if orchestrator never ran.
    """
    trades = int(session.get("trades_executed") or 0)
    reason = (session.get("terminal_reason") or "").lower()
    orch   = [t for t in traces if (t.get("agent") or "").lower() == "orchestrator"]

    if trades > 0:
        return EvalResult("decision_made", "orchestrator", 1.0, True, 0.9,
                          {"trades_executed": trades})
    if reason in good_exits:
        return EvalResult("decision_made", "orchestrator", 1.0, True, 0.9,
                          {"terminal_reason": reason})
    if reason in partial_exits:
        return EvalResult("decision_made", "orchestrator", 0.5, False, 0.9,
                          {"terminal_reason": reason, "reason": "pipeline structurally blocked"})
    if reason in bad_exits:
        return EvalResult("decision_made", "orchestrator", 0.2, False, 0.9,
                          {"terminal_reason": reason,
                           "reason": "bad exit — session did not complete cleanly"})
    if orch:
        return EvalResult("decision_made", "orchestrator", 0.2, False, 0.9,
                          {"reason": "orchestrator ran but no decision or reason recorded",
                           "orch_traces": len(orch)})
    return EvalResult("decision_made", "orchestrator", 0.0, False, 0.9,
                      {"reason": "orchestrator never ran"})


def eval_orchestrator_exit_quality(
    traces: list[dict], session: dict, *,
    good_exits: set = _GOOD_EXITS,
    partial_exits: set = _PARTIAL_EXITS,
    bad_exits: set = _BAD_EXITS,
) -> EvalResult:
    """
    Score the quality of the session's terminal reason.
    1.0 — good named exit; 0.5 — partial exit; 0.2 — bad exit; 0.0 — silent exit.
    """
    reason = (session.get("terminal_reason") or "").lower()
    trades = int(session.get("trades_executed") or 0)

    if reason in good_exits:
        return EvalResult("exit_quality", "orchestrator", 1.0, True, 0.9,
                          {"terminal_reason": reason})
    if reason in partial_exits:
        return EvalResult("exit_quality", "orchestrator", 0.5, False, 0.9,
                          {"terminal_reason": reason,
                           "reason": "structural block prevented clean exit"})
    if reason in bad_exits:
        return EvalResult("exit_quality", "orchestrator", 0.2, False, 0.9,
                          {"terminal_reason": reason, "reason": "bad exit state"})
    if trades > 0:
        return EvalResult("exit_quality", "orchestrator", 0.8, True, 0.9,
                          {"reason": "trades produced but no terminal_reason logged",
                           "trades_executed": trades})
    return EvalResult("exit_quality", "orchestrator", 0.0, False, 0.9,
                      {"reason": "no terminal_reason and 0 trades — silent exit"})


# ── Holistic session evals ────────────────────────────────────────────────────

def eval_session_pipeline_completion(
    traces: list[dict], session: dict, *,
    required_agents: list = REQUIRED_AGENTS,
) -> EvalResult:
    """Score 1.0 if all required agents have at least one trace."""
    agents_present = {(t.get("agent") or "").lower() for t in traces}
    if "market_shadow" in agents_present:
        agents_present.add("market")
    if any(a.startswith("research_") for a in agents_present):
        agents_present.add("research")
    missing = [a for a in required_agents if a not in agents_present]
    score   = len([a for a in required_agents if a in agents_present]) / len(required_agents)
    return EvalResult("pipeline_completion", "session", round(score, 2),
                      not missing, 1.0,
                      {"present": list(agents_present & set(required_agents)),
                       "missing": missing})


def eval_session_cost_anomaly(
    traces: list[dict], session: dict,
    recent_costs: list[float] | None = None, *,
    anomaly_sigma: float = COST_ANOMALY_SIGMA,
) -> EvalResult:
    """Score 0.0 if session cost > mean + anomaly_sigma * stdev of recent sessions."""
    cost = float(session.get("total_cost_usd") or 0)
    if not recent_costs or len(recent_costs) < 5:
        return EvalResult("cost_anomaly", "session", 1.0, True, 0.5,
                          {"reason": "insufficient history", "cost_usd": cost})
    mean  = statistics.mean(recent_costs)
    stdev = statistics.stdev(recent_costs) if len(recent_costs) > 1 else 0
    z     = (cost - mean) / stdev if stdev > 0 else 0
    score = max(0.0, 1.0 - max(0, z - 1) / 2)
    return EvalResult("cost_anomaly", "session", round(score, 2),
                      z <= anomaly_sigma, 0.5,
                      {"cost_usd": cost, "mean": round(mean, 4),
                       "stdev": round(stdev, 4), "z_score": round(z, 2),
                       "threshold_sigma": anomaly_sigma})


def eval_session_outcome_linkage(
    traces: list[dict], session: dict, *,
    good_exits: set = _GOOD_EXITS,
    partial_exits: set = _PARTIAL_EXITS,
    bad_exits: set = _BAD_EXITS,
) -> EvalResult:
    """
    Score whether the session produced a traceable outcome.
    1.0 — trades executed or named good exit; 0.5 — partial exit; 0.0 — silent exit.
    """
    trades = int(session.get("trades_executed") or 0)
    reason = (session.get("terminal_reason") or "").lower()

    if trades > 0:
        return EvalResult("outcome_linkage", "session", 1.0, True, 0.9,
                          {"trades_executed": trades})
    if reason in good_exits:
        return EvalResult("outcome_linkage", "session", 1.0, True, 0.9,
                          {"terminal_reason": reason, "trades_executed": 0})
    if reason in partial_exits:
        return EvalResult("outcome_linkage", "session", 0.5, False, 0.9,
                          {"terminal_reason": reason,
                           "reason": "pipeline structurally blocked"})
    if reason in bad_exits:
        return EvalResult("outcome_linkage", "session", 0.2, False, 0.9,
                          {"terminal_reason": reason, "reason": "bad exit state"})
    return EvalResult("outcome_linkage", "session", 0.0, False, 0.9,
                      {"reason": "0 trades and no terminal_reason — silent exit"})


def eval_session_tokens_per_decision(
    traces: list[dict], session: dict, *,
    spiral_threshold: int = TOKEN_SPIRAL_THRESHOLD,
) -> EvalResult:
    """
    Score proportional to tokens-per-trade vs spiral_threshold.
    0-trade sessions scored on raw token spend to avoid denominator masking.
    """
    tokens  = int((session.get("total_tokens_input") or 0) +
                  (session.get("total_tokens_output") or 0))
    trades  = int(session.get("trades_executed") or 0)
    tok_per = tokens if trades == 0 else tokens / trades
    score   = max(0.0, 1.0 - tok_per / (spiral_threshold * 2))
    return EvalResult("tokens_per_decision", "session", round(score, 2),
                      tok_per < spiral_threshold, 0.5,
                      {"total_tokens": tokens, "trades": trades,
                       "tokens_per_decision": round(tok_per),
                       "threshold": spiral_threshold})


# ── Business outcome evals ────────────────────────────────────────────────────

def eval_cost_per_trade(
    traces: list[dict], session: dict, *,
    cost_threshold: float = COST_PER_TRADE_THRESHOLD,
) -> EvalResult:
    """Score proportional to cost-per-trade vs threshold."""
    cost   = float(session.get("total_cost_usd") or 0)
    trades = int(session.get("trades_executed") or 0)
    cpt    = cost / max(1, trades)
    score  = max(0.0, 1.0 - cpt / (cost_threshold * 3))
    return EvalResult("cost_per_trade", "business", round(score, 2),
                      cpt <= cost_threshold, cost_threshold,
                      {"cost_usd": round(cost, 4), "trades": trades,
                       "cost_per_trade": round(cpt, 4)})


def eval_research_conversion(
    traces: list[dict], session: dict, *,
    conversion_min: float = RESEARCH_CONVERSION_MIN,
) -> EvalResult:
    """
    Trades executed / distinct tickers researched.
    Threshold: conversion_min of researched tickers should produce a trade.
    """
    research_tools = [
        t for t in traces
        if (t.get("agent") or "").lower().startswith("research")
        and (t.get("step_type") or "") == "tool_call"
        and t.get("outcome") != "error"
    ]
    tickers    = {(t.get("entity_id") or "").strip() for t in research_tools
                  if (t.get("entity_id") or "").strip()}
    trades     = int(session.get("trades_executed") or 0)
    n_analyzed = len(tickers)
    if n_analyzed == 0:
        return EvalResult("research_conversion", "business", 0.0, False,
                          conversion_min,
                          {"trades": trades, "tickers_researched": 0,
                           "reason": "no successful research tool calls with entity_id"})
    rate  = trades / n_analyzed
    score = min(1.0, rate / conversion_min)
    return EvalResult("research_conversion", "business", round(score, 2),
                      rate >= conversion_min, conversion_min,
                      {"trades": trades, "tickers_researched": n_analyzed,
                       "tickers": sorted(tickers), "conversion_rate": round(rate, 3)})


def eval_proposal_acceptance(
    traces: list[dict], session: dict, *,
    acceptance_min: float = PROPOSAL_ACCEPTANCE_MIN,
) -> EvalResult:
    """
    Trades executed / trades proposed.
    Threshold: acceptance_min of proposed trades should execute.
    """
    trades   = int(session.get("trades_executed") or 0)
    proposed = int(session.get("trades_proposed") or 0)
    if proposed == 0:
        score = 1.0 if trades == 0 else 0.0
        return EvalResult("proposal_acceptance", "business", score, trades == 0,
                          acceptance_min,
                          {"trades": trades, "proposed": 0,
                           "reason": "no trades proposed this session"})
    rate  = trades / proposed
    score = min(1.0, rate / acceptance_min)
    return EvalResult("proposal_acceptance", "business", round(score, 2),
                      rate >= acceptance_min, acceptance_min,
                      {"trades": trades, "proposed": proposed,
                       "acceptance_rate": round(rate, 3)})


# ── Registry (for reference) ──────────────────────────────────────────────────

PER_AGENT_EVALS = [
    eval_market_data_completeness,
    eval_market_data_freshness,
    eval_research_completion,
    eval_research_token_efficiency,
    eval_research_tool_success_rate,
    eval_research_tool_diversity,
    eval_risk_assessment_complete,
    eval_risk_within_parameters,
    eval_orchestrator_decision_made,
    eval_orchestrator_exit_quality,
]

SESSION_EVALS = [
    eval_session_pipeline_completion,
    eval_session_cost_anomaly,
    eval_session_outcome_linkage,
    eval_session_tokens_per_decision,
]

BUSINESS_EVALS = [
    eval_cost_per_trade,
    eval_research_conversion,
    eval_proposal_acceptance,
]


def run_all_evals(
    session: dict,
    traces: list[dict],
    pipeline_config: dict | None = None,
    recent_costs: list[float] | None = None,
) -> list[EvalResult]:
    """
    Run all 17 evals. Pass pipeline_config (from load_pipeline_config()) to use
    workflow-specific thresholds; omit to use module defaults.
    """
    cfg     = _build_config(pipeline_config)
    good    = set(cfg["good_exits"])
    partial = set(cfg["partial_exits"])
    bad     = set(cfg["bad_exits"])
    return [
        eval_market_data_completeness(traces, session),
        eval_market_data_freshness(traces, session,
            freshness_minutes=int(cfg["data_freshness_minutes"])),
        eval_research_completion(traces, session),
        eval_research_token_efficiency(traces, session,
            efficiency_threshold=int(cfg["token_efficiency_threshold"])),
        eval_research_tool_success_rate(traces, session,
            success_min_rate=float(cfg["tool_success_min_rate"])),
        eval_research_tool_diversity(traces, session,
            diversity_min=int(cfg["tool_diversity_min"])),
        eval_risk_assessment_complete(traces, session),
        eval_risk_within_parameters(traces, session),
        eval_orchestrator_decision_made(traces, session,
            good_exits=good, partial_exits=partial, bad_exits=bad),
        eval_orchestrator_exit_quality(traces, session,
            good_exits=good, partial_exits=partial, bad_exits=bad),
        eval_session_pipeline_completion(traces, session,
            required_agents=list(cfg["required_agents"])),
        eval_session_cost_anomaly(traces, session, recent_costs,
            anomaly_sigma=float(cfg["cost_anomaly_sigma"])),
        eval_session_outcome_linkage(traces, session,
            good_exits=good, partial_exits=partial, bad_exits=bad),
        eval_session_tokens_per_decision(traces, session,
            spiral_threshold=int(cfg["token_spiral_threshold"])),
        eval_cost_per_trade(traces, session,
            cost_threshold=float(cfg["cost_per_trade_threshold"])),
        eval_research_conversion(traces, session,
            conversion_min=float(cfg["research_conversion_min"])),
        eval_proposal_acceptance(traces, session,
            acceptance_min=float(cfg["proposal_acceptance_min"])),
    ]


def run_and_persist(
    session: dict,
    traces: list[dict],
    pipeline_config: dict | None = None,
    recent_costs: list[float] | None = None,
    db=None,
    tenant_id: str = "",
) -> list[EvalResult]:
    """Run all evals and write results to ag_evals if db is provided."""
    results = run_all_evals(session, traces, pipeline_config, recent_costs)
    if db:
        sid  = session.get("id", "")
        rows = [r.to_db_row(sid) for r in results]
        if tenant_id:
            rows = [{**row, "tenant_id": tenant_id, "layer": 3} for row in rows]
        if rows:
            db.table("ag_evals").insert(rows).execute()
    return results
