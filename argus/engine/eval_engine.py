"""
Operational eval engine — 17 rule-based evals across 4 agents + session holistic.

Inputs: ag_sessions dict + ag_traces list (plain dicts — no ORM coupling).
All field names match ag_sessions / ag_traces schema.
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

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


def eval_market_data_freshness(traces: list[dict], session: dict) -> EvalResult:
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
        score = max(0.0, 1.0 - delta / (DATA_FRESHNESS_MINUTES * 2))
        return EvalResult("data_freshness", "market", round(score, 2),
                          delta <= DATA_FRESHNESS_MINUTES, 0.5,
                          {"lag_minutes": round(delta, 1), "threshold_minutes": DATA_FRESHNESS_MINUTES})
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


def eval_research_token_efficiency(traces: list[dict], session: dict) -> EvalResult:
    """Score proportional to research token usage vs 30K threshold."""
    research = [t for t in traces if (t.get("agent") or "").lower().startswith("research")]
    tokens   = sum((t.get("tokens_input") or 0) + (t.get("tokens_output") or 0) for t in research)
    score    = max(0.0, 1.0 - tokens / (TOKEN_EFFICIENCY_THRESHOLD * 2))
    return EvalResult("token_efficiency", "research", round(score, 2),
                      tokens < TOKEN_EFFICIENCY_THRESHOLD, 0.5,
                      {"tokens_used": tokens, "threshold": TOKEN_EFFICIENCY_THRESHOLD})


def eval_research_tool_success_rate(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if >= 80% of research agent tool calls succeeded."""
    tools = [
        t for t in traces
        if (t.get("agent") or "").lower().startswith("research")
        and (t.get("step_type") or "") == "tool_call"
    ]
    if not tools:
        return EvalResult("tool_success_rate", "research", 1.0, True, 0.8,
                          {"reason": "no tool calls made"})
    success = sum(1 for t in tools if t.get("outcome") != "error")
    rate    = success / len(tools)
    return EvalResult("tool_success_rate", "research", round(rate, 2),
                      rate >= TOOL_SUCCESS_MIN_RATE, TOOL_SUCCESS_MIN_RATE,
                      {"total": len(tools), "success": success,
                       "failed": len(tools) - success, "rate": round(rate, 2)})


def eval_research_tool_diversity(traces: list[dict], session: dict) -> EvalResult:
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
    return EvalResult("tool_diversity", "research", score, distinct >= TOOL_DIVERSITY_MIN,
                      TOOL_DIVERSITY_MIN,
                      {"distinct_tools": distinct, "threshold": TOOL_DIVERSITY_MIN,
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

def eval_orchestrator_decision_made(traces: list[dict], session: dict) -> EvalResult:
    """
    Score 1.0 if trades executed or named good exit.
    Score 0.5 if structural_block (partial).
    Score 0.2 if bad exit or orchestrator ran without recording a decision.
    Score 0.0 if orchestrator never ran.
    """
    trades = int(session.get("trades_executed") or 0)
    reason = (session.get("terminal_reason") or "").lower()
    orch   = [t for t in traces if (t.get("agent") or "").lower() == "orchestrator"]

    if trades > 0:
        return EvalResult("decision_made", "orchestrator", 1.0, True, 0.9,
                          {"trades_executed": trades})
    if reason in _GOOD_EXITS:
        return EvalResult("decision_made", "orchestrator", 1.0, True, 0.9,
                          {"terminal_reason": reason})
    if reason in _PARTIAL_EXITS:
        return EvalResult("decision_made", "orchestrator", 0.5, False, 0.9,
                          {"terminal_reason": reason, "reason": "pipeline structurally blocked"})
    if reason in _BAD_EXITS:
        return EvalResult("decision_made", "orchestrator", 0.2, False, 0.9,
                          {"terminal_reason": reason, "reason": "bad exit — session did not complete cleanly"})
    if orch:
        return EvalResult("decision_made", "orchestrator", 0.2, False, 0.9,
                          {"reason": "orchestrator ran but no decision or reason recorded",
                           "orch_traces": len(orch)})
    return EvalResult("decision_made", "orchestrator", 0.0, False, 0.9,
                      {"reason": "orchestrator never ran"})


def eval_orchestrator_exit_quality(traces: list[dict], session: dict) -> EvalResult:
    """
    Score the quality of the session's terminal reason.
    1.0 — good named exit; 0.5 — structural_block; 0.2 — bad exit; 0.0 — silent exit.
    """
    reason = (session.get("terminal_reason") or "").lower()
    trades = int(session.get("trades_executed") or 0)

    if reason in _GOOD_EXITS:
        return EvalResult("exit_quality", "orchestrator", 1.0, True, 0.9,
                          {"terminal_reason": reason})
    if reason in _PARTIAL_EXITS:
        return EvalResult("exit_quality", "orchestrator", 0.5, False, 0.9,
                          {"terminal_reason": reason, "reason": "structural block prevented clean exit"})
    if reason in _BAD_EXITS:
        return EvalResult("exit_quality", "orchestrator", 0.2, False, 0.9,
                          {"terminal_reason": reason, "reason": "bad exit state"})
    if trades > 0:
        return EvalResult("exit_quality", "orchestrator", 0.8, True, 0.9,
                          {"reason": "trades produced but no terminal_reason logged",
                           "trades_executed": trades})
    return EvalResult("exit_quality", "orchestrator", 0.0, False, 0.9,
                      {"reason": "no terminal_reason and 0 trades — silent exit"})


# ── Holistic session evals ────────────────────────────────────────────────────

def eval_session_pipeline_completion(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if all 4 required agents have at least one trace."""
    agents_present = {(t.get("agent") or "").lower() for t in traces}
    if "market_shadow" in agents_present:
        agents_present.add("market")
    if any(a.startswith("research_") for a in agents_present):
        agents_present.add("research")
    missing = [a for a in REQUIRED_AGENTS if a not in agents_present]
    score   = len([a for a in REQUIRED_AGENTS if a in agents_present]) / len(REQUIRED_AGENTS)
    return EvalResult("pipeline_completion", "session", round(score, 2),
                      not missing, 1.0,
                      {"present": list(agents_present & set(REQUIRED_AGENTS)),
                       "missing": missing})


def eval_session_cost_anomaly(traces: list[dict], session: dict,
                               recent_costs: list[float] | None = None) -> EvalResult:
    """Score 0.0 if session cost > mean + 2 sigma of recent sessions."""
    cost = float(session.get("total_cost_usd") or 0)
    if not recent_costs or len(recent_costs) < 5:
        return EvalResult("cost_anomaly", "session", 1.0, True, 0.5,
                          {"reason": "insufficient history", "cost_usd": cost})
    mean  = statistics.mean(recent_costs)
    stdev = statistics.stdev(recent_costs) if len(recent_costs) > 1 else 0
    z     = (cost - mean) / stdev if stdev > 0 else 0
    score = max(0.0, 1.0 - max(0, z - 1) / 2)
    return EvalResult("cost_anomaly", "session", round(score, 2),
                      z <= COST_ANOMALY_SIGMA, 0.5,
                      {"cost_usd": cost, "mean": round(mean, 4),
                       "stdev": round(stdev, 4), "z_score": round(z, 2),
                       "threshold_sigma": COST_ANOMALY_SIGMA})


def eval_session_outcome_linkage(traces: list[dict], session: dict) -> EvalResult:
    """
    Score whether the session produced a traceable outcome.
    1.0 — trades executed or named good exit; 0.5 — structural block; 0.0 — silent exit.
    """
    trades = int(session.get("trades_executed") or 0)
    reason = (session.get("terminal_reason") or "").lower()

    if trades > 0:
        return EvalResult("outcome_linkage", "session", 1.0, True, 0.9,
                          {"trades_executed": trades})
    if reason in _GOOD_EXITS:
        return EvalResult("outcome_linkage", "session", 1.0, True, 0.9,
                          {"terminal_reason": reason, "trades_executed": 0})
    if reason in _PARTIAL_EXITS:
        return EvalResult("outcome_linkage", "session", 0.5, False, 0.9,
                          {"terminal_reason": reason, "reason": "pipeline structurally blocked"})
    if reason in _BAD_EXITS:
        return EvalResult("outcome_linkage", "session", 0.2, False, 0.9,
                          {"terminal_reason": reason, "reason": "bad exit state"})
    return EvalResult("outcome_linkage", "session", 0.0, False, 0.9,
                      {"reason": "0 trades and no terminal_reason — silent exit"})


def eval_session_tokens_per_decision(traces: list[dict], session: dict) -> EvalResult:
    """
    Score proportional to tokens-per-trade vs 40K threshold.
    0-trade sessions scored on raw token spend to avoid denominator masking.
    """
    tokens = int((session.get("total_tokens_input") or 0) +
                 (session.get("total_tokens_output") or 0))
    trades = int(session.get("trades_executed") or 0)
    tok_per = tokens if trades == 0 else tokens / trades
    score = max(0.0, 1.0 - tok_per / (TOKEN_SPIRAL_THRESHOLD * 2))
    return EvalResult("tokens_per_decision", "session", round(score, 2),
                      tok_per < TOKEN_SPIRAL_THRESHOLD, 0.5,
                      {"total_tokens": tokens, "trades": trades,
                       "tokens_per_decision": round(tok_per),
                       "threshold": TOKEN_SPIRAL_THRESHOLD})


# ── Business outcome evals ────────────────────────────────────────────────────

def eval_cost_per_trade(traces: list[dict], session: dict) -> EvalResult:
    """Score proportional to cost-per-trade vs $0.50 threshold."""
    cost   = float(session.get("total_cost_usd") or 0)
    trades = int(session.get("trades_executed") or 0)
    cpt    = cost / max(1, trades)
    score  = max(0.0, 1.0 - cpt / (COST_PER_TRADE_THRESHOLD * 3))
    return EvalResult("cost_per_trade", "business", round(score, 2),
                      cpt <= COST_PER_TRADE_THRESHOLD, COST_PER_TRADE_THRESHOLD,
                      {"cost_usd": round(cost, 4), "trades": trades,
                       "cost_per_trade": round(cpt, 4)})


def eval_research_conversion(traces: list[dict], session: dict) -> EvalResult:
    """
    Trades executed / distinct tickers researched.
    Threshold: 30% of researched tickers should produce a trade.
    """
    research_tools = [
        t for t in traces
        if (t.get("agent") or "").lower().startswith("research")
        and (t.get("step_type") or "") == "tool_call"
        and t.get("outcome") != "error"
    ]
    tickers = {(t.get("entity_id") or "").strip() for t in research_tools
               if (t.get("entity_id") or "").strip()}
    trades = int(session.get("trades_executed") or 0)
    n_analyzed = len(tickers)
    if n_analyzed == 0:
        return EvalResult("research_conversion", "business", 0.0, False,
                          RESEARCH_CONVERSION_MIN,
                          {"trades": trades, "tickers_researched": 0,
                           "reason": "no successful research tool calls with entity_id"})
    rate  = trades / n_analyzed
    score = min(1.0, rate / RESEARCH_CONVERSION_MIN)
    return EvalResult("research_conversion", "business", round(score, 2),
                      rate >= RESEARCH_CONVERSION_MIN, RESEARCH_CONVERSION_MIN,
                      {"trades": trades, "tickers_researched": n_analyzed,
                       "tickers": sorted(tickers), "conversion_rate": round(rate, 3)})


def eval_proposal_acceptance(traces: list[dict], session: dict) -> EvalResult:
    """
    Trades executed / trades proposed.
    Threshold: 40% of proposed trades should execute.
    """
    trades   = int(session.get("trades_executed") or 0)
    proposed = int(session.get("trades_proposed") or 0)
    if proposed == 0:
        score = 1.0 if trades == 0 else 0.0
        return EvalResult("proposal_acceptance", "business", score, trades == 0,
                          PROPOSAL_ACCEPTANCE_MIN,
                          {"trades": trades, "proposed": 0,
                           "reason": "no trades proposed this session"})
    rate  = trades / proposed
    score = min(1.0, rate / PROPOSAL_ACCEPTANCE_MIN)
    return EvalResult("proposal_acceptance", "business", round(score, 2),
                      rate >= PROPOSAL_ACCEPTANCE_MIN, PROPOSAL_ACCEPTANCE_MIN,
                      {"trades": trades, "proposed": proposed,
                       "acceptance_rate": round(rate, 3)})


# ── Registry & runners ────────────────────────────────────────────────────────

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
    recent_costs: list[float] | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for fn in PER_AGENT_EVALS:
        results.append(fn(traces, session))
    for fn in SESSION_EVALS:
        if fn.__name__ == "eval_session_cost_anomaly":
            results.append(fn(traces, session, recent_costs))
        else:
            results.append(fn(traces, session))
    for fn in BUSINESS_EVALS:
        results.append(fn(traces, session))
    return results


def run_and_persist(
    session: dict,
    traces: list[dict],
    recent_costs: list[float] | None = None,
    db=None,
    tenant_id: str = "",
) -> list[EvalResult]:
    """Run all evals and write results to ag_evals if db is provided."""
    results = run_all_evals(session, traces, recent_costs)
    if db:
        sid = session.get("id", "")
        rows = [r.to_db_row(sid) for r in results]
        if tenant_id:
            rows = [{**row, "tenant_id": tenant_id, "layer": 3} for row in rows]
        if rows:
            db.table("ag_evals").insert(rows).execute()
    return results
