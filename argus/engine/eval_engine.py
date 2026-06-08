"""
Generic operational eval engine — executes rule-based evals defined in ag_eval_configs.

Inputs:
  session        — ag_sessions dict
  traces         — ag_traces list (plain dicts)
  eval_configs   — ag_eval_configs rows (layer=3, eval_type=rule) for the workflow
  pipeline_config — ag_pipeline_config rows as flat dict (from load_pipeline_config)

Each eval_configs row must have config.metric set to one of the built-in types:
  agent_presence, tool_success_rate, tool_diversity, token_budget,
  data_freshness, exit_quality, cost_anomaly, pipeline_coverage,
  agent_completion, session_field_ratio

Use @register_eval to plug in custom metric types for domain-specific evals.

Generic defaults (used when pipeline_config is omitted — suitable for tests):
  good_exits   = ["completed", "converged", "no_opportunity"]
  partial_exits = ["partial", "circuit_breaker"]
  bad_exits    = ["error", "in_progress"]
  required_agents = []   (pipeline_coverage scores 0 if not configured)
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# ── Generic defaults ──────────────────────────────────────────────────────────
# These apply when no pipeline_config is supplied (tests, generic pipelines).
# Workflow-specific values always come from ag_pipeline_config via load_pipeline_config().

_DEFAULTS: dict = {
    "required_agents":            [],
    "token_spiral_threshold":     40_000,
    "token_efficiency_threshold": 30_000,
    "tool_success_min_rate":      0.80,
    "cost_anomaly_sigma":         2.0,
    "data_freshness_minutes":     10,
    "tool_diversity_min":         2,
    "cost_per_unit_threshold":    0.50,
    "good_exits":                 ["completed", "converged", "no_opportunity"],
    "partial_exits":              ["partial", "circuit_breaker"],
    "bad_exits":                  ["error", "in_progress"],
}


def _build_config(pipeline_config: dict | None) -> dict:
    cfg = dict(_DEFAULTS)
    if pipeline_config:
        cfg.update(pipeline_config)
    return cfg


# ── EvalResult ────────────────────────────────────────────────────────────────

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


# ── Custom eval registry ──────────────────────────────────────────────────────

_REGISTRY: dict[str, callable] = {}


def register_eval(eval_name: str, workflow_id: str | None = None):
    """
    Decorator to register a custom metric function for evals that don't fit
    the built-in metric types.

    The decorated function receives (traces, session, eval_config, pipeline_config)
    and must return an EvalResult.

    Example:
        @register_eval("diagnosis_confidence", workflow_id="abc123")
        def my_eval(traces, session, ec, cfg):
            scores = [t["payload"]["confidence"] for t in traces if t.get("payload")]
            score = sum(scores) / len(scores) if scores else 0.0
            threshold = float(ec.get("threshold") or 0.85)
            return EvalResult("diagnosis_confidence", ec.get("agent","session"),
                               round(score, 3), score >= threshold, threshold,
                               {"sample_count": len(scores)})
    """
    def decorator(fn: callable) -> callable:
        key = f"{workflow_id}:{eval_name}" if workflow_id else eval_name
        _REGISTRY[key] = fn
        return fn
    return decorator


def get_registry(workflow_id: str | None = None) -> dict:
    """Return registered evals visible to this workflow (global + workflow-specific)."""
    result = {k: v for k, v in _REGISTRY.items() if ":" not in k}
    if workflow_id:
        prefix = f"{workflow_id}:"
        result.update({
            k[len(prefix):]: v
            for k, v in _REGISTRY.items()
            if k.startswith(prefix)
        })
    return result


# ── Agent matching ────────────────────────────────────────────────────────────

def _filter_traces(traces: list[dict], pattern: str, match: str) -> list[dict]:
    """Filter traces to those whose agent name matches pattern using match strategy."""
    if match == "any" or not pattern:
        return traces
    pat = pattern.lower()
    out = []
    for t in traces:
        name = (t.get("agent") or "").lower()
        if match == "exact" and name == pat:
            out.append(t)
        elif match == "prefix" and name.startswith(pat):
            out.append(t)
        elif match == "contains" and pat in name:
            out.append(t)
    return out


# ── Generic metric functions ──────────────────────────────────────────────────
# Each returns (score: float, detail: dict).
# score is always 0.0–1.0. passed = score >= threshold (computed by dispatcher).

def _metric_agent_presence(traces: list[dict]) -> tuple[float, dict]:
    """Score 1.0 if agent has at least one non-error trace."""
    if not traces:
        return 0.0, {"reason": "no agent traces found"}
    success = [t for t in traces
               if t.get("outcome") != "error" and t.get("step_type") != "error"]
    score = len(success) / len(traces)
    return round(score, 2), {"total": len(traces), "success": len(success)}


def _metric_tool_success_rate(traces: list[dict]) -> tuple[float, dict]:
    """Score = fraction of tool calls that succeeded."""
    tools = [t for t in traces if (t.get("step_type") or "") == "tool_call"]
    if not tools:
        return 1.0, {"reason": "no tool calls made"}
    success = sum(1 for t in tools if t.get("outcome") != "error")
    rate = success / len(tools)
    return round(rate, 2), {
        "total": len(tools), "success": success,
        "failed": len(tools) - success, "rate": round(rate, 2),
    }


def _metric_tool_diversity(traces: list[dict], diversity_min: int) -> tuple[float, dict]:
    """Score based on distinct tool names used. Tiered: 0.0 / 0.3 / 0.7 / 1.0."""
    tools = [t for t in traces if (t.get("step_type") or "") == "tool_call"]
    names = {(t.get("tool_name") or "").strip() for t in tools
             if (t.get("tool_name") or "").strip()}
    distinct = len(names)
    if distinct >= diversity_min + 1:
        score = 1.0
    elif distinct == diversity_min:
        score = 0.7
    elif distinct == diversity_min - 1 and diversity_min > 1:
        score = 0.3
    else:
        score = 0.0 if distinct == 0 else 0.3
    return score, {"distinct_tools": distinct, "min_required": diversity_min,
                   "tool_names": sorted(names)}


def _metric_token_budget(traces: list[dict], session: dict,
                          threshold: int) -> tuple[float, dict]:
    """Score proportional to token spend vs threshold. Uses agent traces if present,
    falls back to session totals for session-level evals."""
    if traces:
        tokens = sum((t.get("tokens_input") or 0) + (t.get("tokens_output") or 0)
                     for t in traces)
    else:
        tokens = int((session.get("total_tokens_input") or 0) +
                     (session.get("total_tokens_output") or 0))
        trades = int(session.get("trades_executed") or 0)
        if trades > 0:
            tokens = tokens // trades  # tokens per decision
    score = max(0.0, 1.0 - tokens / (threshold * 2))
    return round(score, 2), {"tokens_used": tokens, "threshold": threshold}


def _metric_data_freshness(traces: list[dict], session: dict,
                            freshness_minutes: int) -> tuple[float, dict]:
    """Score proportional to lag from session start to first agent trace."""
    if not traces:
        return 0.0, {"reason": "no agent traces"}
    started = session.get("started_at") or ""
    first = sorted(traces, key=lambda x: x.get("created_at", ""))[0].get("created_at") or ""
    if not started or not first:
        return 1.0, {"reason": "timestamps unavailable — assuming fresh"}
    try:
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(first.replace("Z", "+00:00"))
        delta = abs((t1 - t0).total_seconds()) / 60
        score = max(0.0, 1.0 - delta / (freshness_minutes * 2))
        return round(score, 2), {
            "lag_minutes": round(delta, 1),
            "threshold_minutes": freshness_minutes,
        }
    except Exception as e:
        return 1.0, {"reason": f"parse error: {e}"}


def _metric_exit_quality(session: dict, good_exits: set,
                          partial_exits: set, bad_exits: set) -> tuple[float, dict]:
    """Score based on terminal_reason classification and trades executed."""
    reason = (session.get("terminal_reason") or "").lower()
    trades = int(session.get("trades_executed") or 0)
    if trades > 0 or reason in good_exits:
        return 1.0, {"terminal_reason": reason, "trades_executed": trades}
    if reason in partial_exits:
        return 0.5, {"terminal_reason": reason, "reason": "pipeline partially completed"}
    if reason in bad_exits:
        return 0.2, {"terminal_reason": reason, "reason": "bad exit state"}
    if trades == 0 and not reason:
        return 0.0, {"reason": "0 trades and no terminal_reason — silent exit"}
    return 0.2, {"terminal_reason": reason, "reason": "unclassified exit"}


def _metric_cost_anomaly(session: dict,
                          recent_costs: list[float] | None,
                          anomaly_sigma: float) -> tuple[float, dict]:
    """Score 0.0 if session cost is an outlier vs recent history."""
    cost = float(session.get("total_cost_usd") or 0)
    if not recent_costs or len(recent_costs) < 5:
        return 1.0, {"reason": "insufficient history", "cost_usd": cost}
    mean  = statistics.mean(recent_costs)
    stdev = statistics.stdev(recent_costs) if len(recent_costs) > 1 else 0
    z     = (cost - mean) / stdev if stdev > 0 else 0
    score = max(0.0, 1.0 - max(0, z - 1) / 2)
    return round(score, 2), {
        "cost_usd": cost, "mean": round(mean, 4),
        "stdev": round(stdev, 4), "z_score": round(z, 2),
        "threshold_sigma": anomaly_sigma,
    }


def _metric_pipeline_coverage(traces: list[dict],
                                required_agents: list[str]) -> tuple[float, dict]:
    """Score = fraction of required agents that produced at least one trace."""
    if not required_agents:
        return 1.0, {"reason": "no required_agents configured"}
    present = {(t.get("agent") or "").lower() for t in traces}
    # expand prefix-matched agent names (e.g. research_AAPL → research)
    expanded: set[str] = set()
    for name in present:
        expanded.add(name)
        for req in required_agents:
            if name.startswith(req.lower() + "_") or name == req.lower() + "_shadow":
                expanded.add(req.lower())
    hit     = [a for a in required_agents if a.lower() in expanded]
    missing = [a for a in required_agents if a.lower() not in expanded]
    score   = len(hit) / len(required_agents)
    return round(score, 2), {"present": hit, "missing": missing}


def _metric_agent_completion(traces: list[dict]) -> tuple[float, dict]:
    """Score 1.0 if agent ran LLM AND at least one tool call succeeded.
    Score 0.3 if LLM ran but all tools failed. Score 0.0 if no LLM."""
    llm_ok = any(
        (t.get("step_type") or "") == "llm_call" and (t.get("outcome") or "") == "success"
        or (t.get("step_type") or "") in ("agent_message", "decision")
        and (t.get("outcome") or "") not in ("", "error")
        for t in traces
    )
    if not llm_ok:
        return 0.0, {"reason": "no successful llm_call or decision trace"}
    tool_calls = [t for t in traces if (t.get("step_type") or "") == "tool_call"]
    tool_ok    = any(t.get("outcome") != "error" for t in tool_calls)
    if tool_calls and not tool_ok:
        return 0.3, {"reason": "llm ran but all tool calls failed",
                     "tool_calls": len(tool_calls)}
    return 1.0, {"llm_ok": True, "tool_calls_ok": True}


def _metric_session_field_ratio(session: dict,
                                  num_field: str, den_field: str,
                                  threshold: float) -> tuple[float, dict]:
    """Score = min(1, session[num] / session[den] / threshold)."""
    num = float(session.get(num_field) or 0)
    den = float(session.get(den_field) or 0)
    if den == 0:
        score = 1.0 if num == 0 else 0.0
        return score, {num_field: num, den_field: 0,
                       "reason": f"{den_field} is 0"}
    ratio = num / den
    score = min(1.0, ratio / threshold) if threshold > 0 else 1.0
    return round(score, 2), {
        num_field: num, den_field: den,
        "ratio": round(ratio, 3), "threshold": threshold,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _dispatch_eval(
    ec: dict,
    traces: list[dict],
    session: dict,
    cfg: dict,
    recent_costs: list[float] | None,
    registry: dict,
) -> EvalResult | None:
    config    = ec.get("config") or {}
    metric    = config.get("metric")
    eval_name = ec.get("eval_name") or "unknown"
    agent     = ec.get("agent") or "session"
    threshold = float(ec.get("threshold") or 0.5)

    # threshold override from pipeline_config
    tkey = config.get("threshold_key")
    if tkey and tkey in cfg:
        threshold = float(cfg[tkey])

    agent_pattern = config.get("agent_pattern") or agent
    agent_match   = config.get("agent_match") or "exact"
    agent_traces  = _filter_traces(traces, agent_pattern, agent_match)

    if metric == "agent_presence":
        score, detail = _metric_agent_presence(agent_traces)

    elif metric == "tool_success_rate":
        score, detail = _metric_tool_success_rate(agent_traces)

    elif metric == "tool_diversity":
        diversity_min = int(cfg.get("tool_diversity_min", 2))
        score, detail = _metric_tool_diversity(agent_traces, diversity_min)

    elif metric == "token_budget":
        t_key = config.get("threshold_key", "token_efficiency_threshold")
        tok_threshold = int(cfg.get(t_key, cfg["token_efficiency_threshold"]))
        score, detail = _metric_token_budget(agent_traces, session, tok_threshold)
        threshold = 0.5  # pass = score > 0.5 (below threshold)

    elif metric == "data_freshness":
        freshness = int(cfg.get("data_freshness_minutes", 10))
        score, detail = _metric_data_freshness(agent_traces, session, freshness)

    elif metric == "exit_quality":
        good    = set(cfg.get("good_exits", []))
        partial = set(cfg.get("partial_exits", []))
        bad     = set(cfg.get("bad_exits", []))
        score, detail = _metric_exit_quality(session, good, partial, bad)

    elif metric == "cost_anomaly":
        sigma = float(cfg.get("cost_anomaly_sigma", 2.0))
        score, detail = _metric_cost_anomaly(session, recent_costs, sigma)

    elif metric == "pipeline_coverage":
        required = list(cfg.get("required_agents", []))
        score, detail = _metric_pipeline_coverage(traces, required)

    elif metric == "agent_completion":
        score, detail = _metric_agent_completion(agent_traces)

    elif metric == "session_field_ratio":
        num_field = config.get("numerator_field", "")
        den_field = config.get("denominator_field", "")
        score, detail = _metric_session_field_ratio(session, num_field, den_field, threshold)

    elif metric == "custom":
        fn = registry.get(eval_name)
        if fn:
            try:
                return fn(traces, session, ec, cfg)
            except Exception as e:
                return EvalResult(eval_name, agent, 0.0, False, threshold,
                                  {"reason": f"custom eval error: {e}"})
        return None  # not registered — skip silently

    else:
        return None  # unknown metric type — skip silently

    passed = score >= threshold
    return EvalResult(eval_name, agent, score, passed, threshold, detail)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_evals_from_config(
    session: dict,
    traces: list[dict],
    eval_configs: list[dict],
    pipeline_config: dict | None = None,
    recent_costs: list[float] | None = None,
    registry: dict | None = None,
) -> list[EvalResult]:
    """
    Run all enabled layer-3 rule evals defined for this workflow.

    eval_configs — list of ag_eval_configs rows (any layer/type; non-L3-rule rows skipped)
    pipeline_config — flat dict from load_pipeline_config(); falls back to generic defaults
    registry — from get_registry(workflow_id); needed only if any eval uses metric=custom
    """
    cfg = _build_config(pipeline_config)
    reg = registry or {}
    results: list[EvalResult] = []
    for ec in eval_configs:
        if not ec.get("enabled", True):
            continue
        if ec.get("layer") != 3 or ec.get("eval_type") != "rule":
            continue
        try:
            result = _dispatch_eval(ec, traces, session, cfg, recent_costs, reg)
            if result:
                results.append(result)
        except Exception:
            pass
    return results


def run_and_persist(
    session: dict,
    traces: list[dict],
    eval_configs: list[dict],
    pipeline_config: dict | None = None,
    recent_costs: list[float] | None = None,
    db=None,
    tenant_id: str = "",
    workflow_id: str = "",
    registry: dict | None = None,
) -> list[EvalResult]:
    """Run evals from config and write results to ag_evals if db is provided."""
    results = run_evals_from_config(
        session, traces, eval_configs, pipeline_config, recent_costs, registry
    )
    if db and results:
        sid  = session.get("id", "")
        rows = [r.to_db_row(sid) for r in results]
        extras: dict = {"layer": 3}
        if tenant_id:
            extras["tenant_id"] = tenant_id
        if workflow_id:
            extras["workflow_id"] = workflow_id
        rows = [{**row, **extras} for row in rows]
        db.table("ag_evals").insert(rows).execute()
    return results
