from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Optional
from uuid import uuid4

import pytz


def _load_env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        try:
            from dotenv import load_dotenv as _ld
            _ld()
            val = os.environ.get(key, "")
        except ImportError:
            pass
    return val


_COST_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00,  "cache_read": 0.08,  "cache_write": 1.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00, "cache_read": 0.30,  "cache_write": 3.75},
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00, "cache_read": 1.50,  "cache_write": 18.75},
}


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    rates = _COST_PER_MTOK.get(model, {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75})
    return (
        input_tokens       * rates["input"]       +
        output_tokens      * rates["output"]       +
        cache_read_tokens  * rates["cache_read"]   +
        cache_write_tokens * rates["cache_write"]
    ) / 1_000_000


class TraceLogger:
    """
    Writes structured trace rows to ag_traces and a summary row to ag_sessions.

    Quickstart:
        tracer = TraceLogger(session_id, workflow_id=os.environ["WORKFLOW_ID"])
        tracer.start_agent_span("research")
        tracer.log_tool_call("research", "web_search", {"query": "..."}, result, latency_ms=320)
        tracer.log_agent_message("research", reasoning, "approved", tokens_input=800, tokens_output=200, model="claude-haiku-4-5-20251001")
        tracer.close_session("completed", result_summary="3 trades executed")

    Environment variables required:
        SUPABASE_URL   — your Supabase project URL
        SUPABASE_KEY   — your Supabase service role key
        TENANT_ID      — your Argus tenant UUID (from Settings)
        WORKFLOW_ID    — your Argus pipeline UUID (from Settings)
    """

    def __init__(
        self,
        session_id: str,
        workflow_id: Optional[str] = None,
        session_type: Optional[str] = None,
        default_model: str = "claude-haiku-4-5-20251001",
    ):
        self.session_id    = session_id
        self._tenant_id    = _load_env("TENANT_ID")
        self._workflow_id  = workflow_id or _load_env("WORKFLOW_ID") or None
        self._session_type = session_type
        self._default_model = default_model
        self._sequence     = 0
        self._agent_spans: dict[str, str] = {}
        self._tokens: dict[str, dict[str, int]] = {}
        self._started_at   = datetime.utcnow()
        self._insert_session_stub()

    # ── Public API ──────────────────────────────────────────────────────────────

    def start_agent_span(self, agent: str) -> str:
        """Register a new span for this agent. Returns the new span_id."""
        span_id = str(uuid4())
        self._agent_spans[agent] = span_id
        return span_id

    def log_tool_call(
        self,
        agent: str,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        entity_id: Optional[str] = None,
        latency_ms: int = 0,
        model: Optional[str] = None,
    ) -> str:
        """Write a tool_call row. Returns the new span_id."""
        return self._write({
            "step_type":   "tool_call",
            "agent":       agent,
            "tool_name":   tool_name,
            "tool_input":  tool_input,
            "tool_output": tool_output if isinstance(tool_output, dict) else {"value": tool_output},
            "entity_id":   entity_id,
            "latency_ms":  latency_ms,
            "model":       model,
        })

    def log_agent_message(
        self,
        agent: str,
        reasoning: str,
        outcome: str,
        entity_id: Optional[str] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        model: Optional[str] = None,
        latency_ms: int = 0,
    ) -> str:
        """Write an agent_message row. Returns the new span_id."""
        return self._write({
            "step_type":       "agent_message",
            "agent":           agent,
            "agent_reasoning": reasoning,
            "outcome":         outcome,
            "entity_id":       entity_id,
            "tokens_input":    tokens_input,
            "tokens_output":   tokens_output,
            "latency_ms":      latency_ms,
            "model":           model,
        })

    def log_decision(
        self,
        agent: str,
        outcome: str,
        detail: Optional[dict] = None,
        latency_ms: int = 0,
        model: Optional[str] = None,
    ) -> str:
        """Write a session-level decision row. Returns span_id."""
        return self._write({
            "step_type":   "decision",
            "agent":       agent,
            "outcome":     outcome,
            "tool_output": detail,
            "latency_ms":  latency_ms,
            "model":       model,
        })

    def log_skip(
        self,
        agent: str,
        reason: str,
        skip_type: str = "design",
    ) -> str:
        """
        Record that an agent was intentionally skipped.

        Args:
            agent:     Base agent name (e.g. 'news', 'risk').
            reason:    Human-readable reason (e.g. 'no_candidates', 'upstream_data_missing').
            skip_type: 'design'  — expected routing (no alarm in Argus).
                       'error'   — upstream failure caused the skip (shown as amber in Argus).

        Returns span_id. Argus uses this to distinguish intentional routing
        from error propagation in the session diagnosis and agents list.
        """
        return self._write({
            "step_type": "skip",
            "agent":     agent,
            "outcome":   "skipped",
            "payload":   {"reason": reason, "skip_type": skip_type},
        })

    def log_error(
        self,
        agent: str,
        error_message: str,
        entity_id: Optional[str] = None,
    ) -> str:
        """Write an error row. Returns span_id."""
        return self._write({
            "step_type": "error",
            "agent":     agent,
            "error":     error_message,
            "entity_id": entity_id,
            "outcome":   "error",
        })

    def log_tokens(self, agent: str, usage: Any) -> None:
        """
        Accumulate token counts for an agent. Pass an Anthropic Usage object or a dict.
        Written to ag_sessions at close_session().
        """
        if hasattr(usage, "input_tokens"):
            inp = usage.input_tokens
            out = usage.output_tokens
            cr  = getattr(usage, "cache_read_input_tokens",    0) or 0
            cw  = getattr(usage, "cache_creation_input_tokens", 0) or 0
        else:
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            cr  = usage.get("cache_read_input_tokens",    0) or 0
            cw  = usage.get("cache_creation_input_tokens", 0) or 0

        if agent not in self._tokens:
            self._tokens[agent] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._tokens[agent]["input"]       += inp
        self._tokens[agent]["output"]      += out
        self._tokens[agent]["cache_read"]  += cr
        self._tokens[agent]["cache_write"] += cw

    def close_session(
        self,
        terminal_reason: str,
        agents_invoked: Optional[list[str]] = None,
        loop_iterations: int = 1,
        trades_proposed: int = 0,
        trades_approved: int = 0,
        trades_executed: int = 0,
        risk_rejections: int = 0,
        retry_triggered: bool = False,
        result_summary: Optional[str] = None,
    ) -> None:
        """Finalize the ag_sessions row for this session."""
        from argus.db import get_client

        completed_at = datetime.utcnow()
        latency_ms   = int((completed_at - self._started_at).total_seconds() * 1000)

        metadata: dict[str, Any] = {
            "date":             date.today().isoformat(),
            "total_steps":      self._sequence,
            "trades_proposed":  trades_proposed,
            "trades_approved":  trades_approved,
            "trades_executed":  trades_executed,
            "risk_rejections":  risk_rejections,
            "retry_triggered":  retry_triggered,
            "total_latency_ms": latency_ms,
        }

        row: dict[str, Any] = {
            "terminal_reason": terminal_reason,
            "ended_at":        completed_at.isoformat(),
            "status":          "completed",
            "metadata":        metadata,
        }
        if result_summary:
            row["result_summary"] = result_summary

        if self._tokens:
            agent_costs: dict[str, Any] = {}
            for agent, v in self._tokens.items():
                model = self._default_model
                cost  = _estimate_cost(model, v["input"], v["output"],
                                       v.get("cache_read", 0), v.get("cache_write", 0))
                agent_costs[agent] = {
                    "model":       model,
                    "input":       v["input"],
                    "output":      v["output"],
                    "cache_read":  v.get("cache_read",  0),
                    "cache_write": v.get("cache_write", 0),
                    "cost_usd":    round(cost, 6),
                }
            total_cost   = sum(a["cost_usd"] for a in agent_costs.values())
            total_input  = sum(v["input"]    for v in self._tokens.values())
            total_output = sum(v["output"]   for v in self._tokens.values())
            row.update({
                "total_tokens_input":  total_input,
                "total_tokens_output": total_output,
                "total_cost_usd":      round(total_cost, 6),
            })
            metadata.update({
                "agents_invoked":   agents_invoked or list(self._tokens.keys()),
                "loop_iterations":  loop_iterations,
                "cost_breakdown":   agent_costs,
            })

        get_client().table("ag_sessions").update(row).eq("id", self.session_id).execute()

    def get_sequence(self) -> int:
        return self._sequence

    def get_agent_span(self, agent: str) -> Optional[str]:
        return self._agent_spans.get(agent)

    # ── Private ─────────────────────────────────────────────────────────────────

    def _insert_session_stub(self) -> None:
        from argus.db import get_client
        stub: dict[str, Any] = {
            "id":              self.session_id,
            "tenant_id":       self._tenant_id,
            "workflow_id":     self._workflow_id,
            "started_at":      self._started_at.isoformat(),
            "status":          "in_progress",
            "terminal_reason": "in_progress",
        }
        if self._session_type:
            stub["session_type"] = self._session_type
        get_client().table("ag_sessions").upsert(stub, on_conflict="id", ignore_duplicates=True).execute()

    def _write(self, fields: dict) -> str:
        from argus.db import get_client
        span_id = str(uuid4())
        self._sequence += 1
        agent     = fields.get("agent", "orchestrator")
        entity_id = fields.get("entity_id")

        tokens_input  = fields.get("tokens_input", 0)
        tokens_output = fields.get("tokens_output", 0)
        model         = fields.get("model") or self._default_model
        cost_usd: Optional[float] = None
        if tokens_input or tokens_output:
            cost_usd = round(_estimate_cost(model, tokens_input, tokens_output), 8)

        payload: dict[str, Any] = {
            "span_id":        span_id,
            "parent_span_id": self._agent_spans.get(agent),
            "entity_id":      entity_id,
            "date":           date.today().isoformat(),
            "sequence":       self._sequence,
            "model":          model,
        }
        if fields.get("tool_input")      is not None: payload["tool_input"]      = fields["tool_input"]
        if fields.get("tool_output")     is not None: payload["tool_output"]     = fields["tool_output"]
        if fields.get("agent_reasoning") is not None: payload["agent_reasoning"] = fields["agent_reasoning"]
        if fields.get("payload")         is not None: payload.update(fields["payload"])

        row: dict[str, Any] = {
            "tenant_id":     self._tenant_id,
            "session_id":    self.session_id,
            "agent":         agent,
            "step_type":     fields.get("step_type"),
            "tool_name":     fields.get("tool_name"),
            "outcome":       fields.get("outcome"),
            "error":         fields.get("error"),
            "latency_ms":    fields.get("latency_ms", 0),
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "cost_usd":      cost_usd,
            "payload":       payload,
            "created_at":    datetime.utcnow().isoformat(),
        }
        get_client().table("ag_traces").insert(row).execute()
        return span_id
