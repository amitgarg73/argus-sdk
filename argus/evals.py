from __future__ import annotations

import os
from uuid import uuid4


def write_eval(
    session_id: str,
    eval_name: str,
    agent: str,
    score: float,
    passed: bool,
    threshold: float,
    reasoning: str = "",
    layer: int = 5,
) -> None:
    """
    Write a single eval row to ag_evals.

    Use this for Layer 5 business outcome metrics — deterministic scores
    you compute yourself from session results.

    Args:
        session_id:  The session ID from TraceLogger.
        eval_name:   Short snake_case name for the metric (e.g. "approval_rate").
        agent:       Which agent this metric belongs to (e.g. "risk").
        score:       Numeric score, 0.0 to 1.0 (or a raw rate like 0.67).
        passed:      Whether the score meets the threshold.
        threshold:   The passing threshold used to derive `passed`.
        reasoning:   One-sentence explanation written to the Argus UI.
        layer:       Eval layer (default 5 for business outcome evals).
    """
    tenant_id = os.environ.get("TENANT_ID", "")
    if not tenant_id:
        return
    from argus.db import get_client
    get_client().table("ag_evals").insert({
        "id":         str(uuid4()),
        "tenant_id":  tenant_id,
        "session_id": session_id,
        "eval_name":  eval_name,
        "agent":      agent,
        "layer":      layer,
        "score":      round(float(score), 4),
        "passed":     passed,
        "threshold":  threshold,
        "detail":     {"reasoning": reasoning},
    }).execute()
