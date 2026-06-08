"""
Example: Layer 5 business outcome evals.

Use write_eval() to record any domain-specific metric you compute from
your pipeline's session results. These appear as L5 evals in Argus.

Pattern: compute your number -> call write_eval() -> it appears in Outcomes.
"""
import os
from uuid import uuid4
from argus import TraceLogger, write_eval

session_id = str(uuid4())
tracer = TraceLogger(session_id, session_type="demo")

# --- simulate your pipeline running ---
proposals_generated = 5
proposals_approved  = 3
# --- pipeline done ---

tracer.close_session("completed", trades_proposed=proposals_generated, trades_approved=proposals_approved)

# Write L5 business evals after close_session
# Metric 1: did the pipeline produce any proposals?
research_score = 1.0 if proposals_generated > 0 else 0.0
write_eval(
    session_id=session_id,
    eval_name="research_yield",
    agent="research",
    score=research_score,
    passed=research_score >= 0.9,
    threshold=0.9,
    reasoning=f"Pipeline generated {proposals_generated} proposals",
)

# Metric 2: what fraction survived the approval step?
if proposals_generated > 0:
    approval_rate = proposals_approved / proposals_generated
    write_eval(
        session_id=session_id,
        eval_name="approval_rate",
        agent="risk",
        score=approval_rate,
        passed=approval_rate >= 0.20,
        threshold=0.20,
        reasoning=f"{proposals_approved} of {proposals_generated} proposals approved",
    )

print(f"Business evals written for session {session_id}")
