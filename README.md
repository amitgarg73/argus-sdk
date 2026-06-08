# argus-sdk

Python SDK for the [Argus](https://argusobs.vercel.app) AI agent observability platform.

Use this to connect your Python pipeline to Argus so every session, agent step, and evaluation appears on your dashboard automatically.

---

## Install

```bash
pip install git+https://github.com/amitgarg73/argus-sdk.git
```

---

## Environment setup

Create a `.env` file (copy from `.env.example`) and fill in four values:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
TENANT_ID=your-argus-tenant-uuid
WORKFLOW_ID=your-argus-pipeline-uuid
```

Find `TENANT_ID` and `WORKFLOW_ID` in Argus under **Settings**.

---

## Quickstart

Wrap each pipeline run with a session open and close. Log each agent step in between.

```python
from uuid import uuid4
from argus import TraceLogger

session_id = str(uuid4())
tracer = TraceLogger(session_id, session_type="scheduled")

# One call per agent step
tracer.start_agent_span("research")
tracer.log_tool_call(
    agent="research",
    tool_name="web_search",
    tool_input={"query": "market conditions"},
    tool_output={"results": [...]},
    latency_ms=320,
)
tracer.log_agent_message(
    agent="research",
    reasoning="Found 3 relevant signals. Confidence high.",
    outcome="success",
    tokens_input=800,
    tokens_output=150,
    model="claude-haiku-4-5-20251001",
)

tracer.close_session("completed", result_summary="3 signals confirmed")
```

Check **Sessions** in Argus — your session should appear within seconds.

---

## Layer 4 — LLM-as-judge (semantic quality scoring)

Once you have eval criteria configured in **Argus Eval Manager**, call `evaluate_session_outputs()` after your agents finish. Argus scores each agent's output against your criteria using Claude Haiku.

```python
from argus import TraceLogger, evaluate_session_outputs

# ... run your pipeline, collect agent outputs as strings ...

agent_outputs = {
    "research": json.dumps(research_result),
    "risk":     json.dumps(risk_result),
}

# Call before close_session
evaluate_session_outputs(session_id, agent_outputs)

tracer.close_session("completed")
```

Requires `ANTHROPIC_API_KEY` in your environment. Eval criteria must be set up in Argus first — go to **Eval Manager** and add criteria for each agent.

---

## Layer 5 — Business outcome evals

Layer 5 is for metrics you compute yourself from session results. There is no LLM involved — you calculate the number and pass it to `write_eval()`. It appears in Argus **Outcomes** as a business KPI.

```python
from argus import write_eval

# After your session runs, compute your business metrics
proposals_generated = 5
proposals_approved  = 3

# Did the pipeline produce any output?
write_eval(
    session_id=session_id,
    eval_name="research_yield",
    agent="research",
    score=1.0 if proposals_generated > 0 else 0.0,
    passed=proposals_generated > 0,
    threshold=0.9,
    reasoning=f"Pipeline generated {proposals_generated} proposals",
)

# What fraction passed the approval step?
if proposals_generated > 0:
    rate = proposals_approved / proposals_generated
    write_eval(
        session_id=session_id,
        eval_name="approval_rate",
        agent="risk",
        score=rate,
        passed=rate >= 0.20,
        threshold=0.20,
        reasoning=f"{proposals_approved} of {proposals_generated} approved",
    )
```

Call `write_eval()` after `close_session()`. One call per metric.

---

## Full example with all three layers

```python
from uuid import uuid4
import json
from argus import TraceLogger, evaluate_session_outputs, write_eval

session_id = str(uuid4())
tracer = TraceLogger(session_id, session_type="scheduled")

# Layer 1-3: trace every agent step
tracer.start_agent_span("research")
tracer.log_agent_message("research", reasoning, "success",
                         tokens_input=1200, tokens_output=300,
                         model="claude-haiku-4-5-20251001")

# Layer 4: LLM-as-judge (criteria from Argus Eval Manager)
evaluate_session_outputs(session_id, {"research": json.dumps(research_output)})

# Close session
tracer.close_session("completed", result_summary="Pipeline complete")

# Layer 5: your business metrics
write_eval(session_id, "research_yield", "research",
           score=1.0, passed=True, threshold=0.9,
           reasoning="Research produced 4 proposals")
```

---

## API reference

### `TraceLogger(session_id, workflow_id=None, session_type=None, default_model="claude-haiku-4-5-20251001")`

| Parameter | Description |
|---|---|
| `session_id` | Unique ID for this pipeline run (use `str(uuid4())`) |
| `workflow_id` | Your Argus pipeline UUID. Defaults to `WORKFLOW_ID` env var. |
| `session_type` | Optional label: `"scheduled"`, `"manual"`, `"intraday"`, etc. |
| `default_model` | Default Claude model for cost estimation when `model=` is not passed per call. |

Key methods:

| Method | When to call |
|---|---|
| `start_agent_span(agent)` | At the start of each agent's turn |
| `log_tool_call(agent, tool_name, tool_input, tool_output, latency_ms)` | After each tool/function call |
| `log_agent_message(agent, reasoning, outcome, tokens_input, tokens_output, model)` | After each LLM call |
| `log_decision(agent, outcome, detail)` | For session-level decisions (no tool, no entity) |
| `log_error(agent, error_message)` | When an agent fails |
| `log_tokens(agent, usage)` | Pass an Anthropic `Usage` object to accumulate cost |
| `close_session(terminal_reason, result_summary)` | At the very end of the pipeline run |

### `evaluate_session_outputs(session_id, agent_outputs)`

Runs LLM-as-judge scoring for all agents in `agent_outputs`. Returns a dict of results. Writes rows to `ag_evals` and opens `ag_incidents` for any failures.

### `write_eval(session_id, eval_name, agent, score, passed, threshold, reasoning, layer=5)`

Writes one business outcome eval row. Use for any metric you compute yourself.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Your Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase service role key |
| `TENANT_ID` | Yes | Your Argus tenant UUID (from Settings) |
| `WORKFLOW_ID` | Yes | Your Argus pipeline UUID (from Settings) |
| `ANTHROPIC_API_KEY` | Only for L4 eval | Required by `evaluate_session_outputs()` |

---

## Examples

- `examples/quickstart.py` — minimal TraceLogger usage
- `examples/business_evals.py` — Layer 5 business metrics pattern

---

## Support

Open an issue on this repo or email amit.thirdeyetrading@gmail.com.
