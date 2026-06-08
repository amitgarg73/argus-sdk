"""
Argus SDK quickstart — minimal working example.

1. Copy .env.example to .env and fill in your values.
2. Run: python examples/quickstart.py
3. Open Argus dashboard and check Sessions — your session should appear.
"""
import os
from uuid import uuid4
from argus import TraceLogger

session_id = str(uuid4())

# Open a session
tracer = TraceLogger(session_id, session_type="demo")

# Log a tool call from one of your agents
tracer.start_agent_span("research")
tracer.log_tool_call(
    agent="research",
    tool_name="web_search",
    tool_input={"query": "latest AI agent frameworks"},
    tool_output={"results": ["LangChain", "CrewAI", "AutoGen"]},
    latency_ms=320,
)

# Log the agent's reasoning and outcome
tracer.log_agent_message(
    agent="research",
    reasoning="Found 3 relevant frameworks. All are production-grade.",
    outcome="success",
    tokens_input=800,
    tokens_output=150,
    model="claude-haiku-4-5-20251001",
)

# Close the session
tracer.close_session("completed", result_summary="Research complete — 3 frameworks identified")

print(f"Session {session_id} written to Argus.")
