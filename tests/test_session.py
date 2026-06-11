"""Tests for TraceLogger — focuses on log_skip and the skip trace payload."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import pytest
from argus.session import TraceLogger


def _make_tracer() -> tuple[TraceLogger, list[dict]]:
    """Return a TraceLogger wired to a mock Supabase client + captured rows."""
    captured: list[dict] = []

    mock_result = MagicMock()
    mock_result.data = [{"id": "span-1"}]
    mock_client = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.return_value = mock_result
    mock_client.table.return_value.upsert.return_value.execute.return_value = mock_result
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_result

    def capture_insert(row):
        captured.append(row)
        return mock_client.table.return_value.insert.return_value

    mock_client.table.return_value.insert.side_effect = capture_insert

    with patch("argus.db.get_client", return_value=mock_client), \
         patch.dict(os.environ, {"TENANT_ID": "t1", "WORKFLOW_ID": "w1", "SUPABASE_URL": "x", "SUPABASE_KEY": "y"}):
        tracer = TraceLogger.__new__(TraceLogger)
        tracer.session_id    = "sess-test"
        tracer._tenant_id    = "t1"
        tracer._workflow_id  = "w1"
        tracer._session_type = None
        tracer._default_model = "claude-haiku-4-5-20251001"
        tracer._sequence     = 0
        tracer._agent_spans  = {}
        tracer._tokens       = {}
        from datetime import datetime
        tracer._started_at   = datetime.utcnow()
        tracer._db           = mock_client

    return tracer, captured


class TestLogSkip:
    def test_design_skip_writes_trace_row(self):
        tracer, captured = _make_tracer()
        with patch("argus.db.get_client", return_value=tracer._db):
            tracer.log_skip("news", reason="no_candidates", skip_type="design")
        row = captured[-1]
        assert row["step_type"] == "skip"
        assert row["agent"] == "news"
        assert row["outcome"] == "skipped"
        assert row["payload"]["reason"] == "no_candidates"
        assert row["payload"]["skip_type"] == "design"

    def test_error_skip_sets_skip_type_error(self):
        tracer, captured = _make_tracer()
        with patch("argus.db.get_client", return_value=tracer._db):
            tracer.log_skip("risk", reason="upstream_data_missing", skip_type="error")
        row = captured[-1]
        assert row["payload"]["skip_type"] == "error"
        assert row["payload"]["reason"] == "upstream_data_missing"

    def test_default_skip_type_is_design(self):
        tracer, captured = _make_tracer()
        with patch("argus.db.get_client", return_value=tracer._db):
            tracer.log_skip("news", reason="no_candidates")
        row = captured[-1]
        assert row["payload"]["skip_type"] == "design"

    def test_log_skip_returns_span_id(self):
        tracer, _ = _make_tracer()
        with patch("argus.db.get_client", return_value=tracer._db):
            result = tracer.log_skip("news", reason="no_candidates")
        assert isinstance(result, str)
        assert len(result) > 0
