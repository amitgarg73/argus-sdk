"""Load pipeline config and eval definitions from Supabase for a given workflow."""
from __future__ import annotations


def load_pipeline_config(workflow_id: str, db) -> dict:
    """
    Fetch all ag_pipeline_config rows for workflow_id and return as flat dict.
    Values are already parsed from JSONB by Supabase (int/float/list/dict).
    Returns empty dict on any error so callers fall back to engine defaults.
    """
    try:
        rows = (
            db.table("ag_pipeline_config")
            .select("key, value")
            .eq("workflow_id", workflow_id)
            .execute()
        )
        return {r["key"]: r["value"] for r in (rows.data or [])}
    except Exception:
        return {}


def load_eval_configs(workflow_id: str, db, layer: int | None = None) -> list[dict]:
    """
    Fetch ag_eval_configs rows for workflow_id.
    Optionally filter by layer (3, 4, or 5).
    Returns empty list on any error.
    """
    try:
        q = (
            db.table("ag_eval_configs")
            .select("*")
            .eq("workflow_id", workflow_id)
            .eq("enabled", True)
        )
        if layer is not None:
            q = q.eq("layer", layer)
        rows = q.execute()
        return rows.data or []
    except Exception:
        return []


def load_pipeline_agents(workflow_id: str, db) -> list[dict]:
    """
    Fetch ag_pipeline_agents rows for workflow_id, ordered by sort_order.
    Each row has: agent_name, display_name, emoji, sort_order.
    Returns empty list on any error.
    """
    try:
        rows = (
            db.table("ag_pipeline_agents")
            .select("agent_name, display_name, emoji, sort_order")
            .eq("workflow_id", workflow_id)
            .order("sort_order")
            .execute()
        )
        return rows.data or []
    except Exception:
        return []
