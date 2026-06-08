"""Load pipeline config from ag_pipeline_config for a given workflow."""
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
