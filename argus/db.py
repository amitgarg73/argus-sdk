from __future__ import annotations

import os
from supabase import create_client, Client

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
    return _client


def reset_client() -> None:
    """Force re-initialization on next get_client() call. Useful in tests."""
    global _client
    _client = None
