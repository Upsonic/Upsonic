"""Scope id helpers for the usage registry."""
from __future__ import annotations

import uuid
from typing import Literal

UsageScope = Literal["chat", "agent", "task", "team", "workflow", "system", "entry"]


def new_usage_id(scope: UsageScope) -> str:
    """Generate a new scope-prefixed usage id.

    The prefix is purely informational — the registry treats ids as opaque
    strings — but it makes logs and storage rows readable.

    Examples:
        >>> new_usage_id("chat")
        'chat-9b1d...'
        >>> new_usage_id("task")
        'task-...'
    """
    return f"{scope}-{uuid.uuid4().hex}"
