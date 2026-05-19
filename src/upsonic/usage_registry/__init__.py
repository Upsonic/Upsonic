"""Centralized usage ledger for LLM / tool / embedding / OCR cost tracking.

This module provides a single source of truth for "who spent how many tokens
on what" across Agents, Tasks, Chats, Teams, Workflows, and system-level
calls (Memory, Reliability layer).

Public surface:
    UsageEntry        — append-only ledger row (one per LLM/tool/etc. call)
    AggregatedUsage   — read-only view derived from a filter over entries
    UsageRegistry     — in-memory registry; queries by scope tag
    new_usage_id      — uuid generator with a scope prefix

Phase 0 keeps everything in-process; cross-process persistence lands in a
later phase via the existing :class:`upsonic.storage.base.Storage` layer.
"""
from __future__ import annotations

from upsonic.usage_registry.entry import UsageEntry, UsageKind
from upsonic.usage_registry.aggregated import AggregatedUsage
from upsonic.usage_registry.registry import UsageRegistry, get_default_registry
from upsonic.usage_registry.ids import new_usage_id
from upsonic.usage_registry.scope import (
    scope,
    current_scope_tags,
    push_scope_tags,
    reset_scope_tags,
)
from upsonic.usage_registry.recorder import (
    record_request_usage,
    record_response_usage,
)

__all__ = (
    "UsageEntry",
    "UsageKind",
    "AggregatedUsage",
    "UsageRegistry",
    "get_default_registry",
    "new_usage_id",
    "scope",
    "current_scope_tags",
    "push_scope_tags",
    "reset_scope_tags",
    "record_request_usage",
    "record_response_usage",
)
