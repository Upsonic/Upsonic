"""Append-only ledger row for the usage registry."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional


UsageKind = Literal["llm", "tool", "embedding", "ocr", "system"]


@dataclass(kw_only=True)
class UsageEntry:
    """A single recorded "spend" event.

    Every LLM call, tool invocation, embedding batch, or OCR pass produces
    exactly one entry. Entries are immutable once recorded and idempotent
    by ``entry_id`` — re-recording the same id replaces (not double-counts)
    the previous row. This is what lets the registry survive retries and
    cross-process resume without the baseline arithmetic that
    ``TaskUsage.snapshot`` / ``subtract`` used to need.

    Scope tags ``*_usage_id`` link the entry to one or more execution
    contexts. A single entry can carry every scope tag that applies — e.g.,
    an LLM call inside a Chat that runs an Agent that executes a Task will
    set ``chat_usage_id``, ``agent_usage_id`` and ``task_usage_id`` all at
    once. Aggregation is then filtering by any one of them.
    """

    entry_id: str = field(default_factory=lambda: f"entry-{uuid.uuid4().hex}")
    """Stable unique id; re-recording the same id is idempotent."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp of when the entry was created."""

    kind: UsageKind = "llm"
    """Type of spend event."""

    model: Optional[str] = None
    """Model identifier (e.g. ``openai/gpt-4o``) when applicable."""

    provider: Optional[str] = None
    """Provider identifier when applicable."""

    # ------------------------------------------------------------------
    # Token & cost
    # ------------------------------------------------------------------
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    input_audio_tokens: int = 0
    output_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    cost_usd: Optional[float] = None

    # ------------------------------------------------------------------
    # Timing (seconds)
    # ------------------------------------------------------------------
    duration: float = 0.0
    model_execution_time: float = 0.0
    tool_execution_time: float = 0.0
    time_to_first_token: Optional[float] = None

    # ------------------------------------------------------------------
    # Scope tags — any subset may be set
    # ------------------------------------------------------------------
    chat_usage_id: Optional[str] = None
    agent_usage_id: Optional[str] = None
    task_usage_id: Optional[str] = None
    team_usage_id: Optional[str] = None
    workflow_usage_id: Optional[str] = None
    system_usage_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Cross-references
    # ------------------------------------------------------------------
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    parent_entry_id: Optional[str] = None
    """Set for sub-agent / nested calls; lets queries roll up a tree."""

    pipeline_step: Optional[str] = None
    """Originating pipeline step name (e.g. ``model_call``, ``verifier``)."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Free-form metadata; never relied on by aggregation."""

    # ------------------------------------------------------------------
    # Computed
    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    # ------------------------------------------------------------------
    # Serialization (used in Phase 4 when storage lands)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageEntry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
