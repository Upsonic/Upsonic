"""Read-only roll-up of :class:`UsageEntry` rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from upsonic.usage_registry.entry import UsageEntry


@dataclass(kw_only=True)
class AggregatedUsage:
    """Result of summing a filter over :class:`UsageEntry` rows.

    Shape intentionally mirrors :class:`upsonic.usage.TaskUsage` so the
    Phase-3 read-through wrappers can drop in without changing callers.
    Unlike ``TaskUsage`` this object is *derived* — no ``incr()``, no
    timer, no in-place mutation.
    """

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
    cost: Optional[float] = None
    """Sum of ``cost_usd`` across contributing entries; ``None`` if no
    contributing entry had a cost (vs ``0.0`` which means "we tried to
    price it and it came out free")."""

    duration: float = 0.0
    model_execution_time: float = 0.0
    tool_execution_time: float = 0.0
    time_to_first_token: Optional[float] = None
    """Earliest non-``None`` TTFT across entries; ``None`` if no entry
    recorded one."""

    entry_count: int = 0
    models: List[str] = field(default_factory=list)
    """Distinct models that contributed, preserving first-seen order."""

    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def upsonic_execution_time(self) -> float:
        """Time spent inside Upsonic (orchestration, pipeline, etc.) —
        wall-clock duration minus what we know was external."""
        return max(
            0.0,
            self.duration - self.model_execution_time - self.tool_execution_time,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_entries(cls, entries: Iterable[UsageEntry]) -> "AggregatedUsage":
        agg = cls()
        any_cost = False
        for e in entries:
            agg.entry_count += 1
            agg.input_tokens += e.input_tokens
            agg.output_tokens += e.output_tokens
            agg.cache_read_tokens += e.cache_read_tokens
            agg.cache_write_tokens += e.cache_write_tokens
            agg.reasoning_tokens += e.reasoning_tokens
            agg.input_audio_tokens += e.input_audio_tokens
            agg.output_audio_tokens += e.output_audio_tokens
            agg.cache_audio_read_tokens += e.cache_audio_read_tokens
            agg.requests += e.requests
            agg.tool_calls += e.tool_calls
            agg.duration += e.duration
            agg.model_execution_time += e.model_execution_time
            agg.tool_execution_time += e.tool_execution_time

            if e.cost_usd is not None:
                any_cost = True
                agg.cost = (agg.cost or 0.0) + e.cost_usd

            if e.time_to_first_token is not None and agg.time_to_first_token is None:
                agg.time_to_first_token = e.time_to_first_token

            if e.model and e.model not in agg.models:
                agg.models.append(e.model)

        if not any_cost:
            agg.cost = None
        return agg

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "input_audio_tokens": self.input_audio_tokens,
            "output_audio_tokens": self.output_audio_tokens,
            "cache_audio_read_tokens": self.cache_audio_read_tokens,
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "cost": self.cost,
            "duration": self.duration,
            "model_execution_time": self.model_execution_time,
            "tool_execution_time": self.tool_execution_time,
            "upsonic_execution_time": self.upsonic_execution_time,
            "time_to_first_token": self.time_to_first_token,
            "entry_count": self.entry_count,
            "models": list(self.models),
        }
