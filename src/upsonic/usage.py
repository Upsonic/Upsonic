from __future__ import annotations as _annotations

import dataclasses
from copy import copy
from dataclasses import dataclass, fields
from typing import Annotated, Any, Dict, Optional, TYPE_CHECKING

from genai_prices.data_snapshot import get_snapshot
from pydantic import AliasChoices, BeforeValidator, Field
from typing_extensions import deprecated, overload

from upsonic import _utils
from upsonic.utils.package.exception import UsageLimitExceeded

if TYPE_CHECKING:
    from upsonic.utils.timer import Timer

__all__ = 'RequestUsage', 'TaskUsage', 'RunUsage', 'AgentUsage', 'Usage', 'UsageLimits'


@dataclass(repr=False, kw_only=True)
class UsageBase:
    input_tokens: Annotated[
        int,
        # `request_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('input_tokens', 'request_tokens')),
    ] = 0
    """Number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Number of tokens written to the cache."""
    cache_read_tokens: int = 0
    """Number of tokens read from the cache."""

    output_tokens: Annotated[
        int,
        # `response_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('output_tokens', 'response_tokens')),
    ] = 0
    """Number of output/completion tokens."""

    input_audio_tokens: int = 0
    """Number of audio input tokens."""
    cache_audio_read_tokens: int = 0
    """Number of audio tokens read from the cache."""
    output_audio_tokens: int = 0
    """Number of audio output tokens."""

    details: Annotated[
        dict[str, int],
        # `details` can not be `None` any longer, but we still want to support deserializing model responses stored in a DB before this was changed
        BeforeValidator(lambda d: d or {}),
    ] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    @property
    @deprecated('`request_tokens` is deprecated, use `input_tokens` instead')
    def request_tokens(self) -> int:
        return self.input_tokens

    @property
    @deprecated('`response_tokens` is deprecated, use `output_tokens` instead')
    def response_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_tokens(self) -> int:
        """Sum of `input_tokens + output_tokens`."""
        return self.input_tokens + self.output_tokens

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token usage values as OpenTelemetry attributes."""
        result: dict[str, int] = {}
        if self.input_tokens:
            result['gen_ai.usage.input_tokens'] = self.input_tokens
        if self.output_tokens:
            result['gen_ai.usage.output_tokens'] = self.output_tokens

        details = self.details.copy()
        if self.cache_write_tokens:
            details['cache_write_tokens'] = self.cache_write_tokens
        if self.cache_read_tokens:
            details['cache_read_tokens'] = self.cache_read_tokens
        if self.input_audio_tokens:
            details['input_audio_tokens'] = self.input_audio_tokens
        if self.cache_audio_read_tokens:
            details['cache_audio_read_tokens'] = self.cache_audio_read_tokens
        if self.output_audio_tokens:
            details['output_audio_tokens'] = self.output_audio_tokens
        if details:
            prefix = 'gen_ai.usage.details.'
            for key, value in details.items():
                # Skipping check for value since spec implies all detail values are relevant
                if value:
                    result[prefix + key] = value
        return result

    def __repr__(self):
        kv_pairs = (f'{f.name}={value!r}' for f in fields(self) if (value := getattr(self, f.name)))
        return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return any(dataclasses.asdict(self).values())


@dataclass(repr=False, kw_only=True)
class RequestUsage(UsageBase):
    """LLM usage associated with a single request.

    Represents token counts from a single LLM API call. This class is immutable
    after creation — it should not be used for accumulation across multiple calls.

    This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the
    request using genai-prices.
    """

    @property
    def requests(self) -> int:
        return 1

    def incr(self, incr_usage: "RequestUsage") -> None:
        """Increment token counts in place (for summing response parts only).

        Args:
            incr_usage: The usage to increment by.
        """
        _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: "RequestUsage") -> "RequestUsage":
        """Add two RequestUsages together.

        This is provided so it's trivial to sum usage information from multiple parts of a response.

        **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    @classmethod
    def extract(
        cls,
        data: Any,
        *,
        provider: str,
        provider_url: str,
        provider_fallback: str,
        api_flavor: str = 'default',
        details: dict[str, Any] | None = None,
    ) -> "RequestUsage":
        """Extract usage information from the response data using genai-prices.

        Args:
            data: The response data from the model API.
            provider: The actual provider ID
            provider_url: The provider base_url
            provider_fallback: The fallback provider ID to use if the actual provider is not found in genai-prices.
                For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID.
            api_flavor: The API flavor to use when extracting usage information,
                e.g. 'chat' or 'responses' for OpenAI.
            details: Becomes the `details` field on the returned `RequestUsage` for convenience.
        """
        details = details or {}
        for provider_id, provider_api_url in [(None, provider_url), (provider, None), (provider_fallback, None)]:
            try:
                provider_obj = get_snapshot().find_provider(None, provider_id, provider_api_url)
                _model_ref, extracted_usage = provider_obj.extract_usage(data, api_flavor=api_flavor)
                return cls(**{k: v for k, v in extracted_usage.__dict__.items() if v is not None}, details=details)
            except Exception:
                pass
        return cls(details=details)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the token usage.
        """
        result: Dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "input_audio_tokens": self.input_audio_tokens,
            "cache_audio_read_tokens": self.cache_audio_read_tokens,
            "output_audio_tokens": self.output_audio_tokens,
        }

        if self.details:
            result["details"] = self.details

        result = {
            k: v for k, v in result.items()
            if v is not None and (not isinstance(v, (int, float)) or v != 0) and (not isinstance(v, dict) or len(v) > 0)
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestUsage":
        """Reconstruct RequestUsage from dictionary.

        Args:
            data: Dictionary containing RequestUsage fields.

        Returns:
            RequestUsage instance.
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            input_audio_tokens=data.get("input_audio_tokens", 0),
            cache_audio_read_tokens=data.get("cache_audio_read_tokens", 0),
            output_audio_tokens=data.get("output_audio_tokens", 0),
            details=data.get("details", {}),
        )


@dataclass(repr=False, kw_only=True)
class TaskUsage(UsageBase):
    """Task-level usage that accumulates metrics across all LLM calls within a single task.

    Lives on ``task._usage`` and ``AgentRunOutput.usage``.
    Accepts ``RequestUsage`` (from model responses) and other ``TaskUsage`` (from
    sub-agents, culture, reflection, policies, etc.) via ``incr()``.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""

    tool_calls: int = 0
    """Number of successful tool calls executed during the run."""

    input_tokens: int = 0
    """Total number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Total number of tokens written to the cache."""

    cache_read_tokens: int = 0
    """Total number of tokens read from the cache."""

    input_audio_tokens: int = 0
    """Total number of audio input tokens."""

    cache_audio_read_tokens: int = 0
    """Total number of audio tokens read from the cache."""

    output_tokens: int = 0
    """Total number of output/completion tokens."""

    output_audio_tokens: int = 0
    """Total number of audio output tokens."""

    reasoning_tokens: int = 0
    """Number of tokens employed in reasoning."""

    details: dict[str, int] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    timer: Optional["Timer"] = None
    """Internal timer utility for tracking execution time."""

    time_to_first_token: Optional[float] = None
    """Time from run start to first token generation, in seconds.

    Only set for streaming runs. Non-streaming runs receive the full
    response in a single request, so this metric is not applicable.
    """

    duration: Optional[float] = None
    """Total run time, in seconds."""

    model_execution_time: Optional[float] = None
    """Total time spent in LLM API calls (model.request()), in seconds."""

    tool_execution_time: Optional[float] = None
    """Total time spent executing tools (external API calls, etc.), in seconds."""

    pause_time: Optional[float] = None
    """Total time spent paused (HITL waiting), in seconds.

    Accumulated automatically by the start/stop timer cycle during
    pause → resume transitions.  Excluded from ``upsonic_execution_time``
    so that framework overhead reflects only active processing.
    """

    _pause_stop_ts: Optional[float] = dataclasses.field(default=None, repr=False)
    """Internal: wall-clock (time.time()) timestamp when the timer was stopped for a pause.
    Uses time.time() instead of perf_counter() so it can be serialized and used across processes."""

    @property
    def upsonic_execution_time(self) -> Optional[float]:
        """Framework overhead time = duration - model_execution_time - tool_execution_time - pause_time, in seconds."""
        if self.duration is None or self.model_execution_time is None:
            return None
        tool_time = self.tool_execution_time or 0.0
        p_time = self.pause_time or 0.0
        return max(0.0, self.duration - self.model_execution_time - tool_time - p_time)

    cost: Optional[float] = None
    """Estimated cost of the run (provider-specific)."""

    provider_metrics: Optional[Dict[str, Any]] = None
    """Provider-specific metrics (e.g., latency breakdown, model info)."""

    additional_metrics: Optional[Dict[str, Any]] = None
    """Any additional custom metrics."""

    def incr(self, incr_usage: "TaskUsage | RequestUsage") -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        if isinstance(incr_usage, TaskUsage):
            self.requests += incr_usage.requests
            self.tool_calls += incr_usage.tool_calls
            self.reasoning_tokens += incr_usage.reasoning_tokens

            if incr_usage.cost is not None:
                if self.cost is None:
                    self.cost = incr_usage.cost
                else:
                    self.cost += incr_usage.cost

            if incr_usage.duration is not None:
                if self.duration is None:
                    self.duration = incr_usage.duration
                else:
                    self.duration += incr_usage.duration

            if incr_usage.model_execution_time is not None:
                if self.model_execution_time is None:
                    self.model_execution_time = incr_usage.model_execution_time
                else:
                    self.model_execution_time += incr_usage.model_execution_time

            if incr_usage.tool_execution_time is not None:
                if self.tool_execution_time is None:
                    self.tool_execution_time = incr_usage.tool_execution_time
                else:
                    self.tool_execution_time += incr_usage.tool_execution_time

            if incr_usage.pause_time is not None:
                if self.pause_time is None:
                    self.pause_time = incr_usage.pause_time
                else:
                    self.pause_time += incr_usage.pause_time

            if incr_usage.time_to_first_token is not None:
                if self.time_to_first_token is None:
                    self.time_to_first_token = incr_usage.time_to_first_token

            if incr_usage.provider_metrics:
                if self.provider_metrics is None:
                    self.provider_metrics = {}
                self.provider_metrics.update(incr_usage.provider_metrics)

            if incr_usage.additional_metrics:
                if self.additional_metrics is None:
                    self.additional_metrics = {}
                self.additional_metrics.update(incr_usage.additional_metrics)
        elif isinstance(incr_usage, RequestUsage):
            self.requests += 1

        details = getattr(incr_usage, "details", None)
        if isinstance(details, dict) and details:
            if "reasoning_tokens" in details:
                self.reasoning_tokens += details.get("reasoning_tokens", 0)

        if isinstance(incr_usage, (TaskUsage, RequestUsage)):
            _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: "TaskUsage | RequestUsage") -> "TaskUsage":
        """Add two usages together."""
        new_usage = copy(self)
        if self.details:
            new_usage.details = self.details.copy()
        if self.provider_metrics:
            new_usage.provider_metrics = self.provider_metrics.copy()
        if self.additional_metrics:
            new_usage.additional_metrics = self.additional_metrics.copy()
        new_usage.incr(other)
        return new_usage

    def __radd__(self, other: "TaskUsage | RequestUsage | int") -> "TaskUsage":
        """Right add to support sum() starting with 0."""
        if other == 0:
            return self
        return self + other

    def start_timer(self) -> None:
        """Start the internal timer for tracking execution time.

        On the very first call a new :class:`Timer` is created and started.
        On subsequent calls (i.e. after a pause → resume) the gap between
        the previous ``stop_timer()`` and this ``start_timer()`` is
        accumulated into ``pause_time`` so that ``duration`` (wall-clock)
        naturally includes the pause period while ``upsonic_execution_time``
        excludes it.
        """
        from time import time as wall_time
        from upsonic.utils.timer import Timer

        # If resuming after a pause, accumulate the pause gap into both
        # pause_time (for breakdown) and duration (for wall-clock total)
        if self._pause_stop_ts is not None:
            gap = wall_time() - self._pause_stop_ts
            if gap < 0:
                gap = 0  # Guard against clock adjustments
            if self.pause_time is None:
                self.pause_time = gap
            else:
                self.pause_time += gap
            if self.duration is not None:
                self.duration += gap
            self._pause_stop_ts = None

        if self.timer is None:
            self.timer = Timer()
        self.timer.start()

    def stop_timer(self, set_duration: bool = True) -> None:
        """Stop the internal timer and optionally accumulate duration.

        Additive: if ``duration`` already holds a value from a prior run
        (e.g. an initial HITL run), the new elapsed time is **added** so
        the total reflects processing time across all runs.

        Also records a ``_pause_stop_ts`` so that if ``start_timer`` is
        called again later (resume), the pause gap can be measured.

        Args:
            set_duration: If True, accumulate timer.elapsed into self.duration.
        """
        from time import time as wall_time

        if self.timer is not None:
            self.timer.stop()
            # Record when we stopped so start_timer can measure pause gap
            # Uses wall-clock time so it works across process boundaries
            self._pause_stop_ts = wall_time()
            if set_duration:
                elapsed: float = self.timer.elapsed
                if self.duration is not None:
                    self.duration += elapsed
                else:
                    self.duration = elapsed

    def set_time_to_first_token(self) -> None:
        """Record the time to first token from timer's elapsed time."""
        if self.timer is not None:
            self.time_to_first_token = self.timer.elapsed

    def add_model_execution_time(self, elapsed: float) -> None:
        """Accumulate model (LLM API) execution time.

        Args:
            elapsed: Time in seconds spent in a single model.request() call.
        """
        if self.model_execution_time is None:
            self.model_execution_time = elapsed
        else:
            self.model_execution_time += elapsed

    def add_tool_execution_time(self, elapsed: float) -> None:
        """Accumulate tool execution time.

        Args:
            elapsed: Time in seconds spent in a single tool execution.
        """
        if self.tool_execution_time is None:
            self.tool_execution_time = elapsed
        else:
            self.tool_execution_time += elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the usage metrics.
        """
        result: Dict[str, Any] = {
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "input_audio_tokens": self.input_audio_tokens,
            "cache_audio_read_tokens": self.cache_audio_read_tokens,
            "output_audio_tokens": self.output_audio_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

        if self.details:
            result["details"] = self.details
        if self.cost is not None:
            result["cost"] = self.cost
        if self.duration is not None:
            result["duration"] = self.duration
        if self.model_execution_time is not None:
            result["model_execution_time"] = self.model_execution_time
        if self.tool_execution_time is not None:
            result["tool_execution_time"] = self.tool_execution_time
        if self.pause_time is not None:
            result["pause_time"] = self.pause_time
        if self.upsonic_execution_time is not None:
            result["upsonic_execution_time"] = self.upsonic_execution_time
        if self.time_to_first_token is not None:
            result["time_to_first_token"] = self.time_to_first_token
        if self.provider_metrics:
            result["provider_metrics"] = self.provider_metrics
        if self.additional_metrics:
            result["additional_metrics"] = self.additional_metrics
        if self._pause_stop_ts is not None:
            result["_pause_stop_ts"] = self._pause_stop_ts

        result = {
            k: v for k, v in result.items()
            if v is not None and (not isinstance(v, (int, float)) or v != 0) and (not isinstance(v, dict) or len(v) > 0)
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskUsage":
        """Reconstruct TaskUsage from dictionary.

        Args:
            data: Dictionary containing TaskUsage fields.

        Returns:
            TaskUsage instance.
        """
        return cls(
            requests=data.get("requests", 0),
            tool_calls=data.get("tool_calls", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            input_audio_tokens=data.get("input_audio_tokens", 0),
            cache_audio_read_tokens=data.get("cache_audio_read_tokens", 0),
            output_audio_tokens=data.get("output_audio_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            details=data.get("details", {}),
            cost=data.get("cost"),
            duration=data.get("duration"),
            model_execution_time=data.get("model_execution_time"),
            tool_execution_time=data.get("tool_execution_time"),
            pause_time=data.get("pause_time"),
            time_to_first_token=data.get("time_to_first_token"),
            provider_metrics=data.get("provider_metrics"),
            additional_metrics=data.get("additional_metrics"),
            _pause_stop_ts=data.get("_pause_stop_ts"),
        )


RunUsage = TaskUsage
"""Backward-compatible alias. Use ``TaskUsage`` in new code."""


@dataclass(repr=False, kw_only=True)
class AgentUsage(UsageBase):
    """Agent-level usage that aggregates ``TaskUsage`` across multiple task runs.

    Lives on ``agent.usage``. Accumulates via ``incr(TaskUsage)`` at the end of
    each task execution.
    """

    requests: int = 0
    """Total number of LLM API requests across all tasks."""

    tool_calls: int = 0
    """Total number of tool calls across all tasks."""

    input_tokens: int = 0
    """Total input tokens across all tasks."""

    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    input_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0

    output_tokens: int = 0
    """Total output tokens across all tasks."""

    output_audio_tokens: int = 0
    reasoning_tokens: int = 0

    details: dict[str, int] = dataclasses.field(default_factory=dict)

    duration: Optional[float] = None
    """Total execution time across all tasks, in seconds."""

    model_execution_time: Optional[float] = None
    """Total time spent in LLM API calls across all tasks, in seconds."""

    tool_execution_time: Optional[float] = None
    """Total time spent executing tools across all tasks, in seconds."""

    pause_time: Optional[float] = None
    """Total time spent paused (HITL waiting) across all tasks, in seconds."""

    @property
    def upsonic_execution_time(self) -> Optional[float]:
        """Framework overhead time = duration - model_execution_time - tool_execution_time - pause_time, in seconds."""
        if self.duration is None or self.model_execution_time is None:
            return None
        tool_time = self.tool_execution_time or 0.0
        p_time = self.pause_time or 0.0
        return max(0.0, self.duration - self.model_execution_time - tool_time - p_time)

    cost: Optional[float] = None
    """Total estimated cost across all tasks."""

    def incr(self, incr_usage: "TaskUsage | AgentUsage") -> None:
        """Increment agent-level usage from a completed task or another agent.

        Args:
            incr_usage: TaskUsage from a completed task, or AgentUsage from a sub-agent.
        """
        if isinstance(incr_usage, (TaskUsage, AgentUsage)):
            self.requests += incr_usage.requests
            self.tool_calls += getattr(incr_usage, "tool_calls", 0)
            self.reasoning_tokens += getattr(incr_usage, "reasoning_tokens", 0)

            if incr_usage.cost is not None:
                if self.cost is None:
                    self.cost = incr_usage.cost
                else:
                    self.cost += incr_usage.cost

            if incr_usage.duration is not None:
                if self.duration is None:
                    self.duration = incr_usage.duration
                else:
                    self.duration += incr_usage.duration

            if incr_usage.model_execution_time is not None:
                if self.model_execution_time is None:
                    self.model_execution_time = incr_usage.model_execution_time
                else:
                    self.model_execution_time += incr_usage.model_execution_time

            if getattr(incr_usage, "tool_execution_time", None) is not None:
                if self.tool_execution_time is None:
                    self.tool_execution_time = incr_usage.tool_execution_time
                else:
                    self.tool_execution_time += incr_usage.tool_execution_time

            if getattr(incr_usage, "pause_time", None) is not None:
                if self.pause_time is None:
                    self.pause_time = incr_usage.pause_time
                else:
                    self.pause_time += incr_usage.pause_time

            details = getattr(incr_usage, "details", None)
            if isinstance(details, dict) and details:
                if "reasoning_tokens" in details:
                    self.reasoning_tokens += details.get("reasoning_tokens", 0)

            _incr_usage_tokens(self, incr_usage)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the agent-level usage metrics.
        """
        result: Dict[str, Any] = {
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "input_audio_tokens": self.input_audio_tokens,
            "cache_audio_read_tokens": self.cache_audio_read_tokens,
            "output_audio_tokens": self.output_audio_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

        if self.details:
            result["details"] = self.details
        if self.cost is not None:
            result["cost"] = self.cost
        if self.duration is not None:
            result["duration"] = self.duration
        if self.model_execution_time is not None:
            result["model_execution_time"] = self.model_execution_time
        if self.tool_execution_time is not None:
            result["tool_execution_time"] = self.tool_execution_time
        if self.pause_time is not None:
            result["pause_time"] = self.pause_time
        if self.upsonic_execution_time is not None:
            result["upsonic_execution_time"] = self.upsonic_execution_time

        result = {
            k: v for k, v in result.items()
            if v is not None and (not isinstance(v, (int, float)) or v != 0) and (not isinstance(v, dict) or len(v) > 0)
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentUsage":
        """Reconstruct AgentUsage from dictionary.

        Args:
            data: Dictionary containing AgentUsage fields.

        Returns:
            AgentUsage instance.
        """
        return cls(
            requests=data.get("requests", 0),
            tool_calls=data.get("tool_calls", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            input_audio_tokens=data.get("input_audio_tokens", 0),
            cache_audio_read_tokens=data.get("cache_audio_read_tokens", 0),
            output_audio_tokens=data.get("output_audio_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            details=data.get("details", {}),
            cost=data.get("cost"),
            duration=data.get("duration"),
            model_execution_time=data.get("model_execution_time"),
            tool_execution_time=data.get("tool_execution_time"),
            pause_time=data.get("pause_time"),
        )


def _incr_usage_tokens(slf: "TaskUsage | AgentUsage | RequestUsage", incr_usage: "TaskUsage | AgentUsage | RequestUsage") -> None:
    """Increment token fields in place.

    Args:
        slf: The usage to increment.
        incr_usage: The usage to increment by.
    """
    slf.input_tokens += incr_usage.input_tokens
    slf.cache_write_tokens += incr_usage.cache_write_tokens
    slf.cache_read_tokens += incr_usage.cache_read_tokens
    slf.input_audio_tokens += incr_usage.input_audio_tokens
    slf.cache_audio_read_tokens += incr_usage.cache_audio_read_tokens
    slf.output_tokens += incr_usage.output_tokens
    slf.output_audio_tokens += incr_usage.output_audio_tokens

    incr_details = getattr(incr_usage, "details", None)
    if isinstance(incr_details, dict):
        for key, value in incr_details.items():
            slf.details[key] = slf.details.get(key, 0) + value


@dataclass(repr=False, kw_only=True)
@deprecated('`Usage` is deprecated, use `TaskUsage` instead')
class Usage(TaskUsage):
    """Deprecated alias for `TaskUsage`."""


@dataclass(repr=False, kw_only=True)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by upsonic, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    tool_calls_limit: int | None = None
    """The maximum number of successful tool calls allowed to be executed."""
    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""
    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""
    count_tokens_before_request: bool = False
    """If True, perform a token counting pass before sending the request to the model,
    to enforce `request_tokens_limit` ahead of time.

    This may incur additional overhead (from calling the model's `count_tokens` API before making the actual request) and is disabled by default.

    Supported by:

    - Anthropic
    - Google
    - Bedrock Converse

    Support for OpenAI is in development
    """

    @property
    @deprecated('`request_tokens_limit` is deprecated, use `input_tokens_limit` instead')
    def request_tokens_limit(self) -> int | None:
        return self.input_tokens_limit

    @property
    @deprecated('`response_tokens_limit` is deprecated, use `output_tokens_limit` instead')
    def response_tokens_limit(self) -> int | None:
        return self.output_tokens_limit

    @overload
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit
        self.output_tokens_limit = output_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    @overload
    @deprecated(
        'Use `input_tokens_limit` instead of `request_tokens_limit` and `output_tokens_limit` and `total_tokens_limit`'
    )
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = request_tokens_limit
        self.output_tokens_limit = response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
        # deprecated:
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
    ):
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit or request_tokens_limit
        self.output_tokens_limit = output_tokens_limit or response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None for limit in (self.input_tokens_limit, self.output_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: TaskUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(
                f'The next request would exceed the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(  # pragma: lax no cover
                f'The next request would exceed the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})'
            )

    def check_tokens(self, usage: TaskUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

        output_tokens = usage.output_tokens
        if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    def check_before_tool_call(self, projected_usage: TaskUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit."""
        tool_calls_limit = self.tool_calls_limit
        tool_calls = projected_usage.tool_calls
        if tool_calls_limit is not None and tool_calls > tool_calls_limit:
            raise UsageLimitExceeded(
                f'The next tool call(s) would exceed the tool_calls_limit of {tool_calls_limit} ({tool_calls=}).'
            )

    __repr__ = _utils.dataclasses_no_defaults_repr