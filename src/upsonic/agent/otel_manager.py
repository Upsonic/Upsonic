"""Agent-level OpenTelemetry manager.

Centralizes all OTel span creation, attribute setting, and error recording
for the Agent (and later Team) execution flow.  ``InstrumentedModel`` remains
responsible **only** for model-level (``chat``) spans.

Usage inside ``Agent``::

    self._otel = AgentOTelManager(settings, tracing_provider)
    with self._otel.agent_run_span(run_id, ...) as span:
        ...
        self._otel.set_run_attributes(span, ...)
"""

from __future__ import annotations

import json as _json
import time as _time
from contextlib import nullcontext
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.models.instrumented import InstrumentationSettings
    from upsonic.integrations.tracing import TracingProvider

ATTR_RUN_ID: str = "upsonic.run_id"
ATTR_AGENT_NAME: str = "upsonic.agent.name"
ATTR_AGENT_MODEL: str = "upsonic.agent.model"
ATTR_TOOL_CALL_COUNT: str = "upsonic.tool_call_count"
ATTR_TOTAL_COST: str = "upsonic.total_cost"
ATTR_EXECUTION_TIME: str = "upsonic.execution_time"
ATTR_MODEL_EXECUTION_TIME: str = "upsonic.model_execution_time"
ATTR_INPUT: str = "upsonic.input"
ATTR_OUTPUT: str = "upsonic.output"
ATTR_TOOL_NAME: str = "upsonic.tool.name"
ATTR_TOOL_CALL_ID: str = "upsonic.tool.call_id"
ATTR_TOOL_EXECUTION_TIME: str = "upsonic.tool.execution_time"
ATTR_TOOL_SUCCESS: str = "upsonic.tool.success"
ATTR_PIPELINE_TOTAL_STEPS: str = "upsonic.pipeline.total_steps"
ATTR_PIPELINE_STREAMING: str = "upsonic.pipeline.streaming"
ATTR_PIPELINE_DEBUG: str = "upsonic.pipeline.debug"
ATTR_STEP_NAME: str = "upsonic.step.name"
ATTR_STEP_DESCRIPTION: str = "upsonic.step.description"
ATTR_STEP_STATUS: str = "upsonic.step.status"
ATTR_STEP_EXECUTION_TIME: str = "upsonic.step.execution_time"
ATTR_PLAN_ACTION: str = "upsonic.plan.action"
ATTR_PLAN_INPUT: str = "upsonic.plan.input"
ATTR_PLAN_OUTPUT: str = "upsonic.plan.output"
ATTR_PLAN_TODO_COUNT: str = "upsonic.plan.todo_count"
ATTR_PLAN_STATUS_PREFIX: str = "upsonic.plan.status"

GEN_AI_OPERATION_NAME: str = "gen_ai.operation.name"
GEN_AI_AGENT_NAME: str = "gen_ai.agent.name"
GEN_AI_AGENT_ID: str = "gen_ai.agent.id"

# Additional run-level metrics
ATTR_USAGE_REQUESTS: str = "upsonic.usage.requests"
ATTR_USAGE_CACHE_WRITE_TOKENS: str = "upsonic.usage.cache_write_tokens"
ATTR_USAGE_CACHE_READ_TOKENS: str = "upsonic.usage.cache_read_tokens"
ATTR_USAGE_REASONING_TOKENS: str = "upsonic.usage.reasoning_tokens"
ATTR_USAGE_TOTAL_TOKENS: str = "upsonic.usage.total_tokens"
ATTR_USAGE_INPUT_AUDIO_TOKENS: str = "upsonic.usage.input_audio_tokens"
ATTR_USAGE_OUTPUT_AUDIO_TOKENS: str = "upsonic.usage.output_audio_tokens"
ATTR_TOOL_TOTAL_EXECUTION_TIME: str = "upsonic.tool_execution_time"
ATTR_FRAMEWORK_OVERHEAD_TIME: str = "upsonic.framework_overhead_time"
ATTR_TIME_TO_FIRST_TOKEN: str = "upsonic.time_to_first_token"
ATTR_STATUS: str = "upsonic.status"
ATTR_MODEL_NAME: str = "upsonic.model_name"
ATTR_MODEL_PROVIDER: str = "upsonic.model_provider"
ATTR_TOOL_LIMIT_REACHED: str = "upsonic.tool_limit_reached"
ATTR_IS_STREAMING: str = "upsonic.is_streaming"
ATTR_PIPELINE_EXECUTED_STEPS: str = "upsonic.pipeline.executed_steps"
ATTR_PIPELINE_STEP_TIMING: str = "upsonic.pipeline.step_timing"
ATTR_PIPELINE_STEP_STATUSES: str = "upsonic.pipeline.step_statuses"

LF_TRACE_NAME: str = "langfuse.trace.name"
LF_TRACE_INPUT: str = "langfuse.trace.input"
LF_TRACE_OUTPUT: str = "langfuse.trace.output"
LF_OBS_INPUT: str = "langfuse.observation.input"
LF_OBS_OUTPUT: str = "langfuse.observation.output"
LF_USER_ID: str = "langfuse.user.id"
LF_SESSION_ID: str = "langfuse.session.id"
GENERIC_USER_ID: str = "user.id"
GENERIC_SESSION_ID: str = "session.id"
_PLAN_OPERATION: str = "plan"
_PLAN_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed", "cancelled")
_CONTENT_ATTRIBUTE_LIMIT: int = 5000


def _serialize_content_attribute(value: Any) -> str:
    """Serialize arbitrary content to an OTel-safe bounded string attribute."""
    try:
        serialized = _json.dumps(value, default=str)
    except Exception:
        serialized = str(value)
    return serialized[:_CONTENT_ATTRIBUTE_LIMIT]


def _set_baggage(
    *,
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Any:
    """Build an OTel context with Baggage entries for Langfuse attributes.

    The ``BaggageSpanProcessor`` copies these entries to span attributes on
    every child span so Langfuse can aggregate by user / session / etc.
    Returns the enriched context, or ``None`` if baggage is unavailable.
    """
    try:
        from opentelemetry import baggage, context as _ctx

        ctx = _ctx.get_current()
        if trace_name:
            ctx = baggage.set_baggage(LF_TRACE_NAME, trace_name, context=ctx)
        if user_id:
            ctx = baggage.set_baggage(LF_USER_ID, user_id, context=ctx)
            ctx = baggage.set_baggage(GENERIC_USER_ID, user_id, context=ctx)
        if session_id:
            ctx = baggage.set_baggage(LF_SESSION_ID, session_id, context=ctx)
            ctx = baggage.set_baggage(GENERIC_SESSION_ID, session_id, context=ctx)
        return ctx
    except Exception:
        return None


class _BaggageSpanContext:
    """Context manager that attaches OTel Baggage before starting a span.

    Ensures ``BaggageSpanProcessor`` sees the baggage entries when child
    spans are created inside the ``with`` block.  After the span ends,
    calls ``flush_fn`` so the ``BatchSpanProcessor`` exports the completed
    span promptly (rather than waiting for its timer or shutdown).
    """

    __slots__ = ("_tracer", "_span_name", "_kind", "_attrs", "_ctx",
                 "_flush_fn", "_token", "_span", "_span_cm")

    def __init__(
        self, tracer: Any, span_name: str, kind: Any,
        attrs: Dict[str, Any], ctx: Any,
        flush_fn: Any = None,
    ) -> None:
        self._tracer = tracer
        self._span_name = span_name
        self._kind = kind
        self._attrs = attrs
        self._ctx = ctx
        self._flush_fn = flush_fn
        self._token: Any = None
        self._span: Any = None
        self._span_cm: Any = None

    def __enter__(self) -> Any:
        # Attach baggage context so child spans inherit it.
        if self._ctx is not None:
            try:
                from opentelemetry import context as _ctx
                self._token = _ctx.attach(self._ctx)
            except Exception:
                pass

        self._span_cm = self._tracer.start_as_current_span(
            self._span_name,
            kind=self._kind,
            attributes=self._attrs,
        )
        self._span = self._span_cm.__enter__()
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = None
        if self._span_cm is not None:
            result = self._span_cm.__exit__(exc_type, exc_val, exc_tb)
        # Detach baggage context.
        if self._token is not None:
            try:
                from opentelemetry import context as _ctx
                _ctx.detach(self._token)
            except Exception:
                pass
        # Flush after span has ended so it's exported promptly.
        if self._flush_fn is not None:
            try:
                self._flush_fn()
            except Exception:
                pass
        return result


def _is_recording(span: Any) -> bool:
    if span is None:
        return False
    try:
        return span.is_recording()
    except Exception:
        return False


def _set_status_ok(span: Any) -> None:
    """Mark *span* as successfully completed (``StatusCode.OK``)."""
    if not _is_recording(span):
        return
    try:
        from opentelemetry.trace import StatusCode
        span.set_status(StatusCode.OK)
    except Exception:
        pass


class AgentOTelManager:
    """Manages all OTel spans and attributes for an Agent execution.

    Holds a reference to :class:`InstrumentationSettings` and exposes
    methods for every span type the agent pipeline emits.  When
    *settings* is ``None`` every method degrades to a safe no-op so
    callers never need ``if`` guards.
    """

    __slots__ = ("_settings", "_tracing_provider")

    def __init__(
        self,
        settings: Optional["InstrumentationSettings"],
        tracing_provider: Optional["TracingProvider"] = None,
    ) -> None:
        self._settings: Optional["InstrumentationSettings"] = settings
        self._tracing_provider: Optional["TracingProvider"] = tracing_provider

    @property
    def enabled(self) -> bool:
        return self._settings is not None

    @property
    def settings(self) -> Optional["InstrumentationSettings"]:
        return self._settings

    def flush(self) -> None:
        """Force-flush pending spans to the backend without shutting down.

        Call after each agent run to ensure span data appears promptly
        in dashboards (Langfuse, Jaeger, etc.) rather than waiting for
        the ``BatchSpanProcessor``'s scheduled export interval.
        """
        if self._tracing_provider is not None:
            self._tracing_provider.flush()

    def agent_run_span(
        self,
        run_id: str,
        *,
        name: str = "",
        model: str = "",
        task_description: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """Create the root ``agent.run`` span, or a no-op context manager.

        Sets OTel Baggage for Langfuse trace-level attributes so that the
        ``BaggageSpanProcessor`` propagates them to every child span.
        """
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        include_content: bool = getattr(self._settings, "include_content", True)
        truncated_desc: str = str(task_description)[:200] if task_description else ""

        trace_name: str = (truncated_desc if include_content else "") or name or "agent.run"

        init_attrs: Dict[str, Any] = {
            ATTR_AGENT_NAME: name,
            ATTR_AGENT_MODEL: model,
            ATTR_RUN_ID: run_id,
            LF_TRACE_NAME: trace_name,
        }

        if user_id:
            init_attrs[LF_USER_ID] = user_id
            init_attrs[GENERIC_USER_ID] = user_id
        if session_id:
            init_attrs[LF_SESSION_ID] = session_id
            init_attrs[GENERIC_SESSION_ID] = session_id

        # Set baggage so BaggageSpanProcessor propagates trace-level
        # attributes to ALL child spans (required by Langfuse).
        ctx = _set_baggage(
            trace_name=trace_name,
            user_id=user_id,
            session_id=session_id,
        )

        return _BaggageSpanContext(
            self._settings.tracer,
            "agent.run",
            SpanKind.SERVER,
            init_attrs,
            ctx,
            flush_fn=self.flush,
        )

    def tool_span(
        self,
        tool_name: str,
        tool_call_id: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a ``tool.execute`` span (``INTERNAL``), or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        attrs: Dict[str, Any] = {
            ATTR_TOOL_NAME: tool_name,
            ATTR_TOOL_CALL_ID: tool_call_id,
        }

        include_content: bool = getattr(self._settings, "include_content", True)
        if include_content and tool_args is not None:
            json_input: str = _json.dumps(tool_args)
            attrs[LF_OBS_INPUT] = json_input

        return self._settings.tracer.start_as_current_span(
            "tool.execute",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )

    def plan_span(
        self,
        *,
        agent_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        input_todos: Optional[Any] = None,
    ) -> Any:
        """Create a GenAI ``plan`` span, or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        span_name = f"plan {agent_name}" if agent_name else "plan"
        attrs: Dict[str, Any] = {
            GEN_AI_OPERATION_NAME: _PLAN_OPERATION,
        }
        if agent_name:
            attrs[GEN_AI_AGENT_NAME] = agent_name
        if agent_id:
            attrs[GEN_AI_AGENT_ID] = agent_id
        if action:
            attrs[ATTR_PLAN_ACTION] = action

        include_content: bool = getattr(self._settings, "include_content", True)
        if include_content and input_todos is not None:
            attrs[ATTR_PLAN_INPUT] = _serialize_content_attribute(input_todos)

        return self._settings.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )

    def pipeline_span(
        self,
        total_steps: int,
        is_streaming: bool,
        debug: bool,
    ) -> Any:
        """Create a ``pipeline.execute`` span (``INTERNAL``), or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        return self._settings.tracer.start_as_current_span(
            "pipeline.execute",
            kind=SpanKind.INTERNAL,
            attributes={
                ATTR_PIPELINE_TOTAL_STEPS: total_steps,
                ATTR_PIPELINE_STREAMING: is_streaming,
                ATTR_PIPELINE_DEBUG: debug,
            },
        )

    def step_span(self, step_name: str, step_description: str) -> Any:
        """Create a ``pipeline.step.<name>`` span (``INTERNAL``), or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        return self._settings.tracer.start_as_current_span(
            f"pipeline.step.{step_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                ATTR_STEP_NAME: step_name,
                ATTR_STEP_DESCRIPTION: step_description,
            },
        )

    def set_run_attributes(
        self,
        span: Any,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_call_count: int = 0,
        total_cost: Optional[float] = None,
        execution_time: Optional[float] = None,
        model_execution_time: Optional[float] = None,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Set final result attributes on the root ``agent.run`` span.

        Token attributes use ``gen_ai.*`` (OTel GenAI semantic conventions)
        so that backends like Langfuse recognise them.

        Langfuse-specific attributes (``langfuse.trace.*``,
        ``langfuse.observation.*``) are set for Input/Output display.
        User and session use only ``langfuse.*`` and ``user.id`` /
        ``session.id`` (no ``upsonic.user_id`` / ``upsonic.session_id``
        duplicates).
        """
        if not _is_recording(span) or self._settings is None:
            return

        use_aggregated: bool = getattr(
            self._settings, "use_aggregated_usage_attribute_names", False
        )
        prefix: str = "gen_ai.aggregated_usage" if use_aggregated else "gen_ai.usage"

        attrs: Dict[str, Any] = {}

        if input_tokens:
            attrs[f"{prefix}.input_tokens"] = input_tokens
        if output_tokens:
            attrs[f"{prefix}.output_tokens"] = output_tokens
        if tool_call_count:
            attrs[ATTR_TOOL_CALL_COUNT] = tool_call_count
        if total_cost is not None and total_cost > 0:
            attrs[ATTR_TOTAL_COST] = total_cost
        if execution_time is not None and execution_time > 0:
            attrs[ATTR_EXECUTION_TIME] = round(execution_time, 3)
        if model_execution_time is not None and model_execution_time > 0:
            attrs[ATTR_MODEL_EXECUTION_TIME] = round(model_execution_time, 3)

        include_content: bool = getattr(self._settings, "include_content", True)
        if include_content:
            if input_text:
                truncated_input: str = str(input_text)[:5000]
                attrs[ATTR_INPUT] = truncated_input
                json_input: str = _json.dumps(truncated_input)
                attrs[LF_TRACE_INPUT] = json_input
                attrs[LF_OBS_INPUT] = json_input
            if output_text:
                truncated_output: str = str(output_text)[:5000]
                attrs[ATTR_OUTPUT] = truncated_output
                json_output: str = _json.dumps(truncated_output)
                attrs[LF_TRACE_OUTPUT] = json_output
                attrs[LF_OBS_OUTPUT] = json_output

        # User and session — only langfuse.* and generic OTel keys (no upsonic.* duplicates)
        if user_id:
            attrs[LF_USER_ID] = user_id
            attrs[GENERIC_USER_ID] = user_id
        if session_id:
            attrs[LF_SESSION_ID] = session_id
            attrs[GENERIC_SESSION_ID] = session_id

        if attrs:
            span.set_attributes(attrs)

    def set_tool_result(
        self,
        span: Any,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
        output: Optional[str] = None,
    ) -> None:
        """Set result attributes on a ``tool.execute`` span."""
        if not _is_recording(span):
            return

        attrs: Dict[str, Any] = {
            ATTR_TOOL_EXECUTION_TIME: execution_time,
            ATTR_TOOL_SUCCESS: success,
        }

        include_content: bool = (
            getattr(self._settings, "include_content", True)
            if self._settings is not None
            else True
        )
        if include_content and output is not None:
            json_output: str = _json.dumps(str(output)[:5000])
            attrs[LF_OBS_OUTPUT] = json_output

        span.set_attributes(attrs)
        if success:
            _set_status_ok(span)
        elif error is not None:
            self.record_error(span, error)

    def set_plan_result(
        self,
        span: Any,
        *,
        success: bool,
        todo_count: Optional[int] = None,
        status_counts: Optional[Dict[str, int]] = None,
        action: Optional[str] = None,
        output: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Set result attributes on a GenAI ``plan`` span."""
        if not _is_recording(span):
            return

        attrs: Dict[str, Any] = {}
        if action:
            attrs[ATTR_PLAN_ACTION] = action
        if todo_count is not None:
            attrs[ATTR_PLAN_TODO_COUNT] = todo_count
        if status_counts:
            for status in _PLAN_STATUSES:
                count = status_counts.get(status)
                if count is not None:
                    attrs[f"{ATTR_PLAN_STATUS_PREFIX}.{status}"] = count

        include_content: bool = (
            getattr(self._settings, "include_content", True)
            if self._settings is not None
            else True
        )
        if include_content and output is not None:
            attrs[ATTR_PLAN_OUTPUT] = _serialize_content_attribute(output)

        if attrs:
            span.set_attributes(attrs)

        if success:
            _set_status_ok(span)
        else:
            self.record_error(span, error or RuntimeError("plan failed"))

    def set_step_result(
        self,
        span: Any,
        status: str,
        execution_time: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Set result attributes on a ``pipeline.step.*`` span."""
        if not _is_recording(span):
            return

        span.set_attributes({
            ATTR_STEP_STATUS: status,
            ATTR_STEP_EXECUTION_TIME: execution_time,
        })
        if error_message:
            try:
                from opentelemetry.trace import StatusCode
                span.set_status(StatusCode.ERROR, error_message)
            except Exception:
                pass
        else:
            _set_status_ok(span)

    @staticmethod
    def extract_trace_id(span: Any) -> Optional[str]:
        """Extract the hex trace ID from an OTel span, or ``None``."""
        if span is None:
            return None
        try:
            ctx = span.get_span_context()
            if ctx is not None and ctx.trace_id:
                return format(ctx.trace_id, "032x")
        except Exception:
            pass
        return None

    @staticmethod
    def mark_success(span: Any) -> None:
        """Explicitly mark *span* as successfully completed (``StatusCode.OK``)."""
        _set_status_ok(span)

    @staticmethod
    def record_error(span: Any, exc: Exception) -> None:
        """Record an exception and ``ERROR`` status on any span."""
        if not _is_recording(span):
            return
        try:
            from opentelemetry.trace import StatusCode
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
        except Exception:
            pass

    def finalize_agent_run(
        self,
        span: Any,
        output: Any,
        *,
        agent_user_id: Optional[str] = None,
        agent_session_id: Optional[str] = None,
        total_cost: Optional[float] = None,
        tool_definitions: Optional[list] = None,
    ) -> None:
        """Extract data from an ``AgentRunOutput`` and set span attributes.

        This is the single call-site an Agent (or Team) needs at the end of
        a run to populate all OTel attributes on the root span.
        """
        if not _is_recording(span) or self._settings is None or output is None:
            return

        run_usage = getattr(output, "usage", None)
        input_tokens: int = (getattr(run_usage, "input_tokens", 0) or 0) if run_usage else 0
        output_tokens: int = (getattr(run_usage, "output_tokens", 0) or 0) if run_usage else 0

        created_at: int = getattr(output, "created_at", 0)
        execution_time: Optional[float] = None
        if created_at:
            execution_time = _time.time() - created_at

        model_execution_time: Optional[float] = None
        exec_stats = getattr(output, "execution_stats", None)
        if exec_stats is not None:
            step_timing: Dict[str, float] = getattr(exec_stats, "step_timing", {})
            model_execution_time = step_timing.get("model_execution")

        task_obj = getattr(output, "task", None)
        input_text: Optional[str] = str(getattr(task_obj, "description", "")) if task_obj else None

        result_output = getattr(output, "output", None)
        output_text: Optional[str] = str(result_output) if result_output is not None else None

        self.set_run_attributes(
            span,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_count=getattr(output, "tool_call_count", 0) or 0,
            total_cost=total_cost,
            execution_time=execution_time,
            model_execution_time=model_execution_time,
            input_text=input_text,
            output_text=output_text,
            user_id=getattr(output, "user_id", None) or agent_user_id,
            session_id=getattr(output, "session_id", None) or agent_session_id,
        )

        # ── Additional TaskUsage metrics ──────────────────────────────
        extra: Dict[str, Any] = {}

        if run_usage is not None:
            requests = getattr(run_usage, "requests", 0) or 0
            if requests:
                extra[ATTR_USAGE_REQUESTS] = requests

            cache_write = getattr(run_usage, "cache_write_tokens", 0) or 0
            if cache_write:
                extra[ATTR_USAGE_CACHE_WRITE_TOKENS] = cache_write

            cache_read = getattr(run_usage, "cache_read_tokens", 0) or 0
            if cache_read:
                extra[ATTR_USAGE_CACHE_READ_TOKENS] = cache_read

            reasoning = getattr(run_usage, "reasoning_tokens", 0) or 0
            if reasoning:
                extra[ATTR_USAGE_REASONING_TOKENS] = reasoning

            total_tok = getattr(run_usage, "total_tokens", 0) or 0
            if total_tok:
                extra[ATTR_USAGE_TOTAL_TOKENS] = total_tok

            input_audio = getattr(run_usage, "input_audio_tokens", 0) or 0
            if input_audio:
                extra[ATTR_USAGE_INPUT_AUDIO_TOKENS] = input_audio

            output_audio = getattr(run_usage, "output_audio_tokens", 0) or 0
            if output_audio:
                extra[ATTR_USAGE_OUTPUT_AUDIO_TOKENS] = output_audio

            tool_exec_time = getattr(run_usage, "tool_execution_time", None)
            if tool_exec_time is not None and tool_exec_time > 0:
                extra[ATTR_TOOL_TOTAL_EXECUTION_TIME] = round(tool_exec_time, 3)

            framework_time = getattr(run_usage, "upsonic_execution_time", None)
            if framework_time is not None and framework_time > 0:
                extra[ATTR_FRAMEWORK_OVERHEAD_TIME] = round(framework_time, 3)

            ttft = getattr(run_usage, "time_to_first_token", None)
            if ttft is not None and ttft > 0:
                extra[ATTR_TIME_TO_FIRST_TOKEN] = round(ttft, 3)

        # ── AgentRunOutput metadata ───────────────────────────────────
        status = getattr(output, "status", None)
        if status is not None:
            extra[ATTR_STATUS] = str(status.value) if hasattr(status, "value") else str(status)

        model_name = getattr(output, "model_name", None)
        if model_name:
            extra[ATTR_MODEL_NAME] = model_name

        model_provider = getattr(output, "model_provider", None)
        if model_provider:
            extra[ATTR_MODEL_PROVIDER] = model_provider

        tool_limit_reached = getattr(output, "tool_limit_reached", False)
        if tool_limit_reached:
            extra[ATTR_TOOL_LIMIT_REACHED] = True

        is_streaming = getattr(output, "is_streaming", False)
        if is_streaming:
            extra[ATTR_IS_STREAMING] = True

        # ── PipelineExecutionStats ────────────────────────────────────
        if exec_stats is not None:
            executed_steps = getattr(exec_stats, "executed_steps", 0) or 0
            if executed_steps:
                extra[ATTR_PIPELINE_EXECUTED_STEPS] = executed_steps

            step_timing_dict = getattr(exec_stats, "step_timing", None)
            if step_timing_dict:
                extra[ATTR_PIPELINE_STEP_TIMING] = _json.dumps(
                    {k: round(v, 3) for k, v in step_timing_dict.items()}
                )

            step_statuses_dict = getattr(exec_stats, "step_statuses", None)
            if step_statuses_dict:
                extra[ATTR_PIPELINE_STEP_STATUSES] = _json.dumps(step_statuses_dict)

        if extra:
            span.set_attributes(extra)

        # ── Tool-level details ────────────────────────────────────────
        include_content: bool = getattr(self._settings, "include_content", True)

        # tools_called: {tool_name: {input: ..., output: ...}, ...}
        tools_list = getattr(output, "tools", None) or []
        if tools_list and include_content:
            tools_called: Dict[str, Any] = {}
            for tool_exec in tools_list:
                name = getattr(tool_exec, "tool_name", None) or "unknown"
                entry: Dict[str, Any] = {}
                args = getattr(tool_exec, "tool_args", None)
                if args is not None:
                    entry["input"] = args
                result = getattr(tool_exec, "result", None)
                if result is not None:
                    entry["output"] = str(result)[:2000]
                tools_called[name] = entry
            span.set_attribute(
                "upsonic.tools_called",
                _json.dumps(tools_called),
            )

        # tool.definitions: list of registered tool schemas
        if tool_definitions:
            defs = []
            for td in tool_definitions:
                d: Dict[str, Any] = {"name": getattr(td, "name", "")}
                desc = getattr(td, "description", None)
                if desc:
                    d["description"] = desc
                params = getattr(td, "parameters_json_schema", None)
                if params:
                    d["parameters"] = params
                defs.append(d)
            span.set_attribute(
                "upsonic.tool.definitions",
                _json.dumps(defs),
            )

        _set_status_ok(span)
        self.flush()
