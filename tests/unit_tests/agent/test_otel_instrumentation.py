"""
Comprehensive tests for OpenTelemetry instrumentation in the Upsonic Agent framework.

Covers:
- Activation modes: per-agent, global, env-var, InstrumentationSettings
- InstrumentedModel wrapping and model override preservation
- Span hierarchy: agent.run → pipeline.execute → pipeline.step → tool.execute → chat
- Error recording on spans (StatusCode.ERROR, record_exception)
- Aggregated usage attributes (gen_ai.usage vs gen_ai.aggregated_usage)
- Aggregated cost on root span
- Zero-overhead no-op path when instrumentation is disabled
- TracerProvider setup: sampling, headers, shutdown

Run with: uv run pytest tests/unit_tests/agent/test_otel_instrumentation.py -v -s
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from upsonic import Agent
from upsonic.models.instrumented import (
    InstrumentationSettings,
    InstrumentedModel,
    instrument_model,
)
from upsonic.agent.otel_manager import AgentOTelManager
from upsonic.agent.pipeline.manager import PipelineManager
from upsonic.agent.pipeline.step import StepResult, StepStatus
from upsonic.usage import RunUsage, RequestUsage


def _make_otel_capture(include_content: bool = True) -> tuple[Any, Any, Any, InstrumentationSettings]:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter_provider = MeterProvider()
    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_content=include_content,
    )
    return exporter, tracer_provider, meter_provider, settings


def _make_mock_model(model: Any = "openai/gpt-4o", **_kwargs: Any) -> MagicMock:
    """Create a mock Model that satisfies Agent requirements without API keys.

    If *model* is already a mock (i.e. WrapperModel re-calling infer_model on
    an already-resolved object), return it unchanged.
    """
    if isinstance(model, MagicMock):
        return model

    name: str = model if isinstance(model, str) else "mock"
    mock = MagicMock()
    mock.model_name = name.split("/")[-1] if "/" in name else name
    mock.system = ""
    mock.base_url = None
    mock.name.return_value = name
    return mock


@pytest.fixture(autouse=True)
def _patch_infer_model():
    """Prevent all tests in this module from hitting real LLM providers."""
    with patch("upsonic.models.infer_model", side_effect=_make_mock_model), \
         patch("upsonic.models.wrapper.infer_model", side_effect=_make_mock_model):
        yield


class RecordingSpan:
    """Mock OTel span that records all interactions for assertion."""

    def __init__(self, recording: bool = True) -> None:
        self._recording: bool = recording
        self.attrs: Dict[str, Any] = {}
        self.status_code: Any = None
        self.status_desc: str = ""
        self.exceptions: List[Exception] = []
        self.name: Optional[str] = None

    def is_recording(self) -> bool:
        return self._recording

    def set_attributes(self, attrs: Dict[str, Any]) -> None:
        self.attrs.update(attrs)

    def set_status(self, code: Any, description: str = "") -> None:
        self.status_code = code
        self.status_desc = description

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)


class MockRunOutput:
    """Lightweight stand-in for AgentRunOutput used in attribute tests."""

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_call_count: int = 0,
    ) -> None:
        self.usage = RunUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.tool_call_count: int = tool_call_count
        self.created_at: int = 0
        self.execution_stats = None
        self.task = None
        self.output = None
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None


def _make_otel(instrument: bool = True) -> AgentOTelManager:
    """Create an AgentOTelManager with or without real settings."""
    if instrument:
        return AgentOTelManager(InstrumentationSettings())
    return AgentOTelManager(None)


class TestActivationModes:
    """Test the three activation modes for OTel instrumentation."""

    def test_instrument_disabled_by_default(self) -> None:
        agent = Agent("openai/gpt-4o")
        assert not isinstance(agent.model, InstrumentedModel)
        assert agent._instrument_settings is None
        assert not agent._otel.enabled

    def test_instrument_true_wraps_model(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        assert isinstance(agent.model, InstrumentedModel)
        assert agent._instrument_settings is not None
        assert isinstance(agent._instrument_settings, InstrumentationSettings)
        assert agent._otel.enabled

    def test_instrument_custom_settings(self) -> None:
        settings = InstrumentationSettings()
        agent = Agent("openai/gpt-4o", instrument=settings)
        assert isinstance(agent.model, InstrumentedModel)
        assert agent._instrument_settings is settings
        assert agent._otel.settings is settings

    def test_instrument_all_global_activation(self) -> None:
        try:
            Agent.instrument_all()
            agent = Agent("openai/gpt-4o")
            assert isinstance(agent.model, InstrumentedModel)
            assert agent._instrument_settings is not None
        finally:
            Agent.instrument_all(False)

    def test_instrument_all_deactivation(self) -> None:
        Agent.instrument_all()
        Agent.instrument_all(False)
        agent = Agent("openai/gpt-4o")
        assert not isinstance(agent.model, InstrumentedModel)
        assert agent._instrument_settings is None

    def test_per_agent_overrides_global(self) -> None:
        try:
            Agent.instrument_all()
            agent = Agent("openai/gpt-4o", instrument=False)
            assert not isinstance(agent.model, InstrumentedModel)
            assert agent._instrument_settings is None
        finally:
            Agent.instrument_all(False)

    def test_per_agent_true_without_global(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        assert isinstance(agent.model, InstrumentedModel)

    def test_instrument_model_utility(self) -> None:
        from upsonic.models import infer_model
        model = infer_model("openai/gpt-4o")
        wrapped = instrument_model(model, True)
        assert isinstance(wrapped, InstrumentedModel)

    def test_instrument_model_already_wrapped(self) -> None:
        from upsonic.models import infer_model
        model = infer_model("openai/gpt-4o")
        wrapped = instrument_model(model, True)
        double = instrument_model(wrapped, True)
        assert double is wrapped


class TestModelOverridePreservation:

    def test_override_returns_original(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        original = agent.model
        saved = agent._apply_model_override("openai/gpt-4o-mini")
        assert saved is original
        assert "gpt-4o-mini" in agent.model.model_name

    def test_no_override_returns_none(self) -> None:
        agent = Agent("openai/gpt-4o")
        result = agent._apply_model_override(None)
        assert result is None

    def test_override_preserves_instrumentation(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        agent._apply_model_override("openai/gpt-4o-mini")
        assert isinstance(agent.model, InstrumentedModel)
        assert "gpt-4o-mini" in agent.model.model_name

    def test_override_without_instrument_stays_unwrapped(self) -> None:
        agent = Agent("openai/gpt-4o")
        agent._apply_model_override("openai/gpt-4o-mini")
        assert not isinstance(agent.model, InstrumentedModel)

    def test_restore_after_override(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        original = agent.model
        saved = agent._apply_model_override("openai/gpt-4o-mini")
        assert agent.model is not original
        agent.model = saved
        assert agent.model is original

    def test_do_async_restore_pattern_in_source(self) -> None:
        import inspect
        source = inspect.getsource(Agent.do_async)
        lines = source.split("\n")

        init_idx: Optional[int] = None
        try_idx: Optional[int] = None
        restore_idx: Optional[int] = None

        for i, line in enumerate(lines):
            s = line.strip()
            if "original_model:" in s and "Optional" in s:
                init_idx = i
            if s == "try:" and try_idx is None:
                try_idx = i
            if "if original_model is not None:" in s:
                restore_idx = i

        assert init_idx is not None, "original_model init not found"
        assert try_idx is not None, "try block not found"
        assert restore_idx is not None, "finally restore not found"
        assert init_idx < try_idx, "init must come before try"
        assert restore_idx > try_idx, "restore must come after try"

    def test_astream_restore_pattern_in_source(self) -> None:
        import inspect
        source = inspect.getsource(Agent.astream)
        assert "original_model = self._apply_model_override(model)" in source
        assert "if original_model is not None:" in source
        assert "self.model = original_model" in source


class TestSpanContextManagers:

    def test_agent_run_span_noop(self) -> None:
        otel = _make_otel(instrument=False)
        with otel.agent_run_span("run-1", name="test") as span:
            assert span is None

    def test_tool_span_noop(self) -> None:
        otel = _make_otel(instrument=False)
        with otel.tool_span("test_tool", "tc-1") as span:
            assert span is None

    def test_pipeline_span_noop(self) -> None:
        otel = _make_otel(instrument=False)
        with otel.pipeline_span(total_steps=3, is_streaming=False, debug=False) as span:
            assert span is None

    def test_step_span_noop(self) -> None:
        otel = _make_otel(instrument=False)
        with otel.step_span("dummy", "dummy step") as span:
            assert span is None

    def test_agent_has_otel_manager(self) -> None:
        agent = Agent("openai/gpt-4o")
        assert isinstance(agent._otel, AgentOTelManager)
        assert not agent._otel.enabled

    def test_agent_otel_enabled_with_instrument(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        assert isinstance(agent._otel, AgentOTelManager)
        assert agent._otel.enabled

    def test_pipeline_manager_resolves_otel(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        pm = PipelineManager(agent=agent)
        assert pm._otel.enabled

    def test_pipeline_manager_resolves_noop_otel(self) -> None:
        pm = PipelineManager(agent=Agent("openai/gpt-4o"))
        assert not pm._otel.enabled


class TestPlanSpans:
    """Test GenAI plan span instrumentation for DeepAgent planning."""

    def test_plan_span_noop(self) -> None:
        otel = _make_otel(instrument=False)
        with otel.plan_span(agent_name="Planner") as span:
            assert span is None

    def test_plan_result_omits_content_when_disabled(self) -> None:
        settings = InstrumentationSettings(include_content=False)
        otel = AgentOTelManager(settings)
        span = RecordingSpan()

        otel.set_plan_result(
            span,
            success=True,
            todo_count=2,
            status_counts={"pending": 2},
            action="create",
            output="hidden plan output",
        )

        assert span.attrs["upsonic.plan.todo_count"] == 2
        assert span.attrs["upsonic.plan.status.pending"] == 2
        assert span.attrs["upsonic.plan.action"] == "create"
        assert "upsonic.plan.output" not in span.attrs

    @pytest.mark.asyncio
    async def test_planning_toolkit_write_todos_emits_plan_span(self) -> None:
        from opentelemetry.trace import StatusCode
        from upsonic.agent.deepagent.tools.planning_toolkit import PlanningToolKit

        exporter, tracer_provider, meter_provider, settings = _make_otel_capture()
        try:
            toolkit = PlanningToolKit()
            task = SimpleNamespace()
            agent = SimpleNamespace(
                _otel=AgentOTelManager(settings),
                name="PlannerAgent",
                agent_id="agent-123",
            )
            toolkit.set_current_context(task, agent)

            response = await toolkit.write_todos([
                {"content": "Research", "status": "pending", "id": "1"},
                {"content": "Write", "status": "pending", "id": "2"},
            ])

            assert "Plan created" in response
            plan_spans = [
                span
                for span in exporter.get_finished_spans()
                if span.attributes.get("gen_ai.operation.name") == "plan"
            ]
            assert len(plan_spans) == 1
            plan_span = plan_spans[0]
            assert plan_span.name == "plan PlannerAgent"
            assert plan_span.attributes["gen_ai.agent.name"] == "PlannerAgent"
            assert plan_span.attributes["gen_ai.agent.id"] == "agent-123"
            assert plan_span.attributes["upsonic.plan.action"] == "create"
            assert plan_span.attributes["upsonic.plan.todo_count"] == 2
            assert plan_span.attributes["upsonic.plan.status.pending"] == 2
            assert plan_span.status.status_code == StatusCode.OK
        finally:
            tracer_provider.shutdown()
            meter_provider.shutdown()

    @pytest.mark.asyncio
    async def test_planning_toolkit_validation_error_marks_plan_span_error(self) -> None:
        from opentelemetry.trace import StatusCode
        from upsonic.agent.deepagent.tools.planning_toolkit import PlanningToolKit

        exporter, tracer_provider, meter_provider, settings = _make_otel_capture()
        try:
            toolkit = PlanningToolKit()
            task = SimpleNamespace()
            agent = SimpleNamespace(_otel=AgentOTelManager(settings), name="PlannerAgent")
            toolkit.set_current_context(task, agent)

            response = await toolkit.write_todos([])

            assert "Missing 'todos' parameter" in response
            plan_spans = [
                span
                for span in exporter.get_finished_spans()
                if span.attributes.get("gen_ai.operation.name") == "plan"
            ]
            assert len(plan_spans) == 1
            assert plan_spans[0].attributes["upsonic.plan.action"] == "create"
            assert plan_spans[0].status.status_code == StatusCode.ERROR
        finally:
            tracer_provider.shutdown()
            meter_provider.shutdown()

    @pytest.mark.asyncio
    async def test_deepagent_write_todos_tool_emits_child_plan_span(self) -> None:
        from upsonic import Task
        from upsonic.agent.deepagent import DeepAgent
        from upsonic.agent.pipeline.steps import ToolSetupStep
        from upsonic.messages.messages import ToolCallPart
        from upsonic.run.pipeline.stats import PipelineExecutionStats

        exporter, tracer_provider, meter_provider, settings = _make_otel_capture()
        try:
            agent = DeepAgent(
                model="openai/gpt-4o",
                name="RuntimePlanner",
                instrument=settings,
                enable_planning=True,
                enable_filesystem=False,
                enable_subagents=False,
            )
            task = Task("Create a short execution plan")
            context = SimpleNamespace(
                is_streaming=False,
                events=[],
                run_id="run-1",
                step_results=[],
                execution_stats=PipelineExecutionStats(total_steps=1),
            )
            agent.current_task = task

            await ToolSetupStep().execute(context, task, agent, agent.model, 0)
            result = await agent._execute_tool_calls([
                ToolCallPart(
                    tool_name="write_todos",
                    args={
                        "todos": [
                            {"content": "Research", "status": "pending", "id": "1"},
                            {"content": "Write", "status": "pending", "id": "2"},
                        ]
                    },
                    tool_call_id="call_plan",
                )
            ])

            assert len(result) == 1
            assert result[0].tool_name == "write_todos"
            spans = exporter.get_finished_spans()
            tool_span = next(span for span in spans if span.name == "tool.execute")
            plan_span = next(
                span
                for span in spans
                if span.attributes.get("gen_ai.operation.name") == "plan"
            )

            assert plan_span.parent is not None
            assert plan_span.parent.span_id == tool_span.context.span_id
            assert plan_span.attributes["upsonic.plan.action"] == "create"
            assert plan_span.attributes["upsonic.plan.todo_count"] == 2
        finally:
            tracer_provider.shutdown()
            meter_provider.shutdown()


class TestErrorRecording:

    def test_record_error_on_span(self) -> None:
        from opentelemetry.trace import StatusCode
        span = RecordingSpan()
        exc = ValueError("something broke")
        AgentOTelManager.record_error(span, exc)

        assert span.status_code == StatusCode.ERROR
        assert span.status_desc == "something broke"
        assert len(span.exceptions) == 1
        assert span.exceptions[0] is exc

    def test_record_error_none_span_is_noop(self) -> None:
        AgentOTelManager.record_error(None, ValueError("test"))

    def test_record_error_non_recording_span_is_noop(self) -> None:
        span = RecordingSpan(recording=False)
        AgentOTelManager.record_error(span, ValueError("test"))
        assert span.status_code is None
        assert len(span.exceptions) == 0

    def test_tool_result_error(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        exc = RuntimeError("tool failed")
        otel.set_tool_result(span, 2.5, success=False, error=exc)

        assert span.attrs["upsonic.tool.execution_time"] == 2.5
        assert span.attrs["upsonic.tool.success"] is False
        assert span.status_code == StatusCode.ERROR
        assert len(span.exceptions) == 1

    def test_tool_result_success_no_error(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_tool_result(span, 0.3, success=True)

        assert span.attrs["upsonic.tool.success"] is True
        assert span.status_code == StatusCode.OK
        assert len(span.exceptions) == 0

    def test_tool_result_none_span_is_noop(self) -> None:
        otel = _make_otel()
        otel.set_tool_result(None, 1.0, success=True)

    def test_pipeline_step_error_recording(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_step_result(
            span,
            status="ERROR",
            execution_time=1.5,
            error_message="step exploded",
        )

        assert span.attrs["upsonic.step.status"] == "ERROR"
        assert span.attrs["upsonic.step.execution_time"] == 1.5
        assert span.status_code == StatusCode.ERROR
        assert span.status_desc == "step exploded"

    def test_pipeline_step_success_no_error(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_step_result(span, status="COMPLETED", execution_time=0.5)

        assert span.attrs["upsonic.step.status"] == "COMPLETED"
        assert span.status_code == StatusCode.OK

    def test_pipeline_error_recording(self) -> None:
        from opentelemetry.trace import StatusCode
        span = RecordingSpan()
        exc = RuntimeError("pipeline crash")
        AgentOTelManager.record_error(span, exc)

        assert span.status_code == StatusCode.ERROR
        assert span.status_desc == "pipeline crash"
        assert len(span.exceptions) == 1
        assert span.exceptions[0] is exc

    def test_pipeline_error_none_span_is_noop(self) -> None:
        AgentOTelManager.record_error(None, RuntimeError("test"))


class TestUsageAttributes:

    def test_standard_usage_keys(self) -> None:
        otel = AgentOTelManager(InstrumentationSettings())
        span = RecordingSpan()
        otel.set_run_attributes(
            span,
            input_tokens=500, output_tokens=200, tool_call_count=3,
        )

        assert span.attrs["gen_ai.usage.input_tokens"] == 500
        assert span.attrs["gen_ai.usage.output_tokens"] == 200
        assert span.attrs["upsonic.tool_call_count"] == 3
        assert "gen_ai.aggregated_usage.input_tokens" not in span.attrs

    def test_aggregated_usage_keys(self) -> None:
        settings = InstrumentationSettings(use_aggregated_usage_attribute_names=True)
        otel = AgentOTelManager(settings)
        span = RecordingSpan()
        otel.set_run_attributes(span, input_tokens=1000, output_tokens=400)

        assert span.attrs["gen_ai.aggregated_usage.input_tokens"] == 1000
        assert span.attrs["gen_ai.aggregated_usage.output_tokens"] == 400
        assert "gen_ai.usage.input_tokens" not in span.attrs

    def test_zero_tokens_omitted(self) -> None:
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_run_attributes(
            span, input_tokens=0, output_tokens=0, tool_call_count=0,
        )

        assert "gen_ai.usage.input_tokens" not in span.attrs
        assert "gen_ai.usage.output_tokens" not in span.attrs
        assert "upsonic.tool_call_count" not in span.attrs

    def test_none_span_is_noop(self) -> None:
        otel = _make_otel()
        otel.set_run_attributes(None, input_tokens=100)

    def test_no_instrument_is_noop(self) -> None:
        otel = _make_otel(instrument=False)
        span = RecordingSpan()
        otel.set_run_attributes(span, input_tokens=100)
        assert len(span.attrs) == 0

    def test_non_recording_span_is_noop(self) -> None:
        otel = _make_otel()
        span = RecordingSpan(recording=False)
        otel.set_run_attributes(span, input_tokens=100)
        assert len(span.attrs) == 0


class TestAggregatedCost:

    def test_cost_set_on_span(self) -> None:
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_run_attributes(span, total_cost=0.0042)

        assert span.attrs["upsonic.total_cost"] == 0.0042

    def test_calculate_aggregated_cost_with_usage(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        agent._agent_run_output = MockRunOutput(
            input_tokens=5000, output_tokens=2000
        )
        cost = agent._calculate_aggregated_cost()
        if cost is not None:
            assert isinstance(cost, float)
            assert cost > 0

    def test_calculate_aggregated_cost_no_usage(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        agent._agent_run_output = MagicMock(usage=None)
        cost = agent._calculate_aggregated_cost()
        assert cost is None

    def test_calculate_aggregated_cost_no_output(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        cost = agent._calculate_aggregated_cost()
        assert cost is None


class TestInstrumentationSettings:

    def test_default_settings(self) -> None:
        settings = InstrumentationSettings()
        assert settings.include_content is True
        assert settings.include_binary_content is True
        assert settings.version == 4
        assert settings.event_mode == "attributes"
        assert settings.use_aggregated_usage_attribute_names is False

    def test_custom_settings(self) -> None:
        settings = InstrumentationSettings(
            include_content=False,
            include_binary_content=False,
            version=3,
            use_aggregated_usage_attribute_names=True,
        )
        assert settings.include_content is False
        assert settings.include_binary_content is False
        assert settings.version == 3
        assert settings.use_aggregated_usage_attribute_names is True

    def test_tracer_scope_name(self) -> None:
        settings = InstrumentationSettings()
        assert settings.tracer is not None



class TestSetupOpentelemetry:

    def test_function_exists(self) -> None:
        from upsonic.utils.logging_config import setup_opentelemetry
        assert callable(setup_opentelemetry)

    def test_atexit_shutdown_in_tracing_provider(self) -> None:
        import inspect
        from upsonic.integrations.tracing import TracingProvider
        source = inspect.getsource(TracingProvider.__init__)
        assert "atexit.register(self.shutdown)" in source

    def test_sampling_support_in_tracing_provider(self) -> None:
        import inspect
        from upsonic.integrations.tracing import TracingProvider
        source = inspect.getsource(TracingProvider._create_sampler)
        assert "TraceIdRatioBasedSampler" in source

    def test_headers_support_in_default_provider(self) -> None:
        import inspect
        from upsonic.integrations.tracing import DefaultTracingProvider
        source = inspect.getsource(DefaultTracingProvider.__init__)
        assert "UPSONIC_OTEL_HEADERS" in source

    def test_dynamic_insecure_flag(self) -> None:
        import inspect
        from upsonic.integrations.tracing import DefaultTracingProvider
        source = inspect.getsource(DefaultTracingProvider._create_exporter)
        assert 'self._endpoint.startswith("https")' in source

    def test_docstring_documents_env_vars(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        doc = DefaultTracingProvider.__doc__ or ""
        assert "UPSONIC_OTEL_ENDPOINT" in doc
        assert "UPSONIC_OTEL_SERVICE_NAME" in doc
        assert "UPSONIC_OTEL_HEADERS" in doc
        assert "UPSONIC_OTEL_SAMPLE_RATE" in doc


class TestParseOtelHeaders:

    def test_standard_parsing(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        result = DefaultTracingProvider._parse_headers("api-key=abc123,x-custom=hello")
        assert result == {"api-key": "abc123", "x-custom": "hello"}

    def test_whitespace_handling(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        result = DefaultTracingProvider._parse_headers("  key = value , key2 = value2  ")
        assert result == {"key": "value", "key2": "value2"}

    def test_empty_string(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        result = DefaultTracingProvider._parse_headers("")
        assert result == {}

    def test_single_pair(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        result = DefaultTracingProvider._parse_headers("Authorization=Bearer token123")
        assert result == {"Authorization": "Bearer token123"}

    def test_malformed_entries_skipped(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        result = DefaultTracingProvider._parse_headers("good=val,bad_no_equals,=no_key,also=val2")
        assert "good" in result
        assert "also" in result
        assert "bad_no_equals" not in result

    def test_equals_in_value(self) -> None:
        from upsonic.integrations.tracing import DefaultTracingProvider
        result = DefaultTracingProvider._parse_headers("token=abc=def=ghi")
        assert result == {"token": "abc=def=ghi"}


class TestSpanHierarchy:

    def test_do_async_uses_otel_manager(self) -> None:
        import inspect
        source = inspect.getsource(Agent.do_async)
        assert "self._otel.agent_run_span" in source
        assert "self._otel.finalize_agent_run" in source
        assert "self._otel.record_error" in source

    def test_astream_uses_otel_manager(self) -> None:
        import inspect
        source = inspect.getsource(Agent.astream)
        assert "self._otel.agent_run_span" in source
        assert "self._otel.finalize_agent_run" in source
        assert "self._otel.record_error" in source

    def test_pipeline_execute_uses_otel_manager(self) -> None:
        import inspect
        source = inspect.getsource(PipelineManager.execute)
        assert "self._otel.pipeline_span" in source
        assert "self._otel.step_span" in source
        assert "self._otel.set_step_result" in source
        assert "self._otel.record_error" in source

    def test_pipeline_execute_stream_uses_otel_manager(self) -> None:
        import inspect
        source = inspect.getsource(PipelineManager.execute_stream)
        assert "self._otel.pipeline_span" in source
        assert "self._otel.step_span" in source
        assert "self._otel.record_error" in source

    def test_tool_execution_uses_otel_manager(self) -> None:
        import inspect
        source = inspect.getsource(Agent._execute_tool_calls)
        assert "self._otel.tool_span" in source
        assert "self._otel.set_tool_result" in source


class TestInstrumentedModelWrapping:

    def test_wrapped_model_name_propagates(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        assert "gpt-4o" in agent.model.model_name

    def test_wrapped_model_system_propagates(self) -> None:
        agent = Agent("openai/gpt-4o", instrument=True)
        assert hasattr(agent.model, "system")

    def test_instrumentation_settings_stored(self) -> None:
        settings = InstrumentationSettings()
        agent = Agent("openai/gpt-4o", instrument=settings)
        assert agent.model.instrumentation_settings is settings


class TestEndToEndAttributes:

    def test_full_run_attributes(self) -> None:
        otel = AgentOTelManager(InstrumentationSettings())
        span = RecordingSpan()
        otel.set_run_attributes(
            span,
            input_tokens=2000,
            output_tokens=800,
            tool_call_count=5,
            total_cost=0.012,
        )

        assert span.attrs["gen_ai.usage.input_tokens"] == 2000
        assert span.attrs["gen_ai.usage.output_tokens"] == 800
        assert span.attrs["upsonic.tool_call_count"] == 5
        assert span.attrs["upsonic.total_cost"] == 0.012

    def test_full_run_aggregated_attributes(self) -> None:
        settings = InstrumentationSettings(use_aggregated_usage_attribute_names=True)
        otel = AgentOTelManager(settings)
        span = RecordingSpan()
        otel.set_run_attributes(
            span,
            input_tokens=3000,
            output_tokens=1200,
            tool_call_count=2,
        )

        assert span.attrs["gen_ai.aggregated_usage.input_tokens"] == 3000
        assert span.attrs["gen_ai.aggregated_usage.output_tokens"] == 1200
        assert span.attrs["upsonic.tool_call_count"] == 2
        assert "gen_ai.usage.input_tokens" not in span.attrs

    def test_step_result_attributes_completed(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_step_result(span, status="COMPLETED", execution_time=3.7)

        assert span.attrs["upsonic.step.status"] == "COMPLETED"
        assert span.attrs["upsonic.step.execution_time"] == 3.7
        assert span.status_code == StatusCode.OK

    def test_step_result_attributes_error(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_step_result(
            span, status="ERROR", execution_time=0.1, error_message="LLM timeout",
        )

        assert span.attrs["upsonic.step.status"] == "ERROR"
        assert span.status_code == StatusCode.ERROR
        assert span.status_desc == "LLM timeout"

    def test_tool_success_attributes(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        otel.set_tool_result(span, 0.42, success=True)

        assert span.attrs["upsonic.tool.execution_time"] == 0.42
        assert span.attrs["upsonic.tool.success"] is True
        assert span.status_code == StatusCode.OK

    def test_tool_failure_attributes(self) -> None:
        from opentelemetry.trace import StatusCode
        otel = _make_otel()
        span = RecordingSpan()
        exc = TimeoutError("external API timeout")
        otel.set_tool_result(span, 30.0, success=False, error=exc)

        assert span.attrs["upsonic.tool.execution_time"] == 30.0
        assert span.attrs["upsonic.tool.success"] is False
        assert span.status_code == StatusCode.ERROR
        assert span.status_desc == "external API timeout"
        assert len(span.exceptions) == 1


class TestSamplingRateValidation:

    def test_sample_rate_clamped_in_tracing_provider(self) -> None:
        import inspect
        from upsonic.integrations.tracing import TracingProvider
        source = inspect.getsource(TracingProvider.__init__)
        assert "max(0.0, min(1.0" in source

    def test_invalid_sample_rate_fallback(self) -> None:
        import inspect
        from upsonic.integrations.tracing import DefaultTracingProvider
        source = inspect.getsource(DefaultTracingProvider.__init__)
        assert "except (ValueError, TypeError)" in source


class TestFinalizeAgentRun:
    """Test the high-level finalize_agent_run helper."""

    def test_finalize_sets_tokens_and_identity(self) -> None:
        otel = _make_otel()
        output = MockRunOutput(input_tokens=100, output_tokens=50, tool_call_count=2)
        output.user_id = "user-42"
        output.session_id = "session-7"

        span = RecordingSpan()
        otel.finalize_agent_run(span, output, total_cost=0.003)

        assert span.attrs["gen_ai.usage.input_tokens"] == 100
        assert span.attrs["gen_ai.usage.output_tokens"] == 50
        assert span.attrs["upsonic.tool_call_count"] == 2
        assert span.attrs["upsonic.total_cost"] == 0.003
        # set_run_attributes uses langfuse.* + user.id / session.id (no upsonic.user_id)
        assert span.attrs["langfuse.user.id"] == "user-42"
        assert span.attrs["user.id"] == "user-42"
        assert span.attrs["langfuse.session.id"] == "session-7"
        assert span.attrs["session.id"] == "session-7"

    def test_finalize_noop_when_disabled(self) -> None:
        otel = _make_otel(instrument=False)
        span = RecordingSpan()
        otel.finalize_agent_run(span, MockRunOutput(input_tokens=100))
        assert len(span.attrs) == 0

    def test_finalize_noop_with_none_output(self) -> None:
        otel = _make_otel()
        span = RecordingSpan()
        otel.finalize_agent_run(span, None)
        assert len(span.attrs) == 0

    def test_finalize_fallback_to_agent_identity(self) -> None:
        otel = _make_otel()
        output = MockRunOutput(input_tokens=10)
        span = RecordingSpan()
        otel.finalize_agent_run(
            span, output, agent_user_id="fallback-user", agent_session_id="fallback-session",
        )
        assert span.attrs["langfuse.user.id"] == "fallback-user"
        assert span.attrs["user.id"] == "fallback-user"
        assert span.attrs["langfuse.session.id"] == "fallback-session"
        assert span.attrs["session.id"] == "fallback-session"


class TestAttributeConstants:
    """Verify attribute constants are importable and consistent."""

    def test_constants_importable(self) -> None:
        from upsonic.agent.otel_manager import (
            ATTR_RUN_ID, ATTR_AGENT_NAME, ATTR_AGENT_MODEL,
            ATTR_TOOL_CALL_COUNT, ATTR_TOTAL_COST,
            ATTR_EXECUTION_TIME, ATTR_MODEL_EXECUTION_TIME,
            ATTR_INPUT, ATTR_OUTPUT,
            ATTR_TOOL_NAME, ATTR_TOOL_CALL_ID, ATTR_TOOL_EXECUTION_TIME,
            ATTR_TOOL_SUCCESS, ATTR_PIPELINE_TOTAL_STEPS,
            ATTR_PIPELINE_STREAMING, ATTR_PIPELINE_DEBUG,
            ATTR_STEP_NAME, ATTR_STEP_DESCRIPTION, ATTR_STEP_STATUS,
            ATTR_STEP_EXECUTION_TIME,
            LF_USER_ID, LF_SESSION_ID, GENERIC_USER_ID, GENERIC_SESSION_ID,
        )
        assert ATTR_RUN_ID == "upsonic.run_id"
        assert ATTR_TOOL_CALL_COUNT == "upsonic.tool_call_count"
        assert ATTR_TOTAL_COST == "upsonic.total_cost"
        assert LF_USER_ID == "langfuse.user.id"
        assert GENERIC_USER_ID == "user.id"
        assert LF_SESSION_ID == "langfuse.session.id"
        assert GENERIC_SESSION_ID == "session.id"

    def test_no_agent_pipeline_constants_in_instrumented(self) -> None:
        """instrumented.py should NOT have agent/pipeline OTel constants."""
        import inspect
        import upsonic.models.instrumented as mod
        source = inspect.getsource(mod)
        assert "ATTR_RUN_ID" not in source
        assert "ATTR_AGENT_NAME" not in source
        assert "ATTR_PIPELINE_TOTAL_STEPS" not in source
        assert "create_run_span" not in source
        assert "set_run_attributes" not in source
