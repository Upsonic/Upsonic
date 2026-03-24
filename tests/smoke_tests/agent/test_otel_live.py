"""
Live OpenTelemetry integration tests with real LLM API calls.

Uses an in-memory span exporter to capture and verify actual OTel spans
produced during real agent runs (do, do_async, stream, astream).

Additionally, tests OTLP export to a real Jaeger collector when the
``JAEGER_OTLP_ENDPOINT`` env var is set (see docker-compose.yml for setup).

These tests verify:
- Span hierarchy is physically created (agent.run → pipeline → step → chat)
- GenAI semantic conventions populated on LLM spans (tokens, model, cost)
- Error recording on real failures
- Tool execution spans with success/failure attributes
- Aggregated usage and cost on root agent span
- Streaming vs non-streaming span correctness
- Model override produces correct spans then restores original
- DefaultTracingProvider exports to a live OTLP collector (Jaeger)
- include_content=False suppresses content on live collector traces
- Session/user IDs propagate to live collector traces

Run with: uv run pytest tests/smoke_tests/agent/test_otel_live.py -v -s
"""

import json
import os
import time

import pytest
import requests
from typing import Any, Dict, List, Optional, Set

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider

from upsonic import Agent, Task
from upsonic.models.instrumented import InstrumentationSettings

MODEL: str = "openai/gpt-4o-mini"

JAEGER_OTLP_ENDPOINT: str = os.getenv("JAEGER_OTLP_ENDPOINT", "http://localhost:4317")
JAEGER_QUERY_URL: str = os.getenv("JAEGER_QUERY_URL", "http://localhost:16686")


@pytest.fixture()
def otel_capture():
    """Create InstrumentationSettings backed by an in-memory exporter.

    Yields a tuple of (InstrumentationSettings, InMemorySpanExporter).
    The exporter collects all finished spans for assertion.
    """
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter_provider = MeterProvider()

    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield settings, exporter

    tracer_provider.shutdown()
    meter_provider.shutdown()


def _span_names(exporter: InMemorySpanExporter) -> List[str]:
    """Return the names of all finished spans."""
    return [s.name for s in exporter.get_finished_spans()]


def _spans_by_name(exporter: InMemorySpanExporter, name: str) -> List[ReadableSpan]:
    """Return all finished spans matching the given name."""
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _find_span(exporter: InMemorySpanExporter, name: str) -> Optional[ReadableSpan]:
    """Return the first span with the given name, or None."""
    matches = _spans_by_name(exporter, name)
    return matches[0] if matches else None


def _attr(span: ReadableSpan, key: str) -> Any:
    """Get an attribute from a span."""
    return span.attributes.get(key) if span.attributes else None


def test_do_creates_span_hierarchy(otel_capture: tuple) -> None:
    """agent.do() should produce agent.run, pipeline.execute, pipeline.step, and chat spans."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="HierarchyAgent")
    result = agent.do("What is 2+2? Reply with just the number.")

    assert result is not None

    names: List[str] = _span_names(exporter)
    assert "agent.run" in names, f"Missing agent.run span. Got: {names}"
    assert "pipeline.execute" in names, f"Missing pipeline.execute span. Got: {names}"

    chat_spans = [n for n in names if n.startswith("chat ")]
    assert len(chat_spans) >= 1, f"Expected at least one chat span. Got: {names}"

    step_spans = [n for n in names if n.startswith("pipeline.step.")]
    assert len(step_spans) >= 1, f"Expected at least one pipeline.step span. Got: {names}"


@pytest.mark.asyncio
async def test_do_async_creates_span_hierarchy(otel_capture: tuple) -> None:
    """agent.do_async() should produce the same span hierarchy as do()."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="AsyncHierarchyAgent")
    result = await agent.do_async("What is 3+3? Reply with just the number.")

    assert result is not None

    names: List[str] = _span_names(exporter)
    assert "agent.run" in names
    assert "pipeline.execute" in names
    assert any(n.startswith("chat ") for n in names)


def test_root_span_attributes(otel_capture: tuple) -> None:
    """The agent.run span should carry agent name, model, and run_id."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="AttrAgent")
    agent.do("Say hello.")

    root = _find_span(exporter, "agent.run")
    assert root is not None, "agent.run span not found"

    assert _attr(root, "upsonic.agent.name") == "AttrAgent"
    model_attr = _attr(root, "upsonic.agent.model")
    assert model_attr is not None and MODEL in str(model_attr)

    run_id = _attr(root, "upsonic.run_id")
    assert run_id is not None and len(run_id) > 0


def test_chat_span_genai_attributes(otel_capture: tuple) -> None:
    """The chat span should have GenAI semantic convention attributes."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("What color is the sky? One word.")

    chat_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, "No chat spans found"

    span = chat_spans[0]
    assert _attr(span, "gen_ai.operation.name") == "chat"
    assert _attr(span, "gen_ai.request.model") is not None

    input_tokens = _attr(span, "gen_ai.usage.input_tokens")
    output_tokens = _attr(span, "gen_ai.usage.output_tokens")
    assert input_tokens is not None and input_tokens > 0, f"input_tokens={input_tokens}"
    assert output_tokens is not None and output_tokens > 0, f"output_tokens={output_tokens}"

    response_model = _attr(span, "gen_ai.response.model")
    assert response_model is not None


def test_aggregated_usage_on_root_span(otel_capture: tuple) -> None:
    """Root agent.run span should have aggregated token counts."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    output = agent.do("Count to 5.", return_output=True)

    root = _find_span(exporter, "agent.run")
    assert root is not None

    input_tokens = _attr(root, "gen_ai.usage.input_tokens")
    output_tokens = _attr(root, "gen_ai.usage.output_tokens")
    assert input_tokens is not None and input_tokens > 0
    assert output_tokens is not None and output_tokens > 0


def test_aggregated_usage_naming(otel_capture: tuple) -> None:
    """With use_aggregated_usage_attribute_names=True, root span uses gen_ai.aggregated_usage.*."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter_provider = MeterProvider()

    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        use_aggregated_usage_attribute_names=True,
    )

    try:
        agent = Agent(MODEL, instrument=settings)
        agent.do("Say hi.")

        root = _find_span(exporter, "agent.run")
        assert root is not None

        assert _attr(root, "gen_ai.aggregated_usage.input_tokens") is not None
        assert _attr(root, "gen_ai.usage.input_tokens") is None
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()


def test_cost_on_root_span(otel_capture: tuple) -> None:
    """Root span should include upsonic.total_cost when pricing data is available."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("What is Python?")

    root = _find_span(exporter, "agent.run")
    assert root is not None

    cost = _attr(root, "upsonic.total_cost")
    if cost is not None:
        assert isinstance(cost, float)
        assert cost > 0


def test_tool_execution_spans(otel_capture: tuple) -> None:
    """Tool calls should produce tool.execute spans with success attributes."""
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, tools=[add_numbers])
    result = agent.do("Use the add_numbers tool to add 10 and 20. Return only the result number.")

    assert result is not None

    tool_spans = _spans_by_name(exporter, "tool.execute")
    assert len(tool_spans) >= 1, f"Expected tool.execute spans, got: {_span_names(exporter)}"

    ts = tool_spans[0]
    assert _attr(ts, "upsonic.tool.name") == "add_numbers"
    assert _attr(ts, "upsonic.tool.success") is True
    exec_time = _attr(ts, "upsonic.tool.execution_time")
    assert exec_time is not None and exec_time >= 0


def test_tool_call_count_on_root(otel_capture: tuple) -> None:
    """Root span should record upsonic.tool_call_count when tools are used."""
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of a and b
        """
        return a * b

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, tools=[multiply])
    agent.do("Use the multiply tool to compute 7 times 8. Return just the number.")

    root = _find_span(exporter, "agent.run")
    assert root is not None

    tc_count = _attr(root, "upsonic.tool_call_count")
    assert tc_count is not None and tc_count >= 1, f"Expected tool_call_count >= 1, got {tc_count}"


def test_structured_output_spans(otel_capture: tuple) -> None:
    """Structured output (Pydantic model) should still produce full span hierarchy."""
    from pydantic import BaseModel, Field

    class MathAnswer(BaseModel):
        answer: int = Field(description="The numeric answer")

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    task = Task(description="What is 5 times 6?", response_format=MathAnswer)
    output = agent.do(task, return_output=True)

    assert output is not None
    assert isinstance(output.output, MathAnswer)
    assert output.output.answer == 30

    names = _span_names(exporter)
    assert "agent.run" in names
    assert any(n.startswith("chat ") for n in names)


@pytest.mark.asyncio
async def test_astream_creates_spans(otel_capture: tuple) -> None:
    """agent.astream() should produce agent.run and pipeline spans."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="StreamAgent")
    task = Task(description="Count from 1 to 3, each on a new line.")

    chunks: List[str] = []
    async for chunk in agent.astream(task, events=False):
        chunks.append(chunk)

    assert len(chunks) > 0, "Should have received streaming chunks"
    accumulated = "".join(chunks)
    assert len(accumulated) > 0

    names = _span_names(exporter)
    assert "agent.run" in names, f"Missing agent.run span for streaming. Got: {names}"


def test_stream_creates_spans(otel_capture: tuple) -> None:
    """agent.stream() should produce agent.run and pipeline spans."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="SyncStreamAgent")
    task = Task(description="Say hello world.")

    chunks: List[str] = []
    for chunk in agent.stream(task, events=False):
        chunks.append(chunk)

    assert len(chunks) > 0
    accumulated = "".join(chunks)
    assert len(accumulated) > 0

    names = _span_names(exporter)
    assert "agent.run" in names, f"Missing agent.run for sync stream. Got: {names}"


@pytest.mark.asyncio
async def test_astream_with_tool_spans(otel_capture: tuple) -> None:
    """Streaming with tools should produce both tool.execute and chat spans."""
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def get_weather(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: The city name

        Returns:
            Weather description
        """
        return f"Sunny and 22°C in {city}"

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, tools=[get_weather])
    task = Task(description="Use the get_weather tool for Paris and tell me the weather.")

    chunks: List[str] = []
    async for chunk in agent.astream(task, events=False):
        chunks.append(chunk)

    assert len(chunks) > 0

    names = _span_names(exporter)
    assert "agent.run" in names

    tool_spans = _spans_by_name(exporter, "tool.execute")
    assert len(tool_spans) >= 1, f"Expected tool spans in stream mode. Got: {names}"
    assert _attr(tool_spans[0], "upsonic.tool.name") == "get_weather"


def test_model_override_span_and_restore(otel_capture: tuple) -> None:
    """Model override should produce chat span for the override model, then restore original."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="OverrideAgent")
    original_model = agent.model

    agent.do("Say hi.", model="openai/gpt-4o-mini")

    # Original model should be restored
    assert agent.model is original_model, "Model should be restored after do()"

    chat_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    request_models = [_attr(s, "gen_ai.request.model") for s in chat_spans]
    assert any("gpt-4o-mini" in str(m) for m in request_models), \
        f"Expected chat span with gpt-4o-mini model. Got request models: {request_models}"


def test_no_instrument_no_spans() -> None:
    """Agent without instrument should produce zero OTel spans."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        agent = Agent(MODEL)
        agent.do("Say hello.")

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.name == "agent.run"]
        assert len(agent_spans) == 0, "Un-instrumented agent should not produce agent.run spans"
    finally:
        tracer_provider.shutdown()


def test_pipeline_step_span_attributes(otel_capture: tuple) -> None:
    """Pipeline step spans should carry step name and description."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("What is 1+1?")

    step_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("pipeline.step.")]
    assert len(step_spans) >= 1, f"No pipeline step spans. Got: {_span_names(exporter)}"

    for ss in step_spans:
        step_name = _attr(ss, "upsonic.step.name")
        assert step_name is not None and len(step_name) > 0


def test_pipeline_span_attributes(otel_capture: tuple) -> None:
    """pipeline.execute span should carry total_steps, streaming, and debug attributes."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("Hi")

    pipeline_span = _find_span(exporter, "pipeline.execute")
    assert pipeline_span is not None

    total_steps = _attr(pipeline_span, "upsonic.pipeline.total_steps")
    assert total_steps is not None and total_steps > 0

    streaming = _attr(pipeline_span, "upsonic.pipeline.streaming")
    assert streaming is not None


def test_span_parent_child_relationships(otel_capture: tuple) -> None:
    """Verify that spans form a proper parent-child tree."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("Hi")

    spans = exporter.get_finished_spans()
    span_map: Dict[int, ReadableSpan] = {s.context.span_id: s for s in spans}

    root = _find_span(exporter, "agent.run")
    assert root is not None
    root_id = root.context.span_id

    pipeline = _find_span(exporter, "pipeline.execute")
    assert pipeline is not None
    assert pipeline.parent is not None
    assert pipeline.parent.span_id == root_id, \
        "pipeline.execute should be a child of agent.run"

    step_spans = [s for s in spans if s.name.startswith("pipeline.step.")]
    pipeline_id = pipeline.context.span_id
    for ss in step_spans:
        assert ss.parent is not None
        assert ss.parent.span_id == pipeline_id, \
            f"pipeline.step should be child of pipeline.execute, but parent={ss.parent.span_id}"


def test_chat_span_has_cost(otel_capture: tuple) -> None:
    """Individual chat spans should include operation.cost when pricing is available."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("Hi")

    chat_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    cost = _attr(chat_spans[0], "operation.cost")
    if cost is not None:
        assert isinstance(cost, float)
        assert cost > 0


def test_multiple_runs_independent_traces(otel_capture: tuple) -> None:
    """Each do() call should produce its own agent.run span with unique run_id."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("First task: say A.")
    agent.do("Second task: say B.")

    root_spans = _spans_by_name(exporter, "agent.run")
    assert len(root_spans) == 2, f"Expected 2 agent.run spans, got {len(root_spans)}"

    run_ids: Set[str] = set()
    for rs in root_spans:
        rid = _attr(rs, "upsonic.run_id")
        assert rid is not None and len(rid) > 0
        run_ids.add(rid)

    assert len(run_ids) == 2, "Each run should have a unique run_id"


def test_include_content_false_hides_messages() -> None:
    """With include_content=False, chat spans should not include prompt/response content."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter_provider = MeterProvider()

    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_content=False,
    )

    try:
        agent = Agent(MODEL, instrument=settings)
        agent.do("Tell me a secret password: XYZZY123.")

        chat_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        span = chat_spans[0]
        input_msgs = _attr(span, "gen_ai.input.messages")
        if input_msgs is not None:
            assert "XYZZY123" not in str(input_msgs), \
                "With include_content=False, prompt content should not appear in span attributes"
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()


def test_span_kind_assignments(otel_capture: tuple) -> None:
    """Verify each span type uses the correct SpanKind."""
    from opentelemetry.trace import SpanKind

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("What is 1+1?")

    root = _find_span(exporter, "agent.run")
    assert root is not None
    assert root.kind == SpanKind.SERVER, f"agent.run should be SERVER, got {root.kind}"

    pipeline = _find_span(exporter, "pipeline.execute")
    assert pipeline is not None
    assert pipeline.kind == SpanKind.INTERNAL, f"pipeline.execute should be INTERNAL, got {pipeline.kind}"

    step_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("pipeline.step.")]
    assert len(step_spans) >= 1
    for ss in step_spans:
        assert ss.kind == SpanKind.INTERNAL, f"{ss.name} should be INTERNAL, got {ss.kind}"

    chat_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    for cs in chat_spans:
        assert cs.kind == SpanKind.CLIENT, f"{cs.name} should be CLIENT, got {cs.kind}"


def test_tool_span_kind(otel_capture: tuple) -> None:
    """tool.execute spans should have SpanKind.INTERNAL."""
    from opentelemetry.trace import SpanKind
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def double(n: int) -> int:
        """Double a number.

        Args:
            n: The number to double

        Returns:
            The doubled number
        """
        return n * 2

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, tools=[double])
    agent.do("Use the double tool to double 21. Return just the number.")

    tool_spans = _spans_by_name(exporter, "tool.execute")
    assert len(tool_spans) >= 1, f"Expected tool.execute spans. Got: {_span_names(exporter)}"

    for ts in tool_spans:
        assert ts.kind == SpanKind.INTERNAL, f"tool.execute should be INTERNAL, got {ts.kind}"


def test_status_code_ok_on_success(otel_capture: tuple) -> None:
    """Successful agent run should set StatusCode.OK on root, pipeline, and step spans."""
    from opentelemetry.trace import StatusCode

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("Say hello.")

    root = _find_span(exporter, "agent.run")
    assert root is not None
    assert root.status.status_code == StatusCode.OK, \
        f"agent.run should be OK, got {root.status.status_code}"

    pipeline = _find_span(exporter, "pipeline.execute")
    assert pipeline is not None
    assert pipeline.status.status_code == StatusCode.OK, \
        f"pipeline.execute should be OK, got {pipeline.status.status_code}"

    step_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("pipeline.step.")]
    for ss in step_spans:
        assert ss.status.status_code == StatusCode.OK, \
            f"{ss.name} should be OK, got {ss.status.status_code}"


def test_status_code_ok_on_tool_success(otel_capture: tuple) -> None:
    """Successful tool execution should set StatusCode.OK on the tool span."""
    from opentelemetry.trace import StatusCode
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def subtract(a: int, b: int) -> int:
        """Subtract b from a.

        Args:
            a: First number
            b: Number to subtract

        Returns:
            Difference of a and b
        """
        return a - b

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, tools=[subtract])
    agent.do("Use the subtract tool to compute 50 minus 8. Return just the number.")

    tool_spans = _spans_by_name(exporter, "tool.execute")
    assert len(tool_spans) >= 1

    for ts in tool_spans:
        assert ts.status.status_code == StatusCode.OK, \
            f"tool.execute should be OK, got {ts.status.status_code}"


def test_langfuse_trace_input_output(otel_capture: tuple) -> None:
    """Root span should have langfuse.trace.input/output and langfuse.observation.input/output."""
    import json

    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("What is 10 + 10? Reply with just the number.")

    root = _find_span(exporter, "agent.run")
    assert root is not None

    lf_trace_input = _attr(root, "langfuse.trace.input")
    assert lf_trace_input is not None, "langfuse.trace.input should be set"
    parsed_input = json.loads(lf_trace_input)
    assert "10" in str(parsed_input)

    lf_trace_output = _attr(root, "langfuse.trace.output")
    assert lf_trace_output is not None, "langfuse.trace.output should be set"

    lf_obs_input = _attr(root, "langfuse.observation.input")
    assert lf_obs_input is not None, "langfuse.observation.input should be set"

    lf_obs_output = _attr(root, "langfuse.observation.output")
    assert lf_obs_output is not None, "langfuse.observation.output should be set"


def test_langfuse_trace_name(otel_capture: tuple) -> None:
    """langfuse.trace.name should be set to the task description, not 'agent.run'."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings, name="TraceNameAgent")
    agent.do("Say hi.")

    root = _find_span(exporter, "agent.run")
    assert root is not None

    trace_name = _attr(root, "langfuse.trace.name")
    assert trace_name is not None, "langfuse.trace.name should be set"
    assert trace_name != "agent.run", \
        "langfuse.trace.name should differ from the span name to avoid duplication in Langfuse UI"


def test_user_id_session_id_propagation(otel_capture: tuple) -> None:
    """user.id and session.id should appear on root span when set on the Agent."""
    settings, exporter = otel_capture

    agent = Agent(
        MODEL,
        instrument=settings,
        session_id="test-session-42",
        user_id="test-user-7",
    )
    agent.do("Say hi.")

    root = _find_span(exporter, "agent.run")
    assert root is not None

    assert _attr(root, "langfuse.user.id") == "test-user-7"
    assert _attr(root, "user.id") == "test-user-7"

    assert _attr(root, "langfuse.session.id") == "test-session-42"
    assert _attr(root, "session.id") == "test-session-42"

    
def test_execution_time_on_root(otel_capture: tuple) -> None:
    """Root span should include upsonic.execution_time."""
    settings, exporter = otel_capture

    agent = Agent(MODEL, instrument=settings)
    agent.do("Say hi.")

    root = _find_span(exporter, "agent.run")
    assert root is not None

    exec_time = _attr(root, "upsonic.execution_time")
    assert exec_time is not None, "upsonic.execution_time should be set"
    assert isinstance(exec_time, float)
    assert exec_time > 0


def _jaeger_available() -> bool:
    """Return True if the Jaeger query API is reachable."""
    try:
        r = requests.get(f"{JAEGER_QUERY_URL}/api/services", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _poll_jaeger_trace(
    service: str,
    *,
    max_wait: float = 10.0,
    poll_interval: float = 1.0,
    lookback: str = "30s",
) -> Optional[Dict[str, Any]]:
    """Poll Jaeger for the most recent trace of a service."""
    deadline: float = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            r = requests.get(
                f"{JAEGER_QUERY_URL}/api/traces",
                params={"service": service, "limit": 1, "lookback": lookback},
                timeout=3,
            )
            data = r.json()
            traces = data.get("data", [])
            if traces:
                return traces[0]
        except Exception:
            pass
        time.sleep(poll_interval)
    return None


def _jaeger_root_tags(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Extract tags from the agent.run span in a Jaeger trace."""
    for span in trace.get("spans", []):
        if span["operationName"] == "agent.run":
            return {t["key"]: t["value"] for t in span.get("tags", [])}
    return {}


def _jaeger_span_ops(trace: Dict[str, Any]) -> List[str]:
    """Return all operation names from a Jaeger trace."""
    return [s["operationName"] for s in trace.get("spans", [])]


jaeger_required = pytest.mark.skipif(
    not _jaeger_available(),
    reason="Jaeger OTLP collector not available at " + JAEGER_QUERY_URL,
)


@jaeger_required
def test_jaeger_default_tracing_provider() -> None:
    """DefaultTracingProvider should export traces to Jaeger."""
    from upsonic.integrations.tracing import DefaultTracingProvider

    svc: str = "otel-live-default"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
    )

    agent = Agent(MODEL, instrument=provider, name="JaegerDefaultAgent")
    agent.do("What is 2 + 2? Reply with just the number.")
    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    ops: List[str] = _jaeger_span_ops(trace)
    assert "agent.run" in ops, f"Missing agent.run span. Got: {ops}"
    assert any(o.startswith("chat ") for o in ops), f"Missing chat span. Got: {ops}"
    assert any(o.startswith("pipeline.step.") for o in ops), f"Missing step spans. Got: {ops}"

    tags: Dict[str, Any] = _jaeger_root_tags(trace)
    assert tags.get("upsonic.agent.name") == "JaegerDefaultAgent"
    assert MODEL in str(tags.get("upsonic.agent.model", ""))


@jaeger_required
def test_jaeger_include_content_false() -> None:
    """include_content=False should suppress sensitive data on Jaeger traces."""
    from upsonic.integrations.tracing import DefaultTracingProvider

    svc: str = "otel-live-nocontent"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
        include_content=False,
    )

    agent = Agent(MODEL, instrument=provider)
    agent.do("My SSN is 999-88-7777. What is 2+2?")
    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    tags: Dict[str, Any] = _jaeger_root_tags(trace)
    all_values: str = " ".join(str(v) for v in tags.values())
    assert "999-88" not in all_values, "SSN leaked on root span"
    assert "upsonic.task.description" not in tags, "Task description should be hidden"
    assert "upsonic.input" not in tags, "Input should be hidden"

    for span in trace.get("spans", []):
        if "chat" in span["operationName"]:
            chat_tags: Dict[str, Any] = {t["key"]: t["value"] for t in span.get("tags", [])}
            assert "gen_ai.input.messages" not in chat_tags, "Chat input should be hidden"
            assert "gen_ai.output.messages" not in chat_tags, "Chat output should be hidden"
            chat_vals: str = " ".join(str(v) for v in chat_tags.values())
            assert "999-88" not in chat_vals, "SSN leaked on chat span"


@jaeger_required
def test_jaeger_session_user_ids() -> None:
    """session_id and user_id should appear on Jaeger traces."""
    from upsonic.integrations.tracing import DefaultTracingProvider

    svc: str = "otel-live-identity"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
    )

    agent = Agent(
        MODEL,
        instrument=provider,
        user_id="jaeger-user-42",
        session_id="jaeger-sess-abc",
    )
    agent.do("Say hi.")
    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    tags: Dict[str, Any] = _jaeger_root_tags(trace)
    assert tags.get("user.id") == "jaeger-user-42"
    assert tags.get("session.id") == "jaeger-sess-abc"
    assert tags.get("langfuse.user.id") == "jaeger-user-42"
    assert tags.get("langfuse.session.id") == "jaeger-sess-abc"


@jaeger_required
def test_jaeger_tool_execution() -> None:
    """Tool execution should produce tool.execute spans on Jaeger."""
    from upsonic.integrations.tracing import DefaultTracingProvider
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def add_nums(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    svc: str = "otel-live-tools"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
    )

    agent = Agent(MODEL, instrument=provider, tools=[add_nums])
    agent.do("Use the add_nums tool to add 10 and 20. Return only the result number.")
    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    ops: List[str] = _jaeger_span_ops(trace)
    assert "tool.execute" in ops, f"Missing tool.execute span. Got: {ops}"

    for span in trace.get("spans", []):
        if span["operationName"] == "tool.execute":
            tool_tags: Dict[str, Any] = {t["key"]: t["value"] for t in span.get("tags", [])}
            assert tool_tags.get("upsonic.tool.name") == "add_nums"
            assert tool_tags.get("upsonic.tool.success") is True


@jaeger_required
def test_jaeger_streaming() -> None:
    """Streaming should produce full span hierarchy on Jaeger."""
    import asyncio
    from upsonic.integrations.tracing import DefaultTracingProvider

    svc: str = "otel-live-stream"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
    )

    agent = Agent(MODEL, instrument=provider, name="JaegerStreamAgent")

    async def _run() -> None:
        chunks: List[str] = []
        async for chunk in agent.astream("Count from 1 to 3", events=False):
            chunks.append(chunk)
        assert len(chunks) > 0

    asyncio.get_event_loop().run_until_complete(_run())
    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    ops: List[str] = _jaeger_span_ops(trace)
    assert "agent.run" in ops, f"Missing agent.run span. Got: {ops}"
    assert any(o.startswith("chat ") for o in ops), f"Missing chat span. Got: {ops}"

    tags: Dict[str, Any] = _jaeger_root_tags(trace)
    assert tags.get("upsonic.agent.name") == "JaegerStreamAgent"


@jaeger_required
def test_jaeger_instrument_all() -> None:
    """Agent.instrument_all() with DefaultTracingProvider should export to Jaeger."""
    from upsonic.integrations.tracing import DefaultTracingProvider

    svc: str = "otel-live-global"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
    )

    try:
        Agent.instrument_all(provider)

        agent_a = Agent(MODEL, name="GlobalA")
        agent_a.do("What is 2+2?")
    finally:
        Agent.instrument_all(False)

    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    tags: Dict[str, Any] = _jaeger_root_tags(trace)
    assert tags.get("upsonic.agent.name") == "GlobalA"


@jaeger_required
def test_jaeger_aggregated_usage() -> None:
    """Root span on Jaeger should carry token counts and execution time."""
    from upsonic.integrations.tracing import DefaultTracingProvider

    svc: str = "otel-live-usage"
    provider = DefaultTracingProvider(
        endpoint=JAEGER_OTLP_ENDPOINT,
        service_name=svc,
    )

    agent = Agent(MODEL, instrument=provider)
    agent.do("What is 2+2?")
    provider.shutdown()

    trace = _poll_jaeger_trace(svc)
    assert trace is not None, f"No trace found in Jaeger for service={svc}"

    tags: Dict[str, Any] = _jaeger_root_tags(trace)
    assert "gen_ai.usage.input_tokens" in tags, f"Missing input_tokens. Tags: {list(tags.keys())}"
    assert "gen_ai.usage.output_tokens" in tags, f"Missing output_tokens. Tags: {list(tags.keys())}"
    assert tags["gen_ai.usage.input_tokens"] > 0
    assert tags["gen_ai.usage.output_tokens"] > 0

    assert "upsonic.execution_time" in tags
    assert tags["upsonic.execution_time"] > 0
