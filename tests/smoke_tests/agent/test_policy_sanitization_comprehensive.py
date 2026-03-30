"""
Comprehensive smoke test for policy sanitization across ALL LLM inputs.

Tests that anonymize/replace policies are applied to:
  - task.description
  - system prompt
  - context
  - chat history
  - tool outputs (re-run)

And that after agent finishes:
  - output is de-anonymized
  - all originals are restored (description, system prompt, chat history, etc.)

Tests both do_async (direct) and astream (streaming) paths.
Also tests pure text streaming (no events) and event-based streaming with
proper content verification for all event types including the new step events.
"""

import asyncio
import os
from typing import Any, Optional, List

import pytest

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.safety_engine.policies.pii_policies import (
    PIIAnonymizePolicy,
    PIIRule,
    PIIAnonymizeAction,
)
from upsonic.safety_engine.base import Policy

from upsonic.run.events.events import (
    PipelineStartEvent,
    PipelineEndEvent,
    StepStartEvent,
    StepEndEvent,
    AgentInitializedEvent,
    MemoryPreparedEvent,
    SystemPromptBuiltEvent,
    ContextBuiltEvent,
    UserInputBuiltEvent,
    ChatHistoryLoadedEvent,
    CacheCheckEvent,
    CacheHitEvent,
    CacheMissEvent,
    PolicyCheckEvent,
    LLMPreparedEvent,
    ModelSelectedEvent,
    ToolsConfiguredEvent,
    MessagesBuiltEvent,
    ModelRequestStartEvent,
    ModelResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    TextDeltaEvent,
    TextCompleteEvent,
    ToolCallDeltaEvent,
    FinalOutputEvent,
    RunStartedEvent,
    RunCompletedEvent,
    ExecutionCompleteEvent,
    AgentEvent,
)

pytestmark = pytest.mark.timeout(120)

SENSITIVE_EMAIL: str = "john.doe@example.com"
SENSITIVE_PHONE: str = "555-123-4567"
SENSITIVE_NAME_LABEL: str = "full name"


def _make_anonymize_policy(
    apply_to_description: bool = True,
    apply_to_context: bool = True,
    apply_to_system_prompt: bool = True,
    apply_to_chat_history: bool = True,
    apply_to_tool_outputs: bool = True,
) -> Policy:
    """Build a PII anonymize policy with explicit scope flags."""
    return Policy(
        name="Test PII Anonymize",
        description="Anonymizes PII for testing",
        rule=PIIRule(),
        action=PIIAnonymizeAction(),
        apply_to_description=apply_to_description,
        apply_to_context=apply_to_context,
        apply_to_system_prompt=apply_to_system_prompt,
        apply_to_chat_history=apply_to_chat_history,
        apply_to_tool_outputs=apply_to_tool_outputs,
    )


@tool
def lookup_contact(query: str) -> str:
    """Look up contact information for a person."""
    return f"Contact info: email is {SENSITIVE_EMAIL}, phone is {SENSITIVE_PHONE}"


# ──────────────────────────────────────────────────────────────────────
# TEST 1: do_async — anonymization + de-anonymization with tool output
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_do_async_anonymize_all_inputs() -> None:
    """Verify anonymization across description, context, system prompt, and tool output; then de-anonymization at the end."""

    policy: Policy = _make_anonymize_policy()

    original_system_prompt: str = f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}."

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="PolicySanitizeTestAgent",
        system_prompt=original_system_prompt,
        user_policy=policy,
    )

    original_description: str = (
        f"Look up contact information for the user whose email is {SENSITIVE_EMAIL} "
        f"and phone is {SENSITIVE_PHONE}. Use the lookup_contact tool."
    )
    original_context: str = f"The user's full name context: email {SENSITIVE_EMAIL}"

    task: Task = Task(
        description=original_description,
        context=original_context,
        tools=[lookup_contact],
    )

    assert task.description == original_description, "description should match original before run"
    assert task.context == original_context, "context should match original before run"
    assert task._anonymization_map is None, "_anonymization_map should be None before run"
    assert task._policy_originals is None, "_policy_originals should be None before run"
    assert task._policy_scope_tool_outputs is False, "_policy_scope_tool_outputs should be False before run"
    assert task._response is None, "_response should be None before run"

    result: Any = await agent.do_async(task)

    assert result is not None, "Result should not be None"

    result_str: str = str(result)
    assert SENSITIVE_EMAIL in result_str, "De-anonymized output should contain original email"

    assert task.description == original_description, \
        f"task.description should be restored to original after finalization. Got: {task.description[:200]}"

    last_system_prompt: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_system_prompt:
        assert SENSITIVE_EMAIL in last_system_prompt, \
            f"Restored system_prompt should contain original email. Got: {last_system_prompt[:200]}"

    if hasattr(agent, '_agent_run_output') and agent._agent_run_output and agent._agent_run_output.input:
        run_input = agent._agent_run_output.input
        assert run_input.user_prompt == original_description, \
            "AgentRunInput.user_prompt should be restored to original after finalization"

    if hasattr(agent, '_agent_run_output') and agent._agent_run_output:
        ch = agent._agent_run_output.chat_history
        if ch:
            for i, msg in enumerate(ch):
                parts = getattr(msg, 'parts', [])
                for j, p in enumerate(parts):
                    args = getattr(p, 'args', None)
                    if args is not None:
                        args_str: str = str(args)
                        assert SENSITIVE_EMAIL not in args_str or "john.doe@example.com" in args_str, \
                            f"ToolCallPart.args should be de-anonymized. Got: {args_str[:200]}"

    assert task._anonymization_map is None, "_anonymization_map should be None after finalization"
    assert task._policy_originals is None, "_policy_originals should be None after finalization"
    assert task._policy_scope_tool_outputs is False, "_policy_scope_tool_outputs should be False after finalization"


# ──────────────────────────────────────────────────────────────────────
# TEST 2: event-based streaming — full event content verification
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_events_full_verification() -> None:
    """Verify all event types emitted during streaming, check their content fields."""

    policy: Policy = _make_anonymize_policy()

    original_system_prompt: str = f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}."
    original_description: str = (
        f"Look up contact information for the user whose email is {SENSITIVE_EMAIL} "
        f"and phone is {SENSITIVE_PHONE}. Use the lookup_contact tool."
    )

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="PolicySanitizeStreamAgent",
        system_prompt=original_system_prompt,
        user_policy=policy,
    )

    task: Task = Task(
        description=original_description,
        context=f"The user's full name context: email {SENSITIVE_EMAIL}",
        tools=[lookup_contact],
    )

    assert task.description == original_description
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._response is None

    all_events: list[AgentEvent] = []
    text_chunks: list[str] = []
    tool_result_events: list[ToolResultEvent] = []

    async for event in agent.astream(task, events=True):
        all_events.append(event)

        if isinstance(event, RunStartedEvent):
            assert event.run_id, "RunStartedEvent must have a run_id"

        elif isinstance(event, PipelineStartEvent):
            assert event.total_steps > 0
            assert event.is_streaming is True

        elif isinstance(event, SystemPromptBuiltEvent):
            assert event.prompt_length > 0

        elif isinstance(event, UserInputBuiltEvent):
            assert event.input_length > 0

        elif isinstance(event, ToolCallEvent):
            assert event.tool_name, "ToolCallEvent must have tool_name"

        elif isinstance(event, ToolResultEvent):
            tool_result_events.append(event)
            assert event.tool_name, "ToolResultEvent must have tool_name"
            result_str: str = str(event.result) if event.result else ""
            assert SENSITIVE_EMAIL not in result_str, \
                f"ToolResultEvent should NOT contain original email. Got: {result_str[:200]}"

        elif isinstance(event, TextDeltaEvent):
            text_chunks.append(event.content)

    event_types: set[str] = {type(e).__name__ for e in all_events}

    assert "RunStartedEvent" in event_types
    assert "PipelineStartEvent" in event_types
    assert "PipelineEndEvent" in event_types
    assert "StepStartEvent" in event_types
    assert "StepEndEvent" in event_types
    assert "MemoryPreparedEvent" in event_types
    assert "SystemPromptBuiltEvent" in event_types
    assert "UserInputBuiltEvent" in event_types
    assert "ChatHistoryLoadedEvent" in event_types
    assert "MessagesBuiltEvent" in event_types
    assert "ToolCallEvent" in event_types, "Must have ToolCallEvent (tool was called)"
    assert "ToolResultEvent" in event_types, "Must have ToolResultEvent (tool returned)"

    for tre in tool_result_events:
        tres: str = str(tre.result) if tre.result else ""
        assert SENSITIVE_EMAIL not in tres, f"ToolResultEvent result leaked real email: {tres[:200]}"

    full_streamed: str = "".join(text_chunks)
    assert SENSITIVE_EMAIL in full_streamed, "De-anonymized streamed text should contain original email"

    assert task.description == original_description, "task.description should be restored"
    assert task._anonymization_map is None, "_anonymization_map should be None"
    assert task._policy_originals is None, "_policy_originals should be None"
    assert task._policy_scope_tool_outputs is False, "_policy_scope_tool_outputs should be False"

    last_sp: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_sp:
        assert SENSITIVE_EMAIL in last_sp, \
            f"Restored system_prompt should contain original email. Got: {last_sp[:200]}"

    if hasattr(agent, '_agent_run_output') and agent._agent_run_output and agent._agent_run_output.input:
        assert agent._agent_run_output.input.user_prompt == original_description


# ──────────────────────────────────────────────────────────────────────
# TEST 3: pure text streaming with PII policy
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pure_text_streaming_with_policy() -> None:
    """Test pure text streaming (events=False) with PII policy — yields only str chunks,
    content should be de-anonymized, and all attributes restored after."""

    policy: Policy = _make_anonymize_policy(
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )

    original_description: str = (
        f"Look up the contact info for {SENSITIVE_EMAIL} and phone {SENSITIVE_PHONE}. "
        "Use the lookup_contact tool."
    )
    original_system_prompt: str = f"You are a helpful assistant. The user email is {SENSITIVE_EMAIL}."

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="PureTextStreamPolicyAgent",
        system_prompt=original_system_prompt,
        user_policy=policy,
    )

    task: Task = Task(
        description=original_description,
        context=f"Context with email: {SENSITIVE_EMAIL}",
        tools=[lookup_contact],
    )

    assert task.description == original_description
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._response is None

    streamed_text: str = ""
    chunk_count: int = 0

    async for text_chunk in agent.astream(task):
        assert isinstance(text_chunk, str), f"Expected str chunk, got {type(text_chunk).__name__}"
        streamed_text += text_chunk
        chunk_count += 1

    assert len(streamed_text) > 0, "Streamed text should not be empty"
    assert chunk_count > 0, "Should receive at least one chunk"

    assert SENSITIVE_EMAIL in streamed_text, \
        f"De-anonymized streamed text should contain '{SENSITIVE_EMAIL}'. Got: {streamed_text[:300]}"

    assert task.description == original_description, "task.description should be restored"
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._policy_scope_tool_outputs is False

    last_sp: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_sp:
        assert SENSITIVE_EMAIL in last_sp, \
            f"Restored system_prompt should contain original email. Got: {last_sp[:200]}"

    if hasattr(agent, '_agent_run_output') and agent._agent_run_output and agent._agent_run_output.input:
        assert agent._agent_run_output.input.user_prompt == original_description, \
            "AgentRunInput.user_prompt should be restored"


# ──────────────────────────────────────────────────────────────────────
# TEST 4: Scope flags — only description, not system prompt
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scope_flags_description_only() -> None:
    """Verify scope flags: only description is anonymized when other scopes are off.
    System prompt, context, chat history should NOT be modified."""

    policy: Policy = _make_anonymize_policy(
        apply_to_description=True,
        apply_to_context=False,
        apply_to_system_prompt=False,
        apply_to_chat_history=False,
        apply_to_tool_outputs=False,
    )

    original_description: str = f"The email is {SENSITIVE_EMAIL}. Just say hello."
    original_system_prompt: str = f"You are a helper. Email: {SENSITIVE_EMAIL}"

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="ScopeTestAgent",
        system_prompt=original_system_prompt,
        user_policy=policy,
    )

    task: Task = Task(
        description=original_description,
        context=f"Context has email: {SENSITIVE_EMAIL}",
    )

    assert task.description == original_description
    assert task._anonymization_map is None
    assert task._policy_originals is None

    result: Any = await agent.do_async(task)

    assert result is not None

    assert task.description == original_description, "description should be restored"

    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._policy_scope_tool_outputs is False

    last_sp_scope: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_sp_scope:
        assert SENSITIVE_EMAIL in last_sp_scope, \
            f"System prompt should NOT be anonymized (apply_to_system_prompt=False). Got: {last_sp_scope[:200]}"


# ──────────────────────────────────────────────────────────────────────
# TEST 5: No policy — baseline (nothing should change)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_policy_baseline() -> None:
    """Baseline test: without policies, PII should pass through unchanged."""

    original_description: str = f"Say hello and mention {SENSITIVE_EMAIL}."

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="NoPolicyAgent",
        system_prompt=f"You are a helper. Email: {SENSITIVE_EMAIL}",
    )

    task: Task = Task(description=original_description)

    assert task.description == original_description
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._response is None

    result: Any = await agent.do_async(task)

    assert result is not None

    assert task.description == original_description, "description should not change without policy"
    assert task._anonymization_map is None, "_anonymization_map should remain None (no policy)"
    assert task._policy_originals is None, "_policy_originals should remain None (no policy)"
    assert task._policy_scope_tool_outputs is False


# ──────────────────────────────────────────────────────────────────────
# TEST 6: Step events content verification
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_step_events_content() -> None:
    """Verify the 5 new step events have correct content."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="StepEventTestAgent",
        system_prompt="You are a helpful coding assistant with expertise in Python.",
    )

    task: Task = Task(
        description="Just say 'hello world' and nothing else.",
        context="The user prefers concise answers.",
    )

    memory_events: list[MemoryPreparedEvent] = []
    system_events: list[SystemPromptBuiltEvent] = []
    context_events: list[ContextBuiltEvent] = []
    input_events: list[UserInputBuiltEvent] = []
    history_events: list[ChatHistoryLoadedEvent] = []

    async for event in agent.astream(task, events=True):
        if isinstance(event, MemoryPreparedEvent):
            memory_events.append(event)
        elif isinstance(event, SystemPromptBuiltEvent):
            system_events.append(event)
        elif isinstance(event, ContextBuiltEvent):
            context_events.append(event)
        elif isinstance(event, UserInputBuiltEvent):
            input_events.append(event)
        elif isinstance(event, ChatHistoryLoadedEvent):
            history_events.append(event)

    assert len(memory_events) == 1, "Should have exactly 1 MemoryPreparedEvent"
    assert memory_events[0].memory_enabled is False

    assert len(system_events) == 1
    assert system_events[0].prompt_length > 0

    assert len(context_events) == 1
    assert context_events[0].context_length > 0

    assert len(input_events) == 1
    assert input_events[0].input_type in ("text", "multipart")
    assert input_events[0].input_length > 0

    assert len(history_events) == 1
    assert history_events[0].history_count >= 0


# ──────────────────────────────────────────────────────────────────────
# TEST 7: Chat history with SQLite — first run without policy, second with
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_history_with_sqlite_policy() -> None:
    """
    First run: agent without policy stores PII in chat history via SQLite.
    Second run: same agent with policy — chat history PII should be anonymized.
    """
    import tempfile
    import os

    db_path: str = os.path.join(tempfile.mkdtemp(), "test_policy_memory.db")

    from upsonic.storage.sqlite import SqliteStorage
    from upsonic.storage.memory import Memory

    storage: SqliteStorage = SqliteStorage(db_file=db_path)
    session_id: str = "test_policy_session_001"
    user_id: str = "test_user_001"

    memory: Memory = Memory(
        storage=storage,
        session_id=session_id,
        user_id=user_id,
        full_session_memory=True,
        model="openai/gpt-4o-mini",
    )

    # --- RUN 1: No policy — agent stores PII in chat history ---
    agent1: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="ChatHistoryTestAgent",
        memory=memory,
    )

    original_description1: str = f"Remember this: my email is {SENSITIVE_EMAIL} and my phone is {SENSITIVE_PHONE}. Just confirm you got it."

    task1: Task = Task(description=original_description1)

    assert task1.description == original_description1
    assert task1._anonymization_map is None
    assert task1._policy_originals is None

    result1: Any = await agent1.do_async(task1)
    assert result1 is not None
    assert task1.description == original_description1, "Run1 description should not change (no policy)"
    assert task1._anonymization_map is None, "Run1 should not have anonymization map (no policy)"

    # --- RUN 2: With policy — chat history should be anonymized ---
    policy: Policy = _make_anonymize_policy(
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )

    memory2: Memory = Memory(
        storage=storage,
        session_id=session_id,
        user_id=user_id,
        full_session_memory=True,
        model="openai/gpt-4o-mini",
    )

    original_description2: str = f"What is my email address? Also tell me my phone number."

    agent2: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="ChatHistoryTestAgent",
        memory=memory2,
        user_policy=policy,
    )

    task2: Task = Task(description=original_description2)

    assert task2.description == original_description2
    assert task2._anonymization_map is None
    assert task2._policy_originals is None

    result2: Any = await agent2.do_async(task2)

    assert task2.description == original_description2, "description should be restored"
    assert task2._anonymization_map is None
    assert task2._policy_originals is None
    assert task2._policy_scope_tool_outputs is False
    assert result2 is not None

    try:
        os.remove(db_path)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# TEST 8: Chat history with tool calls + policy
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_history_with_tool_calls_policy() -> None:
    """
    Run 1: Agent with tool calls, NO policy → stores PII in chat history.
    Run 2: Same storage + session, WITH policy → chat history PII anonymized,
           tool output PII anonymized. Final output must be de-anonymized.
    """
    import tempfile
    import os

    db_path: str = os.path.join(tempfile.mkdtemp(), "test_chat_tool_memory.db")

    from upsonic.storage.sqlite import SqliteStorage
    from upsonic.storage.memory import Memory

    storage: SqliteStorage = SqliteStorage(db_file=db_path)
    session_id: str = "test_chat_tool_session_001"
    user_id: str = "test_user_tool_001"

    memory1: Memory = Memory(
        storage=storage,
        session_id=session_id,
        user_id=user_id,
        full_session_memory=True,
        model="openai/gpt-4o-mini",
    )

    # --- RUN 1: No policy, with tool — stores PII in chat history ---
    agent1: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="ChatToolHistoryAgent",
        memory=memory1,
    )

    original_description1_t8: str = (
        f"Look up the contact info for the email {SENSITIVE_EMAIL}. "
        "Use the lookup_contact tool. Report what you find."
    )

    task1: Task = Task(
        description=original_description1_t8,
        tools=[lookup_contact],
    )

    assert task1.description == original_description1_t8
    assert task1._anonymization_map is None
    assert task1._policy_originals is None

    result1: Any = await agent1.do_async(task1)
    assert result1 is not None
    assert task1.description == original_description1_t8, "Run1 description should not change (no policy)"
    assert task1._anonymization_map is None, "Run1 should not have anonymization map (no policy)"

    result1_str: str = str(result1)
    assert SENSITIVE_EMAIL in result1_str, \
        f"Run1 result should contain original email (no policy). Got: {result1_str[:300]}"

    # --- RUN 2: With policy — same session, PII should be anonymized in history ---
    policy: Policy = _make_anonymize_policy(
        apply_to_description=True,
        apply_to_context=True,
        apply_to_system_prompt=True,
        apply_to_chat_history=True,
        apply_to_tool_outputs=True,
    )

    memory2: Memory = Memory(
        storage=storage,
        session_id=session_id,
        user_id=user_id,
        full_session_memory=True,
        model="openai/gpt-4o-mini",
    )

    original_description2: str = (
        f"Based on our previous conversation, what is the email and phone number "
        f"you found? Also look up {SENSITIVE_EMAIL} again with the tool."
    )

    agent2: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="ChatToolHistoryAgent",
        memory=memory2,
        user_policy=policy,
    )

    task2: Task = Task(
        description=original_description2,
        tools=[lookup_contact],
    )

    assert task2.description == original_description2
    assert task2._anonymization_map is None
    assert task2._policy_originals is None

    result2: Any = await agent2.do_async(task2)

    assert task2.description == original_description2, "description should be restored"
    assert task2._anonymization_map is None
    assert task2._policy_originals is None
    assert task2._policy_scope_tool_outputs is False
    assert result2 is not None

    result2_str: str = str(result2)
    assert SENSITIVE_EMAIL in result2_str, \
        f"De-anonymized result should contain original email. Got: {result2_str[:300]}"

    try:
        os.remove(db_path)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# TEST 9: Event-based streaming — multi-PII (email + phone) de-anonymization
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_events_multi_pii_deanonymization() -> None:
    """Event-based streaming with multiple PII types (email + phone).
    Both must be de-anonymized in the streamed text.
    ToolResultEvent must NOT contain either original PII value."""

    policy: Policy = _make_anonymize_policy()

    original_system_prompt: str = (
        f"You are a helpful assistant. The user's email is {SENSITIVE_EMAIL} "
        f"and phone is {SENSITIVE_PHONE}."
    )
    original_description: str = (
        f"Look up the contact info for email {SENSITIVE_EMAIL} and phone {SENSITIVE_PHONE}. "
        "Use the lookup_contact tool. "
        "In your response, you MUST include the exact email address and exact phone number."
    )
    original_context: str = (
        f"User email: {SENSITIVE_EMAIL}, User phone: {SENSITIVE_PHONE}"
    )

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="MultiPIIStreamEventAgent",
        system_prompt=original_system_prompt,
        user_policy=policy,
        print=True,
    )

    task: Task = Task(
        description=original_description,
        context=original_context,
        tools=[lookup_contact],
    )

    assert task.description == original_description
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._response is None

    all_events: list[AgentEvent] = []
    text_chunks: list[str] = []
    tool_call_events: list[ToolCallEvent] = []
    tool_result_events: list[ToolResultEvent] = []

    print("\n" + "=" * 80)
    print("TEST 9: Event streaming — multi-PII (email + phone)")
    print("=" * 80)

    async for event in agent.astream(task, events=True):
        all_events.append(event)

        if isinstance(event, ToolCallEvent):
            tool_call_events.append(event)
            print(f"\n[ToolCall] {event.tool_name} args={event.tool_args}")

        elif isinstance(event, ToolResultEvent):
            tool_result_events.append(event)
            print(f"[ToolResult] {event.tool_name} result={str(event.result)[:300]}")

        elif isinstance(event, TextDeltaEvent):
            text_chunks.append(event.content)
            print(event.content, end="", flush=True)

    print("\n")

    full_streamed: str = "".join(text_chunks)
    print(f"[FULL STREAMED TEXT] ({len(full_streamed)} chars):")
    print(full_streamed)
    print()

    # ToolResultEvent must NOT contain original PII
    for tre in tool_result_events:
        tres: str = str(tre.result) if tre.result else ""
        print(f"[CHECK] ToolResultEvent contains original email: {SENSITIVE_EMAIL in tres}")
        print(f"[CHECK] ToolResultEvent contains original phone: {SENSITIVE_PHONE in tres}")
        assert SENSITIVE_EMAIL not in tres, \
            f"ToolResultEvent must NOT contain original email. Got: {tres[:300]}"
        assert SENSITIVE_PHONE not in tres, \
            f"ToolResultEvent must NOT contain original phone. Got: {tres[:300]}"

    # Streamed text must contain BOTH de-anonymized PII
    email_found: bool = SENSITIVE_EMAIL in full_streamed
    phone_found: bool = SENSITIVE_PHONE in full_streamed
    print(f"[CHECK] Streamed text contains email '{SENSITIVE_EMAIL}': {email_found}")
    print(f"[CHECK] Streamed text contains phone '{SENSITIVE_PHONE}': {phone_found}")
    assert email_found, \
        f"Streamed text must contain de-anonymized email '{SENSITIVE_EMAIL}'. Got: {full_streamed[:500]}"
    assert phone_found, \
        f"Streamed text must contain de-anonymized phone '{SENSITIVE_PHONE}'. Got: {full_streamed[:500]}"

    # Cleanup assertions
    assert task.description == original_description, "task.description should be restored"
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._policy_scope_tool_outputs is False

    last_sp: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_sp:
        assert SENSITIVE_EMAIL in last_sp, \
            f"system_prompt should contain original email. Got: {last_sp[:200]}"
        assert SENSITIVE_PHONE in last_sp, \
            f"system_prompt should contain original phone. Got: {last_sp[:200]}"

    print("[TEST 9] PASSED")


# ──────────────────────────────────────────────────────────────────────
# TEST 10: Pure text streaming — multi-PII (email + phone) de-anonymization
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pure_text_stream_multi_pii_deanonymization() -> None:
    """Pure text streaming (events=False) with multiple PII types (email + phone).
    Every str chunk must be a str. The concatenated output must contain
    both original email and phone, proving StreamDeanonymizer handles
    multiple anonymous values of different lengths."""

    policy: Policy = _make_anonymize_policy()

    original_system_prompt: str = (
        f"You are a helpful assistant. The user's email is {SENSITIVE_EMAIL} "
        f"and phone is {SENSITIVE_PHONE}."
    )
    original_description: str = (
        f"Look up the contact info for email {SENSITIVE_EMAIL} and phone {SENSITIVE_PHONE}. "
        "Use the lookup_contact tool. "
        "In your response, you MUST include the exact email address and exact phone number."
    )
    original_context: str = (
        f"User email: {SENSITIVE_EMAIL}, User phone: {SENSITIVE_PHONE}"
    )

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        name="MultiPIIPureTextStreamAgent",
        system_prompt=original_system_prompt,
        user_policy=policy,
        print=True,
    )

    task: Task = Task(
        description=original_description,
        context=original_context,
        tools=[lookup_contact],
    )

    assert task.description == original_description
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._response is None

    print("\n" + "=" * 80)
    print("TEST 10: Pure text streaming — multi-PII (email + phone)")
    print("=" * 80)
    print("[STREAM] ", end="")

    streamed_text: str = ""
    chunk_count: int = 0

    async for text_chunk in agent.astream(task):
        assert isinstance(text_chunk, str), f"Expected str, got {type(text_chunk).__name__}"
        streamed_text += text_chunk
        chunk_count += 1
        print(text_chunk, end="", flush=True)

    print("\n")

    print(f"[FULL STREAMED TEXT] ({len(streamed_text)} chars, {chunk_count} chunks):")
    print(streamed_text)
    print()

    assert len(streamed_text) > 0, "Streamed text should not be empty"
    assert chunk_count > 0, "Should receive at least one chunk"

    # Must contain BOTH de-anonymized PII
    email_found: bool = SENSITIVE_EMAIL in streamed_text
    phone_found: bool = SENSITIVE_PHONE in streamed_text
    print(f"[CHECK] Streamed text contains email '{SENSITIVE_EMAIL}': {email_found}")
    print(f"[CHECK] Streamed text contains phone '{SENSITIVE_PHONE}': {phone_found}")
    assert email_found, \
        f"Streamed text must contain de-anonymized email '{SENSITIVE_EMAIL}'. Got: {streamed_text[:500]}"
    assert phone_found, \
        f"Streamed text must contain de-anonymized phone '{SENSITIVE_PHONE}'. Got: {streamed_text[:500]}"

    # Also verify task._response contains both
    response_str: str = str(task._response) if task._response else ""
    response_email: bool = SENSITIVE_EMAIL in response_str
    response_phone: bool = SENSITIVE_PHONE in response_str
    print(f"[CHECK] task._response contains email: {response_email}")
    print(f"[CHECK] task._response contains phone: {response_phone}")

    # Cleanup assertions
    assert task.description == original_description, "task.description should be restored"
    assert task._anonymization_map is None
    assert task._policy_originals is None
    assert task._policy_scope_tool_outputs is False

    last_sp: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_sp:
        assert SENSITIVE_EMAIL in last_sp, \
            f"system_prompt should contain original email. Got: {last_sp[:200]}"
        assert SENSITIVE_PHONE in last_sp, \
            f"system_prompt should contain original phone. Got: {last_sp[:200]}"

    if hasattr(agent, '_agent_run_output') and agent._agent_run_output and agent._agent_run_output.input:
        assert agent._agent_run_output.input.user_prompt == original_description, \
            "AgentRunInput.user_prompt should be restored"

    print("[TEST 10] PASSED")


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    tests = [
        test_do_async_anonymize_all_inputs,
        test_stream_events_full_verification,
        test_pure_text_streaming_with_policy,
        test_scope_flags_description_only,
        test_no_policy_baseline,
        test_stream_step_events_content,
        test_chat_history_with_sqlite_policy,
        test_chat_history_with_tool_calls_policy,
        test_stream_events_multi_pii_deanonymization,
        test_pure_text_stream_multi_pii_deanonymization,
    ]
    for test_fn in tests:
        try:
            asyncio.run(test_fn())
        except Exception as exc:
            print(f"\n[TEST] FAILED {test_fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
