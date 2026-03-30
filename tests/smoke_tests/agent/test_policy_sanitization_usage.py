"""
Pure usage tests for policy sanitization.

Each test demonstrates a real-world usage pattern with minimal boilerplate.
Only final outputs are printed — no debug or attribute inspection.
"""

import asyncio
import os
import tempfile
from typing import Any, Optional

import pytest

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.safety_engine.policies.pii_policies import PIIRule, PIIAnonymizeAction
from upsonic.safety_engine.base import Policy
from upsonic.run.events.events import (
    AgentEvent,
    ToolCallEvent,
    ToolResultEvent,
    TextDeltaEvent,
    TextCompleteEvent,
    PipelineStartEvent,
    PipelineEndEvent,
    MemoryPreparedEvent,
    SystemPromptBuiltEvent,
    ContextBuiltEvent,
    UserInputBuiltEvent,
    ChatHistoryLoadedEvent,
    RunStartedEvent,
    RunCompletedEvent,
)

pytestmark = pytest.mark.timeout(120)

SENSITIVE_EMAIL: str = "john.doe@example.com"
SENSITIVE_PHONE: str = "555-123-4567"


def _pii_policy(
    description: bool = True,
    context: bool = True,
    system_prompt: bool = True,
    chat_history: bool = True,
    tool_outputs: bool = True,
) -> Policy:
    return Policy(
        name="PII Anonymize",
        description="Anonymizes PII",
        rule=PIIRule(),
        action=PIIAnonymizeAction(),
        apply_to_description=description,
        apply_to_context=context,
        apply_to_system_prompt=system_prompt,
        apply_to_chat_history=chat_history,
        apply_to_tool_outputs=tool_outputs,
    )


@tool
def lookup_contact(query: str) -> str:
    """Look up contact information for a person."""
    return f"Contact info: email is {SENSITIVE_EMAIL}, phone is {SENSITIVE_PHONE}"


# ──────────────────────────────────────────────────────────────────────
# 1. do_async with full PII policy + tool
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_do_async_with_policy() -> None:
    """Agent.do_async with PII policy: output must contain de-anonymized PII."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=_pii_policy(),
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} (phone {SENSITIVE_PHONE}). "
            "Use the lookup_contact tool."
        ),
        context=f"User email: {SENSITIVE_EMAIL}",
        tools=[lookup_contact],
    )

    result: Any = await agent.do_async(task)
    print(f"Output: {result}")

    assert result is not None
    assert SENSITIVE_EMAIL in str(result)
    assert task._anonymization_map is None
    assert task._policy_originals is None


# ──────────────────────────────────────────────────────────────────────
# 2. Event-based streaming with PII policy
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_streaming_with_policy() -> None:
    """Stream events with PII policy: tool results anonymized, final text de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. User's email is {SENSITIVE_EMAIL}.",
        user_policy=_pii_policy(),
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} and {SENSITIVE_PHONE}. "
            "Use the lookup_contact tool."
        ),
        tools=[lookup_contact],
    )

    text_chunks: list[str] = []
    tool_results: list[ToolResultEvent] = []

    async for event in agent.astream(task, events=True):
        if isinstance(event, TextDeltaEvent):
            text_chunks.append(event.content)
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[Tool Call] {event.tool_name}")
        elif isinstance(event, ToolResultEvent):
            tool_results.append(event)
            print(f"[Tool Result] {event.tool_name}")
    print()

    full_text: str = "".join(text_chunks)
    print(f"Output: {full_text[:300]}")

    assert SENSITIVE_EMAIL in full_text

    for tr in tool_results:
        assert SENSITIVE_EMAIL not in str(tr.result or ""), \
            "Tool result events must not contain original PII"

    assert task._anonymization_map is None


# ──────────────────────────────────────────────────────────────────────
# 3. Pure text streaming with PII policy
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pure_text_streaming_with_policy() -> None:
    """Pure text streaming (events=False): only str chunks, de-anonymized."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helpful assistant. Email: {SENSITIVE_EMAIL}.",
        user_policy=_pii_policy(),
    )

    task: Task = Task(
        description=(
            f"Look up contact info for {SENSITIVE_EMAIL} (phone {SENSITIVE_PHONE}). "
            "Use the lookup_contact tool."
        ),
        tools=[lookup_contact],
    )

    streamed: str = ""
    async for chunk in agent.astream(task):
        assert isinstance(chunk, str)
        streamed += chunk
        print(chunk, end="", flush=True)
    print()

    print(f"Output: {streamed[:300]}")

    assert len(streamed) > 0
    assert SENSITIVE_EMAIL in streamed
    assert task._anonymization_map is None
    assert task.description.startswith("Look up contact info")


# ──────────────────────────────────────────────────────────────────────
# 4. Scoped policy (description only)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scoped_policy_description_only() -> None:
    """Only description is anonymized; system prompt stays unchanged."""

    original_system_prompt: str = f"You are a helper. Email: {SENSITIVE_EMAIL}"

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=original_system_prompt,
        user_policy=_pii_policy(
            description=True,
            context=False,
            system_prompt=False,
            chat_history=False,
            tool_outputs=False,
        ),
    )

    task: Task = Task(
        description=f"The email is {SENSITIVE_EMAIL}. Just say hello.",
    )

    result: Any = await agent.do_async(task)
    print(f"Output: {result}")

    assert result is not None
    assert task._anonymization_map is None

    last_sp: Optional[str] = getattr(agent, '_last_built_system_prompt', None)
    if last_sp:
        assert SENSITIVE_EMAIL in last_sp, "System prompt must NOT be anonymized"


# ──────────────────────────────────────────────────────────────────────
# 5. No policy baseline
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_policy_baseline() -> None:
    """Without policies, PII passes through unchanged."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt=f"You are a helper. Email: {SENSITIVE_EMAIL}",
    )

    task: Task = Task(
        description=f"Say hello and mention {SENSITIVE_EMAIL}.",
    )

    result: Any = await agent.do_async(task)
    print(f"Output: {result}")

    assert result is not None
    assert task._anonymization_map is None
    assert task._policy_originals is None


# ──────────────────────────────────────────────────────────────────────
# 6. Step events verification
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_events() -> None:
    """All pipeline step events are emitted with valid content."""

    agent: Agent = Agent(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful coding assistant.",
    )

    task: Task = Task(
        description="Just say 'hello world' and nothing else.",
        context="User prefers concise answers.",
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
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    print()

    assert len(memory_events) == 1
    assert memory_events[0].memory_enabled is False
    assert len(system_events) == 1
    assert system_events[0].prompt_length > 0
    assert len(context_events) == 1
    assert context_events[0].context_length > 0
    assert len(input_events) == 1
    assert input_events[0].input_length > 0
    assert len(history_events) == 1
    assert history_events[0].history_count >= 0

    print("All step events verified.")


# ──────────────────────────────────────────────────────────────────────
# 7. Chat history with SQLite + policy (multi-run)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_history_sqlite_policy() -> None:
    """Run 1 stores PII in chat history (no policy). Run 2 with policy de-anonymizes."""

    db_path: str = os.path.join(tempfile.mkdtemp(), "test_policy_memory.db")

    from upsonic.storage.sqlite import SqliteStorage
    from upsonic.storage.memory import Memory

    storage: SqliteStorage = SqliteStorage(db_file=db_path)
    session_id: str = "policy_session"

    # Run 1: no policy
    memory1: Memory = Memory(
        storage=storage, session_id=session_id, user_id="user1",
        full_session_memory=True, model="openai/gpt-4o-mini",
    )
    agent1: Agent = Agent(model="openai/gpt-4o-mini", memory=memory1)
    task1: Task = Task(
        description=f"Remember: my email is {SENSITIVE_EMAIL}, phone {SENSITIVE_PHONE}. Confirm."
    )

    result1: Any = await agent1.do_async(task1)
    print(f"Run 1 output: {result1}")
    assert result1 is not None

    # Run 2: with policy
    memory2: Memory = Memory(
        storage=storage, session_id=session_id, user_id="user1",
        full_session_memory=True, model="openai/gpt-4o-mini",
    )
    agent2: Agent = Agent(model="openai/gpt-4o-mini", memory=memory2, user_policy=_pii_policy())
    task2: Task = Task(description="What is my email and phone number?")

    result2: Any = await agent2.do_async(task2)
    print(f"Run 2 output: {result2}")

    assert result2 is not None
    assert task2._anonymization_map is None

    try:
        os.remove(db_path)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# 8. Chat history with tool calls + policy (multi-run)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_history_tool_calls_policy() -> None:
    """Run 1 with tool calls (no policy). Run 2 with policy: PII de-anonymized."""

    db_path: str = os.path.join(tempfile.mkdtemp(), "test_chat_tool_memory.db")

    from upsonic.storage.sqlite import SqliteStorage
    from upsonic.storage.memory import Memory

    storage: SqliteStorage = SqliteStorage(db_file=db_path)
    session_id: str = "tool_session"

    # Run 1: tool + no policy
    memory1: Memory = Memory(
        storage=storage, session_id=session_id, user_id="user1",
        full_session_memory=True, model="openai/gpt-4o-mini",
    )
    agent1: Agent = Agent(model="openai/gpt-4o-mini", memory=memory1)
    task1: Task = Task(
        description=f"Look up contact info for {SENSITIVE_EMAIL}. Use lookup_contact tool.",
        tools=[lookup_contact],
    )

    result1: Any = await agent1.do_async(task1)
    print(f"Run 1 output: {result1}")
    assert result1 is not None
    assert SENSITIVE_EMAIL in str(result1)

    # Run 2: policy applied
    memory2: Memory = Memory(
        storage=storage, session_id=session_id, user_id="user1",
        full_session_memory=True, model="openai/gpt-4o-mini",
    )
    agent2: Agent = Agent(model="openai/gpt-4o-mini", memory=memory2, user_policy=_pii_policy())
    task2: Task = Task(
        description=f"What email and phone did you find earlier? Also look up {SENSITIVE_EMAIL} again.",
        tools=[lookup_contact],
    )

    result2: Any = await agent2.do_async(task2)
    print(f"Run 2 output: {result2}")

    assert result2 is not None
    assert SENSITIVE_EMAIL in str(result2)
    assert task2._anonymization_map is None

    try:
        os.remove(db_path)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    tests = [
        test_do_async_with_policy,
        test_event_streaming_with_policy,
        test_pure_text_streaming_with_policy,
        test_scoped_policy_description_only,
        test_no_policy_baseline,
        test_step_events,
        test_chat_history_sqlite_policy,
        test_chat_history_tool_calls_policy,
    ]
    for fn in tests:
        try:
            print("\n" + "=" * 70)
            print(f"RUNNING {fn.__name__}")
            print("=" * 70)
            asyncio.run(fn())
            print(f"PASSED {fn.__name__}")
            print("=" * 70)
        except Exception as e:
            print(f"FAILED {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise e
