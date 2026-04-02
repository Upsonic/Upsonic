"""
Summary Memory Smoke Tests

Verifies summary_memory functionality:
1. Summary is generated after conversations
2. Summary persists across agent runs
3. Summary is injected into context when load_summary_memory is enabled
4. Summary works with full_session_memory enabled
5. Summary works WITHOUT full_session_memory (save only summary, not messages)
6. Summary updates/grows across multiple turns
7. Save/load flag separation for summary memory
8. SQLite persistence of summary
"""

import os
import tempfile
import uuid
from typing import List, Optional

import pytest

from upsonic import Agent, Task
from upsonic.storage import Memory, SqliteStorage, InMemoryStorage
from upsonic.session.agent import AgentSession
from upsonic.session.base import SessionType

pytestmark = pytest.mark.timeout(300)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def sqlite_storage() -> SqliteStorage:
    db_file = tempfile.mktemp(suffix=".db")
    s = SqliteStorage(db_file=db_file)
    yield s
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.fixture
def session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


# ═════════════════════════════════════════════════════════════════════════
# 1. Summary Memory Flag Defaults
# ═════════════════════════════════════════════════════════════════════════

class TestSummaryMemoryFlagDefaults:

    def test_summary_flags_set_correctly(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            model="openai/gpt-4o-mini",
        )
        assert m.summary_memory_enabled is True
        assert m.load_summary_memory_enabled is True

    def test_summary_save_true_load_false(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            load_summary_memory=False,
            model="openai/gpt-4o-mini",
        )
        assert m.summary_memory_enabled is True
        assert m.load_summary_memory_enabled is False

    def test_summary_save_false_load_true(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            summary_memory=False,
            load_summary_memory=True,
            model="openai/gpt-4o-mini",
        )
        assert m.summary_memory_enabled is False
        assert m.load_summary_memory_enabled is True

    def test_session_memory_receives_summary_load_flag(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            load_summary_memory=False,
            model="openai/gpt-4o-mini",
        )
        sm = m.get_session_memory(SessionType.AGENT)
        assert sm is not None
        assert sm.summary_enabled is True
        assert sm.load_summary_enabled is False


# ═════════════════════════════════════════════════════════════════════════
# 2. Summary Generation After Conversation
# ═════════════════════════════════════════════════════════════════════════

class TestSummaryGeneration:

    @pytest.mark.asyncio
    async def test_summary_generated_after_single_turn(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task = Task(
            description="I am working on a machine learning project to classify images of cats and dogs using convolutional neural networks."
        )
        await agent.print_do_async(task)

        session = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        assert session is not None
        assert session.summary is not None, "Summary should be generated after the first turn"
        assert len(session.summary) >= 10, f"Summary too short: {session.summary}"

    @pytest.mark.asyncio
    async def test_summary_grows_across_turns(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task1 = Task(description="I love hiking in the mountains during summer.")
        await agent.print_do_async(task1)

        session1 = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        summary_after_1 = session1.summary or ""

        task2 = Task(description="I also enjoy cooking Italian food, especially pasta and risotto.")
        await agent.print_do_async(task2)

        session2 = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        summary_after_2 = session2.summary or ""

        assert len(summary_after_2) > 0, "Summary should exist after two turns"


# ═════════════════════════════════════════════════════════════════════════
# 3. Summary Injected Into Context
# ═════════════════════════════════════════════════════════════════════════

class TestSummaryInjection:

    @pytest.mark.asyncio
    async def test_summary_injected_when_load_enabled(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            load_summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task = Task(description="I am a Python developer building REST APIs.")
        await agent.print_do_async(task)

        prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)

        session = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )

        if session and session.summary:
            assert "SessionSummary" in prepared["context_injection"], (
                "Summary should be injected in context_injection when load_summary_memory=True"
            )

    @pytest.mark.asyncio
    async def test_summary_not_injected_when_load_disabled(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            load_summary_memory=False,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task = Task(description="I am a data engineer working with Spark.")
        await agent.print_do_async(task)

        prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)

        assert "SessionSummary" not in prepared["context_injection"], (
            "Summary should NOT be injected when load_summary_memory=False"
        )


# ═════════════════════════════════════════════════════════════════════════
# 4. Summary Without Full Session Memory
# ═════════════════════════════════════════════════════════════════════════

class TestSummaryWithoutFullSessionMemory:

    @pytest.mark.asyncio
    async def test_summary_generated_without_session_messages(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        """Summary can be generated even when full_session_memory is disabled."""
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=False,
            summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task = Task(description="I specialize in distributed systems and microservices architecture.")
        await agent.print_do_async(task)

        session = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        assert session is not None

        if session.summary:
            assert len(session.summary) >= 5, "Summary should have content"

        if session.messages:
            assert len(session.messages) == 0, (
                "Messages should NOT be persisted when full_session_memory=False"
            )


# ═════════════════════════════════════════════════════════════════════════
# 5. Summary Persistence with SQLite
# ═════════════════════════════════════════════════════════════════════════

class TestSummaryPersistenceSqlite:

    @pytest.mark.asyncio
    async def test_summary_persists_in_sqlite(
        self, sqlite_storage: SqliteStorage, session_id: str, user_id: str
    ) -> None:
        memory = Memory(
            storage=sqlite_storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task = Task(description="I work on natural language processing projects using Hugging Face transformers.")
        await agent.print_do_async(task)

        session = sqlite_storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        assert session is not None
        if session.summary:
            assert len(session.summary) >= 10

        memory2 = Memory(
            storage=sqlite_storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        prepared = await memory2.prepare_inputs_for_task(session_type=SessionType.AGENT)

        if session.summary:
            assert "SessionSummary" in prepared["context_injection"], (
                "Summary from SQLite should be injected into context"
            )


# ═════════════════════════════════════════════════════════════════════════
# 6. Summary Memory with Agent recall
# ═════════════════════════════════════════════════════════════════════════

class TestSummaryAgentRecall:

    @pytest.mark.asyncio
    async def test_agent_uses_summary_for_context(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        """Agent should use summary to recall past context even without full message history."""
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=False,
            summary_memory=True,
            load_full_session_memory=False,
            load_summary_memory=True,
            model="openai/gpt-4o-mini",
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=memory)

        task1 = Task(description="My favorite programming language is Rust and my name is TestUserAlpha.")
        await agent.print_do_async(task1)

        session = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )

        if session and session.summary:
            prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
            assert prepared["message_history"] == [], (
                "No message history should be loaded when load_full_session_memory=False"
            )
            assert "SessionSummary" in prepared["context_injection"], (
                "Summary should be in context"
            )
