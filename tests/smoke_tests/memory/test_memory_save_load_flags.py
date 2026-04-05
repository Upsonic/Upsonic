"""
Save vs Load Flag Separation Smoke Tests

Verifies the separation of save flags (persist to storage) and load flags
(inject into runs) across all memory types:

1. Flag defaults and backward compatibility
2. Session memory: save=True, load=False → messages saved but not injected
3. Session memory: save=False, load=True → nothing to inject, graceful empty
4. Summary memory: independent from session memory save flag
5. User analysis memory: save vs load independence
6. Chat and AutonomousAgent default flags
7. prepare_inputs_for_task respects load flags
"""

import asyncio
import os
import tempfile
import uuid
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from upsonic import Agent, Task
from upsonic.storage import Memory, InMemoryStorage
from upsonic.session.base import SessionType

pytestmark = pytest.mark.timeout(180)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


# ═════════════════════════════════════════════════════════════════════════
# 1. Flag Defaults and Backward Compatibility
# ═════════════════════════════════════════════════════════════════════════

class TestFlagDefaults:

    def test_load_flags_default_to_save_flags(self, storage: InMemoryStorage) -> None:
        """When load flags are not provided, they mirror save flags."""
        m = Memory(
            storage=storage,
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            model="openai/gpt-4o-mini",
        )
        assert m.load_full_session_memory_enabled is True
        assert m.load_summary_memory_enabled is True
        assert m.load_user_analysis_memory_enabled is True

    def test_load_flags_default_false_when_save_false(self, storage: InMemoryStorage) -> None:
        m = Memory(storage=storage)
        assert m.full_session_memory_enabled is False
        assert m.load_full_session_memory_enabled is False
        assert m.summary_memory_enabled is False
        assert m.load_summary_memory_enabled is False
        assert m.user_analysis_memory_enabled is False
        assert m.load_user_analysis_memory_enabled is False

    def test_load_flags_override_save_flags(self, storage: InMemoryStorage) -> None:
        """Load flags can be set independently from save flags."""
        m = Memory(
            storage=storage,
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            load_full_session_memory=False,
            load_summary_memory=False,
            load_user_analysis_memory=False,
            model="openai/gpt-4o-mini",
        )
        assert m.full_session_memory_enabled is True
        assert m.load_full_session_memory_enabled is False
        assert m.summary_memory_enabled is True
        assert m.load_summary_memory_enabled is False
        assert m.user_analysis_memory_enabled is True
        assert m.load_user_analysis_memory_enabled is False

    def test_save_false_load_true(self, storage: InMemoryStorage) -> None:
        """Load flags can enable loading even when save is disabled."""
        m = Memory(
            storage=storage,
            full_session_memory=False,
            summary_memory=False,
            user_analysis_memory=False,
            load_full_session_memory=True,
            load_summary_memory=True,
            load_user_analysis_memory=True,
            model="openai/gpt-4o-mini",
        )
        assert m.full_session_memory_enabled is False
        assert m.load_full_session_memory_enabled is True
        assert m.user_analysis_memory_enabled is False
        assert m.load_user_analysis_memory_enabled is True


# ═════════════════════════════════════════════════════════════════════════
# 2. Session Memory Factory Passes Load Flags
# ═════════════════════════════════════════════════════════════════════════

class TestSessionMemoryLoadFlags:

    def test_session_memory_receives_load_flags(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            load_full_session_memory=False,
            load_summary_memory=True,
        )
        sm = m.get_session_memory(SessionType.AGENT)
        assert sm is not None
        assert sm.enabled is True
        assert sm.load_enabled is False
        assert sm.summary_enabled is True
        assert sm.load_summary_enabled is True

    def test_session_memory_load_flags_default_to_save(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=False,
        )
        sm = m.get_session_memory(SessionType.AGENT)
        assert sm.load_enabled is True
        assert sm.load_summary_enabled is False


# ═════════════════════════════════════════════════════════════════════════
# 3. User Memory Load Flag
# ═════════════════════════════════════════════════════════════════════════

class TestUserMemoryLoadFlag:

    def test_user_memory_created_when_load_enabled(
        self, storage: InMemoryStorage
    ) -> None:
        """User memory instance is created when load is enabled even if save is disabled."""
        m = Memory(
            storage=storage,
            user_analysis_memory=False,
            load_user_analysis_memory=True,
            model="openai/gpt-4o-mini",
        )
        assert m._user_memory is not None
        assert m._user_memory.enabled is False
        assert m._user_memory.load_enabled is True

    def test_user_memory_not_created_when_both_disabled(
        self, storage: InMemoryStorage
    ) -> None:
        m = Memory(
            storage=storage,
            user_analysis_memory=False,
            load_user_analysis_memory=False,
        )
        assert m._user_memory is None

    def test_user_memory_save_enabled_load_disabled(
        self, storage: InMemoryStorage
    ) -> None:
        m = Memory(
            storage=storage,
            user_analysis_memory=True,
            load_user_analysis_memory=False,
            model="openai/gpt-4o-mini",
        )
        assert m._user_memory is not None
        assert m._user_memory.enabled is True
        assert m._user_memory.load_enabled is False


# ═════════════════════════════════════════════════════════════════════════
# 4. prepare_inputs_for_task Respects Load Flags
# ═════════════════════════════════════════════════════════════════════════

class TestPrepareInputsLoadFlags:

    @pytest.mark.asyncio
    async def test_load_disabled_returns_empty(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        """When all load flags are disabled, prepare_inputs returns empty data."""
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            load_full_session_memory=False,
            load_summary_memory=False,
            load_user_analysis_memory=False,
            model="openai/gpt-4o-mini",
        )

        prepared = await m.prepare_inputs_for_task(session_type=SessionType.AGENT)

        assert prepared["message_history"] == []
        assert prepared["context_injection"] == ""
        assert prepared["system_prompt_injection"] == ""

    @pytest.mark.asyncio
    async def test_load_enabled_save_disabled_graceful(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        """When load is enabled but nothing was saved, returns empty gracefully."""
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=False,
            load_full_session_memory=True,
            load_summary_memory=True,
            load_user_analysis_memory=True,
            model="openai/gpt-4o-mini",
        )

        prepared = await m.prepare_inputs_for_task(session_type=SessionType.AGENT)

        assert prepared["message_history"] == []
        assert prepared["context_injection"] == ""


# ═════════════════════════════════════════════════════════════════════════
# 5. Save=True, Load=False → Messages Saved but NOT Injected
# ═════════════════════════════════════════════════════════════════════════

class TestSaveWithoutLoad:

    @pytest.mark.asyncio
    async def test_messages_saved_but_not_loaded(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        """Messages are persisted when save=True, but not injected when load=False."""
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=True,
            load_full_session_memory=False,
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=m)
        await agent.do_async(Task("My name is TestUser"))

        session = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        assert session is not None
        assert session.messages is not None
        assert len(session.messages) >= 2, "Messages should be saved to storage"

        prepared = await m.prepare_inputs_for_task(session_type=SessionType.AGENT)
        assert prepared["message_history"] == [], "Messages should NOT be injected into runs"


# ═════════════════════════════════════════════════════════════════════════
# 6. Save=False → Messages NOT Saved
# ═════════════════════════════════════════════════════════════════════════

class TestSaveDisabledNoMessages:

    @pytest.mark.asyncio
    async def test_no_messages_persisted_when_save_disabled(
        self, storage: InMemoryStorage, session_id: str, user_id: str
    ) -> None:
        """When full_session_memory=False, messages should NOT be persisted."""
        m = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=False,
            summary_memory=False,
            load_full_session_memory=False,
        )

        agent = Agent(model="openai/gpt-4o-mini", memory=m)
        await agent.do_async(Task("My name is TestUser"))

        session = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        if session and session.messages:
            assert len(session.messages) == 0, (
                f"Messages should NOT be saved when full_session_memory=False, "
                f"got {len(session.messages)}"
            )


# ═════════════════════════════════════════════════════════════════════════
# 7. Chat Class Default Flags
# ═════════════════════════════════════════════════════════════════════════

class TestChatDefaultFlags:

    def test_chat_defaults(self) -> None:
        from upsonic import Chat

        agent = Agent(model="openai/gpt-4o-mini", name="TestAgent")
        chat = Chat(
            session_id="chat_test",
            user_id="user_test",
            agent=agent,
        )

        assert chat._memory.full_session_memory_enabled is True
        assert chat._memory.load_full_session_memory_enabled is True
        assert chat._memory.summary_memory_enabled is False
        assert chat._memory.load_summary_memory_enabled is False
        assert chat._memory.user_analysis_memory_enabled is False
        assert chat._memory.load_user_analysis_memory_enabled is False

    def test_chat_with_custom_load_flags(self) -> None:
        from upsonic import Chat

        agent = Agent(model="openai/gpt-4o-mini", name="TestAgent")
        chat = Chat(
            session_id="chat_test_2",
            user_id="user_test_2",
            agent=agent,
            full_session_memory=True,
            summary_memory=True,
            load_full_session_memory=False,
            load_summary_memory=True,
        )

        assert chat._memory.full_session_memory_enabled is True
        assert chat._memory.load_full_session_memory_enabled is False
        assert chat._memory.summary_memory_enabled is True
        assert chat._memory.load_summary_memory_enabled is True


# ═════════════════════════════════════════════════════════════════════════
# 8. AutonomousAgent Default Flags
# ═════════════════════════════════════════════════════════════════════════

class TestAutonomousAgentDefaultFlags:

    def test_autonomous_agent_defaults(self) -> None:
        from upsonic import AutonomousAgent

        agent = AutonomousAgent(model="openai/gpt-4o-mini")

        assert agent.memory is not None
        memory: Memory = agent.memory
        assert memory.full_session_memory_enabled is True
        assert memory.load_full_session_memory_enabled is True
        assert memory.summary_memory_enabled is False
        assert memory.load_summary_memory_enabled is False
        assert memory.user_analysis_memory_enabled is False
        assert memory.load_user_analysis_memory_enabled is False


# ═════════════════════════════════════════════════════════════════════════
# 9. Database Classes Accept Load Flags
# ═════════════════════════════════════════════════════════════════════════

class TestDatabaseLoadFlags:

    def test_inmemory_database_load_flags(self) -> None:
        from upsonic.db.database import InMemoryDatabase

        db = InMemoryDatabase(
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=False,
            load_full_session_memory=False,
            load_summary_memory=True,
            load_user_analysis_memory=False,
        )

        assert db.memory.full_session_memory_enabled is True
        assert db.memory.load_full_session_memory_enabled is False
        assert db.memory.summary_memory_enabled is True
        assert db.memory.load_summary_memory_enabled is True

    def test_inmemory_database_backward_compat(self) -> None:
        from upsonic.db.database import InMemoryDatabase

        db = InMemoryDatabase(
            full_session_memory=True,
            summary_memory=True,
        )

        assert db.memory.load_full_session_memory_enabled is True
        assert db.memory.load_summary_memory_enabled is True
        assert db.memory.load_user_analysis_memory_enabled is False
