"""Phase 4 persistence: round-trip + cross-process simulation across the
three Phase-4b backends (InMemory, JSON, SQLite)."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upsonic import Agent
from upsonic.chat.chat import Chat
from upsonic.models import ModelResponse, TextPart
from upsonic.storage.in_memory.in_memory import InMemoryStorage
from upsonic.storage.json.json import JSONStorage

try:
    from upsonic.storage.sqlite.sqlite import SqliteStorage  # type: ignore
    _SQLITE_AVAILABLE = True
except ImportError:
    SqliteStorage = None  # type: ignore
    _SQLITE_AVAILABLE = False

try:
    import fakeredis  # type: ignore
    from upsonic.storage.redis.redis import RedisStorage  # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:
    fakeredis = None  # type: ignore
    RedisStorage = None  # type: ignore
    _REDIS_AVAILABLE = False
from upsonic.usage import RequestUsage
from upsonic.usage_registry import (
    UsageEntry,
    UsageRegistry,
    get_default_registry,
)


def _entry(**kw) -> UsageEntry:
    kw.setdefault("input_tokens", 10)
    kw.setdefault("output_tokens", 20)
    kw.setdefault("model", "openai/gpt-4o")
    kw.setdefault("pipeline_step", "model_call")
    return UsageEntry(**kw)


class _BackendRoundTripMixin:
    """Subclass overrides ``_make_storage`` to point at a real backend."""

    def _make_storage(self):
        raise NotImplementedError

    def _cleanup(self, storage):
        pass

    def test_upsert_then_query(self):
        storage = self._make_storage()
        try:
            e1 = _entry(
                chat_usage_id="C1", agent_usage_id="A1",
                input_tokens=100, output_tokens=50,
                cost_usd=0.01,
            )
            e2 = _entry(
                chat_usage_id="C1", agent_usage_id="A1",
                input_tokens=5, output_tokens=3,
                cost_usd=0.001,
            )
            e3 = _entry(chat_usage_id="C2", input_tokens=999, output_tokens=999)
            storage.upsert_usage_entry(e1.to_dict())
            storage.upsert_usage_entry(e2.to_dict())
            storage.upsert_usage_entry(e3.to_dict())

            rows = storage.query_usage_entries(chat_usage_id="C1")
            self.assertEqual(len(rows), 2)
            tokens = sum(r["input_tokens"] for r in rows)
            self.assertEqual(tokens, 105)

            rows = storage.query_usage_entries(chat_usage_id="C2")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["input_tokens"], 999)
        finally:
            self._cleanup(storage)

    def test_upsert_is_idempotent(self):
        storage = self._make_storage()
        try:
            e = _entry(entry_id="fixed", input_tokens=10)
            storage.upsert_usage_entry(e.to_dict())
            e2 = _entry(entry_id="fixed", input_tokens=777)
            storage.upsert_usage_entry(e2.to_dict())

            rows = storage.query_usage_entries()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["input_tokens"], 777)
        finally:
            self._cleanup(storage)

    def test_delete_by_scope(self):
        storage = self._make_storage()
        try:
            storage.upsert_usage_entry(_entry(chat_usage_id="X").to_dict())
            storage.upsert_usage_entry(_entry(chat_usage_id="X").to_dict())
            storage.upsert_usage_entry(_entry(chat_usage_id="Y").to_dict())

            removed = storage.delete_usage_entries(chat_usage_id="X")
            self.assertEqual(removed, 2)

            rows = storage.query_usage_entries()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["chat_usage_id"], "Y")
        finally:
            self._cleanup(storage)


class TestInMemoryBackend(_BackendRoundTripMixin, unittest.TestCase):
    def _make_storage(self):
        return InMemoryStorage()


class TestJSONBackend(_BackendRoundTripMixin, unittest.TestCase):
    def _make_storage(self):
        self._tmp = tempfile.mkdtemp(prefix="upsonic-json-")
        return JSONStorage(db_path=self._tmp)

    def _cleanup(self, storage):
        shutil.rmtree(self._tmp, ignore_errors=True)


@unittest.skipUnless(_SQLITE_AVAILABLE, "sqlalchemy not installed")
class TestSQLiteBackend(_BackendRoundTripMixin, unittest.TestCase):
    def _make_storage(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        s = SqliteStorage(db_file=self._tmp.name)
        s._create_all_tables()
        return s

    def _cleanup(self, storage):
        try:
            storage.close()
        except Exception:
            pass
        try:
            os.unlink(self._tmp.name)
        except Exception:
            pass


@unittest.skipUnless(_REDIS_AVAILABLE, "fakeredis not installed")
class TestRedisBackend(_BackendRoundTripMixin, unittest.TestCase):
    def _make_storage(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        return RedisStorage(redis_client=client, db_prefix="upsonic-test")

    def _cleanup(self, storage):
        try:
            storage.redis_client.flushall()
        except Exception:
            pass


class TestRegistryStorageAttach(unittest.TestCase):
    """UsageRegistry.record() should also write-through, and
    load_from_storage() should rehydrate on a fresh registry."""

    def test_record_writes_through_to_storage(self):
        storage = InMemoryStorage()
        reg = UsageRegistry(storage=storage)
        reg.record(_entry(chat_usage_id="C1", input_tokens=10))

        rows = storage.query_usage_entries(chat_usage_id="C1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_tokens"], 10)

    def test_load_from_storage_rehydrates(self):
        storage = InMemoryStorage()
        # Simulate a prior process having written entries.
        for i in range(3):
            storage.upsert_usage_entry(_entry(
                chat_usage_id="C1",
                input_tokens=i * 10,
            ).to_dict())

        reg = UsageRegistry()    # fresh, empty
        n = reg.load_from_storage(storage, chat_usage_id="C1")
        self.assertEqual(n, 3)
        self.assertEqual(reg.by_chat("C1").input_tokens, 0 + 10 + 20)

    def test_flush_to_storage_backfills(self):
        reg = UsageRegistry()
        reg.record(_entry(chat_usage_id="C1", input_tokens=11))
        reg.record(_entry(chat_usage_id="C1", input_tokens=22))

        storage = InMemoryStorage()
        flushed = reg.flush_to_storage(storage)
        self.assertEqual(flushed, 2)
        self.assertEqual(len(storage.query_usage_entries(chat_usage_id="C1")), 2)


def _response(input_tokens=70, output_tokens=30):
    return ModelResponse(
        parts=[TextPart(content="ok")],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        provider_name="test-provider",
        provider_response_id="r-1",
        provider_details={},
        finish_reason="stop",
    )


class TestChatPersistsAndResumes(unittest.TestCase):
    """Open Chat → invoke → close. Open a fresh Chat with the SAME
    chat_usage_id and storage. Historical spend must reappear.

    Uses JSONStorage instead of SQLite so the test runs without the
    optional sqlalchemy dep.
    """

    def setUp(self):
        get_default_registry().clear()
        self._tmp = tempfile.mkdtemp(prefix="upsonic-chat-resume-")
        self.storage = JSONStorage(db_path=self._tmp)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_resume_picks_up_prior_spend(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response(70, 30))

        agent = Agent(name="A", model=mock_model)
        chat = Chat(
            session_id="s", user_id="u",
            agent=agent, storage=self.storage,
            chat_usage_id="durable-chat",
        )
        asyncio.run(chat.invoke("hi"))

        # Simulate process restart: clear in-memory registry, build a new
        # Chat with the SAME chat_usage_id + storage.
        get_default_registry().clear()

        agent2 = Agent(name="A", model=mock_model)
        chat2 = Chat(
            session_id="s", user_id="u",
            agent=agent2, storage=self.storage,
            chat_usage_id="durable-chat",
        )

        # The registry was rehydrated on open — prior tokens are visible.
        self.assertEqual(chat2.usage.input_tokens, 70)
        self.assertEqual(chat2.usage.output_tokens, 30)
        self.assertEqual(chat2.usage.requests, 1)


if __name__ == "__main__":
    unittest.main()
