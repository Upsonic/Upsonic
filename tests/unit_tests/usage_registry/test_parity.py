"""Parity tests: registry totals must equal task.usage / agent.usage / chat.usage.total_tokens.

These run against mocked models so they execute fast and don't need an
API key. The point is to verify the Phase-2 write-through paths line up
with the legacy ``incr``-chain numbers."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upsonic import Agent, Task
from upsonic.chat.chat import Chat
from upsonic.models import ModelResponse, TextPart
from upsonic.storage.in_memory.in_memory import InMemoryStorage
from upsonic.usage import RequestUsage
from upsonic.usage_registry import get_default_registry


def _response_with_usage(text: str = "ok", input_tokens: int = 100, output_tokens: int = 42):
    return ModelResponse(
        parts=[TextPart(content=text)],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        provider_name="test-provider",
        provider_response_id="r-1",
        provider_details={},
        finish_reason="stop",
    )


class TestAgentDoParity(unittest.TestCase):
    def setUp(self):
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_single_call_parity(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response_with_usage(
            input_tokens=100, output_tokens=50,
        ))

        agent = Agent(name="ParityAgent", model=mock_model)
        task = Task("Hi")
        agent.do(task)

        reg = get_default_registry()
        agg_task = reg.by_task(task.task_usage_id)
        agg_agent = reg.by_agent(agent.agent_usage_id)

        # Task-scoped roll-up must equal the legacy TaskUsage tokens.
        self.assertEqual(agg_task.input_tokens, task.usage.input_tokens)
        self.assertEqual(agg_task.output_tokens, task.usage.output_tokens)
        self.assertGreater(agg_task.entry_count, 0)

        # Agent-scoped roll-up must equal AgentUsage tokens.
        self.assertEqual(agg_agent.input_tokens, agent.usage.input_tokens)
        self.assertEqual(agg_agent.output_tokens, agent.usage.output_tokens)

    @patch("upsonic.models.infer_model")
    def test_two_tasks_accumulate_on_agent(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response_with_usage(
            input_tokens=80, output_tokens=20,
        ))

        agent = Agent(name="AgentX", model=mock_model)
        t1 = Task("a")
        t2 = Task("b")
        agent.do(t1)
        agent.do(t2)

        reg = get_default_registry()
        agg_agent = reg.by_agent(agent.agent_usage_id)
        agg_t1 = reg.by_task(t1.task_usage_id)
        agg_t2 = reg.by_task(t2.task_usage_id)

        # Each task scoped to itself.
        self.assertEqual(agg_t1.input_tokens, t1.usage.input_tokens)
        self.assertEqual(agg_t2.input_tokens, t2.usage.input_tokens)

        # Per-task entries do not bleed across.
        self.assertNotEqual(t1.task_usage_id, t2.task_usage_id)
        self.assertEqual(agg_t1.entry_count + agg_t2.entry_count, agg_agent.entry_count)

        # Agent-level total equals AgentUsage.
        self.assertEqual(agg_agent.input_tokens, agent.usage.input_tokens)
        self.assertEqual(agg_agent.output_tokens, agent.usage.output_tokens)


class TestChatInvokeParity(unittest.TestCase):
    def setUp(self):
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_chat_invoke_records_under_chat_scope(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response_with_usage(
            input_tokens=70, output_tokens=30,
        ))

        agent = Agent(name="ChatAgent", model=mock_model)
        chat = Chat(
            session_id="s", user_id="u", agent=agent,
            storage=InMemoryStorage(),
        )

        asyncio.run(chat.invoke("hi"))
        asyncio.run(chat.invoke("hi again"))

        reg = get_default_registry()
        agg_chat = reg.by_chat(chat.chat_usage_id)
        agg_agent = reg.by_agent(agent.agent_usage_id)

        # Chat-scope aggregates both invocations.
        self.assertEqual(agg_chat.entry_count, agg_agent.entry_count)
        self.assertGreater(agg_chat.input_tokens, 0)
        self.assertEqual(agg_chat.input_tokens, agent.usage.input_tokens)
        self.assertEqual(agg_chat.output_tokens, agent.usage.output_tokens)


class TestScopeIsolation(unittest.TestCase):
    def setUp(self):
        get_default_registry().clear()

    @patch("upsonic.models.infer_model")
    def test_two_agents_two_chats_no_crossover(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_response_with_usage(
            input_tokens=50, output_tokens=25,
        ))

        a1 = Agent(name="A1", model=mock_model)
        a2 = Agent(name="A2", model=mock_model)
        chat1 = Chat(session_id="s1", user_id="u1", agent=a1, storage=InMemoryStorage())
        chat2 = Chat(session_id="s2", user_id="u2", agent=a2, storage=InMemoryStorage())

        asyncio.run(chat1.invoke("hi1"))
        asyncio.run(chat2.invoke("hi2"))

        reg = get_default_registry()
        agg1 = reg.by_chat(chat1.chat_usage_id)
        agg2 = reg.by_chat(chat2.chat_usage_id)

        # Each chat sees only its own spend.
        self.assertEqual(agg1.entry_count, 1)
        self.assertEqual(agg2.entry_count, 1)
        self.assertNotEqual(chat1.chat_usage_id, chat2.chat_usage_id)


if __name__ == "__main__":
    unittest.main()
