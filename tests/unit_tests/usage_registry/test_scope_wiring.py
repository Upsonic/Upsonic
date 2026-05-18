"""End-to-end checks that Agent / Chat / Team push the right scope tags
during their public entry points.

These tests use mocked models so no API key is needed. They probe by
mounting a "spy" hook: a tool function that snapshots
:func:`current_scope_tags` at call time, and a custom step that does
the same inside the pipeline. Phase 1c only needs to verify the tags
ARE active by the time the model layer would run — Phase 2 wires the
actual ledger write at that point.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upsonic import Agent, Task
from upsonic.chat.chat import Chat
from upsonic.models import ModelResponse, TextPart
from upsonic.storage.in_memory.in_memory import InMemoryStorage
from upsonic.team.team import Team
from upsonic.usage_registry import current_scope_tags


def _mock_text_response(text: str = "ok") -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        model_name="test-model",
        timestamp="2024-01-01T00:00:00Z",
        usage=None,
        provider_name="test-provider",
        provider_response_id="test-id",
        provider_details={},
        finish_reason="stop",
    )


class _ScopeProbeRequest(AsyncMock):
    """An AsyncMock for ``model.request`` that snapshots scope tags every call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots: list = []
        self.return_value = _mock_text_response("ok")

    async def __call__(self, *args, **kwargs):
        self.snapshots.append(current_scope_tags())
        return await super().__call__(*args, **kwargs)


class TestAgentScopePush(unittest.TestCase):
    @patch("upsonic.models.infer_model")
    def test_agent_do_pushes_agent_and_task_scope(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        probe = _ScopeProbeRequest()
        mock_model.request = probe

        agent = Agent(name="Probe", model=mock_model)
        task = Task("Hello")
        agent.do(task)

        # At least one snapshot — model.request runs at least once per turn.
        self.assertGreater(len(probe.snapshots), 0)
        snap = probe.snapshots[0]
        self.assertEqual(snap["agent_usage_id"], agent.agent_usage_id)
        self.assertEqual(snap["task_usage_id"], task.task_usage_id)
        # Chat / team not active in plain do().
        self.assertIsNone(snap["chat_usage_id"])
        self.assertIsNone(snap["team_usage_id"])

    @patch("upsonic.models.infer_model")
    def test_agent_scope_cleared_after_do(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_mock_text_response("ok"))

        agent = Agent(name="Probe", model=mock_model)
        task = Task("Hello")
        agent.do(task)

        # After do() returns, agent / task scope should be torn down.
        tags = current_scope_tags()
        self.assertIsNone(tags["agent_usage_id"])
        self.assertIsNone(tags["task_usage_id"])


class TestChatScopePush(unittest.TestCase):
    @patch("upsonic.models.infer_model")
    def test_chat_invoke_pushes_chat_scope(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        probe = _ScopeProbeRequest()
        mock_model.request = probe

        agent = Agent(name="ChatAgent", model=mock_model)
        chat = Chat(session_id="s", user_id="u", agent=agent, storage=InMemoryStorage())

        asyncio.run(chat.invoke("Hi"))

        self.assertGreater(len(probe.snapshots), 0)
        snap = probe.snapshots[0]
        self.assertEqual(snap["chat_usage_id"], chat.chat_usage_id)
        self.assertEqual(snap["agent_usage_id"], agent.agent_usage_id)
        # task is created inside invoke, so it should be set but unpredictable here.
        self.assertIsNotNone(snap["task_usage_id"])
        self.assertEqual(snap["user_id"], "u")

    @patch("upsonic.models.infer_model")
    def test_chat_scope_cleared_after_invoke(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        mock_model.request = AsyncMock(return_value=_mock_text_response("ok"))

        agent = Agent(name="ChatAgent", model=mock_model)
        chat = Chat(session_id="s", user_id="u", agent=agent, storage=InMemoryStorage())

        asyncio.run(chat.invoke("Hi"))

        tags = current_scope_tags()
        self.assertIsNone(tags["chat_usage_id"])
        self.assertIsNone(tags["agent_usage_id"])
        self.assertIsNone(tags["task_usage_id"])


class TestTeamScopePush(unittest.TestCase):
    @patch("upsonic.models.infer_model")
    def test_team_do_pushes_team_scope(self, mock_infer_model):
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        probe = _ScopeProbeRequest()
        mock_model.request = probe

        member = Agent(name="Member", model=mock_model)
        team = Team(entities=[member])

        asyncio.run(team.do_async("Hi"))

        self.assertGreater(len(probe.snapshots), 0)
        snap = probe.snapshots[0]
        self.assertEqual(snap["team_usage_id"], team.team_usage_id)
        # Team mode "sequential" spins up its own internal coordinator
        # agent; what we need to assert is just that AN agent scope is
        # active alongside the team scope.
        self.assertIsNotNone(snap["agent_usage_id"])


if __name__ == "__main__":
    unittest.main()
