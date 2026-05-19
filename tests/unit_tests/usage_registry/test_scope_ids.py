"""Unit tests for the scope-id surfaces on Task / Agent / Chat / Team.

Uses ``MagicMock``-backed models everywhere so the tests don't need an
``OPENAI_API_KEY`` in CI."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Agent
from upsonic.chat.chat import Chat
from upsonic.team.team import Team
from upsonic.storage.in_memory.in_memory import InMemoryStorage


def _mock_agent(**kw) -> Agent:
    """Build an Agent with a MagicMock model so no real provider client
    (and therefore no API key) is needed at construction time."""
    return Agent(model=MagicMock(model_name="mock-model"), **kw)


class TestTaskUsageId(unittest.TestCase):
    def test_task_usage_id_auto_generated(self):
        t = Task("hello")
        self.assertTrue(t.task_usage_id.startswith("task-"))

    def test_task_usage_id_distinct_from_task_id(self):
        t = Task("hello")
        self.assertNotEqual(t.task_usage_id, t.task_id)

    def test_task_usage_id_stable_across_reads(self):
        t = Task("hello")
        first = t.task_usage_id
        self.assertEqual(t.task_usage_id, first)
        self.assertEqual(t.task_usage_id, first)

    def test_explicit_task_usage_id_honored(self):
        t = Task("hello", task_usage_id_="custom-123")
        self.assertEqual(t.task_usage_id, "custom-123")

    def test_task_usage_id_distinct_per_task(self):
        ids = {Task("x").task_usage_id for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_task_usage_id_serialized(self):
        t = Task("hello", task_usage_id_="serialized-id")
        d = t.to_dict()
        self.assertEqual(d["task_usage_id_"], "serialized-id")


class TestAgentUsageId(unittest.TestCase):
    def test_agent_usage_id_auto_generated(self):
        a = _mock_agent()
        self.assertTrue(a.agent_usage_id.startswith("agent-"))

    def test_agent_usage_id_stable_across_reads(self):
        a = _mock_agent()
        first = a.agent_usage_id
        self.assertEqual(a.agent_usage_id, first)

    def test_explicit_agent_usage_id_honored(self):
        a = _mock_agent(agent_usage_id="my-id")
        self.assertEqual(a.agent_usage_id, "my-id")

    def test_agent_usage_id_distinct_from_agent_id(self):
        a = _mock_agent()
        self.assertNotEqual(a.agent_usage_id, a.agent_id)


class TestChatUsageId(unittest.TestCase):
    def _make(self, **kw):
        return Chat(
            session_id="s",
            user_id="u",
            agent=_mock_agent(),
            storage=InMemoryStorage(),
            **kw,
        )

    def test_chat_usage_id_auto_generated(self):
        c = self._make()
        self.assertTrue(c.chat_usage_id.startswith("chat-"))

    def test_chat_usage_id_not_alias_of_session_id(self):
        c = self._make()
        self.assertNotEqual(c.chat_usage_id, c.session_id)

    def test_explicit_chat_usage_id_honored(self):
        c = self._make(chat_usage_id="my-chat-id")
        self.assertEqual(c.chat_usage_id, "my-chat-id")


class TestTeamIds(unittest.TestCase):
    def _make(self, **kw):
        return Team(entities=[_mock_agent()], **kw)

    def test_team_id_auto_generated(self):
        t = self._make()
        self.assertTrue(t.team_id.startswith("team-"))

    def test_team_usage_id_auto_generated(self):
        t = self._make()
        self.assertTrue(t.team_usage_id.startswith("team-"))

    def test_team_id_and_team_usage_id_are_independent(self):
        t = self._make()
        self.assertNotEqual(t.team_id, t.team_usage_id)

    def test_explicit_ids_honored(self):
        t = self._make(team_id="t1", team_usage_id="tu1")
        self.assertEqual(t.team_id, "t1")
        self.assertEqual(t.team_usage_id, "tu1")


if __name__ == "__main__":
    unittest.main()
