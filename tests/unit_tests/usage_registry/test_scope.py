"""Unit tests for the scope-tag contextvar plumbing."""
from __future__ import annotations

import asyncio
import unittest

from upsonic.usage_registry import current_scope_tags, scope
from upsonic.usage_registry.scope import (
    _agent_usage_id,
    _chat_usage_id,
    _task_usage_id,
    _team_usage_id,
)


class TestScopeContextManager(unittest.TestCase):
    def test_defaults_all_none(self):
        # Sanity: outside any scope() block, every tag is None.
        # (Other tests share process state, so set defaults explicitly first.)
        for var in (_chat_usage_id, _agent_usage_id, _task_usage_id, _team_usage_id):
            t = var.set(None)
            var.reset(t)
        tags = current_scope_tags()
        self.assertIsNone(tags["chat_usage_id"])
        self.assertIsNone(tags["agent_usage_id"])
        self.assertIsNone(tags["task_usage_id"])
        self.assertIsNone(tags["team_usage_id"])

    def test_scope_sets_only_provided_keys(self):
        with scope(chat_usage_id="c1", agent_usage_id="a1"):
            tags = current_scope_tags()
            self.assertEqual(tags["chat_usage_id"], "c1")
            self.assertEqual(tags["agent_usage_id"], "a1")
            self.assertIsNone(tags["task_usage_id"])
            self.assertIsNone(tags["team_usage_id"])

    def test_scope_restores_on_exit(self):
        with scope(chat_usage_id="outer"):
            self.assertEqual(current_scope_tags()["chat_usage_id"], "outer")
            with scope(chat_usage_id="inner"):
                self.assertEqual(current_scope_tags()["chat_usage_id"], "inner")
            self.assertEqual(current_scope_tags()["chat_usage_id"], "outer")
        self.assertIsNone(current_scope_tags()["chat_usage_id"])

    def test_scope_restores_on_exception(self):
        try:
            with scope(chat_usage_id="c1"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIsNone(current_scope_tags()["chat_usage_id"])

    def test_scope_none_value_does_not_clear_existing(self):
        with scope(chat_usage_id="c1"):
            with scope(agent_usage_id="a1"):   # chat_usage_id NOT passed -> kept
                tags = current_scope_tags()
                self.assertEqual(tags["chat_usage_id"], "c1")
                self.assertEqual(tags["agent_usage_id"], "a1")
            self.assertEqual(current_scope_tags()["chat_usage_id"], "c1")
            self.assertIsNone(current_scope_tags()["agent_usage_id"])


class TestScopeAcrossAsyncio(unittest.TestCase):
    def test_inheritance_into_child_task(self):
        """A child asyncio task inherits the parent's scope via copy_context."""
        captured: dict = {}

        async def child():
            captured.update(current_scope_tags())

        async def parent():
            with scope(chat_usage_id="C", agent_usage_id="A", task_usage_id="T"):
                await asyncio.create_task(child())

        asyncio.run(parent())

        self.assertEqual(captured["chat_usage_id"], "C")
        self.assertEqual(captured["agent_usage_id"], "A")
        self.assertEqual(captured["task_usage_id"], "T")

    def test_child_override_does_not_leak_to_parent(self):
        """When the child opens its own scope(), the parent context is unchanged."""
        results: list = []

        async def child():
            with scope(agent_usage_id="A-child"):
                results.append(current_scope_tags()["agent_usage_id"])

        async def parent():
            with scope(agent_usage_id="A-parent"):
                await asyncio.create_task(child())
                results.append(current_scope_tags()["agent_usage_id"])

        asyncio.run(parent())
        self.assertEqual(results, ["A-child", "A-parent"])


if __name__ == "__main__":
    unittest.main()
