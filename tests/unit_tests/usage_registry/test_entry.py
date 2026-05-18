"""Unit tests for UsageEntry — defaults, totals, serialization."""
from __future__ import annotations

import unittest

from upsonic.usage_registry import UsageEntry


class TestUsageEntry(unittest.TestCase):
    def test_defaults_are_zero_or_none(self):
        e = UsageEntry()
        self.assertEqual(e.input_tokens, 0)
        self.assertEqual(e.output_tokens, 0)
        self.assertEqual(e.requests, 0)
        self.assertEqual(e.tool_calls, 0)
        self.assertEqual(e.duration, 0.0)
        self.assertIsNone(e.cost_usd)
        self.assertIsNone(e.time_to_first_token)
        self.assertEqual(e.kind, "llm")
        self.assertEqual(e.extra, {})

    def test_entry_id_is_unique_per_default(self):
        a = UsageEntry()
        b = UsageEntry()
        self.assertNotEqual(a.entry_id, b.entry_id)
        self.assertTrue(a.entry_id.startswith("entry-"))

    def test_total_tokens_is_sum(self):
        e = UsageEntry(input_tokens=100, output_tokens=42)
        self.assertEqual(e.total_tokens, 142)

    def test_scope_tags_default_none(self):
        e = UsageEntry()
        self.assertIsNone(e.chat_usage_id)
        self.assertIsNone(e.agent_usage_id)
        self.assertIsNone(e.task_usage_id)
        self.assertIsNone(e.team_usage_id)
        self.assertIsNone(e.workflow_usage_id)
        self.assertIsNone(e.system_usage_id)

    def test_round_trip_to_from_dict_preserves_fields(self):
        e = UsageEntry(
            kind="llm",
            model="openai/gpt-4o",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.0042,
            chat_usage_id="chat-1",
            agent_usage_id="agent-1",
            task_usage_id="task-1",
            extra={"k": "v"},
        )
        d = e.to_dict()
        roundtrip = UsageEntry.from_dict(d)
        self.assertEqual(roundtrip.entry_id, e.entry_id)
        self.assertEqual(roundtrip.model, "openai/gpt-4o")
        self.assertEqual(roundtrip.input_tokens, 10)
        self.assertEqual(roundtrip.output_tokens, 20)
        self.assertEqual(roundtrip.cost_usd, 0.0042)
        self.assertEqual(roundtrip.chat_usage_id, "chat-1")
        self.assertEqual(roundtrip.agent_usage_id, "agent-1")
        self.assertEqual(roundtrip.task_usage_id, "task-1")
        self.assertEqual(roundtrip.extra, {"k": "v"})

    def test_from_dict_ignores_unknown_keys(self):
        e = UsageEntry.from_dict({"input_tokens": 5, "garbage_field": "x"})
        self.assertEqual(e.input_tokens, 5)


if __name__ == "__main__":
    unittest.main()
