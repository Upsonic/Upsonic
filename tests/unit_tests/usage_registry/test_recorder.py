"""Unit tests for record_request_usage helper."""
from __future__ import annotations

import unittest

from upsonic.usage import RequestUsage
from upsonic.usage_registry import (
    UsageRegistry,
    current_scope_tags,
    record_request_usage,
    scope,
)


class TestRecordRequestUsage(unittest.TestCase):
    def setUp(self):
        self.reg = UsageRegistry()

    def test_none_input_returns_none_and_records_nothing(self):
        entry = record_request_usage(None, registry=self.reg)
        self.assertIsNone(entry)
        self.assertEqual(len(self.reg), 0)

    def test_zero_token_input_records_nothing(self):
        ru = RequestUsage(input_tokens=0, output_tokens=0)
        entry = record_request_usage(ru, registry=self.reg)
        self.assertIsNone(entry)
        self.assertEqual(len(self.reg), 0)

    def test_basic_token_recording(self):
        ru = RequestUsage(input_tokens=10, output_tokens=20)
        entry = record_request_usage(
            ru,
            model="openai/gpt-4o",
            pipeline_step="model_call",
            cost_usd=0.01,
            registry=self.reg,
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.input_tokens, 10)
        self.assertEqual(entry.output_tokens, 20)
        self.assertEqual(entry.model, "openai/gpt-4o")
        self.assertEqual(entry.pipeline_step, "model_call")
        self.assertEqual(entry.cost_usd, 0.01)
        self.assertEqual(len(self.reg), 1)

    def test_inherits_active_scope_tags(self):
        ru = RequestUsage(input_tokens=5, output_tokens=5)
        with scope(
            chat_usage_id="chat-X",
            agent_usage_id="agent-Y",
            task_usage_id="task-Z",
            user_id="alice",
        ):
            entry = record_request_usage(ru, registry=self.reg)

        self.assertIsNotNone(entry)
        self.assertEqual(entry.chat_usage_id, "chat-X")
        self.assertEqual(entry.agent_usage_id, "agent-Y")
        self.assertEqual(entry.task_usage_id, "task-Z")
        self.assertEqual(entry.user_id, "alice")

    def test_aggregation_via_scope_query(self):
        """Two records under the same scope aggregate correctly."""
        with scope(task_usage_id="T1", agent_usage_id="A1"):
            record_request_usage(
                RequestUsage(input_tokens=10, output_tokens=20),
                registry=self.reg,
                cost_usd=0.01,
            )
            record_request_usage(
                RequestUsage(input_tokens=5, output_tokens=8),
                registry=self.reg,
                cost_usd=0.005,
            )

        agg = self.reg.by_task("T1")
        self.assertEqual(agg.input_tokens, 15)
        self.assertEqual(agg.output_tokens, 28)
        self.assertAlmostEqual(agg.cost, 0.015, places=6)
        self.assertEqual(agg.entry_count, 2)
        self.assertEqual(agg.requests, 2)


if __name__ == "__main__":
    unittest.main()
