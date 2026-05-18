"""Unit tests for AggregatedUsage.from_entries."""
from __future__ import annotations

import unittest

from upsonic.usage_registry import AggregatedUsage, UsageEntry


class TestAggregatedUsage(unittest.TestCase):
    def test_empty_iterable_yields_zeros(self):
        agg = AggregatedUsage.from_entries([])
        self.assertEqual(agg.input_tokens, 0)
        self.assertEqual(agg.total_tokens, 0)
        self.assertEqual(agg.requests, 0)
        self.assertEqual(agg.entry_count, 0)
        self.assertIsNone(agg.cost)
        self.assertEqual(agg.models, [])

    def test_sums_tokens_across_entries(self):
        agg = AggregatedUsage.from_entries([
            UsageEntry(input_tokens=10, output_tokens=20, requests=1),
            UsageEntry(input_tokens=5,  output_tokens=7,  requests=1),
            UsageEntry(input_tokens=1,  output_tokens=2,  requests=1, tool_calls=3),
        ])
        self.assertEqual(agg.input_tokens, 16)
        self.assertEqual(agg.output_tokens, 29)
        self.assertEqual(agg.total_tokens, 45)
        self.assertEqual(agg.requests, 3)
        self.assertEqual(agg.tool_calls, 3)
        self.assertEqual(agg.entry_count, 3)

    def test_cost_is_none_when_no_entry_priced(self):
        agg = AggregatedUsage.from_entries([
            UsageEntry(input_tokens=10, cost_usd=None),
            UsageEntry(input_tokens=5, cost_usd=None),
        ])
        self.assertIsNone(agg.cost)

    def test_cost_sums_priced_entries_only(self):
        agg = AggregatedUsage.from_entries([
            UsageEntry(cost_usd=0.01),
            UsageEntry(cost_usd=None),     # ignored
            UsageEntry(cost_usd=0.005),
        ])
        self.assertAlmostEqual(agg.cost, 0.015, places=6)

    def test_cost_zero_is_distinct_from_none(self):
        agg = AggregatedUsage.from_entries([UsageEntry(cost_usd=0.0)])
        self.assertEqual(agg.cost, 0.0)
        self.assertIsNotNone(agg.cost)

    def test_distinct_models_preserved_in_order(self):
        agg = AggregatedUsage.from_entries([
            UsageEntry(model="openai/gpt-4o"),
            UsageEntry(model="anthropic/claude-sonnet-4-6"),
            UsageEntry(model="openai/gpt-4o"),     # dup
        ])
        self.assertEqual(agg.models, ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"])

    def test_time_to_first_token_takes_earliest_non_none(self):
        agg = AggregatedUsage.from_entries([
            UsageEntry(time_to_first_token=None),
            UsageEntry(time_to_first_token=0.42),
            UsageEntry(time_to_first_token=0.10),   # later entry, but we keep first non-None
        ])
        self.assertEqual(agg.time_to_first_token, 0.42)

    def test_upsonic_execution_time_subtracts_known_external(self):
        agg = AggregatedUsage.from_entries([
            UsageEntry(duration=10.0, model_execution_time=4.0, tool_execution_time=2.0),
        ])
        self.assertEqual(agg.upsonic_execution_time, 4.0)

    def test_upsonic_execution_time_clamped_to_zero(self):
        # If known external time exceeds duration, don't go negative.
        agg = AggregatedUsage.from_entries([
            UsageEntry(duration=1.0, model_execution_time=5.0),
        ])
        self.assertEqual(agg.upsonic_execution_time, 0.0)


if __name__ == "__main__":
    unittest.main()
