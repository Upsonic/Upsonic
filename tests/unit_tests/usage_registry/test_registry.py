"""Unit tests for UsageRegistry — idempotency, scope filtering, aggregation."""
from __future__ import annotations

import threading
import unittest

from upsonic.usage_registry import UsageEntry, UsageRegistry, get_default_registry, new_usage_id


class TestUsageRegistryWrite(unittest.TestCase):
    def test_record_appends(self):
        r = UsageRegistry()
        r.record(UsageEntry(input_tokens=5))
        r.record(UsageEntry(input_tokens=7))
        self.assertEqual(len(r), 2)

    def test_record_is_idempotent_by_entry_id(self):
        """Same entry_id recorded twice → replace, not double-count.
        This is the core invariant that makes retries safe."""
        r = UsageRegistry()
        e1 = UsageEntry(entry_id="fixed", input_tokens=10)
        e2 = UsageEntry(entry_id="fixed", input_tokens=999)
        r.record(e1)
        r.record(e2)
        self.assertEqual(len(r), 1)
        agg = r.aggregate()
        self.assertEqual(agg.input_tokens, 999)

    def test_remove_returns_bool(self):
        r = UsageRegistry()
        e = UsageEntry()
        r.record(e)
        self.assertTrue(r.remove(e.entry_id))
        self.assertFalse(r.remove(e.entry_id))
        self.assertEqual(len(r), 0)

    def test_clear_drops_all(self):
        r = UsageRegistry()
        r.record_many([UsageEntry() for _ in range(5)])
        self.assertEqual(len(r), 5)
        r.clear()
        self.assertEqual(len(r), 0)


class TestUsageRegistryScopeFiltering(unittest.TestCase):
    def setUp(self):
        self.r = UsageRegistry()
        self.r.record_many([
            UsageEntry(entry_id="a", chat_usage_id="c1", agent_usage_id="a1", task_usage_id="t1", input_tokens=10),
            UsageEntry(entry_id="b", chat_usage_id="c1", agent_usage_id="a1", task_usage_id="t2", input_tokens=5),
            UsageEntry(entry_id="c", chat_usage_id="c2", agent_usage_id="a2", task_usage_id="t3", input_tokens=3),
            UsageEntry(entry_id="d", agent_usage_id="a1", task_usage_id="t4", input_tokens=2),  # no chat
        ])

    def test_filter_by_chat_usage_id(self):
        rows = self.r.entries(chat_usage_id="c1")
        self.assertEqual({e.entry_id for e in rows}, {"a", "b"})

    def test_filter_by_agent_usage_id(self):
        rows = self.r.entries(agent_usage_id="a1")
        self.assertEqual({e.entry_id for e in rows}, {"a", "b", "d"})

    def test_filter_by_task_usage_id(self):
        rows = self.r.entries(task_usage_id="t3")
        self.assertEqual({e.entry_id for e in rows}, {"c"})

    def test_filter_and_semantics(self):
        rows = self.r.entries(chat_usage_id="c1", task_usage_id="t1")
        self.assertEqual({e.entry_id for e in rows}, {"a"})

    def test_no_filter_returns_everything(self):
        rows = self.r.entries()
        self.assertEqual(len(rows), 4)


class TestUsageRegistryAggregate(unittest.TestCase):
    def test_aggregate_filters_then_sums(self):
        r = UsageRegistry()
        r.record(UsageEntry(chat_usage_id="c1", input_tokens=10, output_tokens=20, requests=1, cost_usd=0.01))
        r.record(UsageEntry(chat_usage_id="c1", input_tokens=5, output_tokens=8, requests=1, cost_usd=0.005))
        r.record(UsageEntry(chat_usage_id="c2", input_tokens=999, output_tokens=999, cost_usd=99.0))

        agg = r.by_chat("c1")
        self.assertEqual(agg.input_tokens, 15)
        self.assertEqual(agg.output_tokens, 28)
        self.assertEqual(agg.total_tokens, 43)
        self.assertEqual(agg.requests, 2)
        self.assertAlmostEqual(agg.cost, 0.015, places=6)
        self.assertEqual(agg.entry_count, 2)

    def test_aggregate_unknown_id_yields_zero(self):
        r = UsageRegistry()
        r.record(UsageEntry(chat_usage_id="c1", input_tokens=10))
        agg = r.by_chat("nonexistent")
        self.assertEqual(agg.input_tokens, 0)
        self.assertEqual(agg.entry_count, 0)

    def test_multi_scope_entry_rolls_up_into_each_scope(self):
        """A single LLM call inside Chat→Agent→Task should appear in all
        three aggregations — chat / agent / task all show the same spend."""
        r = UsageRegistry()
        r.record(UsageEntry(
            chat_usage_id="C", agent_usage_id="A", task_usage_id="T",
            input_tokens=100, output_tokens=50, cost_usd=0.02,
        ))
        self.assertEqual(r.by_chat("C").total_tokens, 150)
        self.assertEqual(r.by_agent("A").total_tokens, 150)
        self.assertEqual(r.by_task("T").total_tokens, 150)
        self.assertEqual(r.by_chat("C").cost, 0.02)


class TestRegistryThreadSafety(unittest.TestCase):
    def test_concurrent_records_do_not_drop_entries(self):
        r = UsageRegistry()

        def worker(start: int):
            for i in range(100):
                r.record(UsageEntry(entry_id=f"e-{start}-{i}", input_tokens=1))

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(r), 1000)
        self.assertEqual(r.aggregate().input_tokens, 1000)


class TestDefaultRegistrySingleton(unittest.TestCase):
    def test_returns_same_instance(self):
        a = get_default_registry()
        b = get_default_registry()
        self.assertIs(a, b)


class TestNewUsageId(unittest.TestCase):
    def test_scope_prefix_present(self):
        self.assertTrue(new_usage_id("chat").startswith("chat-"))
        self.assertTrue(new_usage_id("agent").startswith("agent-"))
        self.assertTrue(new_usage_id("task").startswith("task-"))

    def test_is_unique(self):
        ids = {new_usage_id("chat") for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
