"""Unit tests for SkillMetrics."""

import time
import unittest

from upsonic.skills.metrics import SkillMetrics


class TestSkillMetricsDefaults(unittest.TestCase):
    def test_default_values(self):
        m = SkillMetrics()
        self.assertEqual(m.load_count, 0)
        self.assertEqual(m.reference_access_count, 0)
        self.assertEqual(m.script_execution_count, 0)
        self.assertEqual(m.total_chars_loaded, 0)
        self.assertIsNone(m.last_used_timestamp)


class TestSkillMetricsRecording(unittest.TestCase):
    def test_record_load(self):
        m = SkillMetrics()
        m.record_load(chars=100)
        self.assertEqual(m.load_count, 1)
        self.assertEqual(m.total_chars_loaded, 100)
        self.assertIsNotNone(m.last_used_timestamp)

    def test_record_load_multiple(self):
        m = SkillMetrics()
        m.record_load(chars=50)
        m.record_load(chars=30)
        self.assertEqual(m.load_count, 2)
        self.assertEqual(m.total_chars_loaded, 80)

    def test_record_reference_access(self):
        m = SkillMetrics()
        m.record_reference_access(chars=200)
        self.assertEqual(m.reference_access_count, 1)
        self.assertEqual(m.total_chars_loaded, 200)
        self.assertIsNotNone(m.last_used_timestamp)

    def test_record_script_execution(self):
        m = SkillMetrics()
        m.record_script_execution()
        self.assertEqual(m.script_execution_count, 1)
        self.assertIsNotNone(m.last_used_timestamp)
        # Script execution does not add chars
        self.assertEqual(m.total_chars_loaded, 0)

    def test_timestamp_updates(self):
        m = SkillMetrics()
        m.record_load()
        t1 = m.last_used_timestamp
        time.sleep(0.01)
        m.record_reference_access()
        t2 = m.last_used_timestamp
        self.assertGreater(t2, t1)

    def test_mixed_operations(self):
        m = SkillMetrics()
        m.record_load(chars=100)
        m.record_reference_access(chars=200)
        m.record_script_execution()
        m.record_load(chars=50)
        self.assertEqual(m.load_count, 2)
        self.assertEqual(m.reference_access_count, 1)
        self.assertEqual(m.script_execution_count, 1)
        self.assertEqual(m.total_chars_loaded, 350)


class TestSkillMetricsSerialization(unittest.TestCase):
    def test_to_dict(self):
        m = SkillMetrics(load_count=5, reference_access_count=3,
                         script_execution_count=1, total_chars_loaded=500,
                         last_used_timestamp=1234.5)
        d = m.to_dict()
        self.assertEqual(d["load_count"], 5)
        self.assertEqual(d["reference_access_count"], 3)
        self.assertEqual(d["script_execution_count"], 1)
        self.assertEqual(d["total_chars_loaded"], 500)
        self.assertEqual(d["last_used_timestamp"], 1234.5)

    def test_from_dict(self):
        data = {"load_count": 10, "total_chars_loaded": 999}
        m = SkillMetrics.from_dict(data)
        self.assertEqual(m.load_count, 10)
        self.assertEqual(m.total_chars_loaded, 999)
        self.assertEqual(m.reference_access_count, 0)
        self.assertIsNone(m.last_used_timestamp)

    def test_roundtrip(self):
        m = SkillMetrics(load_count=7, reference_access_count=2,
                         script_execution_count=4, total_chars_loaded=1234,
                         last_used_timestamp=9999.9)
        restored = SkillMetrics.from_dict(m.to_dict())
        self.assertEqual(restored.load_count, m.load_count)
        self.assertEqual(restored.reference_access_count, m.reference_access_count)
        self.assertEqual(restored.script_execution_count, m.script_execution_count)
        self.assertEqual(restored.total_chars_loaded, m.total_chars_loaded)
        self.assertEqual(restored.last_used_timestamp, m.last_used_timestamp)

    def test_from_dict_empty(self):
        m = SkillMetrics.from_dict({})
        self.assertEqual(m.load_count, 0)
        self.assertIsNone(m.last_used_timestamp)


if __name__ == "__main__":
    unittest.main()
