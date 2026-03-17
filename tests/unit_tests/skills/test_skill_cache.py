"""Unit tests for SkillCache."""

import time
import unittest

from upsonic.skills.cache import SkillCache


class TestSkillCacheBasic(unittest.TestCase):
    def test_set_and_get(self):
        cache = SkillCache(ttl_seconds=60)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_get_missing_key(self):
        cache = SkillCache()
        self.assertIsNone(cache.get("nonexistent"))

    def test_overwrite_key(self):
        cache = SkillCache()
        cache.set("k", "v1")
        cache.set("k", "v2")
        self.assertEqual(cache.get("k"), "v2")

    def test_stores_various_types(self):
        cache = SkillCache()
        cache.set("str", "hello")
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        self.assertEqual(cache.get("str"), "hello")
        self.assertEqual(cache.get("int"), 42)
        self.assertEqual(cache.get("list"), [1, 2, 3])
        self.assertEqual(cache.get("dict"), {"a": 1})


class TestSkillCacheTTL(unittest.TestCase):
    def test_expired_returns_none(self):
        cache = SkillCache(ttl_seconds=0)
        cache.set("expired", "value")
        time.sleep(0.01)
        self.assertIsNone(cache.get("expired"))

    def test_not_expired(self):
        cache = SkillCache(ttl_seconds=60)
        cache.set("fresh", "value")
        self.assertEqual(cache.get("fresh"), "value")

    def test_short_ttl(self):
        cache = SkillCache(ttl_seconds=1)
        cache.set("k", "v")
        self.assertEqual(cache.get("k"), "v")


class TestSkillCacheInvalidation(unittest.TestCase):
    def test_invalidate_specific_key(self):
        cache = SkillCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate("a")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)

    def test_invalidate_all(self):
        cache = SkillCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate()
        self.assertIsNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))

    def test_invalidate_nonexistent_key(self):
        cache = SkillCache()
        # Should not raise
        cache.invalidate("nope")


class TestSkillCacheDunder(unittest.TestCase):
    def test_len(self):
        cache = SkillCache()
        self.assertEqual(len(cache), 0)
        cache.set("a", 1)
        self.assertEqual(len(cache), 1)
        cache.set("b", 2)
        self.assertEqual(len(cache), 2)

    def test_contains_valid(self):
        cache = SkillCache(ttl_seconds=60)
        cache.set("present", "yes")
        self.assertIn("present", cache)
        self.assertNotIn("absent", cache)

    def test_contains_expired(self):
        cache = SkillCache(ttl_seconds=0)
        cache.set("expired", "value")
        time.sleep(0.01)
        self.assertNotIn("expired", cache)


if __name__ == "__main__":
    unittest.main()
