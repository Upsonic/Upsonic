"""Unit tests for skill dependency resolution."""

import unittest

from upsonic.skills.skill import Skill
from upsonic.skills.dependency import (
    detect_cycles,
    get_missing_dependencies,
    resolve_load_order,
)
from upsonic.utils.package.exception import SkillValidationError


def _skill(name, deps=None):
    return Skill(
        name=name, description="d", instructions="i", source_path="",
        dependencies=deps or [],
    )


class TestGetMissingDependencies(unittest.TestCase):
    def test_no_dependencies(self):
        skills = {"a": _skill("a"), "b": _skill("b")}
        self.assertEqual(get_missing_dependencies(skills), {})

    def test_all_dependencies_present(self):
        skills = {"a": _skill("a", ["b"]), "b": _skill("b")}
        self.assertEqual(get_missing_dependencies(skills), {})

    def test_missing_dependency(self):
        skills = {"a": _skill("a", ["missing"])}
        result = get_missing_dependencies(skills)
        self.assertIn("a", result)
        self.assertEqual(result["a"], ["missing"])

    def test_partial_missing(self):
        skills = {
            "a": _skill("a", ["b", "missing"]),
            "b": _skill("b"),
        }
        result = get_missing_dependencies(skills)
        self.assertIn("a", result)
        self.assertEqual(result["a"], ["missing"])
        self.assertNotIn("b", result)


class TestDetectCycles(unittest.TestCase):
    def test_no_cycles(self):
        skills = {"a": _skill("a", ["b"]), "b": _skill("b")}
        cycles = detect_cycles(skills)
        self.assertEqual(cycles, [])

    def test_simple_cycle(self):
        skills = {"a": _skill("a", ["b"]), "b": _skill("b", ["a"])}
        cycles = detect_cycles(skills)
        self.assertTrue(len(cycles) > 0)

    def test_self_cycle(self):
        skills = {"a": _skill("a", ["a"])}
        cycles = detect_cycles(skills)
        self.assertTrue(len(cycles) > 0)

    def test_triangle_cycle(self):
        skills = {
            "a": _skill("a", ["b"]),
            "b": _skill("b", ["c"]),
            "c": _skill("c", ["a"]),
        }
        cycles = detect_cycles(skills)
        self.assertTrue(len(cycles) > 0)

    def test_no_deps(self):
        skills = {"a": _skill("a"), "b": _skill("b"), "c": _skill("c")}
        cycles = detect_cycles(skills)
        self.assertEqual(cycles, [])

    def test_missing_dep_ignored(self):
        skills = {"a": _skill("a", ["nonexistent"])}
        cycles = detect_cycles(skills)
        self.assertEqual(cycles, [])


class TestResolveLoadOrder(unittest.TestCase):
    def test_no_dependencies(self):
        skills = {"a": _skill("a"), "b": _skill("b")}
        order = resolve_load_order(skills)
        self.assertEqual(set(order), {"a", "b"})

    def test_linear_chain(self):
        skills = {
            "c": _skill("c", ["b"]),
            "b": _skill("b", ["a"]),
            "a": _skill("a"),
        }
        order = resolve_load_order(skills)
        self.assertEqual(order.index("a"), 0)
        self.assertLess(order.index("b"), order.index("c"))

    def test_diamond_dependency(self):
        skills = {
            "d": _skill("d", ["b", "c"]),
            "b": _skill("b", ["a"]),
            "c": _skill("c", ["a"]),
            "a": _skill("a"),
        }
        order = resolve_load_order(skills)
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("a"), order.index("c"))
        self.assertLess(order.index("b"), order.index("d"))
        self.assertLess(order.index("c"), order.index("d"))

    def test_cycle_raises(self):
        skills = {"a": _skill("a", ["b"]), "b": _skill("b", ["a"])}
        with self.assertRaises(SkillValidationError):
            resolve_load_order(skills)

    def test_single_skill(self):
        skills = {"only": _skill("only")}
        order = resolve_load_order(skills)
        self.assertEqual(order, ["only"])


if __name__ == "__main__":
    unittest.main()
