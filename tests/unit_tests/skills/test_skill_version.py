"""Unit tests for SkillVersion and VersionConstraint."""

import unittest

from upsonic.skills.version import SkillVersion, VersionConstraint


class TestSkillVersionParsing(unittest.TestCase):
    def test_parse_three_part(self):
        v = SkillVersion.parse("1.2.3")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertEqual(v.patch, 3)

    def test_parse_two_part(self):
        v = SkillVersion.parse("2.5")
        self.assertEqual(v.major, 2)
        self.assertEqual(v.minor, 5)
        self.assertEqual(v.patch, 0)

    def test_parse_strips_whitespace(self):
        v = SkillVersion.parse("  1.0.0  ")
        self.assertEqual(v.major, 1)

    def test_parse_invalid_too_few_parts(self):
        with self.assertRaises(ValueError):
            SkillVersion.parse("1")

    def test_parse_invalid_too_many_parts(self):
        with self.assertRaises(ValueError):
            SkillVersion.parse("1.2.3.4")

    def test_parse_non_integer(self):
        with self.assertRaises(ValueError):
            SkillVersion.parse("1.a.3")

    def test_parse_negative(self):
        with self.assertRaises(ValueError):
            SkillVersion.parse("-1.0.0")


class TestSkillVersionComparison(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(SkillVersion(1, 2, 3), SkillVersion(1, 2, 3))

    def test_not_equal(self):
        self.assertNotEqual(SkillVersion(1, 0, 0), SkillVersion(2, 0, 0))

    def test_less_than_major(self):
        self.assertLess(SkillVersion(1, 0, 0), SkillVersion(2, 0, 0))

    def test_less_than_minor(self):
        self.assertLess(SkillVersion(1, 1, 0), SkillVersion(1, 2, 0))

    def test_less_than_patch(self):
        self.assertLess(SkillVersion(1, 0, 1), SkillVersion(1, 0, 2))

    def test_greater_than(self):
        self.assertGreater(SkillVersion(3, 0, 0), SkillVersion(2, 9, 9))

    def test_le(self):
        self.assertLessEqual(SkillVersion(1, 0, 0), SkillVersion(1, 0, 0))
        self.assertLessEqual(SkillVersion(1, 0, 0), SkillVersion(1, 0, 1))

    def test_ge(self):
        self.assertGreaterEqual(SkillVersion(2, 0, 0), SkillVersion(2, 0, 0))
        self.assertGreaterEqual(SkillVersion(2, 0, 1), SkillVersion(2, 0, 0))

    def test_str(self):
        self.assertEqual(str(SkillVersion(1, 2, 3)), "1.2.3")
        self.assertEqual(str(SkillVersion(0, 0, 0)), "0.0.0")

    def test_frozen(self):
        v = SkillVersion(1, 0, 0)
        with self.assertRaises(AttributeError):
            v.major = 2


class TestVersionConstraintSingle(unittest.TestCase):
    def test_gte(self):
        vc = VersionConstraint(">=1.0.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 0, 0)))
        self.assertTrue(vc.satisfies(SkillVersion(2, 0, 0)))
        self.assertFalse(vc.satisfies(SkillVersion(0, 9, 9)))

    def test_lte(self):
        vc = VersionConstraint("<=2.0.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 0, 0)))
        self.assertTrue(vc.satisfies(SkillVersion(2, 0, 0)))
        self.assertFalse(vc.satisfies(SkillVersion(2, 0, 1)))

    def test_gt(self):
        vc = VersionConstraint(">1.0.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 0, 1)))
        self.assertFalse(vc.satisfies(SkillVersion(1, 0, 0)))

    def test_lt(self):
        vc = VersionConstraint("<2.0.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 9, 9)))
        self.assertFalse(vc.satisfies(SkillVersion(2, 0, 0)))

    def test_eq(self):
        vc = VersionConstraint("==1.2.3")
        self.assertTrue(vc.satisfies(SkillVersion(1, 2, 3)))
        self.assertFalse(vc.satisfies(SkillVersion(1, 2, 4)))

    def test_neq(self):
        vc = VersionConstraint("!=1.0.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 0, 1)))
        self.assertFalse(vc.satisfies(SkillVersion(1, 0, 0)))


class TestVersionConstraintCompound(unittest.TestCase):
    def test_range(self):
        vc = VersionConstraint(">=1.0.0,<2.0.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 5, 0)))
        self.assertFalse(vc.satisfies(SkillVersion(0, 9, 0)))
        self.assertFalse(vc.satisfies(SkillVersion(2, 0, 0)))

    def test_pinned_range(self):
        vc = VersionConstraint(">=1.2.0,<=1.2.9")
        self.assertTrue(vc.satisfies(SkillVersion(1, 2, 5)))
        self.assertFalse(vc.satisfies(SkillVersion(1, 3, 0)))

    def test_two_part_version(self):
        vc = VersionConstraint(">=1.0")
        self.assertTrue(vc.satisfies(SkillVersion(1, 0, 0)))


class TestVersionConstraintErrors(unittest.TestCase):
    def test_empty_string(self):
        with self.assertRaises(ValueError):
            VersionConstraint("")

    def test_invalid_operator(self):
        with self.assertRaises(ValueError):
            VersionConstraint("~=1.0.0")

    def test_no_operator(self):
        with self.assertRaises(ValueError):
            VersionConstraint("1.0.0")

    def test_repr(self):
        vc = VersionConstraint(">=1.0.0")
        self.assertIn(">=1.0.0", repr(vc))


if __name__ == "__main__":
    unittest.main()
