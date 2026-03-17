"""Unit tests for skill validation."""

import tempfile
import unittest
from pathlib import Path

from upsonic.skills.validator import (
    validate_metadata,
    validate_skill_directory,
)


class TestValidateMetadataRequired(unittest.TestCase):
    def test_valid_minimal(self):
        errors = validate_metadata({"name": "test-skill", "description": "A skill"})
        self.assertEqual(errors, [])

    def test_missing_name(self):
        errors = validate_metadata({"description": "d"})
        self.assertTrue(any("name" in e for e in errors))

    def test_missing_description(self):
        errors = validate_metadata({"name": "test"})
        self.assertTrue(any("description" in e for e in errors))

    def test_empty_name(self):
        errors = validate_metadata({"name": "", "description": "d"})
        self.assertTrue(len(errors) > 0)

    def test_empty_description(self):
        errors = validate_metadata({"name": "test", "description": ""})
        self.assertTrue(len(errors) > 0)


class TestValidateMetadataNameRules(unittest.TestCase):
    def test_uppercase_rejected(self):
        errors = validate_metadata({"name": "TestSkill", "description": "d"})
        self.assertTrue(any("lowercase" in e for e in errors))

    def test_leading_hyphen(self):
        errors = validate_metadata({"name": "-test", "description": "d"})
        self.assertTrue(any("hyphen" in e for e in errors))

    def test_trailing_hyphen(self):
        errors = validate_metadata({"name": "test-", "description": "d"})
        self.assertTrue(any("hyphen" in e for e in errors))

    def test_consecutive_hyphens(self):
        errors = validate_metadata({"name": "test--skill", "description": "d"})
        self.assertTrue(any("consecutive" in e.lower() for e in errors))

    def test_special_characters(self):
        errors = validate_metadata({"name": "test_skill", "description": "d"})
        self.assertTrue(any("invalid" in e.lower() for e in errors))

    def test_valid_hyphenated_name(self):
        errors = validate_metadata({"name": "my-cool-skill", "description": "d"})
        self.assertEqual(errors, [])

    def test_name_too_long(self):
        long_name = "a" * 65
        errors = validate_metadata({"name": long_name, "description": "d"})
        self.assertTrue(any("64" in e or "limit" in e.lower() for e in errors))

    def test_description_too_long(self):
        long_desc = "a" * 1025
        errors = validate_metadata({"name": "test", "description": long_desc})
        self.assertTrue(any("1024" in e or "limit" in e.lower() for e in errors))

    def test_description_with_xml_tags_rejected(self):
        errors = validate_metadata({"name": "test", "description": "Use <b>this</b> skill"})
        self.assertTrue(any("xml" in e.lower() or "bracket" in e.lower() for e in errors))

    def test_description_with_angle_bracket_rejected(self):
        errors = validate_metadata({"name": "test", "description": "Use > and < symbols"})
        self.assertTrue(any("xml" in e.lower() or "bracket" in e.lower() for e in errors))

    def test_description_without_xml_tags_allowed(self):
        errors = validate_metadata({"name": "test", "description": "A clean description"})
        self.assertEqual(errors, [])

    def test_name_directory_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "wrong-name"
            skill_dir.mkdir()
            errors = validate_metadata(
                {"name": "correct-name", "description": "d"},
                skill_dir=skill_dir,
            )
            self.assertTrue(any("match" in e.lower() for e in errors))


class TestValidateMetadataOptionalFields(unittest.TestCase):
    def test_version_allowed(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "version": "1.0.0"}
        )
        self.assertEqual(errors, [])

    def test_license_allowed(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "license": "MIT"}
        )
        self.assertEqual(errors, [])

    def test_compatibility_allowed(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "compatibility": "Python 3.8+"}
        )
        self.assertEqual(errors, [])

    def test_allowed_tools_valid(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "allowed-tools": ["tool1", "tool2"]}
        )
        self.assertEqual(errors, [])

    def test_allowed_tools_invalid_type(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "allowed-tools": "not-a-list"}
        )
        self.assertTrue(len(errors) > 0)

    def test_dependencies_valid(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "dependencies": ["dep1"]}
        )
        self.assertEqual(errors, [])

    def test_dependencies_invalid_type(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "dependencies": "not-list"}
        )
        self.assertTrue(len(errors) > 0)

    def test_metadata_dict_valid(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "metadata": {"author": "me"}}
        )
        self.assertEqual(errors, [])

    def test_metadata_invalid_type(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "metadata": "string"}
        )
        self.assertTrue(len(errors) > 0)

    def test_unexpected_field_rejected(self):
        errors = validate_metadata(
            {"name": "test", "description": "d", "unknown-field": "x"}
        )
        self.assertTrue(any("unexpected" in e.lower() for e in errors))


class TestValidateSkillDirectory(unittest.TestCase):
    def test_valid_skill_directory(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: test-skill\ndescription: A test\n---\nInstructions"
            )
            errors = validate_skill_directory(skill_dir)
            self.assertEqual(errors, [])

    def test_missing_skill_md(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "no-md"
            skill_dir.mkdir()
            errors = validate_skill_directory(skill_dir)
            self.assertTrue(any("SKILL.md" in e for e in errors))

    def test_nonexistent_path(self):
        errors = validate_skill_directory(Path("/nonexistent/path"))
        self.assertTrue(len(errors) > 0)

    def test_file_not_directory(self):
        with tempfile.NamedTemporaryFile() as f:
            errors = validate_skill_directory(Path(f.name))
            self.assertTrue(any("directory" in e.lower() for e in errors))

    def test_no_frontmatter(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "bad-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("No frontmatter here")
            errors = validate_skill_directory(skill_dir)
            self.assertTrue(len(errors) > 0)

    def test_unclosed_frontmatter(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "unclosed"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: test\n")
            errors = validate_skill_directory(skill_dir)
            self.assertTrue(len(errors) > 0)


if __name__ == "__main__":
    unittest.main()
