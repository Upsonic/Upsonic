"""
Smoke tests for skill version fields and version constraint filtering.

Tests: Top-level version in SKILL.md frontmatter, nested metadata.version,
version constraint filtering on LocalSkills.
"""

import tempfile
from pathlib import Path

from upsonic.skills.loader import LocalSkills
from upsonic.skills.skills import Skills


def _make_versioned_skill_dir(base, name, *, version="1.0.0", top_level=True):
    """Create a skill directory with version info in SKILL.md."""
    skill_dir = Path(base) / name
    skill_dir.mkdir(parents=True)
    if top_level:
        frontmatter = f"---\nname: {name}\ndescription: Skill {name}\nversion: \"{version}\"\n---\nInstructions for {name}"
    else:
        frontmatter = (
            f"---\nname: {name}\ndescription: Skill {name}\n"
            f"metadata:\n  version: \"{version}\"\n---\nInstructions for {name}"
        )
    (skill_dir / "SKILL.md").write_text(frontmatter)
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('ok')")
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "ref.txt").write_text("Reference content.")
    return skill_dir


class TestVersionFieldParsing:
    def test_top_level_version_parsed(self):
        """Top-level 'version' field in SKILL.md is read correctly."""
        with tempfile.TemporaryDirectory() as d:
            _make_versioned_skill_dir(d, "v-top", version="2.1.0", top_level=True)
            skills = Skills(loaders=[LocalSkills(d)])
            skill = skills.get_skill("v-top")
            assert skill is not None
            assert skill.version is not None
            assert "2.1.0" in str(skill.version)

    def test_nested_metadata_version_parsed(self):
        """Nested metadata.version in SKILL.md is read correctly."""
        with tempfile.TemporaryDirectory() as d:
            _make_versioned_skill_dir(d, "v-nested", version="3.0.0", top_level=False)
            skills = Skills(loaders=[LocalSkills(d)])
            skill = skills.get_skill("v-nested")
            assert skill is not None
            assert skill.version is not None
            assert "3.0.0" in str(skill.version)

    def test_no_version_field(self):
        """Skill without version field loads successfully with None version."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "no-ver"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: no-ver\ndescription: No version\n---\nInstructions"
            )
            (skill_dir / "scripts").mkdir()
            (skill_dir / "references").mkdir()
            skills = Skills(loaders=[LocalSkills(d)])
            skill = skills.get_skill("no-ver")
            assert skill is not None
            assert skill.version is None

    def test_top_level_version_allowed_by_validator(self):
        """Validator does not reject top-level 'version' field."""
        from upsonic.skills.validator import validate_metadata

        frontmatter = {
            "name": "test",
            "description": "test skill",
            "version": "1.0.0",
        }
        errors = validate_metadata(frontmatter)
        assert not errors, f"Unexpected validation errors: {errors}"


class TestVersionConstraintFiltering:
    def test_version_constraint_includes_matching(self):
        """LocalSkills with version_constraint loads skills that match."""
        with tempfile.TemporaryDirectory() as d:
            _make_versioned_skill_dir(d, "match-skill", version="1.2.0", top_level=True)
            loader = LocalSkills(d, version_constraint=">=1.0.0")
            skills = Skills(loaders=[loader])
            assert "match-skill" in skills

    def test_version_constraint_excludes_non_matching(self):
        """LocalSkills with version_constraint skips skills that don't match."""
        with tempfile.TemporaryDirectory() as d:
            _make_versioned_skill_dir(d, "old-skill", version="0.5.0", top_level=True)
            loader = LocalSkills(d, version_constraint=">=1.0.0")
            skills = Skills(loaders=[loader])
            assert "old-skill" not in skills

    def test_no_version_constraint_loads_all(self):
        """Without version_constraint, all skills are loaded."""
        with tempfile.TemporaryDirectory() as d:
            _make_versioned_skill_dir(d, "any-ver", version="0.1.0", top_level=True)
            loader = LocalSkills(d)
            skills = Skills(loaders=[loader])
            assert "any-ver" in skills

    def test_mixed_versions_filtered(self):
        """Multiple skills with different versions are filtered correctly."""
        with tempfile.TemporaryDirectory() as d:
            _make_versioned_skill_dir(d, "new-skill", version="2.0.0", top_level=True)
            _make_versioned_skill_dir(d, "old-skill", version="0.3.0", top_level=True)
            loader = LocalSkills(d, version_constraint=">=1.0.0")
            skills = Skills(loaders=[loader])
            assert "new-skill" in skills
            assert "old-skill" not in skills
