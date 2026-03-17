"""Unit tests for all skill loaders."""

import tempfile
import unittest
from pathlib import Path

from upsonic.skills.skill import Skill
from upsonic.skills.loader import (
    BuiltinSkills,
    InlineSkills,
    LocalSkills,
    SkillLoader,
)


def _make_skill_dir(base, name, *, version=None, deps=None, top_level_ver=True, with_assets=False):
    """Create a minimal skill directory."""
    skill_dir = Path(base) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name}", f"description: Skill {name}"]
    if version and top_level_ver:
        fm_lines.append(f'version: "{version}"')
    if version and not top_level_ver:
        fm_lines.append(f"metadata:\n  version: \"{version}\"")
    if deps:
        fm_lines.append(f"dependencies: [{', '.join(deps)}]")
    frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\nInstructions for " + name
    (skill_dir / "SKILL.md").write_text(frontmatter)
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "run.py").write_text("print('ok')")
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "guide.txt").write_text("Guide content")
    if with_assets:
        assets_dir = skill_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "template.html").write_text("<html>Template</html>")
        (assets_dir / "logo.txt").write_text("Logo placeholder")
    return skill_dir


# ---------------------------------------------------------------------------
# InlineSkills
# ---------------------------------------------------------------------------

class TestInlineSkills(unittest.TestCase):
    def test_load_returns_skills(self):
        skills = [
            Skill(name="a", description="d", instructions="i", source_path=""),
            Skill(name="b", description="d", instructions="i", source_path=""),
        ]
        loader = InlineSkills(skills)
        loaded = loader.load()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].name, "a")

    def test_empty_list(self):
        loader = InlineSkills([])
        self.assertEqual(loader.load(), [])

    def test_is_skill_loader(self):
        self.assertTrue(issubclass(InlineSkills, SkillLoader))


# ---------------------------------------------------------------------------
# LocalSkills
# ---------------------------------------------------------------------------

class TestLocalSkills(unittest.TestCase):
    def test_load_single_skill(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-skill")
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "my-skill")

    def test_load_multiple_skills(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "alpha")
            _make_skill_dir(d, "beta")
            loader = LocalSkills(d)
            skills = loader.load()
            names = {s.name for s in skills}
            self.assertEqual(names, {"alpha", "beta"})

    def test_discovers_scripts(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "scripted")
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertIn("run.py", skills[0].scripts)

    def test_discovers_references(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "referenced")
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertIn("guide.txt", skills[0].references)

    def test_parses_top_level_version(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "versioned", version="2.0.0", top_level_ver=True)
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertIsNotNone(skills[0].version)
            self.assertIn("2.0.0", str(skills[0].version))

    def test_parses_nested_metadata_version(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "nested-ver", version="3.1.0", top_level_ver=False)
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertIsNotNone(skills[0].version)
            self.assertIn("3.1.0", str(skills[0].version))

    def test_version_constraint_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "new-skill", version="2.0.0")
            _make_skill_dir(d, "old-skill", version="0.5.0")
            loader = LocalSkills(d, version_constraint=">=1.0.0")
            skills = loader.load()
            names = {s.name for s in skills}
            self.assertIn("new-skill", names)
            self.assertNotIn("old-skill", names)

    def test_nonexistent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            LocalSkills("/nonexistent/path")

    def test_skips_directories_without_skill_md(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "valid")
            (Path(d) / "not-a-skill").mkdir()
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "valid")

    def test_validation_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "valid-skill")
            loader = LocalSkills(d, validate=True)
            skills = loader.load()
            self.assertEqual(len(skills), 1)

    def test_discovers_assets(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "asset-skill", with_assets=True)
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertIn("template.html", skills[0].assets)
            self.assertIn("logo.txt", skills[0].assets)

    def test_no_assets_directory(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "no-assets")
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertEqual(skills[0].assets, [])

    def test_parses_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "dep-skill", deps=["other-skill"])
            loader = LocalSkills(d)
            skills = loader.load()
            self.assertIn("other-skill", skills[0].dependencies)


# ---------------------------------------------------------------------------
# BuiltinSkills
# ---------------------------------------------------------------------------

class TestBuiltinSkills(unittest.TestCase):
    def test_load_all_builtins(self):
        loader = BuiltinSkills()
        skills = loader.load()
        self.assertTrue(len(skills) >= 3)
        names = {s.name for s in skills}
        self.assertIn("code-review", names)
        self.assertIn("summarization", names)
        self.assertIn("data-analysis", names)

    def test_load_specific_builtin(self):
        loader = BuiltinSkills(skills=["code-review"])
        skills = loader.load()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "code-review")

    def test_load_multiple_specific(self):
        loader = BuiltinSkills(skills=["code-review", "summarization"])
        skills = loader.load()
        names = {s.name for s in skills}
        self.assertEqual(names, {"code-review", "summarization"})

    def test_load_nonexistent_builtin(self):
        loader = BuiltinSkills(skills=["nonexistent-skill"])
        skills = loader.load()
        self.assertEqual(len(skills), 0)

    def test_available_skills(self):
        loader = BuiltinSkills()
        available = loader.available_skills()
        self.assertIn("code-review", available)
        self.assertIn("summarization", available)
        self.assertIn("data-analysis", available)

    def test_builtin_skills_have_instructions(self):
        loader = BuiltinSkills(skills=["code-review"])
        skills = loader.load()
        self.assertTrue(len(skills[0].instructions) > 0)

    def test_builtin_skills_have_version(self):
        loader = BuiltinSkills()
        skills = loader.load()
        for s in skills:
            self.assertIsNotNone(s.version, f"Builtin skill {s.name} has no version")

    def test_is_skill_loader(self):
        self.assertTrue(issubclass(BuiltinSkills, SkillLoader))


# ---------------------------------------------------------------------------
# Remote Loader Attributes (no actual downloads)
# ---------------------------------------------------------------------------

class TestGitHubSkillsAttributes(unittest.TestCase):
    def test_init_attributes(self):
        from upsonic.skills.loader import GitHubSkills
        loader = GitHubSkills(repo="owner/repo", branch="main", path="skills/")
        self.assertEqual(loader.repo, "owner/repo")
        self.assertEqual(loader.branch, "main")

    def test_custom_token(self):
        from upsonic.skills.loader import GitHubSkills
        loader = GitHubSkills(repo="org/repo", token="my-token")
        self.assertEqual(loader.token, "my-token")


class TestURLSkillsAttributes(unittest.TestCase):
    def test_init_attributes(self):
        from upsonic.skills.loader import URLSkills
        loader = URLSkills(url="https://example.com/skills.tar.gz")
        self.assertEqual(loader.url, "https://example.com/skills.tar.gz")

    def test_custom_headers(self):
        from upsonic.skills.loader import URLSkills
        loader = URLSkills(
            url="https://example.com/skills.tar.gz",
            headers={"Authorization": "Bearer token"},
        )
        self.assertEqual(loader.headers["Authorization"], "Bearer token")


if __name__ == "__main__":
    unittest.main()
