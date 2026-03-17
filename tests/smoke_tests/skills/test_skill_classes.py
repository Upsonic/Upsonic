"""
Smoke tests for skill system classes, attributes, and methods.

Tests all core skill components: Skill, Skills, SkillMetrics, SkillVersion,
VersionConstraint, SkillCache, dependency resolution, validators, loaders.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------

class TestSkill:
    """Tests for the Skill dataclass."""

    def test_skill_creation(self):
        from upsonic.skills.skill import Skill

        s = Skill(
            name="test-skill",
            description="A test skill",
            instructions="Do something",
            source_path="/tmp/test",
        )
        assert s.name == "test-skill"
        assert s.description == "A test skill"
        assert s.instructions == "Do something"
        assert s.source_path == "/tmp/test"

    def test_skill_default_fields(self):
        from upsonic.skills.skill import Skill

        s = Skill(name="s", description="d", instructions="i", source_path="")
        assert s.scripts == []
        assert s.references == []
        assert s.assets == []
        assert s.metadata is None
        assert s.license is None
        assert s.compatibility is None
        assert s.allowed_tools is None
        assert s.version is None
        assert s.dependencies == []

    def test_skill_optional_fields(self):
        from upsonic.skills.skill import Skill

        s = Skill(
            name="s",
            description="d",
            instructions="i",
            source_path="",
            scripts=["run.sh"],
            references=["ref.txt"],
            metadata={"key": "val"},
            license="MIT",
            compatibility="python>=3.10",
            allowed_tools=["tool_a"],
            version="1.2.3",
            dependencies=["other-skill"],
        )
        assert s.scripts == ["run.sh"]
        assert s.references == ["ref.txt"]
        assert s.metadata == {"key": "val"}
        assert s.license == "MIT"
        assert s.version == "1.2.3"
        assert s.dependencies == ["other-skill"]
        assert s.allowed_tools == ["tool_a"]
        assert s.compatibility == "python>=3.10"

    def test_skill_to_dict(self):
        from upsonic.skills.skill import Skill

        s = Skill(name="s", description="d", instructions="i", source_path="/p", version="1.0.0")
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "s"
        assert d["description"] == "d"
        assert d["instructions"] == "i"
        assert d["source_path"] == "/p"
        assert d["version"] == "1.0.0"

    def test_skill_from_dict(self):
        from upsonic.skills.skill import Skill

        d = {
            "name": "x",
            "description": "xd",
            "instructions": "xi",
            "source_path": "/x",
            "scripts": ["a.sh"],
            "references": ["b.txt"],
            "version": "2.0.0",
            "dependencies": ["dep1"],
        }
        s = Skill.from_dict(d)
        assert s.name == "x"
        assert s.scripts == ["a.sh"]
        assert s.version == "2.0.0"
        assert s.dependencies == ["dep1"]

    def test_skill_roundtrip(self):
        from upsonic.skills.skill import Skill

        original = Skill(
            name="rt",
            description="roundtrip",
            instructions="body",
            source_path="/rt",
            scripts=["s.py"],
            references=["r.md"],
            metadata={"a": 1},
            license="Apache-2.0",
            version="3.1.0",
            dependencies=["x", "y"],
        )
        restored = Skill.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.scripts == original.scripts
        assert restored.version == original.version
        assert restored.dependencies == original.dependencies
        assert restored.metadata == original.metadata

    def test_skill_repr(self):
        from upsonic.skills.skill import Skill

        s = Skill(name="r", description="d", instructions="i", source_path="")
        r = repr(s)
        assert "r" in r


# ---------------------------------------------------------------------------
# SkillMetrics
# ---------------------------------------------------------------------------

class TestSkillMetrics:
    """Tests for SkillMetrics dataclass."""

    def test_default_values(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        assert m.load_count == 0
        assert m.reference_access_count == 0
        assert m.script_execution_count == 0
        assert m.total_chars_loaded == 0
        assert m.last_used_timestamp is None

    def test_record_load(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        m.record_load(chars=100)
        assert m.load_count == 1
        assert m.total_chars_loaded == 100
        assert m.last_used_timestamp is not None

    def test_record_reference_access(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        m.record_reference_access(chars=50)
        assert m.reference_access_count == 1
        assert m.total_chars_loaded == 50
        assert m.last_used_timestamp is not None

    def test_record_script_execution(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        m.record_script_execution()
        assert m.script_execution_count == 1
        assert m.last_used_timestamp is not None

    def test_multiple_records(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        m.record_load(chars=10)
        m.record_load(chars=20)
        m.record_reference_access(chars=5)
        m.record_script_execution()
        assert m.load_count == 2
        assert m.reference_access_count == 1
        assert m.script_execution_count == 1
        assert m.total_chars_loaded == 35

    def test_to_dict(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        m.record_load(chars=42)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["load_count"] == 1
        assert d["total_chars_loaded"] == 42
        assert d["last_used_timestamp"] is not None

    def test_from_dict(self):
        from upsonic.skills.metrics import SkillMetrics

        d = {"load_count": 5, "reference_access_count": 2, "script_execution_count": 1,
             "total_chars_loaded": 999, "last_used_timestamp": 12345.0}
        m = SkillMetrics.from_dict(d)
        assert m.load_count == 5
        assert m.total_chars_loaded == 999

    def test_roundtrip(self):
        from upsonic.skills.metrics import SkillMetrics

        m = SkillMetrics()
        m.record_load(chars=77)
        m.record_script_execution()
        restored = SkillMetrics.from_dict(m.to_dict())
        assert restored.load_count == m.load_count
        assert restored.total_chars_loaded == m.total_chars_loaded
        assert restored.script_execution_count == m.script_execution_count


# ---------------------------------------------------------------------------
# SkillVersion & VersionConstraint
# ---------------------------------------------------------------------------

class TestSkillVersion:
    """Tests for SkillVersion parsing and comparison."""

    def test_parse_full(self):
        from upsonic.skills.version import SkillVersion

        v = SkillVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_two_parts(self):
        from upsonic.skills.version import SkillVersion

        v = SkillVersion.parse("2.5")
        assert v.major == 2
        assert v.minor == 5
        assert v.patch == 0

    def test_parse_invalid(self):
        from upsonic.skills.version import SkillVersion

        with pytest.raises(ValueError):
            SkillVersion.parse("not_a_version")

    def test_comparison(self):
        from upsonic.skills.version import SkillVersion

        v1 = SkillVersion.parse("1.0.0")
        v2 = SkillVersion.parse("1.0.1")
        v3 = SkillVersion.parse("2.0.0")
        assert v1 < v2
        assert v2 < v3
        assert v3 > v1
        assert v1 <= v1
        assert v3 >= v2

    def test_str(self):
        from upsonic.skills.version import SkillVersion

        v = SkillVersion.parse("3.2.1")
        assert str(v) == "3.2.1"


class TestVersionConstraint:
    """Tests for VersionConstraint parsing and matching."""

    def test_single_constraint(self):
        from upsonic.skills.version import SkillVersion, VersionConstraint

        vc = VersionConstraint(">=1.0.0")
        assert vc.satisfies(SkillVersion.parse("1.0.0"))
        assert vc.satisfies(SkillVersion.parse("2.0.0"))
        assert not vc.satisfies(SkillVersion.parse("0.9.0"))

    def test_multiple_constraints(self):
        from upsonic.skills.version import SkillVersion, VersionConstraint

        vc = VersionConstraint(">=1.0.0,<2.0.0")
        assert vc.satisfies(SkillVersion.parse("1.0.0"))
        assert vc.satisfies(SkillVersion.parse("1.9.9"))
        assert not vc.satisfies(SkillVersion.parse("2.0.0"))
        assert not vc.satisfies(SkillVersion.parse("0.5.0"))

    def test_exact_constraint(self):
        from upsonic.skills.version import SkillVersion, VersionConstraint

        vc = VersionConstraint("==1.5.0")
        assert vc.satisfies(SkillVersion.parse("1.5.0"))
        assert not vc.satisfies(SkillVersion.parse("1.5.1"))

    def test_repr(self):
        from upsonic.skills.version import VersionConstraint

        vc = VersionConstraint(">=1.0.0")
        assert "1.0.0" in repr(vc)


# ---------------------------------------------------------------------------
# Dependency Resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    """Tests for dependency resolution functions."""

    def test_no_dependencies(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.dependency import resolve_load_order, get_missing_dependencies, detect_cycles

        skills = {
            "a": Skill(name="a", description="", instructions="", source_path=""),
            "b": Skill(name="b", description="", instructions="", source_path=""),
        }
        order = resolve_load_order(skills)
        assert set(order) == {"a", "b"}
        assert get_missing_dependencies(skills) == {}
        assert detect_cycles(skills) == []

    def test_simple_dependency(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.dependency import resolve_load_order

        skills = {
            "a": Skill(name="a", description="", instructions="", source_path="", dependencies=["b"]),
            "b": Skill(name="b", description="", instructions="", source_path=""),
        }
        order = resolve_load_order(skills)
        assert order.index("b") < order.index("a")

    def test_missing_dependency(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.dependency import get_missing_dependencies

        skills = {
            "a": Skill(name="a", description="", instructions="", source_path="", dependencies=["missing"]),
        }
        missing = get_missing_dependencies(skills)
        assert "a" in missing
        assert "missing" in missing["a"]

    def test_cycle_detection(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.dependency import detect_cycles

        skills = {
            "a": Skill(name="a", description="", instructions="", source_path="", dependencies=["b"]),
            "b": Skill(name="b", description="", instructions="", source_path="", dependencies=["a"]),
        }
        cycles = detect_cycles(skills)
        assert len(cycles) > 0

    def test_cycle_raises_on_resolve(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.dependency import resolve_load_order
        from upsonic.utils.package.exception import SkillValidationError

        skills = {
            "a": Skill(name="a", description="", instructions="", source_path="", dependencies=["b"]),
            "b": Skill(name="b", description="", instructions="", source_path="", dependencies=["a"]),
        }
        with pytest.raises(SkillValidationError):
            resolve_load_order(skills)


# ---------------------------------------------------------------------------
# SkillCache
# ---------------------------------------------------------------------------

class TestSkillCache:
    """Tests for in-memory TTL SkillCache."""

    def test_set_and_get(self):
        from upsonic.skills.cache import SkillCache

        c = SkillCache(ttl_seconds=60)
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_missing_key(self):
        from upsonic.skills.cache import SkillCache

        c = SkillCache()
        assert c.get("missing") is None

    def test_invalidate_all(self):
        from upsonic.skills.cache import SkillCache

        c = SkillCache()
        c.set("a", 1)
        c.set("b", 2)
        c.invalidate()
        assert c.get("a") is None
        assert c.get("b") is None

    def test_invalidate_key(self):
        from upsonic.skills.cache import SkillCache

        c = SkillCache()
        c.set("a", 1)
        c.set("b", 2)
        c.invalidate("a")
        assert c.get("a") is None
        assert c.get("b") == 2

    def test_len_and_contains(self):
        from upsonic.skills.cache import SkillCache

        c = SkillCache()
        c.set("x", 10)
        assert len(c) == 1
        assert "x" in c
        assert "y" not in c


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TestValidator:
    """Tests for skill validation functions."""

    def test_validate_valid_metadata(self):
        from upsonic.skills.validator import validate_metadata

        errors = validate_metadata({"name": "my-skill", "description": "A fine skill"})
        assert errors == []

    def test_validate_missing_name(self):
        from upsonic.skills.validator import validate_metadata

        errors = validate_metadata({"description": "no name"})
        assert any("name" in e.lower() for e in errors)

    def test_validate_missing_description(self):
        from upsonic.skills.validator import validate_metadata

        errors = validate_metadata({"name": "x"})
        assert any("description" in e.lower() for e in errors)

    def test_validate_name_too_long(self):
        from upsonic.skills.validator import validate_metadata

        errors = validate_metadata({"name": "a" * 100, "description": "ok"})
        assert len(errors) > 0

    def test_validate_skill_directory(self):
        from upsonic.skills.validator import validate_skill_directory

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\nBody")
            errors = validate_skill_directory(skill_dir)
            assert errors == []

    def test_validate_skill_directory_no_skill_md(self):
        from upsonic.skills.validator import validate_skill_directory

        with tempfile.TemporaryDirectory() as d:
            errors = validate_skill_directory(Path(d))
            assert any("SKILL.md" in e for e in errors)

    def test_validate_description_rejects_xml_tags(self):
        from upsonic.skills.validator import validate_metadata

        errors = validate_metadata({"name": "test", "description": "Use <b>bold</b>"})
        assert any("xml" in e.lower() or "bracket" in e.lower() for e in errors)

    def test_validate_description_allows_plain_text(self):
        from upsonic.skills.validator import validate_metadata

        errors = validate_metadata({"name": "test", "description": "A plain description"})
        assert errors == []


# ---------------------------------------------------------------------------
# InlineSkills Loader
# ---------------------------------------------------------------------------

class TestInlineSkills:
    """Tests for InlineSkills loader."""

    def test_load(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.loader import InlineSkills

        s1 = Skill(name="a", description="da", instructions="ia", source_path="")
        s2 = Skill(name="b", description="db", instructions="ib", source_path="")
        loader = InlineSkills([s1, s2])
        loaded = loader.load()
        assert len(loaded) == 2
        assert loaded[0].name == "a"
        assert loaded[1].name == "b"

    def test_load_returns_copy(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.loader import InlineSkills

        s = Skill(name="x", description="d", instructions="i", source_path="")
        loader = InlineSkills([s])
        a = loader.load()
        b = loader.load()
        assert a is not b


# ---------------------------------------------------------------------------
# LocalSkills Loader
# ---------------------------------------------------------------------------

class TestLocalSkills:
    """Tests for LocalSkills loader with real filesystem."""

    def _create_skill_dir(self, base: str, name: str, version: str = None) -> Path:
        skill_dir = Path(base) / name
        skill_dir.mkdir(parents=True)
        frontmatter = f"---\nname: {name}\ndescription: Test skill {name}\n"
        if version:
            frontmatter += f"metadata:\n  version: {version}\n"
        frontmatter += "---\n"
        body = f"Instructions for {name}"
        (skill_dir / "SKILL.md").write_text(frontmatter + body)
        return skill_dir

    def test_load_single_skill(self):
        from upsonic.skills.loader import LocalSkills

        with tempfile.TemporaryDirectory() as d:
            self._create_skill_dir(d, "my-skill")
            loader = LocalSkills(str(Path(d) / "my-skill"))
            skills = loader.load()
            assert len(skills) == 1
            assert skills[0].name == "my-skill"

    def test_load_directory_of_skills(self):
        from upsonic.skills.loader import LocalSkills

        with tempfile.TemporaryDirectory() as d:
            self._create_skill_dir(d, "skill-a")
            self._create_skill_dir(d, "skill-b")
            loader = LocalSkills(d)
            skills = loader.load()
            names = {s.name for s in skills}
            assert "skill-a" in names
            assert "skill-b" in names

    def test_load_with_scripts_and_references(self):
        from upsonic.skills.loader import LocalSkills

        with tempfile.TemporaryDirectory() as d:
            skill_dir = self._create_skill_dir(d, "full-skill")
            (skill_dir / "scripts").mkdir()
            (skill_dir / "scripts" / "run.sh").write_text("#!/bin/bash\necho hello")
            (skill_dir / "references").mkdir()
            (skill_dir / "references" / "doc.txt").write_text("Reference content")
            loader = LocalSkills(str(skill_dir))
            skills = loader.load()
            assert len(skills) == 1
            assert "run.sh" in skills[0].scripts
            assert "doc.txt" in skills[0].references

    def test_load_with_assets(self):
        from upsonic.skills.loader import LocalSkills

        with tempfile.TemporaryDirectory() as d:
            skill_dir = self._create_skill_dir(d, "asset-skill")
            (skill_dir / "assets").mkdir()
            (skill_dir / "assets" / "template.html").write_text("<html>Template</html>")
            (skill_dir / "assets" / "logo.txt").write_text("Logo placeholder")
            loader = LocalSkills(str(skill_dir))
            skills = loader.load()
            assert len(skills) == 1
            assert "template.html" in skills[0].assets
            assert "logo.txt" in skills[0].assets

    def test_load_without_assets_dir(self):
        from upsonic.skills.loader import LocalSkills

        with tempfile.TemporaryDirectory() as d:
            self._create_skill_dir(d, "no-assets")
            loader = LocalSkills(str(Path(d) / "no-assets"))
            skills = loader.load()
            assert skills[0].assets == []


# ---------------------------------------------------------------------------
# BuiltinSkills Loader
# ---------------------------------------------------------------------------

class TestBuiltinSkills:
    """Tests for BuiltinSkills loader."""

    def test_available_skills(self):
        from upsonic.skills.loader import BuiltinSkills

        b = BuiltinSkills()
        available = b.available_skills()
        assert isinstance(available, list)
        assert "code-review" in available
        assert "summarization" in available
        assert "data-analysis" in available

    def test_load_all(self):
        from upsonic.skills.loader import BuiltinSkills

        b = BuiltinSkills()
        skills = b.load()
        assert len(skills) >= 3
        names = {s.name for s in skills}
        assert "code-review" in names

    def test_load_filtered(self):
        from upsonic.skills.loader import BuiltinSkills

        b = BuiltinSkills(skills=["code-review"])
        skills = b.load()
        assert len(skills) == 1
        assert skills[0].name == "code-review"


# ---------------------------------------------------------------------------
# Skills Container
# ---------------------------------------------------------------------------

class TestSkillsContainer:
    """Tests for the Skills container class."""

    def _make_skills(self, *names):
        from upsonic.skills.skill import Skill
        from upsonic.skills.loader import InlineSkills
        from upsonic.skills.skills import Skills

        skill_list = [
            Skill(name=n, description=f"desc-{n}", instructions=f"instr-{n}", source_path="")
            for n in names
        ]
        return Skills(loaders=[InlineSkills(skill_list)])

    def test_load_and_len(self):
        s = self._make_skills("a", "b", "c")
        assert len(s) == 3

    def test_contains(self):
        s = self._make_skills("alpha", "beta")
        assert "alpha" in s
        assert "beta" in s
        assert "gamma" not in s

    def test_get_skill(self):
        s = self._make_skills("x")
        skill = s.get_skill("x")
        assert skill is not None
        assert skill.name == "x"
        assert s.get_skill("nonexistent") is None

    def test_get_all_skills(self):
        s = self._make_skills("a", "b")
        all_skills = s.get_all_skills()
        assert len(all_skills) == 2

    def test_get_skill_names(self):
        s = self._make_skills("x", "y", "z")
        names = s.get_skill_names()
        assert set(names) == {"x", "y", "z"}

    def test_get_tools_returns_callables(self):
        s = self._make_skills("t")
        tools = s.get_tools()
        assert len(tools) >= 4
        for tool in tools:
            assert callable(tool)

    def test_get_tools_includes_asset_tool(self):
        s = self._make_skills("t")
        tool_names = [t.__name__ for t in s.get_tools()]
        assert "get_skill_asset" in tool_names

    def test_get_system_prompt_section(self):
        s = self._make_skills("prompt-test")
        section = s.get_system_prompt_section()
        assert isinstance(section, str)
        assert "prompt-test" in section

    def test_get_metrics(self):
        s = self._make_skills("m")
        metrics = s.get_metrics()
        assert isinstance(metrics, dict)
        assert "m" in metrics

    def test_reload(self):
        s = self._make_skills("r")
        assert len(s) == 1
        s.reload()
        assert len(s) == 1

    def test_callbacks_invoked(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.loader import InlineSkills
        from upsonic.skills.skills import Skills

        loaded_skills = []
        skill = Skill(name="cb", description="d", instructions="body", source_path="")
        s = Skills(
            loaders=[InlineSkills([skill])],
            on_load=lambda name, desc: loaded_skills.append(name),
        )
        # Trigger load via tool
        tools = s.get_tools()
        instr_tool = tools[0]  # get_skill_instructions
        result = instr_tool(skill_name="cb")
        assert "cb" in loaded_skills

    def test_merge(self):
        from upsonic.skills.skills import Skills

        s1 = self._make_skills("a", "b")
        s2 = self._make_skills("b", "c")
        merged = Skills.merge(s1, s2)
        assert "a" in merged
        assert "b" in merged
        assert "c" in merged

    def test_merge_override_order(self):
        from upsonic.skills.skill import Skill
        from upsonic.skills.loader import InlineSkills
        from upsonic.skills.skills import Skills

        s1 = Skills(loaders=[InlineSkills([
            Skill(name="x", description="first", instructions="first-body", source_path=""),
        ])])
        s2 = Skills(loaders=[InlineSkills([
            Skill(name="x", description="second", instructions="second-body", source_path=""),
        ])])
        merged = Skills.merge(s1, s2)
        assert merged.get_skill("x").description == "second"

    def test_caching(self):
        s = self._make_skills("cached")
        s2 = self._make_skills("cached")
        # With cache_ttl, second access should be faster
        from upsonic.skills.skill import Skill
        from upsonic.skills.loader import InlineSkills
        from upsonic.skills.skills import Skills

        skill = Skill(name="c", description="d", instructions="body", source_path="")
        s = Skills(loaders=[InlineSkills([skill])], cache_ttl=60)
        tools = s.get_tools()
        r1 = tools[0](skill_name="c")
        r2 = tools[0](skill_name="c")
        assert r1 == r2

    def test_repr(self):
        s = self._make_skills("repr-test")
        r = repr(s)
        assert "repr-test" in r  # should contain skill name


# ---------------------------------------------------------------------------
# Remote Loader Base
# ---------------------------------------------------------------------------

class TestRemoteSkillLoaderBase:
    """Tests for RemoteSkillLoader attributes."""

    def test_attributes(self):
        from upsonic.skills.loader import RemoteSkillLoader

        # Cannot instantiate abstract class, just check it exists
        assert hasattr(RemoteSkillLoader, "load")
        assert hasattr(RemoteSkillLoader, "_download")
        assert hasattr(RemoteSkillLoader, "_source_key")


# ---------------------------------------------------------------------------
# GitHub/URL/Registry Loader attributes
# ---------------------------------------------------------------------------

class TestGitHubSkillsAttributes:
    def test_init_attributes(self):
        from upsonic.skills.loader import GitHubSkills

        g = GitHubSkills(repo="owner/repo", branch="dev", path="custom/", token="tok")
        assert g.repo == "owner/repo"
        assert g.branch == "dev"
        assert g.path == "custom"  # trailing slash is stripped
        assert g.token == "tok"


class TestURLSkillsAttributes:
    def test_init_attributes(self):
        from upsonic.skills.loader import URLSkills

        u = URLSkills(url="https://example.com/skills.tar.gz", headers={"X-Key": "val"})
        assert u.url == "https://example.com/skills.tar.gz"
        assert u.headers == {"X-Key": "val"}


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Tests that all expected classes are importable."""

    def test_skill_imports(self):
        from upsonic.skills import (
            Skill,
            Skills,
            SkillLoader,
            LocalSkills,
            InlineSkills,
            BuiltinSkills,
            RemoteSkillLoader,
            GitHubSkills,
            URLSkills,
            SkillMetrics,
            SkillError,
            SkillParseError,
            SkillValidationError,
        )
        assert Skill is not None
        assert Skills is not None
        assert SkillLoader is not None
        assert LocalSkills is not None
        assert InlineSkills is not None
        assert BuiltinSkills is not None
        assert RemoteSkillLoader is not None
        assert GitHubSkills is not None
        assert URLSkills is not None
        assert SkillMetrics is not None
        assert SkillError is not None
        assert SkillParseError is not None
        assert SkillValidationError is not None

    def test_exception_hierarchy(self):
        from upsonic.utils.package.exception import (
            SkillError,
            SkillParseError,
            SkillValidationError,
            SkillDownloadError,
            SkillIntegrityError,
            SkillRegistryError,
        )
        assert issubclass(SkillParseError, SkillError)
        assert issubclass(SkillValidationError, SkillError)
        assert issubclass(SkillDownloadError, SkillError)
        assert issubclass(SkillIntegrityError, SkillError)
        assert issubclass(SkillRegistryError, SkillError)
