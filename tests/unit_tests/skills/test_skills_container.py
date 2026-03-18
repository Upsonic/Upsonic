"""Unit tests for the Skills container class."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from upsonic.skills.skill import Skill
from upsonic.skills.loader import InlineSkills, LocalSkills, BuiltinSkills
from upsonic.skills.skills import Skills
from upsonic.skills.metrics import SkillMetrics


def _skill(name, deps=None, allowed_tools=None):
    return Skill(
        name=name, description=f"Desc of {name}", instructions=f"Instr for {name}",
        source_path="", scripts=["run.sh"], references=["ref.txt"],
        dependencies=deps or [], allowed_tools=allowed_tools,
    )


def _skills(*names):
    return Skills(loaders=[InlineSkills([_skill(n) for n in names])])


def _make_skill_dir(base, name, with_assets=False):
    skill_dir = Path(base) / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill {name}\n---\nInstructions for {name}"
    )
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "hello.py").write_text("#!/usr/bin/env python3\nprint('hello')")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.txt").write_text("Reference content.")
    if with_assets:
        (skill_dir / "assets").mkdir()
        (skill_dir / "assets" / "template.html").write_text("<html>Template</html>")
    return skill_dir


# ---------------------------------------------------------------------------
# Basic container operations
# ---------------------------------------------------------------------------

class TestSkillsInit(unittest.TestCase):
    def test_loads_skills_on_init(self):
        s = _skills("a", "b")
        self.assertEqual(len(s), 2)

    def test_empty_loaders(self):
        s = Skills(loaders=[InlineSkills([])])
        self.assertEqual(len(s), 0)

    def test_multiple_loaders(self):
        loader1 = InlineSkills([_skill("from-1")])
        loader2 = InlineSkills([_skill("from-2")])
        s = Skills(loaders=[loader1, loader2])
        self.assertIn("from-1", s)
        self.assertIn("from-2", s)

    def test_later_loader_overrides(self):
        s1 = Skill(name="dup", description="first", instructions="i1", source_path="")
        s2 = Skill(name="dup", description="second", instructions="i2", source_path="")
        s = Skills(loaders=[InlineSkills([s1]), InlineSkills([s2])])
        self.assertEqual(s.get_skill("dup").description, "second")


class TestSkillsAccessors(unittest.TestCase):
    def test_get_skill(self):
        s = _skills("find-me")
        skill = s.get_skill("find-me")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "find-me")

    def test_get_skill_not_found(self):
        s = _skills("a")
        self.assertIsNone(s.get_skill("nonexistent"))

    def test_get_all_skills(self):
        s = _skills("x", "y", "z")
        all_skills = s.get_all_skills()
        self.assertEqual(len(all_skills), 3)

    def test_get_skill_names(self):
        s = _skills("alpha", "beta")
        names = s.get_skill_names()
        self.assertIn("alpha", names)
        self.assertIn("beta", names)

    def test_contains(self):
        s = _skills("present")
        self.assertIn("present", s)
        self.assertNotIn("absent", s)

    def test_len(self):
        s = _skills("a", "b", "c")
        self.assertEqual(len(s), 3)

    def test_repr(self):
        s = _skills("r1", "r2")
        r = repr(s)
        self.assertIn("r1", r)
        self.assertIn("r2", r)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestSkillsMetrics(unittest.TestCase):
    def test_metrics_initialized_for_each_skill(self):
        s = _skills("m1", "m2")
        metrics = s.get_metrics()
        self.assertIn("m1", metrics)
        self.assertIn("m2", metrics)
        self.assertIsInstance(metrics["m1"], SkillMetrics)

    def test_metrics_start_at_zero(self):
        s = _skills("zero")
        m = s.get_metrics()["zero"]
        self.assertEqual(m.load_count, 0)
        self.assertEqual(m.reference_access_count, 0)
        self.assertEqual(m.script_execution_count, 0)

    def test_metrics_increment_on_instruction_load(self):
        s = _skills("tracked")
        tools = s.get_tools()
        instr_fn = tools[0]
        instr_fn(skill_name="tracked")
        m = s.get_metrics()["tracked"]
        self.assertEqual(m.load_count, 1)
        self.assertGreater(m.total_chars_loaded, 0)


# ---------------------------------------------------------------------------
# Tool generation
# ---------------------------------------------------------------------------

class TestSkillsGetTools(unittest.TestCase):
    def test_returns_four_tools(self):
        s = _skills("t1")
        tools = s.get_tools()
        self.assertEqual(len(tools), 4)

    def test_tool_names(self):
        s = _skills("t1")
        names = [t.__name__ for t in s.get_tools()]
        self.assertEqual(names, [
            "get_skill_instructions",
            "get_skill_reference",
            "get_skill_script",
            "get_skill_asset",
        ])

    def test_prefix_applied(self):
        s = _skills("t1")
        names = [t.__name__ for t in s.get_tools(prefix="task_")]
        self.assertEqual(names, [
            "task_get_skill_instructions",
            "task_get_skill_reference",
            "task_get_skill_script",
            "task_get_skill_asset",
        ])

    def test_get_skill_instructions_returns_json(self):
        s = _skills("json-test")
        tools = s.get_tools()
        result = json.loads(tools[0](skill_name="json-test"))
        self.assertEqual(result["skill_name"], "json-test")
        self.assertIn("instructions", result)

    def test_get_skill_instructions_not_found(self):
        s = _skills("exists")
        tools = s.get_tools()
        result = json.loads(tools[0](skill_name="nope"))
        self.assertIn("error", result)

    def test_get_skill_reference_with_real_files(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "ref-test")
            s = Skills(loaders=[LocalSkills(d)])
            tools = s.get_tools()
            result = json.loads(tools[1](skill_name="ref-test", reference_path="guide.txt"))
            self.assertIn("content", result)
            self.assertIn("Reference content", result["content"])

    def test_get_skill_reference_not_found(self):
        s = _skills("ref-nf")
        tools = s.get_tools()
        result = json.loads(tools[1](skill_name="ref-nf", reference_path="missing.txt"))
        self.assertIn("error", result)

    def test_get_skill_script_read_mode(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "script-read")
            s = Skills(loaders=[LocalSkills(d)])
            tools = s.get_tools()
            result = json.loads(tools[2](
                skill_name="script-read", script_path="hello.py", execute=False
            ))
            self.assertIn("content", result)

    def test_get_skill_script_execute_mode(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "script-exec")
            s = Skills(loaders=[LocalSkills(d)])
            tools = s.get_tools()
            result = json.loads(tools[2](
                skill_name="script-exec", script_path="hello.py", execute=True
            ))
            self.assertEqual(result["returncode"], 0)
            self.assertIn("hello", result["stdout"])

    def test_get_skill_asset_with_real_files(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "asset-test", with_assets=True)
            s = Skills(loaders=[LocalSkills(d)])
            tools = s.get_tools()
            asset_fn = tools[3]
            result = json.loads(asset_fn(skill_name="asset-test", asset_path="template.html"))
            self.assertIn("content", result)
            self.assertIn("Template", result["content"])

    def test_get_skill_asset_not_found(self):
        s = _skills("asset-nf")
        tools = s.get_tools()
        asset_fn = tools[3]
        result = json.loads(asset_fn(skill_name="asset-nf", asset_path="missing.txt"))
        self.assertIn("error", result)

    def test_get_skill_asset_skill_not_found(self):
        s = _skills("exists")
        tools = s.get_tools()
        asset_fn = tools[3]
        result = json.loads(asset_fn(skill_name="nope", asset_path="any.txt"))
        self.assertIn("error", result)

    def test_tools_are_callable(self):
        s = _skills("callable")
        for tool in s.get_tools():
            self.assertTrue(callable(tool))


# ---------------------------------------------------------------------------
# Knowledge base tool
# ---------------------------------------------------------------------------

class TestSkillsKnowledgeBase(unittest.TestCase):
    def test_no_kb_four_tools(self):
        s = _skills("no-kb")
        self.assertEqual(len(s.get_tools()), 4)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSkillsSystemPrompt(unittest.TestCase):
    def test_contains_skill_names(self):
        s = _skills("prompt-a", "prompt-b")
        section = s.get_system_prompt_section()
        self.assertIn("prompt-a", section)
        self.assertIn("prompt-b", section)

    def test_contains_descriptions(self):
        s = _skills("desc-check")
        section = s.get_system_prompt_section()
        self.assertIn("Desc of desc-check", section)

    def test_empty_skills_returns_string(self):
        s = Skills(loaders=[InlineSkills([])])
        section = s.get_system_prompt_section()
        self.assertIsInstance(section, str)

    def test_contains_skills_xml_tag(self):
        s = _skills("xml-tag")
        section = s.get_system_prompt_section()
        self.assertIn("skills_system", section.lower())


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestSkillsMerge(unittest.TestCase):
    def test_merge_two(self):
        s1 = _skills("from-a")
        s2 = _skills("from-b")
        merged = Skills.merge(s1, s2)
        self.assertIn("from-a", merged)
        self.assertIn("from-b", merged)

    def test_merge_override(self):
        sk1 = Skill(name="dup", description="first", instructions="i1", source_path="")
        sk2 = Skill(name="dup", description="second", instructions="i2", source_path="")
        s1 = Skills(loaders=[InlineSkills([sk1])])
        s2 = Skills(loaders=[InlineSkills([sk2])])
        merged = Skills.merge(s1, s2)
        self.assertEqual(merged.get_skill("dup").description, "second")

    def test_merge_three(self):
        s1 = _skills("a")
        s2 = _skills("b")
        s3 = _skills("c")
        merged = Skills.merge(s1, s2, s3)
        self.assertEqual(len(merged), 3)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

class TestSkillsCopy(unittest.TestCase):
    def test_copy_has_same_skills(self):
        s = _skills("orig")
        c = s.copy()
        self.assertIn("orig", c)

    def test_copy_has_independent_metrics(self):
        s = _skills("indep")
        c = s.copy()
        tools = s.get_tools()
        tools[0](skill_name="indep")
        self.assertEqual(s.get_metrics()["indep"].load_count, 1)
        self.assertEqual(c.get_metrics()["indep"].load_count, 0)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

class TestSkillsReload(unittest.TestCase):
    def test_reload_reloads_skills(self):
        s = _skills("reloadable")
        self.assertIn("reloadable", s)
        s.reload()
        self.assertIn("reloadable", s)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestSkillsCallbacks(unittest.TestCase):
    def test_on_load_callback(self):
        called = []
        s = Skills(
            loaders=[InlineSkills([_skill("cb-load")])],
            on_load=lambda name, desc: called.append((name, desc)),
        )
        tools = s.get_tools()
        tools[0](skill_name="cb-load")
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][0], "cb-load")

    def test_on_reference_callback(self):
        called = []
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "cb-ref")
            s = Skills(
                loaders=[LocalSkills(d)],
                on_reference_access=lambda name, path: called.append((name, path)),
            )
            tools = s.get_tools()
            tools[1](skill_name="cb-ref", reference_path="guide.txt")
            self.assertEqual(len(called), 1)
            self.assertEqual(called[0][0], "cb-ref")
            self.assertEqual(called[0][1], "guide.txt")

    def test_on_script_callback(self):
        called = []
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "cb-script")
            s = Skills(
                loaders=[LocalSkills(d)],
                on_script_execute=lambda name, path, rc: called.append((name, path, rc)),
            )
            tools = s.get_tools()
            tools[2](skill_name="cb-script", script_path="hello.py", execute=True)
            self.assertEqual(len(called), 1)
            self.assertEqual(called[0][0], "cb-script")
            self.assertEqual(called[0][2], 0)

    def test_callback_error_does_not_crash(self):
        def bad_callback(*args):
            raise RuntimeError("callback boom")

        s = Skills(
            loaders=[InlineSkills([_skill("cb-err")])],
            on_load=bad_callback,
        )
        tools = s.get_tools()
        # Should not raise
        result = tools[0](skill_name="cb-err")
        self.assertIn("cb-err", result)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestSkillsCaching(unittest.TestCase):
    def test_cache_enabled_with_ttl(self):
        s = Skills(loaders=[InlineSkills([_skill("cached")])], cache_ttl=300)
        self.assertIsNotNone(s._cache)

    def test_cache_disabled_without_ttl(self):
        s = _skills("no-cache")
        self.assertIsNone(s._cache)

    def test_cached_instructions_same_result(self):
        s = Skills(loaders=[InlineSkills([_skill("cache-hit")])], cache_ttl=300)
        tools = s.get_tools()
        r1 = tools[0](skill_name="cache-hit")
        r2 = tools[0](skill_name="cache-hit")
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# Tool binding (allowed_tools)
# ---------------------------------------------------------------------------

class TestSkillsToolBinding(unittest.TestCase):
    def test_active_skill_tools_empty_initially(self):
        s = _skills("binding")
        self.assertEqual(s.get_active_skill_tools(), set())

    def test_active_skill_tools_after_load(self):
        sk = _skill("bound", allowed_tools=["tool-a", "tool-b"])
        s = Skills(loaders=[InlineSkills([sk])])
        tools = s.get_tools()
        tools[0](skill_name="bound")
        active = s.get_active_skill_tools()
        self.assertIn("tool-a", active)
        self.assertIn("tool-b", active)

    def test_active_skill_tools_without_allowed_tools(self):
        s = _skills("no-tools")
        tools = s.get_tools()
        tools[0](skill_name="no-tools")
        self.assertEqual(s.get_active_skill_tools(), set())


# ---------------------------------------------------------------------------
# Dependency handling in Skills
# ---------------------------------------------------------------------------

class TestSkillsDependencyHandling(unittest.TestCase):
    def test_strict_deps_raises_on_cycle(self):
        from upsonic.utils.package.exception import SkillValidationError

        sk_a = _skill("a", deps=["b"])
        sk_b = _skill("b", deps=["a"])
        with self.assertRaises(SkillValidationError):
            Skills(loaders=[InlineSkills([sk_a, sk_b])], strict_deps=True)

    def test_non_strict_deps_warns_on_cycle(self):
        sk_a = _skill("a", deps=["b"])
        sk_b = _skill("b", deps=["a"])
        # Should not raise
        s = Skills(loaders=[InlineSkills([sk_a, sk_b])], strict_deps=False)
        self.assertIn("a", s)
        self.assertIn("b", s)

    def test_missing_deps_logged_not_raised(self):
        sk = _skill("lonely", deps=["missing-dep"])
        s = Skills(loaders=[InlineSkills([sk])])
        self.assertIn("lonely", s)


# ---------------------------------------------------------------------------
# Safety / Policy
# ---------------------------------------------------------------------------

class TestSkillsSafetyPolicy(unittest.TestCase):
    def test_no_policy_returns_content(self):
        s = _skills("safe")
        tools = s.get_tools()
        result = json.loads(tools[0](skill_name="safe"))
        self.assertNotIn("error", result)

    def test_policy_blocks_content(self):
        mock_result = MagicMock()
        mock_result.confidence = 0.95
        mock_result.details = "Content blocked by policy"
        mock_policy = MagicMock()
        mock_policy.check.return_value = mock_result
        s = Skills(
            loaders=[InlineSkills([_skill("blocked")])],
            policy=mock_policy,
        )
        tools = s.get_tools()
        result = json.loads(tools[0](skill_name="blocked"))
        self.assertIn("error", result)
        self.assertIn("blocked", result["error"].lower())


# ---------------------------------------------------------------------------
# Multiple loaders with builtins
# ---------------------------------------------------------------------------

class TestSkillsMultipleLoadersWithBuiltins(unittest.TestCase):
    def test_inline_plus_builtin(self):
        s = Skills(loaders=[
            InlineSkills([_skill("custom")]),
            BuiltinSkills(skills=["code-review"]),
        ])
        self.assertIn("custom", s)
        self.assertIn("code-review", s)

    def test_inline_plus_local(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "local-skill")
            s = Skills(loaders=[
                InlineSkills([_skill("inline-skill")]),
                LocalSkills(d),
            ])
            self.assertIn("inline-skill", s)
            self.assertIn("local-skill", s)

    def test_builtin_plus_local(self):
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-local")
            s = Skills(loaders=[
                BuiltinSkills(skills=["summarization"]),
                LocalSkills(d),
            ])
            self.assertIn("summarization", s)
            self.assertIn("my-local", s)


if __name__ == "__main__":
    unittest.main()
