"""Unit tests for the Skill dataclass."""

import unittest

from upsonic.skills.skill import Skill


class TestSkillCreation(unittest.TestCase):
    def test_minimal_creation(self):
        s = Skill(name="test", description="desc", instructions="do X", source_path="/tmp")
        self.assertEqual(s.name, "test")
        self.assertEqual(s.description, "desc")
        self.assertEqual(s.instructions, "do X")
        self.assertEqual(s.source_path, "/tmp")

    def test_default_fields(self):
        s = Skill(name="s", description="d", instructions="i", source_path="")
        self.assertEqual(s.scripts, [])
        self.assertEqual(s.references, [])
        self.assertEqual(s.assets, [])
        self.assertIsNone(s.metadata)
        self.assertIsNone(s.license)
        self.assertIsNone(s.compatibility)
        self.assertIsNone(s.allowed_tools)
        self.assertIsNone(s.version)
        self.assertEqual(s.dependencies, [])

    def test_all_fields(self):
        s = Skill(
            name="full",
            description="Full skill",
            instructions="Full instructions",
            source_path="/skills/full",
            scripts=["run.sh", "test.py"],
            references=["guide.md", "api.md"],
            assets=["template.html", "logo.png"],
            metadata={"author": "test", "tags": ["a", "b"]},
            license="MIT",
            compatibility="Python 3.8+",
            allowed_tools=["tool1", "tool2"],
            version="2.1.0",
            dependencies=["dep-a", "dep-b"],
        )
        self.assertEqual(s.scripts, ["run.sh", "test.py"])
        self.assertEqual(s.references, ["guide.md", "api.md"])
        self.assertEqual(s.assets, ["template.html", "logo.png"])
        self.assertEqual(s.metadata["author"], "test")
        self.assertEqual(s.license, "MIT")
        self.assertEqual(s.compatibility, "Python 3.8+")
        self.assertEqual(s.allowed_tools, ["tool1", "tool2"])
        self.assertEqual(s.version, "2.1.0")
        self.assertEqual(s.dependencies, ["dep-a", "dep-b"])


class TestSkillSerialization(unittest.TestCase):
    def test_to_dict(self):
        s = Skill(
            name="ser", description="d", instructions="i", source_path="/p",
            scripts=["a.sh"], references=["b.txt"], assets=["t.html"], version="1.0.0",
        )
        d = s.to_dict()
        self.assertEqual(d["name"], "ser")
        self.assertEqual(d["scripts"], ["a.sh"])
        self.assertEqual(d["assets"], ["t.html"])
        self.assertEqual(d["version"], "1.0.0")
        self.assertIn("instructions", d)

    def test_from_dict(self):
        data = {
            "name": "from-dict",
            "description": "from dict",
            "instructions": "instr",
            "source_path": "/p",
            "scripts": ["x.py"],
            "references": ["y.md"],
            "version": "3.0.0",
            "dependencies": ["dep1"],
        }
        s = Skill.from_dict(data)
        self.assertEqual(s.name, "from-dict")
        self.assertEqual(s.version, "3.0.0")
        self.assertEqual(s.dependencies, ["dep1"])

    def test_roundtrip(self):
        original = Skill(
            name="roundtrip", description="d", instructions="i",
            source_path="/p", scripts=["s.sh"], references=["r.txt"],
            metadata={"key": "val"}, version="1.2.3",
            dependencies=["dep1"],
        )
        restored = Skill.from_dict(original.to_dict())
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.version, original.version)
        self.assertEqual(restored.dependencies, original.dependencies)
        self.assertEqual(restored.metadata, original.metadata)

    def test_from_dict_missing_optional_fields(self):
        data = {"name": "min", "description": "d", "instructions": "i", "source_path": ""}
        s = Skill.from_dict(data)
        self.assertEqual(s.scripts, [])
        self.assertIsNone(s.version)

    def test_repr(self):
        s = Skill(name="repr-test", description="d", instructions="i", source_path="")
        r = repr(s)
        self.assertIn("repr-test", r)


if __name__ == "__main__":
    unittest.main()
