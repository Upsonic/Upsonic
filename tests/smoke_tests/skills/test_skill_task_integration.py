"""
Smoke tests for Task + Skills integration.

Tests: Task accepts skills, exposes metrics, serialization roundtrips.
"""

from upsonic.skills.skill import Skill
from upsonic.skills.loader import InlineSkills
from upsonic.skills.skills import Skills


def _make_skill(name="test-skill", description="A test skill"):
    return Skill(
        name=name, description=description,
        instructions=f"Instructions for {name}",
        source_path="", scripts=["run.sh"], references=["ref.txt"],
    )


def _make_skills(*names):
    return Skills(loaders=[InlineSkills(
        [_make_skill(n, f"Description of {n}") for n in names]
    )])


class TestTaskSkillsIntegration:
    def test_task_accepts_skills(self):
        from upsonic import Task
        task = Task(description="Do something", skills=_make_skills("task-skill"))
        assert task.skills is not None

    def test_task_skills_none_by_default(self):
        from upsonic import Task
        assert Task(description="plain").skills is None

    def test_task_get_skill_metrics_empty(self):
        from upsonic import Task
        assert Task(description="no skills").get_skill_metrics() == {}

    def test_task_get_skill_metrics_with_skills(self):
        from upsonic import Task
        task = Task(description="with skills", skills=_make_skills("t-metric"))
        metrics = task.get_skill_metrics()
        assert isinstance(metrics, dict)
        assert "t-metric" in metrics

    def test_task_to_dict_without_serialize(self):
        from upsonic import Task
        task = Task(description="test", skills=_make_skills("ser-test"))
        d = task.to_dict(serialize_flag=False)
        assert "skills" in d
        assert d["skills"] is None

    def test_task_to_dict_with_serialize(self):
        from upsonic import Task
        task = Task(description="test", skills=_make_skills("ser-test"))
        d = task.to_dict(serialize_flag=True)
        assert "skills" in d
        assert d["skills"] is not None

    def test_task_roundtrip_with_serialize(self):
        from upsonic import Task
        task = Task(description="test roundtrip", skills=_make_skills("roundtrip"))
        d = task.to_dict(serialize_flag=True)
        restored = Task.from_dict(d, deserialize_flag=True)
        assert restored.skills is not None
        assert "roundtrip" in restored.skills

    def test_task_from_dict_no_skills(self):
        from upsonic import Task
        task = Task(description="no skills")
        restored = Task.from_dict(task.to_dict())
        assert restored.skills is None


class TestAgentRunOutputSkillMetrics:
    def test_skill_metrics_field_exists(self):
        from upsonic.run.agent.output import AgentRunOutput
        o = AgentRunOutput()
        assert hasattr(o, "skill_metrics")
        assert o.skill_metrics is None

    def test_skill_metrics_to_dict(self):
        from upsonic.run.agent.output import AgentRunOutput
        o = AgentRunOutput()
        o.skill_metrics = {"code-review": {"load_count": 3, "reference_access_count": 1,
                           "script_execution_count": 0, "total_chars_loaded": 500,
                           "last_used_timestamp": 12345.0}}
        d = o.to_dict()
        assert d["skill_metrics"]["code-review"]["load_count"] == 3

    def test_skill_metrics_from_dict(self):
        from upsonic.run.agent.output import AgentRunOutput
        data = {"skill_metrics": {"my-skill": {"load_count": 5, "total_chars_loaded": 999}}}
        o = AgentRunOutput.from_dict(data)
        assert o.skill_metrics["my-skill"]["load_count"] == 5

    def test_skill_metrics_roundtrip(self):
        from upsonic.run.agent.output import AgentRunOutput
        o = AgentRunOutput()
        o.skill_metrics = {
            "a": {"load_count": 1, "reference_access_count": 2, "script_execution_count": 3,
                   "total_chars_loaded": 100, "last_used_timestamp": 99.9},
            "b": {"load_count": 0, "reference_access_count": 0, "script_execution_count": 0,
                   "total_chars_loaded": 0, "last_used_timestamp": None},
        }
        restored = AgentRunOutput.from_dict(o.to_dict())
        assert restored.skill_metrics == o.skill_metrics

    def test_skill_metrics_none_roundtrip(self):
        from upsonic.run.agent.output import AgentRunOutput
        o = AgentRunOutput()
        restored = AgentRunOutput.from_dict(o.to_dict())
        assert restored.skill_metrics is None
