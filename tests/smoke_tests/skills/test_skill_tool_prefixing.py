"""
Smoke tests for skill tool name prefixing.

Tests: Agent vs task skill tools have distinct prefixed names,
prefixed tools still work, no duplicate definitions when combined.
"""

import json
import os

import pytest

from upsonic.skills.skill import Skill
from upsonic.skills.loader import InlineSkills
from upsonic.skills.skills import Skills


requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real LLM tests",
)


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


class TestSkillToolNamePrefixing:
    def test_default_tools_have_no_prefix(self):
        skills = _make_skills("s1")
        tool_names = [t.__name__ for t in skills.get_tools()]
        assert tool_names == [
            "get_skill_instructions",
            "get_skill_reference",
            "get_skill_script",
        ]

    def test_prefixed_tools_have_prefix(self):
        skills = _make_skills("s1")
        tool_names = [t.__name__ for t in skills.get_tools(prefix="task_")]
        assert tool_names == [
            "task_get_skill_instructions",
            "task_get_skill_reference",
            "task_get_skill_script",
        ]

    def test_prefixed_tools_still_work(self):
        skills = _make_skills("prefixed-call")
        tools = skills.get_tools(prefix="task_")
        instr_fn = tools[0]
        result = instr_fn(skill_name="prefixed-call")
        parsed = json.loads(result)
        assert parsed["skill_name"] == "prefixed-call"
        assert "instructions" in parsed

    def test_agent_and_task_skill_tools_have_different_names(self):
        agent_skills = _make_skills("agent-s")
        task_skills = _make_skills("task-s")
        agent_tool_names = [t.__name__ for t in agent_skills.get_tools()]
        task_tool_names = [t.__name__ for t in task_skills.get_tools(prefix="task_")]
        assert agent_tool_names != task_tool_names
        for a_name, t_name in zip(agent_tool_names, task_tool_names):
            assert t_name == f"task_{a_name}"

    def test_agent_and_task_skill_tools_are_different_objects(self):
        agent_skills = _make_skills("agent-s")
        task_skills = _make_skills("task-s")
        agent_tools = agent_skills.get_tools()
        task_tools = task_skills.get_tools(prefix="task_")
        for at, tt in zip(agent_tools, task_tools):
            assert at is not tt

    @requires_anthropic
    def test_combined_tool_definitions_no_duplicates(self):
        from upsonic import Agent, Task

        agent_skills = _make_skills("agent-dup")
        task_skills = _make_skills("task-dup")
        agent = Agent("anthropic/claude-sonnet-4-5", skills=agent_skills)
        task = Task(description="Say 'ok'.", skills=task_skills)

        agent._setup_task_tools(task)
        agent.current_task = task
        combined_defs = agent._get_combined_tool_definitions()
        combined_names = [d.name for d in combined_defs]

        assert "get_skill_instructions" in combined_names
        assert "task_get_skill_instructions" in combined_names
        assert combined_names.count("get_skill_instructions") == 1
        assert combined_names.count("task_get_skill_instructions") == 1

    @requires_anthropic
    def test_resolve_tool_manager_agent_tools(self):
        from upsonic import Agent, Task

        agent_skills = _make_skills("agent-resolve")
        task_skills = _make_skills("task-resolve")
        agent = Agent("anthropic/claude-sonnet-4-5", skills=agent_skills)
        task = Task(description="Say 'ok'.", skills=task_skills)

        agent._setup_task_tools(task)
        agent.current_task = task
        resolved = agent._resolve_tool_manager("get_skill_instructions")
        assert resolved is agent.tool_manager

    @requires_anthropic
    def test_resolve_tool_manager_task_tools(self):
        from upsonic import Agent, Task

        agent_skills = _make_skills("agent-resolve")
        task_skills = _make_skills("task-resolve")
        agent = Agent("anthropic/claude-sonnet-4-5", skills=agent_skills)
        task = Task(description="Say 'ok'.", skills=task_skills)

        agent._setup_task_tools(task)
        agent.current_task = task
        resolved = agent._resolve_tool_manager("task_get_skill_instructions")
        assert resolved is task.tool_manager
