"""
Smoke tests for Agent + Skills full pipeline integration.

Tests: Agent accepts skills, registers skill tools, exposes metrics,
injects skills into system prompt, tool registration, and full
agent.do(task) pipeline with skills — including advanced verification
of tool registration, system prompt injection, and skill metrics tracking.
"""

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


# ---------------------------------------------------------------------------
# Agent + Skills basics
# ---------------------------------------------------------------------------

class TestAgentSkillsIntegration:
    def test_agent_accepts_skills(self):
        from upsonic import Agent
        skills = _make_skills("s1", "s2")
        agent = Agent("anthropic/claude-sonnet-4-5", skills=skills)
        assert agent.skills is skills

    def test_agent_registers_skill_tools(self):
        from upsonic import Agent
        skills = _make_skills("tool-test")
        agent = Agent("anthropic/claude-sonnet-4-5", skills=skills)
        tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in agent.tools]
        assert any("skill" in name.lower() for name in tool_names)

    def test_agent_get_skill_metrics_empty(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5")
        assert agent.get_skill_metrics() == {}

    def test_agent_get_skill_metrics_with_skills(self):
        from upsonic import Agent
        skills = _make_skills("metrics-test")
        agent = Agent("anthropic/claude-sonnet-4-5", skills=skills)
        metrics = agent.get_skill_metrics()
        assert isinstance(metrics, dict)
        assert "metrics-test" in metrics

    def test_agent_skills_none_by_default(self):
        from upsonic import Agent
        assert Agent("anthropic/claude-sonnet-4-5").skills is None


# ---------------------------------------------------------------------------
# Full pipeline: agent.do(task) with skills — ADVANCED VERIFICATION
# ---------------------------------------------------------------------------

class TestAgentSkillsFullPipeline:
    @requires_anthropic
    def test_agent_do_with_skills_verifies_output_and_metrics(self):
        """Full pipeline: verify AgentRunOutput structure and skill metrics."""
        from upsonic import Agent, Task
        skills = _make_skills("math-helper")
        agent = Agent("anthropic/claude-sonnet-4-5", name="Skilled Agent", skills=skills)
        task = Task(description="What is 2 + 2? Answer with just the number.")

        # Use return_output=True to get full AgentRunOutput
        output = agent.do(task, return_output=True)

        # 1. Basic output verification
        assert output is not None
        assert output.output is not None
        assert "4" in str(output.output)

        # 2. AgentRunOutput has expected attributes
        assert hasattr(output, "tools")
        assert hasattr(output, "tool_call_count")
        assert hasattr(output, "events")
        assert hasattr(output, "skill_metrics")

        # 3. Tool call tracking is populated as integer
        assert isinstance(output.tool_call_count, int)
        assert output.tool_call_count >= 0

        # 4. If tools were called, verify ToolExecution structure
        if output.tools:
            for tool_exec in output.tools:
                assert hasattr(tool_exec, "tool_name")
                assert hasattr(tool_exec, "tool_args")
                assert hasattr(tool_exec, "result")
                assert tool_exec.tool_name is not None

        # 5. Skill metrics on output match agent skills
        assert output.skill_metrics is not None
        assert isinstance(output.skill_metrics, dict)
        assert "math-helper" in output.skill_metrics
        m = output.skill_metrics["math-helper"]
        assert "load_count" in m
        assert "reference_access_count" in m
        assert "script_execution_count" in m

    @requires_anthropic
    def test_agent_do_verifies_system_prompt_injection(self):
        """Verify skills are injected into agent system prompt after execution."""
        from upsonic import Agent, Task
        skills = _make_skills("prompt-verify-skill")
        agent = Agent(
            "anthropic/claude-sonnet-4-5", name="Prompt Verify Agent",
            skills=skills, system_prompt="You are a helpful assistant.",
        )

        # Before execution: system prompt is the custom one
        assert agent.system_prompt == "You are a helpful assistant."

        agent.do(Task(description="Say 'ok'."))

        # After execution: system prompt contains both custom and skill info
        assert "You are a helpful assistant" in agent.system_prompt
        assert "prompt-verify-skill" in agent.system_prompt
        assert "Description of prompt-verify-skill" in agent.system_prompt

    @requires_anthropic
    def test_agent_do_verifies_tool_registration(self):
        """Verify skill tools are properly registered on the agent."""
        from upsonic import Agent, Task
        skills = _make_skills("reg-verify")
        agent = Agent("anthropic/claude-sonnet-4-5", name="Reg Verify Agent", skills=skills)

        # Verify tool registration before execution
        registered_names = list(agent.registered_agent_tools.keys())
        for expected_tool in ("get_skill_instructions", "get_skill_reference", "get_skill_script", "get_skill_asset"):
            assert any(expected_tool in n for n in registered_names), \
                f"{expected_tool} not found in registered tools: {registered_names}"

        # Verify tool definitions include skill tools
        def_names = [d.name for d in agent.tool_manager.get_tool_definitions()]
        assert "get_skill_instructions" in def_names

        # Execute and verify output
        output = agent.do(
            Task(description="Say 'ok'."),
            return_output=True,
        )
        assert output is not None
        assert output.output is not None

    @requires_anthropic
    def test_agent_do_verifies_skill_metrics_tracking(self):
        """Verify skill metrics are tracked before and after execution."""
        from upsonic import Agent, Task
        skills = _make_skills("metrics-track")
        agent = Agent("anthropic/claude-sonnet-4-5", name="Metrics Agent", skills=skills)

        # Metrics should exist before execution
        metrics_before = agent.get_skill_metrics()
        assert "metrics-track" in metrics_before
        m = metrics_before["metrics-track"]
        assert "load_count" in m

        # Execute
        output = agent.do(
            Task(description="Say 'ok'."),
            return_output=True,
        )

        # Metrics should still exist after execution
        metrics_after = agent.get_skill_metrics()
        assert "metrics-track" in metrics_after

        # skill_metrics on output should match
        assert output.skill_metrics is not None
        assert "metrics-track" in output.skill_metrics

    @requires_anthropic
    def test_agent_do_with_both_agent_and_task_skills_verified(self):
        """Verify merged agent+task skills appear in output and system prompt."""
        from upsonic import Agent, Task
        agent_skills = _make_skills("agent-both")
        task_skills = _make_skills("task-both")
        agent = Agent("anthropic/claude-sonnet-4-5", name="Both Agent", skills=agent_skills)
        task = Task(description="What is 1+1? Just the number.", skills=task_skills)

        output = agent.do(task, return_output=True)

        assert output is not None
        assert "2" in str(output.output)

        # System prompt should contain both skill sets after execution
        assert "agent-both" in agent.system_prompt
        assert "task-both" in agent.system_prompt

        # Skill metrics should contain both
        assert output.skill_metrics is not None
        assert "agent-both" in output.skill_metrics
        assert "task-both" in output.skill_metrics

    @requires_anthropic
    def test_task_with_skills_executed_by_plain_agent_verified(self):
        """Verify task-level skills are properly injected into a plain agent."""
        from upsonic import Agent, Task
        skills = _make_skills("task-level-skill")
        agent = Agent("anthropic/claude-sonnet-4-5", name="Plain Agent")
        assert agent.skills is None  # No agent-level skills

        task = Task(description="What is 3 * 7? Answer with just the number.", skills=skills)
        output = agent.do(task, return_output=True)

        assert output is not None
        assert "21" in str(output.output)

        # Task-level skills should be injected into system prompt
        assert "task-level-skill" in agent.system_prompt

        # Skill metrics should reflect task-level skills
        assert output.skill_metrics is not None
        assert "task-level-skill" in output.skill_metrics

    @requires_anthropic
    def test_agent_do_string_task_with_skills_verified(self):
        """Verify string task (not Task object) works with skills."""
        from upsonic import Agent
        skills = _make_skills("string-task-skill")
        agent = Agent("anthropic/claude-sonnet-4-5", name="String Agent", skills=skills)
        result = agent.do("What is the capital of France? Answer with just the city name.")
        assert result is not None
        assert "Paris" in str(result)

        # Verify system prompt injection happened
        assert "string-task-skill" in agent.system_prompt


# ---------------------------------------------------------------------------
# System Prompt Integration
# ---------------------------------------------------------------------------

class TestSystemPromptSkills:
    def test_system_prompt_contains_skill_info(self):
        skills = _make_skills("prompt-skill")
        section = skills.get_system_prompt_section()
        assert "prompt-skill" in section
        assert "Description of prompt-skill" in section

    def test_system_prompt_multiple_skills(self):
        skills = _make_skills("alpha", "beta", "gamma")
        section = skills.get_system_prompt_section()
        for name in ("alpha", "beta", "gamma"):
            assert name in section

    def test_system_prompt_empty_skills(self):
        skills = Skills(loaders=[InlineSkills([])])
        assert isinstance(skills.get_system_prompt_section(), str)


class TestSystemPromptInjection:
    def test_agent_system_prompt_before_execution(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5", skills=_make_skills("prompt-inject"),
                      system_prompt="Base prompt.")
        assert agent.system_prompt == "Base prompt."

    def test_agent_system_prompt_none_without_custom(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5", skills=_make_skills("no-custom-prompt"))
        assert agent.system_prompt is None

    @requires_anthropic
    def test_agent_system_prompt_contains_skills_after_execution(self):
        from upsonic import Agent, Task
        agent = Agent("anthropic/claude-sonnet-4-5", name="Prompt Agent",
                      skills=_make_skills("injected-skill"))
        agent.do(Task(description="Say 'ok'."))
        assert "injected-skill" in agent.system_prompt

    @requires_anthropic
    def test_agent_system_prompt_contains_custom_and_skills(self):
        from upsonic import Agent, Task
        agent = Agent("anthropic/claude-sonnet-4-5", name="Custom Prompt Agent",
                      skills=_make_skills("dual-prompt-skill"),
                      system_prompt="You are a helpful math tutor.")
        agent.do(Task(description="Say 'ok'."))
        assert "You are a helpful math tutor" in agent.system_prompt
        assert "dual-prompt-skill" in agent.system_prompt

    @requires_anthropic
    def test_task_skills_in_system_prompt(self):
        from upsonic import Agent, Task
        agent = Agent("anthropic/claude-sonnet-4-5", name="Task Prompt Agent")
        task = Task(description="Say 'ok'.", skills=_make_skills("task-prompt-skill"))
        agent.do(task)
        assert "task-prompt-skill" in agent.system_prompt

    @requires_anthropic
    def test_merged_skills_in_system_prompt(self):
        from upsonic import Agent, Task
        agent = Agent("anthropic/claude-sonnet-4-5", name="Merged Prompt Agent",
                      skills=_make_skills("agent-prompt-skill"))
        task = Task(description="Say 'ok'.", skills=_make_skills("task-prompt-skill"))
        agent.do(task)
        assert "agent-prompt-skill" in agent.system_prompt
        assert "task-prompt-skill" in agent.system_prompt

    def test_skills_system_prompt_section_format(self):
        section = _make_skills("format-skill").get_system_prompt_section()
        assert "format-skill" in section
        assert "Description of format-skill" in section
        assert "skills" in section.lower()


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_agent_tools_contain_skill_tools(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5", skills=_make_skills("reg-test"))
        tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in agent.tools]
        for expected in ("get_skill_instructions", "get_skill_reference", "get_skill_script", "get_skill_asset"):
            assert any(expected in n for n in tool_names)

    def test_agent_registered_tools_contain_skill_tools(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5", skills=_make_skills("registered-check"))
        assert any("get_skill_instructions" in n for n in agent.registered_agent_tools.keys())

    def test_agent_tool_definitions_include_skill_tools(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5", skills=_make_skills("def-check"))
        def_names = [d.name for d in agent.tool_manager.get_tool_definitions()]
        assert "get_skill_instructions" in def_names

    def test_skill_tools_are_callable(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5", skills=_make_skills("callable-test"))
        skill_tools = [t for t in agent.tools if hasattr(t, '__name__') and 'skill' in t.__name__.lower()]
        assert len(skill_tools) >= 4
        for tool in skill_tools:
            assert callable(tool)

    def test_skill_tools_count(self):
        tools = _make_skills("count-test").get_tools()
        assert len(tools) == 4

    def test_no_skill_tools_without_skills(self):
        from upsonic import Agent
        agent = Agent("anthropic/claude-sonnet-4-5")
        tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in agent.tools]
        assert not any("skill" in n.lower() for n in tool_names)
