"""
Smoke tests for GitHubSkills loader — load skills from a public GitHub repository.

Tests: downloading, parsing, system prompt injection, tool registration,
skill metrics, and agent execution with GitHub-sourced skills.

Uses the public Anthropic skills repo: https://github.com/anthropics/skills
"""

import os

import pytest

from upsonic.skills.loader import GitHubSkills
from upsonic.skills.skills import Skills


requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

REPO = "anthropics/skills"
BRANCH = "main"
PATH = "skills/"


class TestGitHubSkillsLoading:
    """Verify GitHubSkills downloads, parses, and exposes skills correctly."""

    def test_load_specific_skills_from_github(self):
        """Load two specific skills and verify they are present."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api", "pdf"])
        skills = Skills(loaders=[loader])

        assert len(skills) >= 2
        assert "claude-api" in skills
        assert "pdf" in skills

    def test_skill_has_instructions(self):
        """Each loaded skill must have non-empty instructions from SKILL.md."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api"])
        skills = Skills(loaders=[loader])

        skill = skills.get_skill("claude-api")
        assert skill is not None
        assert skill.instructions is not None
        assert len(skill.instructions) > 50, "Instructions should be substantial"

    def test_skill_has_name_and_description(self):
        """Parsed skill metadata must include name and description."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["pdf"])
        skills = Skills(loaders=[loader])

        skill = skills.get_skill("pdf")
        assert skill is not None
        assert skill.name == "pdf"
        assert skill.description is not None
        assert len(skill.description) > 0

    def test_system_prompt_contains_skill_info(self):
        """System prompt section must reference loaded GitHub skills."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api", "pdf"])
        skills = Skills(loaders=[loader])

        prompt = skills.get_system_prompt_section()
        assert "claude-api" in prompt
        assert "pdf" in prompt

    def test_tools_registered(self):
        """Skills container must expose callable skill tools."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api"])
        skills = Skills(loaders=[loader])

        tools = skills.get_tools()
        assert len(tools) >= 4, f"Expected >=4 skill tools, got {len(tools)}"

        tool_names = [t.__name__ for t in tools]
        for expected in ("get_skill_instructions", "get_skill_reference", "get_skill_script", "get_skill_asset"):
            assert any(expected in n for n in tool_names), f"{expected} tool not found in {tool_names}"

    def test_metrics_initialized(self):
        """Skill metrics must be initialized for each loaded skill."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api", "pdf"])
        skills = Skills(loaders=[loader])

        metrics = skills.get_metrics()
        assert "claude-api" in metrics
        assert "pdf" in metrics

    def test_get_instructions_tool_returns_content(self):
        """Calling the get_skill_instructions tool should return actual content."""
        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api"])
        skills = Skills(loaders=[loader])

        tools = skills.get_tools()
        instr_tool = next(t for t in tools if "get_skill_instructions" in t.__name__)
        result = instr_tool(skill_name="claude-api")
        assert result is not None
        assert len(result) > 50
        assert "claude" in result.lower() or "api" in result.lower()


class TestGitHubSkillsAgentIntegration:
    """Verify Agent works end-to-end with GitHub-loaded skills."""

    @requires_anthropic
    def test_agent_registers_github_skill_tools(self):
        """Agent with GitHub skills must have skill tools registered."""
        from upsonic import Agent

        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api"])
        skills = Skills(loaders=[loader])
        agent = Agent("anthropic/claude-sonnet-4-5", name="GH Skill Agent", skills=skills)

        tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in agent.tools]
        assert any("get_skill_instructions" in n for n in tool_names)
        assert any("get_skill_reference" in n for n in tool_names)

        registered = list(agent.registered_agent_tools.keys())
        assert any("get_skill_instructions" in n for n in registered)

    @requires_anthropic
    def test_agent_system_prompt_has_github_skills(self):
        """Agent system prompt must mention loaded GitHub skills after execution."""
        from upsonic import Agent, Task

        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api"])
        skills = Skills(loaders=[loader])
        agent = Agent(
            "anthropic/claude-sonnet-4-5",
            name="GH Prompt Agent",
            skills=skills,
            system_prompt="You are a helpful assistant.",
        )

        task = Task(description="What is 1 + 1? Answer with just the number.")
        output = agent.do(task, return_output=True)

        assert output is not None
        assert "You are a helpful assistant" in agent.system_prompt
        assert "claude-api" in agent.system_prompt

    @requires_anthropic
    def test_agent_do_with_github_skills_returns_output_and_metrics(self):
        """Full pipeline: agent.do with GitHub skills returns output and skill metrics."""
        from upsonic import Agent, Task

        loader = GitHubSkills(repo=REPO, branch=BRANCH, path=PATH, skills=["claude-api"])
        skills = Skills(loaders=[loader])
        agent = Agent("anthropic/claude-sonnet-4-5", name="GH Pipeline Agent", skills=skills)

        task = Task(description="Say hello. Answer in one word.")
        output = agent.do(task, return_output=True)

        assert output is not None
        assert output.output is not None
        assert hasattr(output, "skill_metrics")
        assert output.skill_metrics is not None
        assert "claude-api" in output.skill_metrics
        assert isinstance(output.tool_call_count, int)
        assert output.tool_call_count >= 0
