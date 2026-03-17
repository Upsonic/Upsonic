"""
Smoke tests for Team + Skills full pipeline integration.

Tests: Team accepts skills, propagates to agents, merges with existing,
handles nested teams, and full team.do(tasks) pipeline with skills —
including advanced verification of skill propagation, tool registration,
system prompt injection, and metrics tracking.
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
# Team + Skills basics
# ---------------------------------------------------------------------------

class TestTeamSkillsIntegration:
    def test_team_accepts_skills(self):
        from upsonic import Agent, Team
        agent = Agent("anthropic/claude-sonnet-4-5", name="A1")
        team = Team(entities=[agent], skills=_make_skills("team-skill"))
        assert team.skills is not None

    def test_team_propagates_skills_to_agent(self):
        from upsonic import Agent, Team
        agent = Agent("anthropic/claude-sonnet-4-5", name="A1")
        assert agent.skills is None
        Team(entities=[agent], skills=_make_skills("propagated"))
        assert agent.skills is not None
        assert "propagated" in agent.skills

    def test_team_merges_with_existing_agent_skills(self):
        from upsonic import Agent, Team
        agent = Agent("anthropic/claude-sonnet-4-5", name="A1", skills=_make_skills("agent-s"))
        Team(entities=[agent], skills=_make_skills("team-s"))
        assert "team-s" in agent.skills
        assert "agent-s" in agent.skills

    def test_team_no_skills(self):
        from upsonic import Agent, Team
        agent = Agent("anthropic/claude-sonnet-4-5", name="A1")
        Team(entities=[agent])
        assert agent.skills is None

    def test_nested_team_propagation(self):
        from upsonic import Agent, Team
        inner_agent = Agent("anthropic/claude-sonnet-4-5", name="Inner")
        inner_team = Team(entities=[inner_agent])
        Team(entities=[inner_team], skills=_make_skills("nested-skill"))
        assert inner_agent.skills is not None
        assert "nested-skill" in inner_agent.skills

    def test_team_propagation_verifies_tool_registration(self):
        """Verify propagated skills register tools on the agent."""
        from upsonic import Agent, Team
        agent = Agent("anthropic/claude-sonnet-4-5", name="ToolCheck")
        assert agent.skills is None

        Team(entities=[agent], skills=_make_skills("tool-prop"))

        # Skills propagated — verify tools are registered
        tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in agent.tools]
        for expected in ("get_skill_instructions", "get_skill_reference", "get_skill_script", "get_skill_asset"):
            assert any(expected in n for n in tool_names), \
                f"{expected} not registered after team propagation"

    def test_team_propagation_verifies_metrics(self):
        """Verify propagated skills have metrics accessible on the agent."""
        from upsonic import Agent, Team
        agent = Agent("anthropic/claude-sonnet-4-5", name="MetricsCheck")
        Team(entities=[agent], skills=_make_skills("metrics-prop"))

        metrics = agent.get_skill_metrics()
        assert isinstance(metrics, dict)
        assert "metrics-prop" in metrics
        assert "load_count" in metrics["metrics-prop"]


# ---------------------------------------------------------------------------
# Full pipeline: team.do(tasks) with skills — ADVANCED VERIFICATION
# ---------------------------------------------------------------------------

class TestTeamSkillsFullPipeline:
    @requires_anthropic
    def test_team_do_with_skills_verifies_propagation(self):
        """Verify skills are propagated and accessible after team.do()."""
        from upsonic import Agent, Task, Team
        skills = _make_skills("team-exec-skill")
        agent = Agent(
            model="anthropic/claude-sonnet-4-5", name="Worker",
            role="General Assistant", goal="Answer questions accurately",
        )
        team = Team(
            agents=[agent], skills=skills,
            model="anthropic/claude-sonnet-4-5",
        )

        # Verify skills propagated to agent before execution
        assert agent.skills is not None
        assert "team-exec-skill" in agent.skills

        # Verify tool registration
        tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in agent.tools]
        assert any("get_skill_instructions" in n for n in tool_names)

        task = Task(description="What is 10 + 5? Answer with just the number.")
        result = team.do(tasks=task)
        assert result is not None
        assert "15" in str(result)

        # After execution: verify system prompt injection
        assert agent.system_prompt is not None
        assert "team-exec-skill" in agent.system_prompt

        # Verify metrics are available
        metrics = agent.get_skill_metrics()
        assert "team-exec-skill" in metrics

    @requires_anthropic
    def test_team_skills_propagated_during_execution_verified(self):
        """Verify full propagation chain: None → skills → tools → prompt."""
        from upsonic import Agent, Task, Team
        skills = _make_skills("propagated-exec")
        agent = Agent(
            model="anthropic/claude-sonnet-4-5", name="Executor",
            role="Task Executor", goal="Execute tasks",
        )

        # Before team creation: no skills
        assert agent.skills is None

        team = Team(
            agents=[agent], skills=skills,
            model="anthropic/claude-sonnet-4-5",
        )

        # After team creation: skills propagated
        assert agent.skills is not None
        assert "propagated-exec" in agent.skills

        # Verify registered tools include skill tools
        registered_names = list(agent.registered_agent_tools.keys())
        assert any("get_skill_instructions" in n for n in registered_names)

        result = team.do(tasks=Task(description="Say 'acknowledged'."))
        assert result is not None

        # After execution: verify prompt contains skill
        assert "propagated-exec" in agent.system_prompt

    @requires_anthropic
    def test_team_sequential_mode_verifies_all_agents_have_skills(self):
        """Sequential mode: verify all agents receive skills and metrics."""
        from upsonic import Agent, Task, Team
        skills = _make_skills("seq-skill")
        researcher = Agent(
            model="anthropic/claude-sonnet-4-5", name="Researcher",
            role="Research Specialist", goal="Find accurate information",
        )
        writer = Agent(
            model="anthropic/claude-sonnet-4-5", name="Writer",
            role="Content Writer", goal="Create clear content",
        )
        team = Team(
            entities=[researcher, writer],
            skills=skills,
            mode="sequential",
        )

        # Verify both agents got skills
        assert researcher.skills is not None
        assert writer.skills is not None
        assert "seq-skill" in researcher.skills
        assert "seq-skill" in writer.skills

        # Verify both agents have skill tools registered
        for ag in (researcher, writer):
            tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in ag.tools]
            assert any("get_skill_instructions" in n for n in tool_names), \
                f"Agent {ag.name} missing skill tools"

        tasks = [
            Task(description="What is 9 - 4? Answer with just the number."),
        ]
        result = team.do(tasks=tasks)
        assert result is not None
        assert "5" in str(result)

        # After execution: verify at least one agent has skills in system prompt
        # (In sequential mode, not all agents may execute for a single task)
        executed_agents = [ag for ag in (researcher, writer) if ag.system_prompt is not None]
        assert len(executed_agents) > 0, "At least one agent should have executed"
        for ag in executed_agents:
            assert "seq-skill" in ag.system_prompt

    @requires_anthropic
    def test_team_coordinate_mode_with_skills_verified(self):
        """Coordinate mode: verify skill propagation and execution."""
        from upsonic import Agent, Task, Team
        skills = _make_skills("coord-skill")
        agent = Agent(
            model="anthropic/claude-sonnet-4-5", name="CoordWorker",
            role="General Assistant", goal="Answer questions accurately",
        )
        team = Team(
            agents=[agent], skills=skills, mode="coordinate",
            model="anthropic/claude-sonnet-4-5",
        )

        assert "coord-skill" in agent.skills

        task = Task(description="What is 7 + 3? Answer with just the number.")
        result = team.do(tasks=task)
        assert result is not None
        assert "10" in str(result)

        # Verify post-execution state
        assert "coord-skill" in agent.system_prompt
        metrics = agent.get_skill_metrics()
        assert "coord-skill" in metrics

    @requires_anthropic
    def test_team_route_mode_with_skills_verified(self):
        """Route mode: verify skill propagation and execution."""
        from upsonic import Agent, Task, Team
        skills = _make_skills("route-skill")
        agent = Agent(
            model="anthropic/claude-sonnet-4-5", name="RouteWorker",
            role="General Assistant", goal="Answer questions accurately",
        )
        team = Team(
            agents=[agent], skills=skills, mode="route",
            model="anthropic/claude-sonnet-4-5",
        )

        assert "route-skill" in agent.skills

        task = Task(description="What is 6 * 2? Answer with just the number.")
        result = team.do(tasks=task)
        assert result is not None
        assert "12" in str(result)

        # Verify post-execution
        assert "route-skill" in agent.system_prompt

    @requires_anthropic
    def test_nested_team_do_verifies_deep_propagation(self):
        """Nested teams: verify skills propagate to inner agents."""
        from upsonic import Agent, Task, Team
        skills = _make_skills("nested-exec")
        inner_agent = Agent(
            model="anthropic/claude-sonnet-4-5", name="InnerWorker",
            role="General Assistant", goal="Answer questions accurately",
        )
        inner_team = Team(
            entities=[inner_agent],
            name="Inner Team", role="Inner Department", goal="Process tasks",
            mode="sequential",
        )
        outer_team = Team(
            entities=[inner_team],
            skills=skills,
            mode="sequential",
        )

        # Verify deep propagation: outer team → inner team → inner agent
        assert inner_agent.skills is not None
        assert "nested-exec" in inner_agent.skills

        # Verify tools registered on inner agent
        tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in inner_agent.tools]
        assert any("get_skill_instructions" in n for n in tool_names)

        task = Task(description="What is 8 + 1? Answer with just the number.")
        result = outer_team.do(tasks=task)
        assert result is not None
        assert "9" in str(result)

        # Verify metrics accessible on inner agent
        metrics = inner_agent.get_skill_metrics()
        assert "nested-exec" in metrics
