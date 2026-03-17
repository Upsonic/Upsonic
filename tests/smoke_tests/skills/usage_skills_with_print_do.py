"""
Usage examples: Skills with Agent, Task, and Team using print_do.

Demonstrates three scenarios:
  1. Skills added via Agent class
  2. Skills added via Task class
  3. Skills added via Team class

Run with: uv run python tests/smoke_tests/skills/usage_skills_with_print_do.py
Requires: ANTHROPIC_API_KEY environment variable set.
"""

from typing import Any, Dict

from upsonic import Agent, Task, Team
from upsonic.skills.skill import Skill
from upsonic.skills.loader import InlineSkills
from upsonic.skills.skills import Skills


# ---------------------------------------------------------------------------
# Helper: create a simple skill set
# ---------------------------------------------------------------------------

def skill_tools_were_called(metrics: Dict[str, Any]) -> bool:
    """Return True if any skill had tool invocations (load, reference, or script)."""
    for skill_data in metrics.values():
        if isinstance(skill_data, dict):
            if (
                skill_data.get("load_count", 0) > 0
                or skill_data.get("reference_access_count", 0) > 0
                or skill_data.get("script_execution_count", 0) > 0
            ):
                return True
    return False


def make_skills(*names: str) -> Skills:
    """Create a Skills instance with inline skills for testing."""
    skill_list = [
        Skill(
            name=name,
            description=f"Skill for {name}",
            instructions=f"You are an expert at {name}. Apply your {name} expertise to the task.",
            source_path="",
            scripts=[],
            references=[],
        )
        for name in names
    ]
    return Skills(loaders=[InlineSkills(skill_list)])


# ===========================================================================
# CASE 1: Skills added via Agent class
# ===========================================================================

def case_skills_via_agent():
    """Skills are attached directly to the Agent."""
    print("\n" + "=" * 70)
    print("CASE 1: Skills added via Agent class")
    print("=" * 70)

    skills = make_skills("math-reasoning", "step-by-step-explanation")

    agent = Agent(
        model="anthropic/claude-sonnet-4-5",
        name="Math Agent",
        role="Mathematics Expert",
        goal="Solve math problems with clear explanations",
        skills=skills,
    )

    # Verify skills are registered
    print(f"Agent skills: {agent.skills.get_skill_names()}")
    print(f"Agent skill metrics (before): {agent.get_skill_metrics()}")

    # Execute with print_do
    result = agent.print_do("What is the derivative of x^3 + 2x? Explain step by step.")

    print(f"\nAgent skill metrics (after): {agent.get_skill_metrics()}")
    return result


# ===========================================================================
# CASE 2: Skills added via Task class
# ===========================================================================

def case_skills_via_task():
    """Skills are attached to the Task, not the Agent."""
    print("\n" + "=" * 70)
    print("CASE 2: Skills added via Task class")
    print("=" * 70)

    skills = make_skills("creative-writing", "poetry")

    # Agent has NO skills
    agent = Agent(
        model="anthropic/claude-sonnet-4-5",
        name="Writer Agent",
        role="Content Writer",
        goal="Create engaging content",
    )
    print(f"Agent skills (before task): {agent.skills}")

    # Task carries the skills
    task = Task(
        description="Write a short haiku about programming.",
        skills=skills,
    )
    print(f"Task skills: {task.skills.get_skill_names()}")
    print(f"Task skill metrics (before): {task.get_skill_metrics()}")

    # Execute with print_do
    result = agent.print_do(task)

    print(f"\nTask skill metrics (after): {task.get_skill_metrics()}")
    return result


# ===========================================================================
# CASE 3: Skills added via Team class
# ===========================================================================

def case_skills_via_team():
    """Skills are attached to the Team and propagated to all agents."""
    print("\n" + "=" * 70)
    print("CASE 3: Skills added via Team class")
    print("=" * 70)

    skills = make_skills("data-analysis", "report-writing")

    analyst = Agent(
        model="anthropic/claude-sonnet-4-5",
        name="Data Analyst",
        role="Data Analysis Expert",
        goal="Analyze data and extract insights",
    )

    writer = Agent(
        model="anthropic/claude-sonnet-4-5",
        name="Report Writer",
        role="Business Report Specialist",
        goal="Create professional summaries",
    )

    print(f"Analyst skills (before team): {analyst.skills}")
    print(f"Writer skills (before team): {writer.skills}")

    team = Team(
        agents=[analyst, writer],
        skills=skills,
        mode="coordinate",
        model="anthropic/claude-sonnet-4-5",
    )

    # Skills should now be propagated
    print(f"Analyst skills (after team): {analyst.skills.get_skill_names()}")
    print(f"Writer skills (after team): {writer.skills.get_skill_names()}")

    # Execute with print_do
    task = Task(description="Summarize key benefits of using AI in healthcare in 3 bullet points.")
    result = team.print_do(tasks=task)

    analyst_metrics = analyst.get_skill_metrics()
    writer_metrics = writer.get_skill_metrics()
    print(f"\nAnalyst skill metrics: {analyst_metrics}")
    print(f"Writer skill metrics: {writer_metrics}")
    agents_who_called: list[str] = []
    if skill_tools_were_called(analyst_metrics):
        agents_who_called.append(analyst.name)
    if skill_tools_were_called(writer_metrics):
        agents_who_called.append(writer.name)
    if agents_who_called:
        print(f"\nSkill tools were called by: {', '.join(agents_who_called)}")
    else:
        print("\nNo skill tools were called by any agent.")
    return result


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Skills Usage Examples — print_do with Agent, Task, and Team")
    print("=" * 70)

    print("\n--- Running Case 1: Skills via Agent ---")
    case_skills_via_agent()

    print("\n--- Running Case 2: Skills via Task ---")
    case_skills_via_task()

    print("\n--- Running Case 3: Skills via Team ---")
    case_skills_via_team()

    print("\n" + "=" * 70)
    print("All three cases completed successfully!")
    print("=" * 70)
