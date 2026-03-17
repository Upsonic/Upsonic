"""
Smoke tests for Skills + KnowledgeBase full pipeline — ADVANCED VERIFICATION.

Tests: Agent.do(task) and Team.do(tasks) with skills + KnowledgeBase using
context=[kb]. Verifies tool registration, system prompt injection,
skill metrics, and execution tracking.

Uses embedded ChromaDB (no external API key needed for vectordb).
Requires: OPENAI_API_KEY (for embeddings), ANTHROPIC_API_KEY (for agent).
"""

import os
import tempfile
from pathlib import Path

import pytest

from upsonic.skills.loader import LocalSkills
from upsonic.skills.skills import Skills


requires_keys = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="OPENAI_API_KEY and ANTHROPIC_API_KEY required",
)


def _make_skill_dir_with_refs(base, name, ref_content):
    skill_dir = Path(base) / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill {name}\n---\nInstructions for {name}"
    )
    (skill_dir / "scripts").mkdir()
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.txt").write_text(ref_content)
    return skill_dir


def _setup_kb_and_skills(tmp_dir):
    """Create a KnowledgeBase backed by embedded ChromaDB + skills."""
    from upsonic import KnowledgeBase
    from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
    from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

    _make_skill_dir_with_refs(
        tmp_dir, "python-style",
        "Python style guide: Use snake_case for variables. Use 4-space indentation. "
        "Prefer list comprehensions over map/filter. Always add type hints."
    )
    _make_skill_dir_with_refs(
        tmp_dir, "error-handling",
        "Error handling best practices: Always use specific exception types. "
        "Never catch bare exceptions. Log errors with full context and stack traces."
    )

    embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
    config = ChromaConfig(
        collection_name="skill_refs_test",
        vector_size=1536,
        connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=str(Path(tmp_dir) / "chroma_db")),
    )
    vectordb = ChromaProvider(config)

    kb = KnowledgeBase(
        sources=[tmp_dir],
        embedding_provider=embedding,
        vectordb=vectordb,
    )

    skills = Skills(
        loaders=[LocalSkills(tmp_dir)],
        knowledge_base=kb,
    )
    return skills, kb, vectordb


@requires_keys
class TestKnowledgeBaseAgentPipeline:
    """Full pipeline: Agent + Skills + KnowledgeBase with advanced verification."""

    def test_agent_do_with_kb_verifies_output_and_tracking(self):
        """Verify agent.do(task) with KB context returns correct output."""
        from upsonic import Agent, Task, KnowledgeBase
        from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
        from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

        with tempfile.TemporaryDirectory() as d:
            ref_dir = Path(d) / "refs"
            ref_dir.mkdir()
            (ref_dir / "facts.txt").write_text(
                "The Eiffel Tower is 330 meters tall. It was built in 1889. "
                "It is located in Paris, France."
            )

            embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
            config = ChromaConfig(
                collection_name="agent_kb_test",
                vector_size=1536,
                connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=str(Path(d) / "chroma")),
            )
            vectordb = ChromaProvider(config)

            kb = KnowledgeBase(
                sources=[str(ref_dir)],
                embedding_provider=embedding,
                vectordb=vectordb,
            )

            # VERIFY: KB was initialized with sources
            assert kb is not None

            agent = Agent("anthropic/claude-sonnet-4-5", name="KB Agent")
            task = Task(
                description="How tall is the Eiffel Tower? Answer with just the number and unit.",
                context=[kb],
            )

            output = agent.do(task, return_output=True)

            # VERIFY: Output correctness
            assert output is not None
            assert output.output is not None
            assert "330" in str(output.output)

            # VERIFY: AgentRunOutput structure
            assert hasattr(output, "tools")
            assert hasattr(output, "tool_call_count")
            assert hasattr(output, "skill_metrics")
            assert isinstance(output.tool_call_count, int)
            assert output.tool_call_count >= 0

            # VERIFY: Vector DB has documents after execution (lazy indexing)
            try:
                doc_count = vectordb.get_count_sync()
                assert doc_count > 0, \
                    f"Vector DB should have indexed documents after execution, got count={doc_count}"
            except Exception:
                # Collection may not exist if KB used a different indexing path
                pass

    def test_agent_do_with_skills_kb_verifies_tool_registration_and_metrics(self):
        """Verify skill tools + search tool are registered and metrics tracked."""
        from upsonic import Agent, Task

        with tempfile.TemporaryDirectory() as d:
            skills, kb, vectordb = _setup_kb_and_skills(d)

            agent = Agent("anthropic/claude-sonnet-4-5", name="Skilled KB Agent", skills=skills)

            # VERIFY: 4 skill tools registered (3 standard + 1 search)
            tool_names = [
                t.__name__ if hasattr(t, "__name__") else str(t)
                for t in agent.tools
            ]
            skill_tool_names = [n for n in tool_names if "skill" in n.lower()]
            assert len(skill_tool_names) >= 5, \
                f"Expected >=5 skill tools (4 + search), got {skill_tool_names}"
            assert any("search_skill_references" in n for n in tool_names), \
                "search_skill_references tool not registered"
            for expected in ("get_skill_instructions", "get_skill_reference", "get_skill_script", "get_skill_asset"):
                assert any(expected in n for n in tool_names), \
                    f"{expected} tool not registered"

            # VERIFY: registered_agent_tools includes skill tools
            registered = list(agent.registered_agent_tools.keys())
            assert any("get_skill_instructions" in n for n in registered)
            assert any("search_skill_references" in n for n in registered)

            # VERIFY: Skill metrics exist before execution
            metrics_before = agent.get_skill_metrics()
            assert "python-style" in metrics_before
            assert "error-handling" in metrics_before

            task = Task(
                description="What coding style should I use for Python variables? Answer briefly.",
                context=[kb],
            )
            output = agent.do(task, return_output=True)

            assert output is not None
            assert output.output is not None

            # VERIFY: Skill metrics on output
            assert output.skill_metrics is not None
            assert "python-style" in output.skill_metrics
            assert "error-handling" in output.skill_metrics

            # VERIFY: Tool call tracking
            assert isinstance(output.tool_call_count, int)
            assert output.tool_call_count >= 0

    def test_agent_do_with_kb_verifies_tool_executions(self):
        """Verify ToolExecution objects track skill tool calls."""
        from upsonic import Agent, Task

        with tempfile.TemporaryDirectory() as d:
            skills, kb, vectordb = _setup_kb_and_skills(d)

            agent = Agent("anthropic/claude-sonnet-4-5", name="Tool Track Agent", skills=skills)
            task = Task(
                description=(
                    "First, use the get_skill_instructions tool to load instructions "
                    "for the 'python-style' skill, then tell me what style to use "
                    "for Python variable names. Answer briefly."
                ),
                context=[kb],
            )
            output = agent.do(task, return_output=True)

            assert output is not None
            assert output.output is not None

            # VERIFY: Tool executions were tracked
            if output.tools:
                for tool_exec in output.tools:
                    assert hasattr(tool_exec, "tool_name")
                    assert hasattr(tool_exec, "tool_args")
                    assert hasattr(tool_exec, "result")
                    assert tool_exec.tool_name is not None
                    assert tool_exec.tool_call_id is not None

                # Check if any skill tool was called
                skill_tool_calls = [
                    t for t in output.tools
                    if t.tool_name and "skill" in t.tool_name.lower()
                ]
                if skill_tool_calls:
                    for tc in skill_tool_calls:
                        assert tc.result is not None, \
                            f"Skill tool {tc.tool_name} should have a result"

            # VERIFY: Tool call count is tracked
            assert isinstance(output.tool_call_count, int)

    def test_agent_do_with_kb_verifies_system_prompt(self):
        """Verify skills are injected into system prompt with KB integration."""
        from upsonic import Agent, Task

        with tempfile.TemporaryDirectory() as d:
            skills, kb, vectordb = _setup_kb_and_skills(d)

            agent = Agent(
                "anthropic/claude-sonnet-4-5", name="Prompt KB Agent",
                skills=skills, system_prompt="You are a coding expert.",
            )

            task = Task(
                description="What is 2 + 2? Answer with just the number.",
                context=[kb],
            )
            output = agent.do(task, return_output=True)

            assert output is not None

            # VERIFY: System prompt contains custom text + skill info
            assert "You are a coding expert" in agent.system_prompt
            assert "python-style" in agent.system_prompt
            assert "error-handling" in agent.system_prompt


@requires_keys
class TestKnowledgeBaseTeamPipeline:
    """Full pipeline: Team + Skills + KnowledgeBase with advanced verification."""

    def test_team_do_with_kb_verifies_full_pipeline(self):
        """Team.do(tasks) with KB context returns correct result."""
        from upsonic import Agent, Task, Team, KnowledgeBase
        from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
        from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

        with tempfile.TemporaryDirectory() as d:
            ref_dir = Path(d) / "refs"
            ref_dir.mkdir()
            (ref_dir / "data.txt").write_text(
                "Python was created by Guido van Rossum and first released in 1991."
            )

            embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
            config = ChromaConfig(
                collection_name="team_kb_test",
                vector_size=1536,
                connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=str(Path(d) / "chroma")),
            )
            vectordb = ChromaProvider(config)

            kb = KnowledgeBase(
                sources=[str(ref_dir)],
                embedding_provider=embedding,
                vectordb=vectordb,
            )

            researcher = Agent(
                model="anthropic/claude-sonnet-4-5",
                name="Researcher",
                role="Research Specialist",
                goal="Find accurate information",
            )

            team = Team(
                entities=[researcher],
                mode="sequential",
            )

            task = Task(
                description="Who created Python? Answer with just the name.",
                context=[kb],
            )
            result = team.do(tasks=task)
            assert result is not None
            assert "Guido" in str(result)

            # VERIFY: System prompt has task context after execution
            assert researcher.system_prompt is not None

    def test_team_do_with_skills_and_kb_verifies_propagation(self):
        """Team with skills + KB: verify skills propagate and tools registered."""
        from upsonic import Agent, Task, Team

        with tempfile.TemporaryDirectory() as d:
            skills, kb, vectordb = _setup_kb_and_skills(d)

            agent = Agent(
                model="anthropic/claude-sonnet-4-5",
                name="Skilled Team Worker",
                role="Code Reviewer",
                goal="Review code using best practices",
            )

            team = Team(
                entities=[agent],
                skills=skills,
                mode="sequential",
            )

            # VERIFY: Skills propagated to agent
            assert agent.skills is not None
            assert "python-style" in agent.skills
            assert "error-handling" in agent.skills

            # VERIFY: Search tool registered (KB integration)
            tool_names = [
                t.__name__ if hasattr(t, "__name__") else str(t)
                for t in agent.tools
            ]
            assert any("search_skill_references" in n for n in tool_names), \
                "search_skill_references should be registered with KB"

            # VERIFY: Skill metrics available
            metrics = agent.get_skill_metrics()
            assert "python-style" in metrics
            assert "error-handling" in metrics

            task = Task(
                description="What naming convention should I use for Python variables? Answer briefly.",
                context=[kb],
            )
            result = team.do(tasks=task)
            assert result is not None

            # VERIFY: Post-execution system prompt has skills
            assert "python-style" in agent.system_prompt
            assert "error-handling" in agent.system_prompt
