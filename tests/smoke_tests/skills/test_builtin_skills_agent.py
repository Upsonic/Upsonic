"""
Smoke tests for built-in skills (code-review, summarization, data-analysis)
running through real Agent + Task pipelines with LLM calls.

Tests verify:
- Skills load via BuiltinSkills loader
- System prompt contains skill metadata (name, description, scripts, references)
- Agent calls skill tools (get_skill_instructions, get_skill_reference)
- Skill metrics are tracked after execution
- Agent output is relevant to the task
"""

import os

import pytest

from upsonic.skills.loader import BuiltinSkills
from upsonic.skills.skills import Skills

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real LLM tests",
)

try:
    import pandas  # noqa: F401
    _has_pandas = True
except ImportError:
    _has_pandas = False

requires_pandas = pytest.mark.skipif(
    not _has_pandas,
    reason="pandas not installed — skipping script execution tests",
)

MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_builtin(*names):
    """Load specific built-in skills into a Skills container."""
    return Skills(loaders=[BuiltinSkills(skills=list(names))])


def _load_all_builtins():
    """Load all built-in skills."""
    return Skills(loaders=[BuiltinSkills()])


# ---------------------------------------------------------------------------
# System Prompt Injection — No LLM needed
# ---------------------------------------------------------------------------

class TestBuiltinSystemPromptInjection:
    """Verify built-in skill metadata appears in the system prompt."""

    def test_code_review_in_system_prompt(self):
        skills = _load_builtin("code-review")
        prompt = skills.get_system_prompt_section()
        assert "code-review" in prompt
        assert "owasp-top-10.md" in prompt or "severity-guide.md" in prompt
        assert "<scripts>none</scripts>" in prompt  # code-review has no scripts

    def test_summarization_in_system_prompt(self):
        skills = _load_builtin("summarization")
        prompt = skills.get_system_prompt_section()
        assert "summarization" in prompt
        assert "summary-templates.md" in prompt
        assert "<scripts>none</scripts>" in prompt

    def test_data_analysis_in_system_prompt(self):
        skills = _load_builtin("data-analysis")
        prompt = skills.get_system_prompt_section()
        assert "data-analysis" in prompt
        assert "profile_data.py" in prompt
        assert "statistical-tests-guide.md" in prompt

    def test_all_builtins_in_system_prompt(self):
        skills = _load_all_builtins()
        prompt = skills.get_system_prompt_section()
        assert "code-review" in prompt
        assert "summarization" in prompt
        assert "data-analysis" in prompt

    def test_system_prompt_has_tool_instructions(self):
        skills = _load_builtin("code-review")
        prompt = skills.get_system_prompt_section()
        assert "get_skill_instructions" in prompt
        assert "get_skill_reference" in prompt
        assert "get_skill_script" in prompt
        assert "get_skill_asset" in prompt

    def test_system_prompt_injected_into_agent(self):
        from upsonic import Agent
        skills = _load_builtin("code-review", "data-analysis")
        agent = Agent(MODEL, name="Prompt Agent", skills=skills)
        # Before execution, check agent has the skills registered
        registered = list(agent.registered_agent_tools.keys())
        assert any("get_skill_instructions" in n for n in registered)
        assert any("get_skill_reference" in n for n in registered)
        assert any("get_skill_asset" in n for n in registered)


# ---------------------------------------------------------------------------
# Tool Registration for Builtins — No LLM needed
# ---------------------------------------------------------------------------

class TestBuiltinToolRegistration:
    """Verify built-in skills register the correct tools on the agent."""

    def test_code_review_tools_on_agent(self):
        from upsonic import Agent
        skills = _load_builtin("code-review")
        agent = Agent(MODEL, name="CR Agent", skills=skills)
        tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in agent.tools]
        for expected in ("get_skill_instructions", "get_skill_reference", "get_skill_script", "get_skill_asset"):
            assert any(expected in n for n in tool_names), \
                f"{expected} not in agent tools: {tool_names}"

    def test_data_analysis_tools_on_agent(self):
        from upsonic import Agent
        skills = _load_builtin("data-analysis")
        agent = Agent(MODEL, name="DA Agent", skills=skills)
        tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in agent.tools]
        assert any("get_skill_instructions" in n for n in tool_names)

    def test_all_builtins_tools_on_agent(self):
        from upsonic import Agent
        skills = _load_all_builtins()
        agent = Agent(MODEL, name="All Skills Agent", skills=skills)
        tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in agent.tools]
        skill_tools = [n for n in tool_names if "skill" in n.lower()]
        # Should have 4 skill tools (instructions, reference, script, asset)
        assert len(skill_tools) >= 4, f"Expected >=4 skill tools, got {skill_tools}"


# ---------------------------------------------------------------------------
# Code Review — Full Agent Pipeline (LLM)
# ---------------------------------------------------------------------------

class TestCodeReviewBuiltin:
    """Test code-review skill through a real agent pipeline."""

    @requires_anthropic
    def test_code_review_agent_reviews_code(self):
        """Agent with code-review skill should produce a meaningful review."""
        from upsonic import Agent, Task

        skills = _load_builtin("code-review")
        agent = Agent(
            MODEL,
            name="Code Reviewer",
            role="Senior Software Engineer",
            goal="Review code for bugs, security issues, and quality",
            skills=skills,
        )

        buggy_code = '''
def login(username, password):
    query = "SELECT * FROM users WHERE name='" + username + "' AND pass='" + password + "'"
    result = db.execute(query)
    if result:
        return True
    return False
'''

        task = Task(
            description=f"Review this Python code for issues:\n```python\n{buggy_code}\n```"
        )

        output = agent.print_do(task, return_output=True)

        # 1. Output exists and is substantial
        assert output is not None
        assert output.output is not None
        result_text = str(output.output).lower()
        assert len(result_text) > 50, "Review should be substantial"

        # 2. Should detect the SQL injection
        assert any(term in result_text for term in ("sql injection", "injection", "sql", "concatenat")), \
            f"Should detect SQL injection in the review. Got: {result_text[:200]}"

        # 3. Agent should have called get_skill_instructions for code-review
        skill_tool_calls = [
            t for t in output.tools
            if hasattr(t, "tool_name") and "skill" in t.tool_name.lower()
        ]
        assert len(skill_tool_calls) > 0, \
            f"Agent should call at least one skill tool. Tools called: {[t.tool_name for t in output.tools]}"

        # Check specifically that get_skill_instructions was called
        instr_calls = [t for t in output.tools if "get_skill_instructions" in t.tool_name]
        assert len(instr_calls) > 0, \
            f"Agent should call get_skill_instructions. Tools called: {[t.tool_name for t in output.tools]}"

        # 4. Skill metrics should show the skill was loaded
        metrics = output.skill_metrics
        assert "code-review" in metrics
        m = metrics["code-review"]
        assert m["load_count"] >= 1, f"code-review should have been loaded at least once, got: {m}"

    @requires_anthropic
    def test_code_review_system_prompt_after_execution(self):
        """After execution, agent system prompt should contain code-review skill info."""
        from upsonic import Agent, Task

        skills = _load_builtin("code-review")
        agent = Agent(
            MODEL,
            name="CR Prompt Check",
            skills=skills,
            system_prompt="You are a code reviewer.",
        )

        agent.print_do(Task(description="Review this: `x = 1 + 1`. Just say 'looks good'."))

        # System prompt should contain both custom prompt and skill info
        assert "You are a code reviewer" in agent.system_prompt
        assert "code-review" in agent.system_prompt
        assert "owasp-top-10.md" in agent.system_prompt or "severity-guide.md" in agent.system_prompt


# ---------------------------------------------------------------------------
# Summarization — Full Agent Pipeline (LLM)
# ---------------------------------------------------------------------------

class TestSummarizationBuiltin:
    """Test summarization skill through a real agent pipeline."""

    @requires_anthropic
    def test_summarization_agent_summarizes_text(self):
        """Agent with summarization skill should produce a concise summary."""
        from upsonic import Agent, Task

        skills = _load_builtin("summarization")
        agent = Agent(
            MODEL,
            name="Summarizer",
            role="Content Analyst",
            goal="Summarize content accurately and concisely",
            skills=skills,
        )

        long_text = """
        The quarterly financial results show that revenue increased by 23% year-over-year
        to $4.2 million, driven primarily by expansion in the enterprise segment which
        grew 45%. However, operating expenses also rose by 18% due to increased hiring
        in engineering and sales departments. The company added 12 new enterprise clients
        during the quarter, bringing the total to 89. Customer retention rate remained
        strong at 94%. The board approved a new $2 million investment in R&D for the
        next fiscal year, focusing on AI-powered features. Cash reserves stand at $8.5
        million, providing approximately 18 months of runway at current burn rate. The
        management team recommended expanding into the European market in Q2 2025,
        with an estimated initial investment of $500,000.
        """

        task = Task(
            description=f"Summarize this quarterly report:\n{long_text}"
        )

        output = agent.print_do(task, return_output=True)

        # 1. Output exists
        assert output is not None
        assert output.output is not None
        result_text = str(output.output)
        assert len(result_text) > 30, "Summary should be meaningful"

        # 2. Summary should preserve key metrics from the text
        result_lower = result_text.lower()
        # Should mention at least some of the key numbers
        key_terms = ["23%", "4.2", "revenue", "enterprise", "94%", "r&d", "european"]
        found_terms = [t for t in key_terms if t.lower() in result_lower]
        assert len(found_terms) >= 2, \
            f"Summary should preserve key data. Found: {found_terms}. Output: {result_text[:300]}"

        # 3. Agent should have called skill tools
        skill_tool_calls = [
            t for t in output.tools
            if hasattr(t, "tool_name") and "skill" in t.tool_name.lower()
        ]
        assert len(skill_tool_calls) > 0, \
            f"Agent should call skill tools. Tools called: {[t.tool_name for t in output.tools]}"

        # 4. Skill metrics should track usage
        metrics = output.skill_metrics
        assert "summarization" in metrics
        assert metrics["summarization"]["load_count"] >= 1

    @requires_anthropic
    def test_summarization_system_prompt_after_execution(self):
        """After execution, agent system prompt should contain summarization skill info."""
        from upsonic import Agent, Task

        skills = _load_builtin("summarization")
        agent = Agent(MODEL, name="Sum Prompt Check", skills=skills)

        agent.print_do(Task(description="Summarize: 'The cat sat on the mat.' Just repeat it shorter."))

        assert "summarization" in agent.system_prompt
        assert "summary-templates.md" in agent.system_prompt


# ---------------------------------------------------------------------------
# Data Analysis — Full Agent Pipeline (LLM)
# ---------------------------------------------------------------------------

class TestDataAnalysisBuiltin:
    """Test data-analysis skill through a real agent pipeline."""

    @requires_anthropic
    def test_data_analysis_agent_analyzes_data(self):
        """Agent with data-analysis skill should provide analytical insights."""
        from upsonic import Agent, Task

        skills = _load_builtin("data-analysis")
        agent = Agent(
            MODEL,
            name="Data Analyst",
            role="Senior Data Analyst",
            goal="Analyze data and extract actionable insights",
            skills=skills,
        )

        data_description = """
        Here is monthly website traffic data for 2024:

        Jan: 10,000 visitors, 2.1% conversion
        Feb: 11,200 visitors, 2.3% conversion
        Mar: 13,500 visitors, 2.5% conversion
        Apr: 12,800 visitors, 2.2% conversion
        May: 15,000 visitors, 2.8% conversion
        Jun: 18,500 visitors, 3.1% conversion
        Jul: 16,200 visitors, 2.7% conversion
        Aug: 14,800 visitors, 2.4% conversion
        Sep: 19,000 visitors, 3.3% conversion
        Oct: 22,000 visitors, 3.5% conversion
        Nov: 25,000 visitors, 3.8% conversion
        Dec: 21,500 visitors, 3.2% conversion

        What are the key trends and insights?
        """

        task = Task(description=data_description)

        output = agent.print_do(task, return_output=True)

        # 1. Output exists and is meaningful
        assert output is not None
        assert output.output is not None
        result_text = str(output.output)
        assert len(result_text) > 50, "Analysis should be substantial"

        # 2. Should mention trends or patterns
        result_lower = result_text.lower()
        trend_terms = ["trend", "growth", "increas", "pattern", "peak", "conversion", "traffic", "visitor"]
        found = [t for t in trend_terms if t in result_lower]
        assert len(found) >= 2, \
            f"Analysis should discuss trends/patterns. Found: {found}. Output: {result_text[:300]}"

        # 3. Agent should have called skill tools
        skill_tool_calls = [
            t for t in output.tools
            if hasattr(t, "tool_name") and "skill" in t.tool_name.lower()
        ]
        assert len(skill_tool_calls) > 0, \
            f"Agent should call skill tools. Tools called: {[t.tool_name for t in output.tools]}"

        # 4. Skill metrics
        metrics = output.skill_metrics
        assert "data-analysis" in metrics
        assert metrics["data-analysis"]["load_count"] >= 1

    @requires_anthropic
    def test_data_analysis_system_prompt_after_execution(self):
        """After execution, agent system prompt should contain data-analysis skill info."""
        from upsonic import Agent, Task

        skills = _load_builtin("data-analysis")
        agent = Agent(MODEL, name="DA Prompt Check", skills=skills)

        agent.print_do(Task(description="What is the average of 1, 2, 3? Just give the number."))

        assert "data-analysis" in agent.system_prompt
        assert "profile_data.py" in agent.system_prompt
        assert "statistical-tests-guide.md" in agent.system_prompt


# ---------------------------------------------------------------------------
# All Builtins Together — Full Agent Pipeline (LLM)
# ---------------------------------------------------------------------------

class TestAllBuiltinsTogether:
    """Test loading all built-in skills on a single agent."""

    @requires_anthropic
    def test_all_builtins_agent_uses_correct_skill(self):
        """Agent with all builtins should pick the right skill for the task."""
        from upsonic import Agent, Task

        skills = _load_all_builtins()
        agent = Agent(
            MODEL,
            name="Multi-Skill Agent",
            role="Expert Assistant",
            goal="Use the appropriate skill for each task",
            skills=skills,
        )

        # Give a code review task — agent should use code-review skill
        task = Task(
            description="Review this code for issues: `def f(x): return x / 0`"
        )

        output = agent.print_do(task, return_output=True)

        assert output is not None
        assert output.output is not None

        # Agent should have loaded at least one skill
        total_loads = sum(
            m.get("load_count", 0) if isinstance(m, dict) else 0
            for m in output.skill_metrics.values()
        )
        assert total_loads >= 1, \
            f"Agent should load at least one skill. Metrics: {output.skill_metrics}"

        # Check that skill tools were called
        skill_tool_calls = [
            t for t in output.tools
            if hasattr(t, "tool_name") and "skill" in t.tool_name.lower()
        ]
        assert len(skill_tool_calls) > 0, \
            f"Agent should call skill tools. Tools: {[t.tool_name for t in output.tools]}"

    @requires_anthropic
    def test_all_builtins_system_prompt_contains_all_skills(self):
        """System prompt should list all built-in skills after execution."""
        from upsonic import Agent, Task

        skills = _load_all_builtins()
        agent = Agent(MODEL, name="All Builtins", skills=skills)

        agent.print_do(Task(description="Say 'hello'."))

        # All skill names should appear in system prompt
        for name in ("code-review", "summarization", "data-analysis"):
            assert name in agent.system_prompt, \
                f"{name} not found in agent system prompt after execution"

    @requires_anthropic
    def test_all_builtins_metrics_initialized(self):
        """All built-in skills should have metrics initialized."""
        from upsonic import Agent, Task

        skills = _load_all_builtins()
        agent = Agent(MODEL, name="Metrics Init", skills=skills)

        output = agent.print_do(Task(description="Say 'ok'."), return_output=True)

        for name in ("code-review", "summarization", "data-analysis"):
            assert name in output.skill_metrics, \
                f"{name} missing from skill_metrics: {list(output.skill_metrics.keys())}"
            m = output.skill_metrics[name]
            assert "load_count" in m
            assert "reference_access_count" in m
            assert "script_execution_count" in m
            assert "total_chars_loaded" in m


# ---------------------------------------------------------------------------
# Builtin Skill Metadata Quality Checks — No LLM needed
# ---------------------------------------------------------------------------

class TestBuiltinSkillQuality:
    """Verify built-in skills have proper metadata and content."""

    def test_all_builtins_have_version(self):
        loader = BuiltinSkills()
        skills = loader.load()
        for s in skills:
            assert s.version is not None, f"Builtin skill {s.name} has no version"

    def test_all_builtins_have_substantial_instructions(self):
        loader = BuiltinSkills()
        skills = loader.load()
        for s in skills:
            assert len(s.instructions) > 500, \
                f"Builtin skill {s.name} has short instructions ({len(s.instructions)} chars)"

    def test_all_builtins_have_descriptive_descriptions(self):
        loader = BuiltinSkills()
        skills = loader.load()
        for s in skills:
            # Descriptions should be trigger-rich (like Anthropic's skills)
            assert len(s.description) > 100, \
                f"Builtin skill {s.name} has short description ({len(s.description)} chars)"

    def test_code_review_has_references(self):
        loader = BuiltinSkills(skills=["code-review"])
        skills = loader.load()
        assert len(skills[0].references) >= 2, \
            f"code-review should have >=2 references, got: {skills[0].references}"
        assert "owasp-top-10.md" in skills[0].references
        assert "severity-guide.md" in skills[0].references

    def test_summarization_has_references(self):
        loader = BuiltinSkills(skills=["summarization"])
        skills = loader.load()
        assert len(skills[0].references) >= 1, \
            f"summarization should have >=1 reference, got: {skills[0].references}"
        assert "summary-templates.md" in skills[0].references

    def test_data_analysis_has_scripts_and_references(self):
        loader = BuiltinSkills(skills=["data-analysis"])
        skills = loader.load()
        assert len(skills[0].scripts) >= 1, \
            f"data-analysis should have >=1 script, got: {skills[0].scripts}"
        assert "profile_data.py" in skills[0].scripts
        assert len(skills[0].references) >= 1, \
            f"data-analysis should have >=1 reference, got: {skills[0].references}"
        assert "statistical-tests-guide.md" in skills[0].references

    def test_builtin_instructions_reference_their_files(self):
        """SKILL.md instructions should mention their reference/script files."""
        loader = BuiltinSkills()
        skills = loader.load()
        for s in skills:
            for ref in s.references:
                assert ref in s.instructions, \
                    f"Skill {s.name} has reference '{ref}' but doesn't mention it in instructions"
            for script in s.scripts:
                assert script in s.instructions, \
                    f"Skill {s.name} has script '{script}' but doesn't mention it in instructions"


# ---------------------------------------------------------------------------
# Script Execution — Agent must call get_skill_script with execute=True (LLM)
# ---------------------------------------------------------------------------

class TestSkillScriptExecution:
    """Test that the agent actually executes skill scripts via get_skill_script."""

    @requires_anthropic
    @requires_pandas
    def test_data_analysis_agent_executes_profile_script(self, tmp_path):
        """Agent should execute profile_data.py on a real CSV file."""
        import json
        from upsonic import Agent, Task

        # Create a real CSV file for the agent to profile
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text(
            "month,revenue,customers\n"
            "Jan,10000,150\n"
            "Feb,12000,180\n"
            "Mar,9500,140\n"
            "Apr,15000,210\n"
            "May,18000,250\n"
            "Jun,22000,300\n"
        )

        skills = _load_builtin("data-analysis")
        agent = Agent(
            MODEL,
            name="Script Executor",
            role="Data Analyst",
            goal="Analyze data files using available skill scripts",
            skills=skills,
        )

        task = Task(
            description=(
                f"I have a CSV file at: {csv_file}\n\n"
                "Use the data-analysis skill's profile_data.py script to "
                "profile this dataset. Execute the script with the file path "
                "as argument, then summarize what the profiling reveals."
            ),
        )

        output = agent.print_do(task, return_output=True)

        assert output is not None
        assert output.output is not None

        # 1. Agent MUST have called get_skill_script
        script_tool_calls = [
            t for t in output.tools
            if hasattr(t, "tool_name") and t.tool_name == "get_skill_script"
        ]
        assert len(script_tool_calls) >= 1, (
            f"Agent must call get_skill_script. "
            f"Tools called: {[t.tool_name for t in output.tools]}"
        )

        # 2. The script call should target profile_data.py with execute=True
        executed_script_calls = []
        for t in script_tool_calls:
            args = t.tool_args or {}
            if args.get("script_path") == "profile_data.py" and args.get("execute") is True:
                executed_script_calls.append(t)

        assert len(executed_script_calls) >= 1, (
            f"Agent must execute profile_data.py (execute=True). "
            f"Script tool calls: {[(t.tool_args) for t in script_tool_calls]}"
        )

        # 3. The execution should have succeeded (result contains profiling output)
        for t in executed_script_calls:
            # Result may be wrapped in {'func': '...'} or be raw JSON
            raw = t.result if isinstance(t.result, str) else str(t.result)
            # Parse: try direct JSON, fallback to extracting inner value
            try:
                result = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                result = {"_raw": raw}

            # If wrapped in {'func': '<json_string>'}, unwrap
            if isinstance(result, dict) and "func" in result:
                try:
                    result = json.loads(result["func"])
                except (json.JSONDecodeError, TypeError):
                    result = {"_raw": result["func"]}

            # Check for success
            result_str = str(result)
            assert "error" not in result_str.lower() or "returncode" in result_str, (
                f"Script execution should succeed. Result: {result_str[:500]}"
            )
            # The profiling output should mention dataset shape
            assert "rows" in result_str.lower() or "columns" in result_str.lower() or "dataset" in result_str.lower(), (
                f"Script stdout should contain dataset profile. Got: {result_str[:500]}"
            )

        # 4. Skill metrics should record script execution
        metrics = output.skill_metrics
        assert "data-analysis" in metrics
        assert metrics["data-analysis"]["script_execution_count"] >= 1, (
            f"script_execution_count should be >= 1. Metrics: {metrics['data-analysis']}"
        )

        # 5. Output should reference the profiling results
        result_text = str(output.output).lower()
        data_terms = ["revenue", "customers", "month", "rows", "columns", "mean", "min", "max"]
        found = [t for t in data_terms if t in result_text]
        assert len(found) >= 2, (
            f"Output should reference profiling results. Found terms: {found}. "
            f"Output: {str(output.output)[:300]}"
        )
