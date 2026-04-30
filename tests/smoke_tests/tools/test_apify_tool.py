"""
Smoke tests for ApifyTools — real API requests against Apify.

Requires a valid APIFY_API_TOKEN or APIFY_API_KEY environment variable.
Automatically skipped when the key is not set.

Coverage:
- Constructor validation (API key, missing key, single/multiple actors)
- Multi-actor registration and tool discovery via ToolProcessor
- Docstring generation with Args/Returns
- End-to-end Agent + Task usage with print_do for multiple actors
- Same print structure as framework (Agent Started, Tool Calls, LLM Result, Task Metrics, Agent Metrics)

Run:
    uv run pytest tests/smoke_tests/tools/test_apify_tool.py -v --tb=short
"""

import os
import sys
from typing import Any, List, Optional

import pytest

APIFY_API_KEY: Optional[str] = os.getenv("APIFY_API_KEY") or os.getenv("APIFY_API_TOKEN")

pytestmark = [
    pytest.mark.skipif(
        not APIFY_API_KEY,
        reason="APIFY_API_KEY/APIFY_API_TOKEN not set; skipping Apify smoke tests",
    ),
]

# ── Actor registry ───────────────────────────────────────────────────
ALL_ACTORS: List[str] = [
    "apify/rag-web-browser",
    "apify/website-content-crawler",
    "apify/google-search-scraper",
    "apify/web-scraper",
]

ACTOR_TOOL_NAMES = {
    "apify/rag-web-browser": "apify_actor_apify_rag_web_browser",
    "apify/website-content-crawler": "apify_actor_apify_website_content_crawler",
    "apify/google-search-scraper": "apify_actor_apify_google_search_scraper",
    "apify/web-scraper": "apify_actor_apify_web_scraper",
}


def _process_toolkit(toolkit: Any) -> None:
    from upsonic.tools import ToolManager
    ToolManager().register_tools([toolkit])


def _enable_print_capture() -> None:
    """Patch Rich console to use sys.stdout so capsys captures agent/tool prints."""
    import upsonic.utils.printing as _printing
    from rich.console import Console
    _printing.console = Console(file=sys.stdout)


def _assert_print_do_output_has_same_structure(
    captured_out: str,
    *,
    expect_tool_calls: bool = True,
) -> None:
    """Assert that captured print_do output matches the same structure as framework output.

    Verifies: Agent Started, Tool Calls (with Parameters/Result), LLM Result,
    Task Metrics, Agent Metrics — so tools are called properly without error.
    """
    assert "Agent Started" in captured_out or "Agent Status" in captured_out, (
        f"Expected 'Agent Started' or 'Agent Status' in output; got:\n{captured_out[:800]}"
    )
    assert "Agent Name:" in captured_out, (
        f"Expected 'Agent Name:' in output; got:\n{captured_out[:800]}"
    )
    assert "Tool Calls" in captured_out and "executed" in captured_out, (
        f"Expected 'Tool Calls' and 'executed' in output; got:\n{captured_out[:800]}"
    )
    if expect_tool_calls:
        assert "Parameters" in captured_out, (
            f"Expected 'Parameters' (tool call params) in output; got:\n{captured_out[:800]}"
        )
        assert "Result" in captured_out, (
            f"Expected 'Result' (tool result) in output; got:\n{captured_out[:800]}"
        )
    assert "LLM Result" in captured_out, (
        f"Expected 'LLM Result' in output; got:\n{captured_out[:800]}"
    )
    assert "Task Metrics" in captured_out, (
        f"Expected 'Task Metrics' in output; got:\n{captured_out[:800]}"
    )
    assert "Input Tokens" in captured_out and "Output Tokens" in captured_out, (
        f"Expected 'Input Tokens' and 'Output Tokens' in output; got:\n{captured_out[:800]}"
    )
    assert "Agent Metrics" in captured_out, (
        f"Expected 'Agent Metrics' in output; got:\n{captured_out[:800]}"
    )
    assert "Total Tool Calls" in captured_out, (
        f"Expected 'Total Tool Calls' in output; got:\n{captured_out[:800]}"
    )


def _echo_captured_to_terminal(captured_out: str) -> None:
    """Echo captured print_do output to real terminal so it is visible with pytest -s."""
    sys.__stdout__.write(captured_out)
    sys.__stdout__.flush()


# ──────────────────────────────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────────────────────────────

class TestInit:

    def test_init_with_api_key(self) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        t = ApifyTools(actors=[ALL_ACTORS[0]], apify_api_token=APIFY_API_KEY)
        assert t.apify_api_token == APIFY_API_KEY

    def test_init_missing_key_raises(self) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        from unittest.mock import patch as _patch
        with _patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Apify API token is required"):
                ApifyTools(apify_api_token="")

    def test_init_single_actor_string(self) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        t = ApifyTools(actors="apify/rag-web-browser", apify_api_token=APIFY_API_KEY)
        assert hasattr(t, "apify_actor_apify_rag_web_browser")

    def test_init_all_actors(self) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        t = ApifyTools(actors=ALL_ACTORS, apify_api_token=APIFY_API_KEY)
        for actor_id, tool_name in ACTOR_TOOL_NAMES.items():
            assert hasattr(t, tool_name), f"Missing method for {actor_id}"


# ──────────────────────────────────────────────────────────────────────
#  Tool discovery via ToolProcessor — all actors
# ──────────────────────────────────────────────────────────────────────

class TestToolDiscovery:

    def test_processor_discovers_all_actor_tools(self) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        t = ApifyTools(actors=ALL_ACTORS, apify_api_token=APIFY_API_KEY)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        for actor_id, tool_name in ACTOR_TOOL_NAMES.items():
            assert tool_name in names, f"{tool_name} not discovered for {actor_id}"

    def test_tool_count_matches_actors(self) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        t = ApifyTools(actors=ALL_ACTORS, apify_api_token=APIFY_API_KEY)
        _process_toolkit(t)
        assert len(t.functions) == len(ALL_ACTORS)

    @pytest.mark.parametrize("actor_id", ALL_ACTORS)
    def test_actor_tool_has_docstring(self, actor_id: str) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        tool_name = ACTOR_TOOL_NAMES[actor_id]
        t = ApifyTools(actors=[actor_id], apify_api_token=APIFY_API_KEY)
        method = getattr(t, tool_name)
        assert method.__doc__ is not None
        assert len(method.__doc__) > 0
        assert "Args:" in method.__doc__
        assert "Returns:" in method.__doc__

    @pytest.mark.parametrize("actor_id", ALL_ACTORS)
    def test_actor_tool_is_marked(self, actor_id: str) -> None:
        from upsonic.tools.custom_tools.apify import ApifyTools
        tool_name = ACTOR_TOOL_NAMES[actor_id]
        t = ApifyTools(actors=[actor_id], apify_api_token=APIFY_API_KEY)
        method = getattr(t, tool_name)
        assert getattr(method, '_upsonic_is_tool', False) or \
            getattr(method.__func__, '_upsonic_is_tool', False)


# ──────────────────────────────────────────────────────────────────────
#  End-to-end Agent + Task tests with print_do
# ──────────────────────────────────────────────────────────────────────

class TestAgentIntegration:

    @pytest.mark.timeout(120)
    def test_rag_web_browser_with_print_do(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test apify/rag-web-browser — fetches a URL and summarizes content."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.apify import ApifyTools

        _enable_print_capture()
        capsys.readouterr()

        apify_tools = ApifyTools(
            actors=["apify/rag-web-browser"],
            apify_api_token=APIFY_API_KEY,
        )

        agent = Agent(
            "anthropic/claude-sonnet-4-5",
            name="RAG Browser Agent",
            tools=[apify_tools],
        )

        task = Task(
            "Use the apify rag web browser tool to find information on "
            "https://docs.upsonic.ai/get-started/introduction and summarize it in 2-3 sentences.",
            agent=agent,
        )

        result = agent.print_do(task, return_output=True)
        captured = capsys.readouterr()

        assert result is not None
        response_text = str(result)
        assert len(response_text) > 0
        _assert_print_do_output_has_same_structure(captured.out, expect_tool_calls=True)
        _echo_captured_to_terminal(captured.out)

    @pytest.mark.timeout(120)
    def test_google_search_scraper_with_print_do(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test apify/google-search-scraper — searches Google and returns results."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.apify import ApifyTools

        _enable_print_capture()
        capsys.readouterr()

        apify_tools = ApifyTools(
            actors=["apify/google-search-scraper"],
            apify_api_token=APIFY_API_KEY,
        )

        agent = Agent(
            "anthropic/claude-sonnet-4-5",
            name="Google Search Agent",
            tools=[apify_tools],
        )

        task = Task(
            "Use the apify google search scraper tool to search for 'Upsonic AI agent framework' "
            "and tell me the top 3 results with their titles and URLs.",
            agent=agent,
        )

        result = agent.print_do(task, return_output=True)
        captured = capsys.readouterr()

        assert result is not None
        response_text = str(result)
        assert len(response_text) > 0
        _assert_print_do_output_has_same_structure(captured.out, expect_tool_calls=True)
        _echo_captured_to_terminal(captured.out)

    @pytest.mark.timeout(180)
    def test_multiple_actors_with_print_do(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test registering multiple actors and using them in a single agent."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.apify import ApifyTools

        _enable_print_capture()
        capsys.readouterr()

        apify_tools = ApifyTools(
            actors=[
                "apify/rag-web-browser",
                "apify/google-search-scraper",
            ],
            apify_api_token=APIFY_API_KEY,
        )

        agent = Agent(
            "anthropic/claude-sonnet-4-5",
            name="Multi-Tool Agent",
            tools=[apify_tools],
        )

        task = Task(
            "Use the apify google search scraper tool to search for 'Upsonic framework' "
            "and give me a one-sentence summary of what you find.",
            agent=agent,
        )

        result = agent.print_do(task, return_output=True)
        captured = capsys.readouterr()

        assert result is not None
        response_text = str(result)
        assert len(response_text) > 0
        _assert_print_do_output_has_same_structure(captured.out, expect_tool_calls=True)
        _echo_captured_to_terminal(captured.out)
