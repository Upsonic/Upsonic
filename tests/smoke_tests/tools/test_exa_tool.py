"""
Smoke tests for ExaTools — real API requests against Exa.

Requires a valid EXA_API_KEY environment variable.
Automatically skipped when the key is not set.

Coverage:
- Constructor with all init parameters and defaults
- Missing key raises ValueError
- exclude_tools / include_tools filtering
- search: all parameters (query, num_results, search_type, category,
          include_domains, exclude_domains, start_published_date,
          end_published_date, include_text, exclude_text,
          text, highlights, summary, max_characters, max_age_hours)
- get_contents: all parameters (urls, text, highlights, summary,
                max_characters, max_age_hours, subpages, subpage_target)
- find_similar: all parameters (url, num_results, include_domains,
                exclude_domains, start_published_date, end_published_date,
                text, highlights, summary, max_characters)
- answer: all parameters (query, text)
- Async counterparts for every tool (asearch, aget_contents, afind_similar, aanswer)
- Proper output structure assertions for every method

Run:
    EXA_API_KEY="your-key" uv run pytest tests/smoke_tests/tools/test_exa_tool.py -v --tb=short
"""

import json
import os
from typing import Any, Dict, List, Optional

import pytest

EXA_API_KEY: Optional[str] = os.getenv("EXA_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not EXA_API_KEY,
        reason="EXA_API_KEY not set; skipping Exa smoke tests",
    ),
]

TEST_SEARCH_QUERY: str = "Python programming language"
TEST_URL: str = "https://github.com/anthropics/anthropic-sdk-python"
TEST_CONTENT_URL: str = "https://docs.exa.ai/reference/getting-started"
BATCH_URLS: List[str] = [
    "https://docs.exa.ai/reference/getting-started",
    "https://github.com/anthropics/anthropic-sdk-python",
]

ALL_TOOL_NAMES: List[str] = [
    "search",
    "get_contents",
    "find_similar",
    "answer",
]


@pytest.fixture(scope="module")
def tools() -> Any:
    from upsonic.tools.custom_tools.exa import ExaTools
    return ExaTools(api_key=EXA_API_KEY)


_SKIP_ERROR_PATTERNS: List[str] = [
    "Rate Limit",
    "rate limit",
    "Insufficient credits",
    "insufficient credits",
    "quota",
]


def _parse(raw: str) -> Dict[str, Any]:
    """Parse a JSON response and handle API errors gracefully."""
    parsed: Any = json.loads(raw)
    if isinstance(parsed, dict) and parsed.get("error"):
        error_msg: str = parsed["error"]
        for pattern in _SKIP_ERROR_PATTERNS:
            if pattern in error_msg:
                pytest.skip(f"Skipped due to API limit: {error_msg[:120]}")
        pytest.fail(f"API returned error: {error_msg}")
    assert isinstance(parsed, dict), (
        f"Expected dict, got {type(parsed).__name__}: {str(parsed)[:200]}"
    )
    return parsed


def _assert_search_results(parsed: Dict[str, Any], min_results: int = 1) -> None:
    """Assert that search results have the expected structure."""
    assert "results" in parsed, f"Missing 'results' key. Keys: {list(parsed.keys())}"
    results = parsed["results"]
    assert isinstance(results, list)
    assert len(results) >= min_results, f"Expected >= {min_results} results, got {len(results)}"
    for r in results:
        assert "url" in r, f"Result missing 'url': {list(r.keys())}"
        assert "title" in r, f"Result missing 'title': {list(r.keys())}"


def _assert_has_text(parsed: Dict[str, Any]) -> None:
    """Assert that results contain text content."""
    for r in parsed["results"]:
        assert r.get("text") is not None, f"Result missing 'text': {r.get('title')}"
        assert isinstance(r["text"], str)
        assert len(r["text"]) > 0


def _assert_has_highlights(parsed: Dict[str, Any]) -> None:
    """Assert that results contain highlights."""
    for r in parsed["results"]:
        assert r.get("highlights") is not None, f"Result missing 'highlights': {r.get('title')}"
        assert isinstance(r["highlights"], list)


def _assert_answer_response(parsed: Dict[str, Any]) -> None:
    """Assert that an answer response has the expected structure."""
    assert "answer" in parsed, f"Missing 'answer' key. Keys: {list(parsed.keys())}"
    assert isinstance(parsed["answer"], str)
    assert len(parsed["answer"]) > 0
    assert "citations" in parsed, f"Missing 'citations' key. Keys: {list(parsed.keys())}"
    assert isinstance(parsed["citations"], list)


def _process_toolkit(toolkit: Any) -> None:
    from upsonic.tools import ToolManager
    ToolManager().register_tools([toolkit])


# ──────────────────────────────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────────────────────────────

class TestInit:

    def test_init_with_api_key(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY)
        assert t.api_key == EXA_API_KEY

    def test_init_defaults(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY)
        assert t.default_num_results == 10
        assert t.default_search_type == "auto"
        assert t.default_text is True
        assert t.default_highlights is False
        assert t.default_summary is False

    def test_init_custom_config(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(
            api_key=EXA_API_KEY,
            default_num_results=5,
            default_search_type="neural",
            default_text=False,
            default_highlights=True,
            default_summary=True,
        )
        assert t.default_num_results == 5
        assert t.default_search_type == "neural"
        assert t.default_text is False
        assert t.default_highlights is True
        assert t.default_summary is True

    def test_init_missing_key_raises(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        from unittest.mock import patch as _patch
        with _patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Exa API key is required"):
                ExaTools(api_key="")

    def test_init_from_env_var(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        from unittest.mock import patch as _patch
        with _patch.dict("os.environ", {"EXA_API_KEY": "test-key-123"}, clear=False):
            t: ExaTools = ExaTools()
            assert t.api_key == "test-key-123"


# ──────────────────────────────────────────────────────────────────────
#  functions / exclude_tools filtering
# ──────────────────────────────────────────────────────────────────────

class TestFunctions:

    def test_default_functions(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        for expected in ALL_TOOL_NAMES:
            assert expected in names, f"{expected} missing from default functions"

    def test_all_tools_count(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert len(names) == 4

    def test_exclude_all_except_search(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "search"]
        t: ExaTools = ExaTools(api_key=EXA_API_KEY, exclude_tools=excluded)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["search"]

    def test_exclude_all_except_answer(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "answer"]
        t: ExaTools = ExaTools(api_key=EXA_API_KEY, exclude_tools=excluded)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["answer"]

    def test_exclude_all_tools(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY, exclude_tools=ALL_TOOL_NAMES)
        _process_toolkit(t)
        assert t.functions == []

    def test_exclude_search_keeps_rest(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY, exclude_tools=["search"])
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "search" not in names
        assert len(names) == 3

    def test_include_tools_is_additive(self) -> None:
        from upsonic.tools.custom_tools.exa import ExaTools
        t: ExaTools = ExaTools(api_key=EXA_API_KEY, include_tools=["search"])
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "search" in names
        assert len(names) == len(ALL_TOOL_NAMES)


# ──────────────────────────────────────────────────────────────────────
#  search — all parameters (sync)
# ──────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_basic(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY))
        _assert_search_results(parsed)
        _assert_has_text(parsed)  # default_text=True

    def test_num_results(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=3))
        _assert_search_results(parsed)
        assert len(parsed["results"]) <= 3

    def test_search_type_neural(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=3, search_type="neural"))
        _assert_search_results(parsed)

    def test_search_type_keyword(self, tools: Any) -> None:
        parsed = _parse(tools.search("Python programming", num_results=3, search_type="keyword"))
        _assert_search_results(parsed)

    def test_search_type_auto(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=3, search_type="auto"))
        _assert_search_results(parsed)

    def test_category_news(self, tools: Any) -> None:
        parsed = _parse(tools.search("artificial intelligence", num_results=3, category="news"))
        _assert_search_results(parsed)

    def test_category_research_paper(self, tools: Any) -> None:
        parsed = _parse(tools.search("transformer architecture", num_results=3, category="research paper"))
        _assert_search_results(parsed)

    def test_category_company(self, tools: Any) -> None:
        parsed = _parse(tools.search("AI startup healthcare", num_results=3, category="company"))
        _assert_search_results(parsed)

    def test_include_domains(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            TEST_SEARCH_QUERY, num_results=3,
            include_domains=["github.com", "stackoverflow.com"],
        ))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            url = r["url"].lower()
            assert "github.com" in url or "stackoverflow.com" in url, f"Unexpected domain: {r['url']}"

    def test_exclude_domains(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            TEST_SEARCH_QUERY, num_results=5,
            exclude_domains=["wikipedia.org"],
        ))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert "wikipedia.org" not in r["url"].lower(), f"Excluded domain found: {r['url']}"

    def test_include_and_exclude_domains(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            TEST_SEARCH_QUERY, num_results=3,
            include_domains=["github.com"],
            exclude_domains=["gist.github.com"],
        ))
        _assert_search_results(parsed)

    def test_start_published_date(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            "artificial intelligence", num_results=3,
            start_published_date="2024-01-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    def test_end_published_date(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            "artificial intelligence", num_results=3,
            end_published_date="2025-12-31T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    def test_start_and_end_published_date(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            "artificial intelligence", num_results=3,
            start_published_date="2024-01-01T00:00:00.000Z",
            end_published_date="2025-06-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    def test_include_text(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            "machine learning", num_results=3,
            include_text=["neural network"],
        ))
        _assert_search_results(parsed)

    def test_exclude_text(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            "machine learning", num_results=3,
            exclude_text=["cryptocurrency"],
        ))
        _assert_search_results(parsed)

    def test_text_true(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=2, text=True))
        _assert_search_results(parsed)
        _assert_has_text(parsed)

    def test_text_false(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=2, text=False))
        _assert_search_results(parsed)
        # With text=False, results should not have text content
        for r in parsed["results"]:
            assert r.get("text") is None or r.get("text") == "", \
                f"Expected no text when text=False, got: {str(r.get('text'))[:100]}"

    def test_highlights_true(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=2, highlights=True))
        _assert_search_results(parsed)
        _assert_has_highlights(parsed)

    def test_summary_true(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=2, summary=True))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("summary") is not None, f"Result missing 'summary': {r.get('title')}"

    def test_max_characters(self, tools: Any) -> None:
        parsed = _parse(tools.search(TEST_SEARCH_QUERY, num_results=2, max_characters=500))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        for r in parsed["results"]:
            # Allow some tolerance for the max_characters limit
            assert len(r["text"]) <= 600, \
                f"Text too long ({len(r['text'])} chars) with max_characters=500"

    def test_max_age_hours(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            "latest AI news", num_results=2, max_age_hours=24,
        ))
        _assert_search_results(parsed)

    def test_all_content_options_combined(self, tools: Any) -> None:
        parsed = _parse(tools.search(
            TEST_SEARCH_QUERY,
            num_results=2,
            text=True,
            highlights=True,
            summary=True,
            max_characters=1000,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)
        for r in parsed["results"]:
            assert r.get("summary") is not None

    def test_all_parameters_combined(self, tools: Any) -> None:
        """Test search with every parameter set at once."""
        parsed = _parse(tools.search(
            query="Python web framework",
            num_results=3,
            search_type="auto",
            include_domains=["github.com"],
            start_published_date="2023-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
            include_text=["python"],
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
            max_age_hours=168,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)


# ──────────────────────────────────────────────────────────────────────
#  search — async
# ──────────────────────────────────────────────────────────────────────

class TestSearchAsync:

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed = _parse(await tools.asearch(TEST_SEARCH_QUERY, num_results=3))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_category(self, tools: Any) -> None:
        parsed = _parse(await tools.asearch(
            "artificial intelligence", num_results=3, category="news",
        ))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_domains(self, tools: Any) -> None:
        parsed = _parse(await tools.asearch(
            TEST_SEARCH_QUERY, num_results=3,
            include_domains=["github.com"],
            exclude_domains=["gist.github.com"],
        ))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_dates(self, tools: Any) -> None:
        parsed = _parse(await tools.asearch(
            "AI safety", num_results=3,
            start_published_date="2024-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_text_filters(self, tools: Any) -> None:
        parsed = _parse(await tools.asearch(
            "machine learning", num_results=3,
            include_text=["deep learning"],
            exclude_text=["cryptocurrency"],
        ))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_all_content_options(self, tools: Any) -> None:
        parsed = _parse(await tools.asearch(
            TEST_SEARCH_QUERY, num_results=2,
            text=True, highlights=True, summary=True,
            max_characters=500,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)

    @pytest.mark.asyncio
    async def test_async_all_parameters(self, tools: Any) -> None:
        """Async search with every parameter set at once."""
        parsed = _parse(await tools.asearch(
            query="Python web framework",
            num_results=3,
            search_type="auto",
            include_domains=["github.com"],
            start_published_date="2023-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
            include_text=["python"],
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
            max_age_hours=168,
        ))
        _assert_search_results(parsed)


# ──────────────────────────────────────────────────────────────────────
#  get_contents — all parameters (sync)
# ──────────────────────────────────────────────────────────────────────

class TestGetContents:

    def test_basic(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL]))
        _assert_search_results(parsed)
        _assert_has_text(parsed)

    def test_multiple_urls(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents(BATCH_URLS))
        _assert_search_results(parsed, min_results=2)

    def test_text_true(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], text=True))
        _assert_search_results(parsed)
        _assert_has_text(parsed)

    def test_text_false(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], text=False))
        _assert_search_results(parsed)
        # When text=False and no other content option is set, summary is used
        # as fallback to prevent the SDK from injecting default text.
        for r in parsed["results"]:
            assert r.get("text") is None or r.get("text") == "" or r.get("summary") is not None

    def test_highlights_true(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], highlights=True))
        _assert_search_results(parsed)
        _assert_has_highlights(parsed)

    def test_summary_true(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], summary=True))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("summary") is not None

    def test_max_characters(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], max_characters=300))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        for r in parsed["results"]:
            assert len(r["text"]) <= 400, \
                f"Text too long ({len(r['text'])} chars) with max_characters=300"

    def test_max_age_hours(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], max_age_hours=168))
        _assert_search_results(parsed)

    def test_max_age_hours_zero_always_fresh(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], max_age_hours=0))
        _assert_search_results(parsed)

    def test_subpages(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents([TEST_CONTENT_URL], subpages=2))
        _assert_search_results(parsed)

    def test_subpage_target(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents(
            [TEST_CONTENT_URL], subpages=1, subpage_target="search",
        ))
        _assert_search_results(parsed)

    def test_all_content_options_combined(self, tools: Any) -> None:
        parsed = _parse(tools.get_contents(
            [TEST_CONTENT_URL],
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)

    def test_all_parameters_combined(self, tools: Any) -> None:
        """Test get_contents with every parameter set at once."""
        parsed = _parse(tools.get_contents(
            urls=BATCH_URLS,
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
            max_age_hours=168,
            subpages=1,
            subpage_target="getting started",
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)


# ──────────────────────────────────────────────────────────────────────
#  get_contents — async
# ──────────────────────────────────────────────────────────────────────

class TestGetContentsAsync:

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed = _parse(await tools.aget_contents([TEST_CONTENT_URL]))
        _assert_search_results(parsed)
        _assert_has_text(parsed)

    @pytest.mark.asyncio
    async def test_async_multiple_urls(self, tools: Any) -> None:
        parsed = _parse(await tools.aget_contents(BATCH_URLS))
        _assert_search_results(parsed, min_results=2)

    @pytest.mark.asyncio
    async def test_async_with_highlights(self, tools: Any) -> None:
        parsed = _parse(await tools.aget_contents([TEST_CONTENT_URL], highlights=True))
        _assert_search_results(parsed)
        _assert_has_highlights(parsed)

    @pytest.mark.asyncio
    async def test_async_with_summary(self, tools: Any) -> None:
        parsed = _parse(await tools.aget_contents([TEST_CONTENT_URL], summary=True))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("summary") is not None

    @pytest.mark.asyncio
    async def test_async_all_parameters(self, tools: Any) -> None:
        """Async get_contents with every parameter set at once."""
        parsed = _parse(await tools.aget_contents(
            urls=BATCH_URLS,
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
            max_age_hours=168,
            subpages=1,
            subpage_target="getting started",
        ))
        _assert_search_results(parsed)


# ──────────────────────────────────────────────────────────────────────
#  find_similar — all parameters (sync)
# ──────────────────────────────────────────────────────────────────────

class TestFindSimilar:

    def test_basic(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL))
        _assert_search_results(parsed)

    def test_num_results(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL, num_results=3))
        _assert_search_results(parsed)
        assert len(parsed["results"]) <= 3

    def test_include_domains(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(
            TEST_URL, num_results=3,
            include_domains=["github.com"],
        ))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert "github.com" in r["url"].lower(), f"Unexpected domain: {r['url']}"

    def test_exclude_domains(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(
            TEST_URL, num_results=5,
            exclude_domains=["pypi.org"],
        ))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert "pypi.org" not in r["url"].lower(), f"Excluded domain found: {r['url']}"

    def test_start_published_date(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(
            TEST_URL, num_results=3,
            start_published_date="2023-01-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    def test_end_published_date(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(
            TEST_URL, num_results=3,
            end_published_date="2026-01-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    def test_start_and_end_published_date(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(
            TEST_URL, num_results=3,
            start_published_date="2023-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    def test_text_true(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL, num_results=2, text=True))
        _assert_search_results(parsed)
        _assert_has_text(parsed)

    def test_text_false(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL, num_results=2, text=False))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("text") is None or r.get("text") == ""

    def test_highlights_true(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL, num_results=2, highlights=True))
        _assert_search_results(parsed)
        _assert_has_highlights(parsed)

    def test_summary_true(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL, num_results=2, summary=True))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("summary") is not None

    def test_max_characters(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(TEST_URL, num_results=2, max_characters=500))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        for r in parsed["results"]:
            assert len(r["text"]) <= 600

    def test_all_content_options_combined(self, tools: Any) -> None:
        parsed = _parse(tools.find_similar(
            TEST_URL, num_results=2,
            text=True, highlights=True, summary=True,
            max_characters=500,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)

    def test_all_parameters_combined(self, tools: Any) -> None:
        """Test find_similar with every parameter set at once."""
        parsed = _parse(tools.find_similar(
            url=TEST_URL,
            num_results=3,
            include_domains=["github.com"],
            start_published_date="2023-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)


# ──────────────────────────────────────────────────────────────────────
#  find_similar — async
# ──────────────────────────────────────────────────────────────────────

class TestFindSimilarAsync:

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed = _parse(await tools.afind_similar(TEST_URL, num_results=3))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_domains(self, tools: Any) -> None:
        parsed = _parse(await tools.afind_similar(
            TEST_URL, num_results=3,
            include_domains=["github.com"],
            exclude_domains=["gist.github.com"],
        ))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_dates(self, tools: Any) -> None:
        parsed = _parse(await tools.afind_similar(
            TEST_URL, num_results=3,
            start_published_date="2023-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
        ))
        _assert_search_results(parsed)

    @pytest.mark.asyncio
    async def test_async_with_all_content_options(self, tools: Any) -> None:
        parsed = _parse(await tools.afind_similar(
            TEST_URL, num_results=2,
            text=True, highlights=True, summary=True,
            max_characters=500,
        ))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)

    @pytest.mark.asyncio
    async def test_async_all_parameters(self, tools: Any) -> None:
        """Async find_similar with every parameter set at once."""
        parsed = _parse(await tools.afind_similar(
            url=TEST_URL,
            num_results=3,
            include_domains=["github.com"],
            start_published_date="2023-01-01T00:00:00.000Z",
            end_published_date="2026-01-01T00:00:00.000Z",
            text=True,
            highlights=True,
            summary=True,
            max_characters=500,
        ))
        _assert_search_results(parsed)


# ──────────────────────────────────────────────────────────────────────
#  answer — all parameters (sync)
# ──────────────────────────────────────────────────────────────────────

class TestAnswer:

    def test_basic(self, tools: Any) -> None:
        parsed = _parse(tools.answer("What is Python programming language?"))
        _assert_answer_response(parsed)

    def test_text_true(self, tools: Any) -> None:
        parsed = _parse(tools.answer("What is Exa AI?", text=True))
        _assert_answer_response(parsed)
        # When text=True, citations should include text
        for c in parsed["citations"]:
            assert c.get("text") is not None, f"Citation missing 'text': {c.get('title')}"

    def test_text_false(self, tools: Any) -> None:
        parsed = _parse(tools.answer("What is Python?", text=False))
        _assert_answer_response(parsed)
        # When text=False, citations should not have full text
        for c in parsed["citations"]:
            assert c.get("text") is None or c.get("text") == "", \
                f"Expected no text in citation when text=False"

    def test_long_query(self, tools: Any) -> None:
        parsed = _parse(tools.answer(
            "What are the main differences between Python and JavaScript for web development, "
            "and which one is better for building REST APIs?"
        ))
        _assert_answer_response(parsed)

    def test_answer_has_citations_with_urls(self, tools: Any) -> None:
        parsed = _parse(tools.answer("What is machine learning?"))
        _assert_answer_response(parsed)
        for c in parsed["citations"]:
            assert "url" in c, f"Citation missing 'url': {list(c.keys())}"
            assert isinstance(c["url"], str)


# ──────────────────────────────────────────────────────────────────────
#  answer — async
# ──────────────────────────────────────────────────────────────────────

class TestAnswerAsync:

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed = _parse(await tools.aanswer("What is Python programming language?"))
        _assert_answer_response(parsed)

    @pytest.mark.asyncio
    async def test_async_text_true(self, tools: Any) -> None:
        parsed = _parse(await tools.aanswer("What is Exa AI?", text=True))
        _assert_answer_response(parsed)
        for c in parsed["citations"]:
            assert c.get("text") is not None

    @pytest.mark.asyncio
    async def test_async_text_false(self, tools: Any) -> None:
        parsed = _parse(await tools.aanswer("What is Python?", text=False))
        _assert_answer_response(parsed)


# ──────────────────────────────────────────────────────────────────────
#  Default config behavior tests
# ──────────────────────────────────────────────────────────────────────

class TestDefaultConfigBehavior:

    def test_default_text_true_includes_text(self) -> None:
        """When default_text=True (default), search results should include text."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_text=True)
        parsed = _parse(t.search(TEST_SEARCH_QUERY, num_results=2))
        _assert_search_results(parsed)
        _assert_has_text(parsed)

    def test_default_text_false_excludes_text(self) -> None:
        """When default_text=False, search results should not include text."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_text=False)
        parsed = _parse(t.search(TEST_SEARCH_QUERY, num_results=2))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("text") is None or r.get("text") == ""

    def test_default_highlights_true(self) -> None:
        """When default_highlights=True, search results should include highlights."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_highlights=True)
        parsed = _parse(t.search(TEST_SEARCH_QUERY, num_results=2))
        _assert_search_results(parsed)
        _assert_has_highlights(parsed)

    def test_default_summary_true(self) -> None:
        """When default_summary=True, search results should include summaries."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_summary=True)
        parsed = _parse(t.search(TEST_SEARCH_QUERY, num_results=2))
        _assert_search_results(parsed)
        for r in parsed["results"]:
            assert r.get("summary") is not None

    def test_default_num_results(self) -> None:
        """When default_num_results=5, search should return at most 5 results."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_num_results=5)
        parsed = _parse(t.search(TEST_SEARCH_QUERY))
        _assert_search_results(parsed)
        assert len(parsed["results"]) <= 5

    def test_default_search_type_neural(self) -> None:
        """When default_search_type='neural', search should use neural type."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_search_type="neural")
        parsed = _parse(t.search(TEST_SEARCH_QUERY, num_results=2))
        _assert_search_results(parsed)

    def test_parameter_overrides_default(self) -> None:
        """Explicit parameter should override default config."""
        from upsonic.tools.custom_tools.exa import ExaTools
        t = ExaTools(api_key=EXA_API_KEY, default_text=False, default_highlights=False)
        # Explicitly request text and highlights even though defaults are False
        parsed = _parse(t.search(TEST_SEARCH_QUERY, num_results=2, text=True, highlights=True))
        _assert_search_results(parsed)
        _assert_has_text(parsed)
        _assert_has_highlights(parsed)


# ──────────────────────────────────────────────────────────────────────
#  Agent + Task integration tests (real LLM + real Exa API calls)
# ──────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

agent_task_skip = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set; skipping Agent/Task integration tests",
)

MODEL: str = "anthropic/claude-sonnet-4-5"


@pytest.fixture(scope="module")
def exa_tools() -> Any:
    from upsonic.tools.custom_tools.exa import ExaTools
    return ExaTools(api_key=EXA_API_KEY, default_num_results=3, default_highlights=False, default_summary=False)


class TestAgentWithExaTools:
    """Test that ExaTools works correctly when passed to an Agent."""

    @agent_task_skip
    def test_agent_search_tool(self, exa_tools: Any) -> None:
        """Agent uses the Exa search tool to answer a question."""
        from upsonic import Agent, Task

        agent = Agent(MODEL, tools=[exa_tools])
        task = Task(
            "Use the search tool to find information about 'Upsonic AI agent framework'. "
            "Return the title and URL of the first result.",
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        # Verify the agent actually called an Exa tool
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "search" in tool_names, f"Expected 'search' in tool_calls, got: {tool_names}"

    @agent_task_skip
    def test_agent_get_contents_tool(self, exa_tools: Any) -> None:
        """Agent uses the Exa get_contents tool to fetch a URL."""
        from upsonic import Agent, Task

        agent = Agent(MODEL, tools=[exa_tools])
        task = Task(
            "Use the get_contents tool to fetch the content of https://docs.exa.ai/reference/getting-started "
            "and tell me the first heading on the page.",
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "get_contents" in tool_names, f"Expected 'get_contents' in tool_calls, got: {tool_names}"

    @agent_task_skip
    def test_agent_find_similar_tool(self, exa_tools: Any) -> None:
        """Agent uses the Exa find_similar tool to find related pages."""
        from upsonic import Agent, Task

        agent = Agent(MODEL, tools=[exa_tools])
        task = Task(
            "Use the find_similar tool to find pages similar to https://github.com/anthropics/anthropic-sdk-python. "
            "List the titles of the results.",
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "find_similar" in tool_names, f"Expected 'find_similar' in tool_calls, got: {tool_names}"

    @agent_task_skip
    def test_agent_answer_tool(self, exa_tools: Any) -> None:
        """Agent uses the Exa answer tool to get a cited answer."""
        from upsonic import Agent, Task

        agent = Agent(MODEL, tools=[exa_tools])
        task = Task(
            "Use the answer tool to answer: 'What is Exa AI?'. Return the answer you get.",
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "answer" in tool_names, f"Expected 'answer' in tool_calls, got: {tool_names}"


class TestTaskWithExaTools:
    """Test that ExaTools works correctly when passed directly to a Task."""

    @agent_task_skip
    def test_task_level_search_tool(self) -> None:
        """ExaTools passed at the Task level (not Agent level)."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.exa import ExaTools

        exa = ExaTools(api_key=EXA_API_KEY, default_num_results=3)
        agent = Agent(MODEL)
        task = Task(
            "Use the search tool to search for 'Python asyncio tutorial'. "
            "Return the title of the first result.",
            tools=[exa],
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "search" in tool_names, f"Expected 'search' in tool_calls, got: {tool_names}"

    @agent_task_skip
    def test_task_level_get_contents_tool(self) -> None:
        """get_contents passed at the Task level."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.exa import ExaTools

        exa = ExaTools(api_key=EXA_API_KEY)
        agent = Agent(MODEL)
        task = Task(
            "Use the get_contents tool to retrieve the content from "
            "https://github.com/anthropics/anthropic-sdk-python and summarize it in one sentence.",
            tools=[exa],
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "get_contents" in tool_names, f"Expected 'get_contents' in tool_calls, got: {tool_names}"

    @agent_task_skip
    def test_task_level_answer_tool(self) -> None:
        """answer tool passed at the Task level."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.exa import ExaTools

        exa = ExaTools(api_key=EXA_API_KEY)
        agent = Agent(MODEL)
        task = Task(
            "Use the answer tool to answer: 'What are the main features of FastAPI?'. "
            "Return the answer.",
            tools=[exa],
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "answer" in tool_names, f"Expected 'answer' in tool_calls, got: {tool_names}"

    @agent_task_skip
    def test_task_with_exclude_tools(self) -> None:
        """ExaTools with exclude_tools should only expose allowed tools."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.exa import ExaTools

        # Only expose search and answer, exclude get_contents and find_similar
        exa = ExaTools(
            api_key=EXA_API_KEY,
            default_num_results=3,
            exclude_tools=["get_contents", "find_similar"],
        )
        agent = Agent(MODEL, tools=[exa])
        task = Task(
            "Use the search tool to find 'Exa AI search engine'. "
            "Return the URL of the first result.",
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "search" in tool_names
        # Ensure excluded tools were NOT called
        assert "get_contents" not in tool_names
        assert "find_similar" not in tool_names

    @agent_task_skip
    def test_agent_multi_tool_chain(self, exa_tools: Any) -> None:
        """Agent uses multiple Exa tools in a single task (search then get_contents)."""
        from upsonic import Agent, Task

        agent = Agent(MODEL, tools=[exa_tools])
        task = Task(
            "First, use the search tool to find 'Anthropic Claude documentation'. "
            "Then use get_contents to fetch the content of the first URL from the search results. "
            "Summarize the fetched content in one sentence.",
            agent=agent,
        )
        result = agent.do(task)

        assert result is not None
        result_str = str(result)
        assert len(result_str) > 0
        tool_names = [tc["tool_name"] for tc in task.tool_calls]
        assert "search" in tool_names, f"Expected 'search' in tool_calls, got: {tool_names}"
        assert "get_contents" in tool_names, f"Expected 'get_contents' in tool_calls, got: {tool_names}"
