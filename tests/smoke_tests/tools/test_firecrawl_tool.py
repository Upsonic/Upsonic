"""
Smoke tests for FirecrawlTools — real API requests against Firecrawl.

Requires a valid FIRECRAWL_API_KEY environment variable.
Automatically skipped when the key is not set.

Coverage:
- Constructor and exclude_tools filtering (include_tools is additive)
- Every sync and async method with ALL parameters
- Every scrape format mode: markdown, links, html, rawHtml, summary, images
- Crawling with all options (blocking + non-blocking + management)
- Mapping with/without limit
- Searching with all options (limit, scrape_options, location, tbs, timeout, sources, categories)
- Batch scraping with all formats (blocking + non-blocking + management)
- Extraction with all options (blocking + non-blocking + management)
- Proper output structure assertions for every method

Uses www.nike.com.tr as the test target.

Run:
    uv run pytest tests/smoke_tests/tools/test_firecrawl_tool.py -v --tb=short
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import pytest

FIRECRAWL_API_KEY: Optional[str] = os.getenv("FIRECRAWL_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not FIRECRAWL_API_KEY,
        reason="FIRECRAWL_API_KEY not set; skipping Firecrawl smoke tests",
    ),
]

TEST_URL: str = "https://www.nike.com.tr"
TEST_SEARCH_QUERY: str = "Python programming language"
BATCH_URLS: List[str] = ["https://www.nike.com.tr", "https://www.nike.com.tr/w/erkek-6yleepznik1"]
ALL_FORMATS: List[str] = ["markdown", "links", "html", "rawHtml", "summary", "images"]

CRAWL_RATE_LIMIT_SLEEP: int = 25


@pytest.fixture(scope="module")
def tools() -> Any:
    from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
    return FirecrawlTools(api_key=FIRECRAWL_API_KEY)


_SKIP_ERROR_PATTERNS: List[str] = [
    "Rate Limit",
    "rate limit",
    "Insufficient credits",
    "insufficient credits",
]


def _parse(raw: str) -> Dict[str, Any]:
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


def _assert_document(parsed: Dict[str, Any]) -> None:
    assert "metadata" in parsed
    metadata: Any = parsed["metadata"]
    assert isinstance(metadata, dict)
    assert "url" in metadata


def _assert_crawl_job(parsed: Dict[str, Any]) -> None:
    assert "status" in parsed
    assert parsed["status"] in ("scraping", "completed", "failed", "cancelled")
    assert "total" in parsed
    assert isinstance(parsed["total"], int)
    assert "completed" in parsed
    assert isinstance(parsed["completed"], int)
    assert "data" in parsed
    assert isinstance(parsed["data"], list)


def _assert_crawl_response(parsed: Dict[str, Any]) -> None:
    assert "id" in parsed
    assert isinstance(parsed["id"], str)
    assert len(parsed["id"]) > 0
    assert "url" in parsed
    assert isinstance(parsed["url"], str)


def _assert_map_data(parsed: Dict[str, Any]) -> None:
    assert "links" in parsed
    assert isinstance(parsed["links"], list)


def _assert_search_data(parsed: Dict[str, Any]) -> None:
    has_results: bool = (
        parsed.get("web") is not None
        or parsed.get("news") is not None
        or parsed.get("images") is not None
    )
    assert has_results, f"SearchData has no result lists: {list(parsed.keys())}"


def _assert_batch_scrape_job(parsed: Dict[str, Any]) -> None:
    assert "status" in parsed
    assert parsed["status"] in ("scraping", "completed", "failed", "cancelled")
    assert "total" in parsed
    assert isinstance(parsed["total"], int)
    assert "completed" in parsed
    assert isinstance(parsed["completed"], int)
    assert "data" in parsed
    assert isinstance(parsed["data"], list)


def _assert_batch_scrape_response(parsed: Dict[str, Any]) -> None:
    assert "id" in parsed
    assert isinstance(parsed["id"], str)
    assert len(parsed["id"]) > 0
    assert "url" in parsed


def _assert_extract_response(parsed: Dict[str, Any]) -> None:
    assert "success" in parsed or "status" in parsed or "data" in parsed


def _assert_extract_job_started(parsed: Dict[str, Any]) -> None:
    assert "id" in parsed
    assert isinstance(parsed["id"], str)
    assert len(parsed["id"]) > 0


def _assert_extract_status(parsed: Dict[str, Any]) -> None:
    assert "status" in parsed
    assert parsed["status"] in ("processing", "completed", "failed", "cancelled")


def _process_toolkit(toolkit: Any) -> None:
    from upsonic.tools import ToolManager
    ToolManager().register_tools([toolkit])


# ──────────────────────────────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────────────────────────────

ALL_TOOL_NAMES: List[str] = [
    "scrape_url", "crawl_website", "start_crawl",
    "map_website", "search_web",
    "batch_scrape", "start_batch_scrape",
    "extract_data", "start_extract",
    "get_crawl_status", "cancel_crawl",
    "get_batch_scrape_status", "get_extract_status",
]


class TestInit:

    def test_init_with_api_key(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        assert t.api_key == FIRECRAWL_API_KEY

    def test_init_defaults(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        assert t.default_formats == ["markdown"]
        assert t.default_scrape_limit == 100
        assert t.default_search_limit == 5
        assert t.fc_timeout == 120
        assert t.poll_interval == 2
        assert t.api_url is None

    def test_init_custom_config(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            api_url="https://custom.firecrawl.example.com",
            default_formats=["markdown", "html"],
            default_scrape_limit=50,
            default_search_limit=10,
            fc_timeout=60,
            poll_interval=5,
        )
        assert t.api_url == "https://custom.firecrawl.example.com"
        assert t.default_formats == ["markdown", "html"]
        assert t.default_scrape_limit == 50
        assert t.default_search_limit == 10
        assert t.fc_timeout == 60
        assert t.poll_interval == 5

    def test_init_missing_key_raises(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        from unittest.mock import patch as _patch
        with _patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Firecrawl API key is required"):
                FirecrawlTools(api_key="")


# ──────────────────────────────────────────────────────────────────────
#  functions / exclude_tools filtering (include_tools is additive)
# ──────────────────────────────────────────────────────────────────────

class TestFunctions:

    def test_default_functions(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        for expected in ALL_TOOL_NAMES:
            assert expected in names, f"{expected} missing from default functions"

    def test_all_tools_count(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert len(names) == 13

    def test_include_tools_is_additive(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            include_tools=["scrape_url"],
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" in names, "included tool should be present"
        assert len(names) == len(ALL_TOOL_NAMES), (
            "include_tools is additive -- all @tool methods + included should be registered"
        )

    def test_exclude_all_except_scrape(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "scrape_url"]
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=excluded,
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["scrape_url"]

    def test_exclude_all_except_search(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "search_web"]
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=excluded,
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["search_web"]

    def test_exclude_all_except_crawl_management(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        keep: set = {"crawl_website", "start_crawl", "get_crawl_status", "cancel_crawl"}
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n not in keep]
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=excluded,
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert set(names) == keep

    def test_exclude_all_tools(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=ALL_TOOL_NAMES,
        )
        _process_toolkit(t)
        assert t.functions == []

    def test_exclude_scrape_keeps_rest(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=["scrape_url"],
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" not in names
        assert len(names) == 12

    def test_exclude_batch_tools(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=["batch_scrape", "start_batch_scrape", "get_batch_scrape_status"],
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "batch_scrape" not in names
        assert "start_batch_scrape" not in names
        assert "get_batch_scrape_status" not in names
        assert len(names) == 10

    def test_exclude_extract_tools(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            exclude_tools=["extract_data", "start_extract", "get_extract_status"],
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "extract_data" not in names
        assert "start_extract" not in names
        assert "get_extract_status" not in names
        assert len(names) == 10


# ──────────────────────────────────────────────────────────────────────
#  scrape_url — all formats (sync)
# ──────────────────────────────────────────────────────────────────────

class TestScrapeUrlFormats:

    def test_format_markdown(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=["markdown"]))
        _assert_document(parsed)
        assert parsed.get("markdown") is not None
        assert isinstance(parsed["markdown"], str)
        assert len(parsed["markdown"]) > 0

    def test_format_links(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=["links"]))
        _assert_document(parsed)
        assert "links" in parsed
        assert isinstance(parsed["links"], list)

    def test_format_html(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=["html"]))
        _assert_document(parsed)
        assert parsed.get("html") is not None
        assert isinstance(parsed["html"], str)
        assert "<" in parsed["html"]

    def test_format_rawHtml(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=["rawHtml"]))
        _assert_document(parsed)
        assert parsed.get("raw_html") is not None
        assert isinstance(parsed["raw_html"], str)
        assert "<" in parsed["raw_html"]

    def test_format_summary(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=["summary"]))
        _assert_document(parsed)
        assert parsed.get("summary") is not None
        assert isinstance(parsed["summary"], str)

    def test_format_images(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=["images"]))
        _assert_document(parsed)
        assert "images" in parsed

    def test_all_formats_combined(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, formats=ALL_FORMATS))
        _assert_document(parsed)
        assert parsed.get("markdown") is not None
        assert parsed.get("html") is not None
        assert parsed.get("raw_html") is not None
        assert "links" in parsed
        assert "images" in parsed


# ──────────────────────────────────────────────────────────────────────
#  scrape_url — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestScrapeUrlAttributes:

    def test_only_main_content(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, only_main_content=True))
        _assert_document(parsed)

    def test_include_tags(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, include_tags=["h1", "p"]))
        _assert_document(parsed)

    def test_exclude_tags(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, exclude_tags=["nav", "footer"]))
        _assert_document(parsed)

    def test_include_and_exclude_tags(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(
            TEST_URL,
            include_tags=["h1", "p", "a"],
            exclude_tags=["nav", "footer", "script"],
        ))
        _assert_document(parsed)

    def test_wait_for(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, wait_for=1000))
        _assert_document(parsed)

    def test_timeout(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, timeout=30000))
        _assert_document(parsed)

    def test_location(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, location={"country": "TR"}))
        _assert_document(parsed)

    def test_mobile(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, mobile=True))
        _assert_document(parsed)

    def test_skip_tls_verification(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, skip_tls_verification=True))
        _assert_document(parsed)

    def test_remove_base64_images(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, remove_base64_images=True))
        _assert_document(parsed)

    def test_json_schema_extraction(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
        }
        parsed: Dict[str, Any] = _parse(tools.scrape_url(
            TEST_URL,
            json_schema=schema,
            json_prompt="Extract the page title and description",
        ))
        _assert_document(parsed)
        assert parsed.get("json") is not None

    def test_all_attributes_combined(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(
            TEST_URL,
            formats=["markdown", "html"],
            only_main_content=True,
            include_tags=["h1", "p"],
            exclude_tags=["nav"],
            wait_for=500,
            timeout=30000,
            location={"country": "TR"},
            mobile=False,
            skip_tls_verification=False,
            remove_base64_images=True,
        ))
        _assert_document(parsed)
        assert parsed.get("markdown") is not None
        assert parsed.get("html") is not None

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(TEST_URL))
        _assert_document(parsed)
        assert parsed.get("markdown") is not None

    @pytest.mark.asyncio
    async def test_async_all_formats(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(TEST_URL, formats=ALL_FORMATS))
        _assert_document(parsed)
        assert parsed.get("markdown") is not None
        assert parsed.get("html") is not None
        assert "links" in parsed

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(
            TEST_URL,
            formats=["markdown"],
            only_main_content=True,
            include_tags=["h1"],
            exclude_tags=["footer"],
            wait_for=500,
            timeout=30000,
            location={"country": "TR"},
            mobile=True,
            skip_tls_verification=False,
            remove_base64_images=True,
        ))
        _assert_document(parsed)

    @pytest.mark.asyncio
    async def test_async_json_schema(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(
            TEST_URL,
            json_schema=schema,
            json_prompt="Extract the page title",
        ))
        _assert_document(parsed)
        assert parsed.get("json") is not None


# ──────────────────────────────────────────────────────────────────────
#  crawl_website — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestCrawlWebsite:

    @pytest.fixture(autouse=True)
    def _rate_limit(self) -> None:
        time.sleep(CRAWL_RATE_LIMIT_SLEEP)

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(TEST_URL, limit=2, max_discovery_depth=1))
        _assert_crawl_job(parsed)

    def test_with_scrape_formats(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, limit=2, max_discovery_depth=1, scrape_formats=["markdown", "links"],
        ))
        _assert_crawl_job(parsed)

    def test_with_exclude_paths(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, limit=2, max_discovery_depth=1, exclude_paths=["/private/*"],
        ))
        _assert_crawl_job(parsed)

    def test_with_include_paths(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, limit=2, max_discovery_depth=1, include_paths=["/*"],
        ))
        _assert_crawl_job(parsed)

    def test_with_sitemap_skip(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, limit=2, max_discovery_depth=1, sitemap="skip",
        ))
        _assert_crawl_job(parsed)

    def test_with_poll_interval_and_timeout(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, limit=2, max_discovery_depth=1, poll_interval=3, timeout=60,
        ))
        _assert_crawl_job(parsed)

    def test_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL,
            limit=2,
            scrape_formats=["markdown"],
            exclude_paths=["/admin/*"],
            include_paths=["/*"],
            max_discovery_depth=1,
            sitemap="skip",
            poll_interval=3,
            timeout=60,
        ))
        _assert_crawl_job(parsed)

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.acrawl_website(TEST_URL, limit=2, max_discovery_depth=1))
        _assert_crawl_job(parsed)

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.acrawl_website(
            TEST_URL,
            limit=2,
            scrape_formats=["markdown", "html"],
            exclude_paths=["/private/*"],
            include_paths=["/*"],
            max_discovery_depth=1,
            sitemap="skip",
            poll_interval=3,
            timeout=60,
        ))
        _assert_crawl_job(parsed)


# ──────────────────────────────────────────────────────────────────────
#  Crawl management: start_crawl / get_crawl_status / cancel_crawl
# ──────────────────────────────────────────────────────────────────────

class TestCrawlManagement:

    @pytest.fixture(autouse=True)
    def _rate_limit(self) -> None:
        time.sleep(CRAWL_RATE_LIMIT_SLEEP)

    def test_start_crawl_basic_then_status_and_cancel(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_crawl(TEST_URL, limit=2, max_discovery_depth=1))
        _assert_crawl_response(start_parsed)
        job_id: str = start_parsed["id"]

        time.sleep(3)
        status_parsed: Dict[str, Any] = _parse(tools.get_crawl_status(job_id))
        _assert_crawl_job(status_parsed)

        cancel_raw: str = tools.cancel_crawl(job_id)
        cancel_parsed: Dict[str, Any] = json.loads(cancel_raw)
        assert isinstance(cancel_parsed, dict)
        assert "cancelled" in cancel_parsed or "error" in cancel_parsed

    def test_start_crawl_all_attributes(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_crawl(
            TEST_URL,
            limit=2,
            scrape_formats=["markdown", "links"],
            exclude_paths=["/private/*"],
            include_paths=["/*"],
            max_discovery_depth=1,
            sitemap="skip",
        ))
        _assert_crawl_response(start_parsed)
        tools.cancel_crawl(start_parsed["id"])

    @pytest.mark.asyncio
    async def test_async_start_status_cancel(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(await tools.astart_crawl(TEST_URL, limit=2, max_discovery_depth=1))
        _assert_crawl_response(start_parsed)
        job_id: str = start_parsed["id"]

        time.sleep(3)
        status_parsed: Dict[str, Any] = _parse(await tools.aget_crawl_status(job_id))
        _assert_crawl_job(status_parsed)

        cancel_raw: str = await tools.acancel_crawl(job_id)
        cancel_parsed: Dict[str, Any] = json.loads(cancel_raw)
        assert "cancelled" in cancel_parsed or "error" in cancel_parsed

    @pytest.mark.asyncio
    async def test_async_start_crawl_all_attributes(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(await tools.astart_crawl(
            TEST_URL,
            limit=2,
            scrape_formats=["html"],
            exclude_paths=["/admin/*"],
            include_paths=["/*"],
            max_discovery_depth=1,
            sitemap="skip",
        ))
        _assert_crawl_response(start_parsed)
        await tools.acancel_crawl(start_parsed["id"])


# ──────────────────────────────────────────────────────────────────────
#  map_website — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestMapWebsite:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.map_website(TEST_URL))
        _assert_map_data(parsed)

    def test_with_limit(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.map_website(TEST_URL, limit=5))
        _assert_map_data(parsed)
        assert len(parsed["links"]) <= 5

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.amap_website(TEST_URL))
        _assert_map_data(parsed)

    @pytest.mark.asyncio
    async def test_async_with_limit(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.amap_website(TEST_URL, limit=5))
        _assert_map_data(parsed)
        assert len(parsed["links"]) <= 5


# ──────────────────────────────────────────────────────────────────────
#  search_web — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestSearchWeb:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY))
        _assert_search_data(parsed)

    def test_with_limit(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=3))
        _assert_search_data(parsed)

    def test_with_scrape_options(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(
            TEST_SEARCH_QUERY,
            limit=2,
            scrape_options={"formats": ["markdown"]},
        ))
        _assert_search_data(parsed)

    def test_with_location(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, location="TR"))
        _assert_search_data(parsed)

    def test_with_tbs_past_week(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, tbs="qdr:w"))
        _assert_search_data(parsed)

    def test_with_tbs_past_month(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, tbs="qdr:m"))
        _assert_search_data(parsed)

    def test_with_tbs_past_year(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, tbs="qdr:y"))
        _assert_search_data(parsed)

    def test_with_timeout(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, timeout=30000))
        _assert_search_data(parsed)

    def test_with_sources_web(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, sources=["web"]))
        _assert_search_data(parsed)
        assert parsed.get("web") is not None
        assert isinstance(parsed["web"], list)

    def test_with_sources_news(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, sources=["news"]))
        _assert_search_data(parsed)

    def test_with_categories_research(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, categories=["research"]))
        _assert_search_data(parsed)

    def test_with_categories_github(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(TEST_SEARCH_QUERY, limit=2, categories=["github"]))
        _assert_search_data(parsed)

    def test_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.search_web(
            TEST_SEARCH_QUERY,
            limit=3,
            scrape_options={"formats": ["markdown"]},
            location="TR",
            tbs="qdr:m",
            timeout=30000,
            sources=["web"],
            categories=["research"],
        ))
        _assert_search_data(parsed)

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.asearch_web(TEST_SEARCH_QUERY, limit=2))
        _assert_search_data(parsed)

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.asearch_web(
            TEST_SEARCH_QUERY,
            limit=2,
            scrape_options={"formats": ["markdown"]},
            location="TR",
            tbs="qdr:w",
            timeout=30000,
            sources=["web", "news"],
            categories=["research"],
        ))
        _assert_search_data(parsed)


# ──────────────────────────────────────────────────────────────────────
#  batch_scrape — all formats + attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestBatchScrape:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS))
        _assert_batch_scrape_job(parsed)

    def test_format_markdown(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=["markdown"]))
        _assert_batch_scrape_job(parsed)
        for doc in parsed["data"]:
            assert doc.get("markdown") is not None

    def test_format_html(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=["html"]))
        _assert_batch_scrape_job(parsed)
        for doc in parsed["data"]:
            assert doc.get("html") is not None

    def test_format_links(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=["links"]))
        _assert_batch_scrape_job(parsed)

    def test_format_rawHtml(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=["rawHtml"]))
        _assert_batch_scrape_job(parsed)
        for doc in parsed["data"]:
            assert doc.get("raw_html") is not None

    def test_format_summary(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=["summary"]))
        _assert_batch_scrape_job(parsed)

    def test_format_images(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=["images"]))
        _assert_batch_scrape_job(parsed)

    def test_all_formats(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, formats=ALL_FORMATS))
        _assert_batch_scrape_job(parsed)
        for doc in parsed["data"]:
            assert doc.get("markdown") is not None
            assert doc.get("html") is not None

    def test_with_poll_interval(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(BATCH_URLS, poll_interval=3))
        _assert_batch_scrape_job(parsed)

    def test_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.batch_scrape(
            BATCH_URLS, formats=["markdown", "html"], poll_interval=3,
        ))
        _assert_batch_scrape_job(parsed)
        for doc in parsed["data"]:
            assert doc.get("markdown") is not None
            assert doc.get("html") is not None

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.abatch_scrape(BATCH_URLS))
        _assert_batch_scrape_job(parsed)

    @pytest.mark.asyncio
    async def test_async_all_formats(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.abatch_scrape(BATCH_URLS, formats=ALL_FORMATS))
        _assert_batch_scrape_job(parsed)

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.abatch_scrape(
            BATCH_URLS, formats=["markdown", "links"], poll_interval=3,
        ))
        _assert_batch_scrape_job(parsed)


# ──────────────────────────────────────────────────────────────────────
#  Batch scrape management: start_batch_scrape / get_batch_scrape_status
# ──────────────────────────────────────────────────────────────────────

class TestBatchScrapeManagement:

    def test_start_and_get_status(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_batch_scrape(BATCH_URLS))
        _assert_batch_scrape_response(start_parsed)
        job_id: str = start_parsed["id"]

        time.sleep(5)
        status_parsed: Dict[str, Any] = _parse(tools.get_batch_scrape_status(job_id))
        _assert_batch_scrape_job(status_parsed)

    def test_start_with_formats(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_batch_scrape(BATCH_URLS, formats=["markdown", "links"]))
        _assert_batch_scrape_response(start_parsed)

    def test_start_with_all_formats(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_batch_scrape(BATCH_URLS, formats=ALL_FORMATS))
        _assert_batch_scrape_response(start_parsed)

    @pytest.mark.asyncio
    async def test_async_start_and_get_status(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(await tools.astart_batch_scrape(BATCH_URLS))
        _assert_batch_scrape_response(start_parsed)
        job_id: str = start_parsed["id"]

        time.sleep(5)
        status_parsed: Dict[str, Any] = _parse(await tools.aget_batch_scrape_status(job_id))
        _assert_batch_scrape_job(status_parsed)

    @pytest.mark.asyncio
    async def test_async_start_with_formats(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(await tools.astart_batch_scrape(
            BATCH_URLS, formats=["html", "summary"],
        ))
        _assert_batch_scrape_response(start_parsed)


# ──────────────────────────────────────────────────────────────────────
#  extract_data — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestExtractData:

    def test_with_prompt(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title and description",
        ))
        _assert_extract_response(parsed)
        assert parsed.get("data") is not None

    def test_with_schema(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
        }
        parsed: Dict[str, Any] = _parse(tools.extract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title and description",
            schema=schema,
        ))
        _assert_extract_response(parsed)
        assert parsed.get("data") is not None

    def test_with_enable_web_search(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            enable_web_search=False,
        ))
        _assert_extract_response(parsed)

    def test_with_scrape_options(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            scrape_options={"formats": ["markdown"]},
        ))
        _assert_extract_response(parsed)

    def test_all_attributes(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        parsed: Dict[str, Any] = _parse(tools.extract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            schema=schema,
            enable_web_search=False,
            scrape_options={"formats": ["markdown"]},
        ))
        _assert_extract_response(parsed)
        assert parsed.get("data") is not None

    @pytest.mark.asyncio
    async def test_async_with_prompt(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aextract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title and description",
        ))
        _assert_extract_response(parsed)
        assert parsed.get("data") is not None

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        parsed: Dict[str, Any] = _parse(await tools.aextract_data(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            schema=schema,
            enable_web_search=False,
            scrape_options={"formats": ["markdown"]},
        ))
        _assert_extract_response(parsed)
        assert parsed.get("data") is not None


# ──────────────────────────────────────────────────────────────────────
#  Extract management: start_extract / get_extract_status
# ──────────────────────────────────────────────────────────────────────

class TestExtractManagement:

    def test_start_and_get_status(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_extract(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
        ))
        _assert_extract_job_started(start_parsed)

        time.sleep(5)
        status_parsed: Dict[str, Any] = _parse(tools.get_extract_status(start_parsed["id"]))
        _assert_extract_status(status_parsed)

    def test_start_with_schema(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        start_parsed: Dict[str, Any] = _parse(tools.start_extract(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            schema=schema,
        ))
        _assert_extract_job_started(start_parsed)

    def test_start_with_enable_web_search(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(tools.start_extract(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            enable_web_search=False,
        ))
        _assert_extract_job_started(start_parsed)

    def test_start_all_attributes(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        start_parsed: Dict[str, Any] = _parse(tools.start_extract(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            schema=schema,
            enable_web_search=False,
        ))
        _assert_extract_job_started(start_parsed)

    @pytest.mark.asyncio
    async def test_async_start_and_get_status(self, tools: Any) -> None:
        start_parsed: Dict[str, Any] = _parse(await tools.astart_extract(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
        ))
        _assert_extract_job_started(start_parsed)

        time.sleep(5)
        status_parsed: Dict[str, Any] = _parse(await tools.aget_extract_status(start_parsed["id"]))
        _assert_extract_status(status_parsed)

    @pytest.mark.asyncio
    async def test_async_start_all_attributes(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        start_parsed: Dict[str, Any] = _parse(await tools.astart_extract(
            urls=["https://www.nike.com.tr"],
            prompt="Extract the page title",
            schema=schema,
            enable_web_search=False,
        ))
        _assert_extract_job_started(start_parsed)
