"""Unit tests for ``ToolRegistry``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from upsonic.tools import tool, ToolKit
from upsonic.tools.normalizer import NormalizationResult, ToolNormalizer
from upsonic.tools.registry import ToolRegistry


class TestToolRegistry:
    """Coverage for add/remove + cascade-delete + lookups."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return ToolRegistry()

    @pytest.fixture
    def normalizer(self) -> ToolNormalizer:
        return ToolNormalizer()

    # ------------------------------------------------------------------
    # add() — atomic merge
    # ------------------------------------------------------------------
    def test_add_atomic_merge(self, registry, normalizer):
        @tool
        def fn(x: int) -> int:
            """fn."""
            return x

        result = normalizer.normalize([fn], already_registered=set())
        added = registry.add(result)

        assert "fn" in added
        assert "fn" in registry.registered_tools
        assert id(fn) in registry.raw_object_ids

    def test_add_returns_only_new(self, registry, normalizer):
        @tool
        def fn(x: int) -> int:
            """fn."""
            return x

        result = normalizer.normalize([fn], already_registered=set())
        added = registry.add(result)
        assert added is result.tools

    # ------------------------------------------------------------------
    # remove() — by name (1:1 function)
    # ------------------------------------------------------------------
    def test_remove_by_name(self, registry, normalizer):
        @tool
        def fn(x: int) -> int:
            """fn."""
            return x

        registry.add(normalizer.normalize([fn], set()))
        removed_names, originals = registry.remove("fn")

        assert "fn" in removed_names
        assert fn in originals
        assert "fn" not in registry.registered_tools

    # ------------------------------------------------------------------
    # remove() — ToolKit cascade
    # ------------------------------------------------------------------
    def test_remove_toolkit_cascades(self, registry, normalizer):
        class TK(ToolKit):
            @tool
            def a(self) -> str:
                """a."""
                return "a"

            @tool
            def b(self) -> str:
                """b."""
                return "b"

        tk = TK()
        registry.add(normalizer.normalize([tk], set()))

        removed_names, originals = registry.remove(tk)

        assert set(removed_names) == {"a", "b"}
        assert tk in originals
        assert "a" not in registry.registered_tools
        assert "b" not in registry.registered_tools

    # ------------------------------------------------------------------
    # remove() — partial vs full MCP handler split (R4)
    # ------------------------------------------------------------------
    def test_remove_one_mcp_tool_keeps_handler(self, registry, normalizer):
        from upsonic.tools.mcp import MCPHandler

        handler = MagicMock(spec=MCPHandler)

        t1 = MagicMock()
        t1.name = "mcp_a"
        t1.handler = handler
        t2 = MagicMock()
        t2.name = "mcp_b"
        t2.handler = handler
        handler.get_tools = MagicMock(return_value=[t1, t2])

        registry.add(normalizer.normalize([handler], set()))
        assert "mcp_a" in registry.registered_tools
        assert "mcp_b" in registry.registered_tools

        # Remove ONE tool by name — handler must NOT appear in originals
        # because handler still owns "mcp_b".
        removed_names, originals = registry.remove("mcp_a")
        assert "mcp_a" in removed_names
        assert handler not in originals
        # Handler still tracked
        assert id(handler) in registry.mcp_handler_to_tools
        assert "mcp_b" in registry.registered_tools

    def test_remove_full_mcp_handler(self, registry, normalizer):
        from upsonic.tools.mcp import MCPHandler

        handler = MagicMock(spec=MCPHandler)

        t1 = MagicMock()
        t1.name = "only_one"
        t1.handler = handler
        handler.get_tools = MagicMock(return_value=[t1])

        registry.add(normalizer.normalize([handler], set()))

        # Remove the only tool — handler should be fully cleaned up
        removed_names, _ = registry.remove("only_one")
        assert "only_one" in removed_names
        assert id(handler) not in registry.mcp_handler_to_tools
        assert handler not in registry.mcp_handlers

    # ------------------------------------------------------------------
    # collect_instructions — toolkit-level
    # ------------------------------------------------------------------
    def test_collect_instructions_from_toolkit(self, registry, normalizer):
        class TK(ToolKit):
            @tool
            def hello(self) -> str:
                """hello."""
                return "hi"

        tk = TK(instructions="be terse", add_instructions=True)
        registry.add(normalizer.normalize([tk], set()))

        instructions = registry.collect_instructions()
        assert any("be terse" in s for s in instructions)

    # ------------------------------------------------------------------
    # all_definitions
    # ------------------------------------------------------------------
    def test_all_definitions(self, registry, normalizer):
        @tool
        def fn(x: int) -> int:
            """fn."""
            return x

        registry.add(normalizer.normalize([fn], set()))

        defs = registry.all_definitions()
        assert len(defs) == 1
        assert defs[0].name == "fn"

    # ------------------------------------------------------------------
    # store_wrapped / get_wrapped
    # ------------------------------------------------------------------
    def test_store_and_get_wrapped(self, registry):
        async def fake(**kwargs):
            return "ok"

        registry.store_wrapped("alpha", fake)
        assert registry.get_wrapped("alpha") is fake

    def test_get_wrapped_missing(self, registry):
        assert registry.get_wrapped("nope") is None
