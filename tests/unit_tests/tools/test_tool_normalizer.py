"""Unit tests for ``ToolNormalizer``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from upsonic.tools import tool, ToolKit
from upsonic.tools.base import ToolValidationError
from upsonic.tools.normalizer import NormalizationResult, ToolNormalizer


class TestToolNormalizer:
    """Coverage for the eight input kinds + dedup + error path."""

    @pytest.fixture
    def normalizer(self) -> ToolNormalizer:
        return ToolNormalizer()

    # ------------------------------------------------------------------
    # 1) raw function
    # ------------------------------------------------------------------
    def test_normalize_raw_function(self, normalizer):
        @tool
        def adder(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = normalizer.normalize([adder], already_registered=set())

        assert "adder" in result.tools
        assert id(adder) in result.raw_object_ids

    # ------------------------------------------------------------------
    # 2) bound method
    # ------------------------------------------------------------------
    def test_normalize_bound_method(self, normalizer):
        class Box:
            def value(self) -> int:
                """Return constant."""
                return 42

        b = Box()
        result = normalizer.normalize([b.value], already_registered=set())

        assert "value" in result.tools

    # ------------------------------------------------------------------
    # 3) ToolKit class
    # ------------------------------------------------------------------
    def test_normalize_toolkit_class(self, normalizer):
        class TK(ToolKit):
            @tool
            def hello(self) -> str:
                """Greet."""
                return "hi"

        result = normalizer.normalize([TK], already_registered=set())
        assert "hello" in result.tools
        assert len(result.toolkit_instances) == 1
        assert any(
            "hello" in names
            for names in result.class_instance_owners.values()
        )

    # ------------------------------------------------------------------
    # 4) ToolKit instance
    # ------------------------------------------------------------------
    def test_normalize_toolkit_instance(self, normalizer):
        class TK(ToolKit):
            @tool
            def echo(self, x: str) -> str:
                """Echo."""
                return x

        tk = TK()
        result = normalizer.normalize([tk], already_registered=set())
        assert "echo" in result.tools
        assert tk in result.toolkit_instances

    # ------------------------------------------------------------------
    # 5) tool-provider (object with get_tools, not a ToolKit)
    # ------------------------------------------------------------------
    def test_normalize_tool_provider(self, normalizer):
        @tool
        def provided(x: int) -> int:
            """Return x doubled."""
            return x * 2

        class Provider:
            def get_tools(self):
                return [provided]

        provider = Provider()
        result = normalizer.normalize([provider], already_registered=set())

        assert "provided" in result.tools
        assert provider in result.tool_provider_instances

    # ------------------------------------------------------------------
    # 6) agent instance
    # ------------------------------------------------------------------
    def test_normalize_agent_instance(self, normalizer):
        # A real lightweight class — must have ``name`` and one of
        # ``do_async`` / ``do`` / ``agent_id`` but NOT ``get_tools`` (which
        # would route it to the tool-provider branch).
        class FakeAgent:
            name = "my-agent"
            agent_id = "abc"
            description = "An agent"
            system_prompt = ""
            response_format = None

            def do(self, *args, **kwargs):
                return None

        agent = FakeAgent()
        result = normalizer.normalize([agent], already_registered=set())

        # AgentTool wraps the agent; the tool name is derived from agent.name
        assert any(name for name in result.tools)

    # ------------------------------------------------------------------
    # 7) MCP handler
    # ------------------------------------------------------------------
    def test_normalize_mcp_handler(self, normalizer):
        from upsonic.tools.mcp import MCPHandler

        handler = MagicMock(spec=MCPHandler)

        # Build fake MCP tools
        t1 = MagicMock()
        t1.name = "mcp_a"
        t2 = MagicMock()
        t2.name = "mcp_b"
        handler.get_tools = MagicMock(return_value=[t1, t2])

        result = normalizer.normalize([handler], already_registered=set())

        assert "mcp_a" in result.tools
        assert "mcp_b" in result.tools
        assert handler in result.mcp_handlers
        owners = result.mcp_handler_owners[id(handler)]
        assert sorted(owners) == ["mcp_a", "mcp_b"]

    # ------------------------------------------------------------------
    # 8) plain class instance (with public methods that look like tools)
    # ------------------------------------------------------------------
    def test_normalize_plain_class_instance(self, normalizer):
        class Plain:
            def greet(self, name: str) -> str:
                """Greet."""
                return f"hi {name}"

        p = Plain()
        result = normalizer.normalize([p], already_registered=set())
        assert "greet" in result.tools

    # ------------------------------------------------------------------
    # Dedup via already_registered
    # ------------------------------------------------------------------
    def test_dedup_already_registered_raw_function(self, normalizer):
        @tool
        def f() -> int:
            """f."""
            return 1

        already = {id(f)}
        result = normalizer.normalize([f], already_registered=already)
        assert result.tools == {}

    def test_dedup_within_same_call(self, normalizer):
        @tool
        def f(x: int) -> int:
            """f."""
            return x

        # Same object passed twice — second occurrence is filtered
        result = normalizer.normalize([f, f], already_registered=set())
        assert "f" in result.tools

    # ------------------------------------------------------------------
    # ToolValidationError path
    # ------------------------------------------------------------------
    def test_invalid_tool_raises_validation_error(self, normalizer):
        from upsonic.tools.config import ToolConfig

        # require_parameter_descriptions=True with no docstring/params
        # forces the schema generator to fail.
        def bare(a):  # missing type hint
            return a

        bare._upsonic_tool_config = ToolConfig(
            require_parameter_descriptions=True,
        )

        with pytest.raises(ToolValidationError):
            normalizer.normalize([bare], already_registered=set())

    # ------------------------------------------------------------------
    # NormalizationResult shape
    # ------------------------------------------------------------------
    def test_default_result_is_empty(self, normalizer):
        result = normalizer.normalize([], already_registered=set())
        assert isinstance(result, NormalizationResult)
        assert result.tools == {}
        assert result.raw_object_ids == set()
        assert result.mcp_handlers == []
        assert result.mcp_handler_owners == {}
        assert result.class_instance_owners == {}
        assert result.knowledge_base_instances == []
        assert result.toolkit_instances == []
        assert result.tool_provider_instances == []
