"""Unit tests for ``ToolWrapper``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from upsonic.tools.config import ToolConfig, ToolHooks
from upsonic.tools.execution import ToolWrapper
from upsonic.tools.hitl import (
    ConfirmationPause,
    ExternalExecutionPause,
    UserInputPause,
)
from upsonic.tools.registry import ToolRegistry


def _make_tool(name="tool_under_test", config=None):
    tool = MagicMock()
    tool.name = name
    tool.config = config or ToolConfig()
    tool.execute = AsyncMock(return_value="OK")
    tool.record_execution = MagicMock()
    return tool


class TestToolWrapper:
    """Coverage for the 9-aspect behavioral pipeline."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return ToolRegistry()

    @pytest.fixture
    def wrapper_factory(self, registry):
        return ToolWrapper(registry=registry)

    @pytest.mark.asyncio
    async def test_wrap_returns_async_callable(self, wrapper_factory):
        tool = _make_tool()
        wrapped = wrapper_factory.wrap(tool)
        out = await wrapped()
        assert out["func"] == "OK"
        tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pause_confirmation_propagates(self, wrapper_factory):
        cfg = ToolConfig(requires_confirmation=True)
        tool = _make_tool(config=cfg)
        wrapped = wrapper_factory.wrap(tool)
        with pytest.raises(ConfirmationPause):
            await wrapped()
        # tool.execute must NOT have been called for the pause path
        tool.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pause_user_input_propagates(self, wrapper_factory):
        cfg = ToolConfig(requires_user_input=True)
        tool = _make_tool(config=cfg)
        wrapped = wrapper_factory.wrap(tool)
        with pytest.raises(UserInputPause):
            await wrapped()

    @pytest.mark.asyncio
    async def test_pause_external_execution_propagates(self, wrapper_factory):
        cfg = ToolConfig(external_execution=True)
        tool = _make_tool(config=cfg)
        wrapped = wrapper_factory.wrap(tool)
        with pytest.raises(ExternalExecutionPause):
            await wrapped()

    @pytest.mark.asyncio
    async def test_pause_exceptions_not_swallowed_by_retry(self, wrapper_factory):
        """Pause exceptions raised from inside execute() must propagate even
        when ``max_retries`` > 0.
        """
        cfg = ToolConfig(max_retries=3)
        tool = _make_tool(config=cfg)
        tool.execute = AsyncMock(side_effect=ConfirmationPause())
        wrapped = wrapper_factory.wrap(tool)
        with pytest.raises(ConfirmationPause):
            await wrapped()
        # Only ONE attempt — retry must NOT swallow pause exceptions
        assert tool.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_general_exception(self, wrapper_factory):
        cfg = ToolConfig(max_retries=2)
        tool = _make_tool(config=cfg)
        tool.execute = AsyncMock(
            side_effect=[RuntimeError("boom"), RuntimeError("boom"), "ok"]
        )

        # Patch sleep to avoid real backoff
        with patch("upsonic.tools.execution.asyncio.sleep", new=AsyncMock()):
            wrapped = wrapper_factory.wrap(tool)
            out = await wrapped()
        assert out["func"] == "ok"
        assert tool.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, wrapper_factory):
        cfg = ToolConfig(max_retries=2, timeout=0.01)
        tool = _make_tool(config=cfg)
        # Force timeout once, succeed second time
        side_effects = [asyncio.TimeoutError(), AsyncMock(return_value="late")()]

        async def maybe_timeout(**kwargs):
            effect = side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return await effect if asyncio.iscoroutine(effect) else effect

        tool.execute = AsyncMock(side_effect=maybe_timeout)

        with patch("upsonic.tools.execution.asyncio.sleep", new=AsyncMock()):
            wrapped = wrapper_factory.wrap(tool)
            out = await wrapped()
        assert out["func"] == "late"

    @pytest.mark.asyncio
    async def test_hooks_fire_in_order(self, wrapper_factory):
        order = []

        def before(**kwargs):
            order.append("before")
            return "B"

        def after(result):
            order.append("after")
            return "A"

        cfg = ToolConfig(tool_hooks=ToolHooks(before=before, after=after))
        tool = _make_tool(config=cfg)
        wrapped = wrapper_factory.wrap(tool)

        out = await wrapped()
        assert order == ["before", "after"]
        assert out["func_before"] == "B"
        assert out["func_after"] == "A"
        assert out["func"] == "OK"

    @pytest.mark.asyncio
    async def test_kb_setup_called_when_kb_owns_tool(self, registry, wrapper_factory):
        """KB ``setup_async`` should be awaited when the tool belongs to a KB."""
        kb = MagicMock()
        kb.setup_async = AsyncMock()
        kb_id = id(kb)
        registry.knowledge_base_instances[kb_id] = kb
        registry.class_instance_to_tools[kb_id] = ["the_tool"]

        tool = _make_tool(name="the_tool")
        wrapped = wrapper_factory.wrap(tool)
        await wrapped()

        kb.setup_async.assert_awaited_once()
