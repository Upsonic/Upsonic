"""Cancellation must survive MCP cleanup after all resources are released."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from upsonic.tools.mcp import MCPHandler, MultiMCPHandler


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("cancel scope"), asyncio.CancelledError()])
async def test_cleanup_releases_resources_and_preserves_cancellation(failure):
    handler = MCPHandler(url="https://example.org/mcp", transport="streamable-http")
    session = AsyncMock()
    session.__aexit__.side_effect = failure
    transport = AsyncMock()
    client = AsyncMock()
    handler._session_ctx = session
    handler.session = session
    handler._transport_ctx = transport
    handler._managed_http_client = client
    handler._initialized = True
    loop = asyncio.get_running_loop()
    original_exception_handler = loop.get_exception_handler()

    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await handler.close()
    else:
        await handler.close()

    session.__aexit__.assert_awaited_once()
    transport.__aexit__.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert handler.session is None and handler._session_ctx is None
    assert handler._transport_ctx is None and handler._managed_http_client is None
    assert not handler._initialized
    assert loop.get_exception_handler() is original_exception_handler


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("cleanup error"), asyncio.CancelledError()])
async def test_multi_cleanup_closes_all_handlers_before_cancellation(failure):
    coordinator = MultiMCPHandler(urls=["https://example.org/mcp"])
    first = AsyncMock()
    first.close.side_effect = failure
    second = AsyncMock()
    coordinator.handlers = [first, second]
    coordinator.tools = [object()]
    coordinator._initialized = True

    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await coordinator.close()
    else:
        await coordinator.close()

    first.close.assert_awaited_once()
    second.close.assert_awaited_once()
    assert coordinator.handlers == [] and coordinator.tools == []
    assert not coordinator._initialized
