import pytest
from fastapi.testclient import TestClient
from upsonic.tools_server.server.api import (
    app,
    timeout,
    timeout_handler,
    TimeoutException,
)
import asyncio
from httpx import AsyncClient

client = TestClient(app)


@pytest.mark.asyncio
async def test_timeout_handler():
    # Test successful completion within timeout
    async def quick_operation():
        return "success"

    result = await timeout_handler(1.0, quick_operation())
    assert result == "success"

    # Test timeout exception
    async def slow_operation():
        await asyncio.sleep(2)
        return "too late"

    with pytest.raises(TimeoutException):
        await timeout_handler(1.0, slow_operation())


@pytest.mark.asyncio
async def test_timeout_decorator():
    @timeout(1.0)
    async def test_endpoint():
        await asyncio.sleep(2)
        return {"message": "success"}

    # Test that the decorator raises HTTPException on timeout
    with pytest.raises(Exception) as exc_info:
        await test_endpoint()
    assert "408" in str(exc_info.value)

    # Test successful completion
    @timeout(2.0)
    async def quick_endpoint():
        return {"message": "success"}

    result = await quick_endpoint()
    assert result == {"message": "success"}






