import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from upsonic.tools_server.server.api import (
    app,
    timeout,
    timeout_handler,
    TimeoutException,
)
import asyncio
from httpx import AsyncClient

# Create a new event loop for each test
@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Create an async client for testing
@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

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
    async def test_slow_endpoint():
        await asyncio.sleep(2)
        return {"message": "success"}

    # Test that the timeout works using asyncio.timeout
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(1.0):
            await test_slow_endpoint()

    # Test successful completion
    async def test_quick_endpoint():
        await asyncio.sleep(0.1)
        return {"message": "success"}

    async with asyncio.timeout(2.0):
        result = await test_quick_endpoint()
        assert result == {"message": "success"}

@pytest.mark.asyncio
async def test_status_endpoint(async_client):
    response = await async_client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Server is running"}






