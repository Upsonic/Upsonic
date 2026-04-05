"""Detect async storage and run async storage calls from synchronous memory APIs when needed."""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, TypeVar

_T = TypeVar("_T")


def is_async_storage_backend(storage: Any) -> bool:
    """Return True if ``storage`` implements :class:`~upsonic.storage.base.AsyncStorage`."""
    from upsonic.storage.base import AsyncStorage

    return isinstance(storage, AsyncStorage)


def run_awaitable_sync(awaitable: Awaitable[_T]) -> _T:
    """Run ``awaitable`` from synchronous code (new loop, or worker thread if a loop is running)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()
