import asyncio
import concurrent.futures
from typing import Awaitable, TypeVar

T = TypeVar('T')

class AsyncExecutionMixin:
    """
    A mixin class that provides a robust method for calling an async function
    from a synchronous context.

    This utility is the cornerstone of the dual sync/async API. It intelligently
    detects the presence of a running asyncio event loop to avoid common

    `RuntimeError` exceptions in mixed environments (like Jupyter notebooks or
    when integrating with other async frameworks).
    """

    def _run_async_from_sync(self, awaitable: Awaitable[T]) -> T:
        """
        Executes an awaitable from a synchronous method, managing the event loop
        intelligently.

        Args:
            awaitable: The coroutine or other awaitable object to run.

        Returns:
            The result of the awaitable.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, awaitable)
                return future.result()
        else:
            loop.run_until_complete(awaitable)