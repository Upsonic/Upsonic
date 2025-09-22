from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional

# Use TYPE_CHECKING to import types for type hinting only,
# preventing circular import errors at runtime.
if TYPE_CHECKING:
    from upsonic.messages import ModelRequest, ModelResponse
    from upsonic.messages.streaming import ModelResponseStreamEvent
    # Import the newly defined settings base class
    from .settings import BaseModelSettings

# A generic type alias for provider-specific settings.
ModelSettings = Dict[str, Any]


class ModelProvider(ABC):
    """
    Abstract Base Class for all AI model providers.

    This class defines the standard interface that the agent framework will use
    to interact with any underlying language model. Its primary purpose is to
    ensure modularity and interchangeability of providers.

    The core methods are asynchronous (`_async` suffix) to support high-performance,
    non-blocking I/O. For convenience, synchronous wrappers are provided that handle
    the async event loop internally.
    """

    @abstractmethod
    async def run_async(
        self, request: "ModelRequest", settings: Optional[ModelSettings] = None
    ) -> "ModelResponse":
        """
        Asynchronously executes a single, non-streaming request to the model.

        This is the primary method for getting a complete response from the model
        in one shot.

        Args:
            request: The ModelRequest object containing the prompt and message history.
            settings: A dictionary of provider-specific settings to override defaults.

        Returns:
            A complete ModelResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    async def run_stream_async(
        self, request: "ModelRequest", settings: Optional[ModelSettings] = None
    ) -> AsyncIterator["ModelResponseStreamEvent"]:
        """
        Asynchronously executes a streaming request to the model.

        This method is used when you need to process the model's response in
        real-time as it's being generated. It yields events representing
        the incremental updates to the response.

        Args:
            request: The ModelRequest object.
            settings: A dictionary of provider-specific settings.

        Yields:
            An asynchronous iterator of ModelResponseStreamEvent objects.
        """
        # The 'yield' keyword is required to make this an async generator.
        if False:
            yield

    @abstractmethod
    async def run_batch_async(
        self, requests: List["ModelRequest"], settings: Optional[ModelSettings] = None
    ) -> List["ModelResponse"]:
        """
        Asynchronously executes a batch of requests to the model.

        This is designed for processing multiple, independent requests efficiently.
        Concrete implementations should use the provider's native batch API if
        available, or fall back to concurrent execution (e.g., using asyncio.gather).

        Args:
            requests: A list of ModelRequest objects to process.
            settings: A dictionary of settings applied to all requests in the batch.

        Returns:
            A list of ModelResponse objects, corresponding to the input requests.
        """
        raise NotImplementedError

    # --- Synchronous Wrappers ---

    def run(
        self, request: "ModelRequest", settings: Optional[ModelSettings] = None
    ) -> "ModelResponse":
        """
        Synchronously executes a single, non-streaming request.
        This is a convenience wrapper around `run_async`.
        """
        import asyncio
        import concurrent.futures

        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to use a different approach
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.run_async(request, settings))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
                
        except RuntimeError:  # No running event loop
            return asyncio.run(self.run_async(request, settings))

    def run_stream(
        self, request: "ModelRequest", settings: Optional[ModelSettings] = None
    ) -> Iterator["ModelResponseStreamEvent"]:
        """
        Synchronously executes a streaming request.
        This is a convenience wrapper around `run_stream_async`.
        """
        import asyncio
        import concurrent.futures
        import threading

        async def _aiter_to_iter():
            """Helper to convert the async iterator to a sync one."""
            async for item in self.run_stream_async(request, settings):
                yield item

        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to use a different approach
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    gen = _aiter_to_iter()
                    results = []
                    while True:
                        try:
                            results.append(new_loop.run_until_complete(gen.__anext__()))
                        except StopAsyncIteration:
                            break
                    return results
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                results = future.result()
                for result in results:
                    yield result
                    
        except RuntimeError:  # No running event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                gen = _aiter_to_iter()
                while True:
                    try:
                        yield loop.run_until_complete(gen.__anext__())
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

    def run_batch(
        self, requests: List["ModelRequest"], settings: Optional[ModelSettings] = None
    ) -> List["ModelResponse"]:
        """
        Synchronously executes a batch of requests.
        This is a convenience wrapper around `run_batch_async`.
        """
        import asyncio
        import concurrent.futures

        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to use a different approach
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.run_batch_async(requests, settings))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
                
        except RuntimeError:  # No running event loop
            return asyncio.run(self.run_batch_async(requests, settings))