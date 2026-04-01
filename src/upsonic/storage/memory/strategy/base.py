"""Shared base for all memory strategy implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from upsonic.storage.base import Storage
    from upsonic.models import Model


class BaseMemoryStrategy(ABC):
    """Common configuration and entrypoints for memory strategies.

    Implementations choose async (``a*``) or sync storage calls based on whether the
    backend is :class:`~upsonic.storage.base.AsyncStorage`. Sync entrypoints may run
    async storage work via :func:`~upsonic.storage.memory.storage_dispatch.run_awaitable_sync`
    when the backend is async.
    """

    def __init__(
        self,
        storage: "Storage",
        enabled: bool = True,
        model: Optional[Union["Model", str]] = None,
        debug: bool = False,
        debug_level: int = 1,
    ) -> None:
        self.storage: "Storage" = storage
        self.enabled: bool = enabled
        self.model: Optional[Union["Model", str]] = model
        self.debug: bool = debug
        self.debug_level: int = debug_level

    @abstractmethod
    async def aget(self, *args: Any, **kwargs: Any) -> Any:
        """Load memory (async storage path when applicable)."""
        ...

    @abstractmethod
    async def asave(self, *args: Any, **kwargs: Any) -> Any:
        """Persist memory (async storage path when applicable)."""
        ...

    @abstractmethod
    def get(self, *args: Any, **kwargs: Any) -> Any:
        """Load memory (sync API; supports sync and async storage backends)."""
        ...

    @abstractmethod
    def save(self, *args: Any, **kwargs: Any) -> Any:
        """Persist memory (sync API; supports sync and async storage backends)."""
        ...
