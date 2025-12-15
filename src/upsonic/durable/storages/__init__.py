from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import InMemoryDurableStorage
    from .file import FileDurableStorage
    from .sqlite import SQLiteDurableStorage
    from .redis import RedisDurableStorage

def _get_durable_storage_classes():
    """Lazy import of durable storage classes."""
    from .memory import InMemoryDurableStorage
    from .file import FileDurableStorage
    from .sqlite import SQLiteDurableStorage
    from .redis import RedisDurableStorage
    
    return {
        'InMemoryDurableStorage': InMemoryDurableStorage,
        'FileDurableStorage': FileDurableStorage,
        'SQLiteDurableStorage': SQLiteDurableStorage,
        'RedisDurableStorage': RedisDurableStorage,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    storage_classes = _get_durable_storage_classes()
    if name in storage_classes:
        return storage_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "InMemoryDurableStorage",
    "FileDurableStorage",
    "SQLiteDurableStorage",
    "RedisDurableStorage",
]

