from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution import DurableExecution
    from .storage import DurableExecutionStorage
    from .storages.memory import InMemoryDurableStorage
    from .storages.file import FileDurableStorage
    from .storages.sqlite import SQLiteDurableStorage
    from .storages.redis import RedisDurableStorage

def _get_durable_classes():
    """Lazy import of durable execution classes."""
    from .execution import DurableExecution
    from .storage import DurableExecutionStorage
    from .storages.memory import InMemoryDurableStorage
    from .storages.file import FileDurableStorage
    from .storages.sqlite import SQLiteDurableStorage
    from .storages.redis import RedisDurableStorage
    
    return {
        'DurableExecution': DurableExecution,
        'DurableExecutionStorage': DurableExecutionStorage,
        'InMemoryDurableStorage': InMemoryDurableStorage,
        'FileDurableStorage': FileDurableStorage,
        'SQLiteDurableStorage': SQLiteDurableStorage,
        'RedisDurableStorage': RedisDurableStorage,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    durable_classes = _get_durable_classes()
    if name in durable_classes:
        return durable_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "DurableExecution",
    "DurableExecutionStorage",
    "InMemoryDurableStorage",
    "FileDurableStorage",
    "SQLiteDurableStorage",
    "RedisDurableStorage",
]

