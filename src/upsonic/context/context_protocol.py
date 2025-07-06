from typing import Protocol, runtime_checkable

from .context import ContextSections


@runtime_checkable
class ContextItem(Protocol):
    """Protocol defining the interface for context items."""

    def to_context_string(self) -> str:
        """Return context string representation for this item."""
        ...

    def get_context_section(self) -> ContextSections:
        """Return the context section this item belongs to."""
        ...
