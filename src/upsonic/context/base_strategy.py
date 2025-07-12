"""
Base strategy interface for context processing.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ContextProcessingStrategy(ABC):
    """Abstract base class for context processing strategies."""

    @abstractmethod
    def can_process(self, context_item: Any) -> bool:
        """
        Check if this strategy can process the given context item.

        Args:
            context_item: The context item to check

        Returns:
            True if this strategy can process the item, False otherwise
        """
        pass

    @abstractmethod
    def process(self, context_item: Any) -> str:
        """
        Process the context item and return formatted string.

        Args:
            context_item: The context item to process

        Returns:
            Formatted string representation of the context item
        """
        pass

    @abstractmethod
    def get_section_name(self) -> str:
        """
        Get the XML section name for this strategy.

        Returns:
            The XML section name (e.g., "Tasks", "Agents")
        """
        pass

    def validate(self, context_item: Any) -> Optional[str]:
        """
        Validate the context item before processing.

        Args:
            context_item: The context item to validate

        Returns:
            None if valid, error message if invalid
        """
        return None
