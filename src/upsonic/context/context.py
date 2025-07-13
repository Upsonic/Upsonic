"""
Context processing module using strategy pattern.
"""

from typing import Any

from .context_processor import ContextProcessor

# Global instance for backward compatibility
_context_processor = ContextProcessor()


def context_proceess(context: Any) -> str:
    """
    Process context items and return formatted XML string.

    This function maintains backward compatibility while using the new
    strategy-based context processing system.

    Args:
        context: Context items to process

    Returns:
        Formatted XML string with all context sections
    """
    return _context_processor.process_context(context)


def get_context_processor() -> ContextProcessor:
    """
    Get the global context processor instance.

    This allows users to customize the processor by adding/removing strategies.

    Returns:
        The global ContextProcessor instance
    """
    return _context_processor


def set_context_processor(processor: ContextProcessor) -> None:
    """
    Set a custom context processor instance.

    Args:
        processor: The ContextProcessor instance to use
    """
    global _context_processor
    _context_processor = processor


# Note: Legacy imports are now handled in __init__.py for proper module organization
