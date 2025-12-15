from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gmail import GmailInterface
    from .schemas import CheckEmailsResponse

def _get_gmail_classes():
    """Lazy import of Gmail classes."""
    from .gmail import GmailInterface
    from .schemas import CheckEmailsResponse
    
    return {
        'GmailInterface': GmailInterface,
        'CheckEmailsResponse': CheckEmailsResponse,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    gmail_classes = _get_gmail_classes()
    if name in gmail_classes:
        return gmail_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ["GmailInterface", "CheckEmailsResponse"]

