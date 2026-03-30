from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mail import MailInterface
    from .schemas import CheckEmailsResponse, EmailListResponse, MailboxStatusResponse

def _get_mail_classes():
    """Lazy import of Mail classes."""
    from .mail import MailInterface
    from .schemas import CheckEmailsResponse, EmailListResponse, MailboxStatusResponse

    return {
        'MailInterface': MailInterface,
        'CheckEmailsResponse': CheckEmailsResponse,
        'EmailListResponse': EmailListResponse,
        'MailboxStatusResponse': MailboxStatusResponse,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    mail_classes = _get_mail_classes()
    if name in mail_classes:
        return mail_classes[name]

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ["MailInterface", "CheckEmailsResponse", "EmailListResponse", "MailboxStatusResponse"]
