"""
WhatsApp Integration Module for Upsonic.

This module provides comprehensive WhatsApp Business API integration
for the Upsonic AI Agent framework.

Components:
    - WhatsAppInterface: Main interface class for WhatsApp integration
    - WhatsApp-specific schemas and utilities
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .whatsapp import WhatsAppInterface
    from .schemas import WhatsAppWebhookPayload

def _get_whatsapp_classes():
    """Lazy import of WhatsApp classes."""
    from .whatsapp import WhatsAppInterface
    from .schemas import WhatsAppWebhookPayload
    
    return {
        'WhatsAppInterface': WhatsAppInterface,
        'WhatsAppWebhookPayload': WhatsAppWebhookPayload,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    whatsapp_classes = _get_whatsapp_classes()
    if name in whatsapp_classes:
        return whatsapp_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "WhatsAppInterface",
    "WhatsAppWebhookPayload",
]

