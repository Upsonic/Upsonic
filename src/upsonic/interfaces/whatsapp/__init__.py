"""
WhatsApp Integration Module for Upsonic.

This module provides comprehensive WhatsApp Business API integration
for the Upsonic AI Agent framework.

Components:
    - WhatsAppInterface: Main interface class for WhatsApp integration
    - WhatsApp-specific schemas and utilities
"""

from upsonic.interfaces.whatsapp.interface import WhatsAppInterface
from upsonic.interfaces.whatsapp.schemas import WhatsAppWebhookPayload

__all__ = [
    "WhatsAppInterface",
    "WhatsAppWebhookPayload",
]

