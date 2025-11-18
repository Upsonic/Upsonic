"""
Schemas for WhatsApp Business API Integration.

This module contains all Pydantic models specific to WhatsApp webhooks,
messages, and API requests/responses.
"""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================================
# WhatsApp Webhook Structures
# ============================================================================

class WhatsAppContact(BaseModel):
    """Model for WhatsApp contact information."""
    
    profile: Dict[str, str] = Field(..., description="Contact profile")
    wa_id: str = Field(..., description="WhatsApp ID")


class WhatsAppValue(BaseModel):
    """Model for WhatsApp webhook value."""
    
    messaging_product: str = Field(default="whatsapp", description="Messaging product")
    metadata: Dict[str, Any] = Field(..., description="Metadata")
    contacts: Optional[List[WhatsAppContact]] = Field(None, description="Contacts")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Messages")
    statuses: Optional[List[Dict[str, Any]]] = Field(None, description="Message statuses")


class WhatsAppChange(BaseModel):
    """Model for WhatsApp webhook change."""
    
    value: WhatsAppValue = Field(..., description="Change value")
    field: str = Field(..., description="Changed field")


class WhatsAppEntry(BaseModel):
    """Model for WhatsApp webhook entry."""
    
    id: str = Field(..., description="Entry ID")
    changes: List[WhatsAppChange] = Field(..., description="List of changes")


class WhatsAppWebhookPayload(BaseModel):
    """Model for incoming WhatsApp webhook payload."""
    
    object: str = Field(..., description="Webhook object type")
    entry: List[WhatsAppEntry] = Field(..., description="List of entries")


class WhatsAppVerificationRequest(BaseModel):
    """Model for WhatsApp webhook verification."""
    
    mode: str = Field(..., alias="hub.mode", description="Verification mode")
    token: str = Field(..., alias="hub.verify_token", description="Verification token")
    challenge: str = Field(..., alias="hub.challenge", description="Challenge string")
