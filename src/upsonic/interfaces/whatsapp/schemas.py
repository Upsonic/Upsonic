"""
Schemas for WhatsApp Business API Integration.

This module contains all Pydantic models specific to WhatsApp webhooks,
messages, and API requests/responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field



class WhatsAppValue(BaseModel):
    """Model for WhatsApp webhook value."""
    
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Messages")
    statuses: Optional[List[Dict[str, Any]]] = Field(None, description="Message statuses")


class WhatsAppChange(BaseModel):
    """Model for WhatsApp webhook change."""
    
    value: WhatsAppValue = Field(..., description="Change value")


class WhatsAppEntry(BaseModel):
    """Model for WhatsApp webhook entry."""
    
    changes: List[WhatsAppChange] = Field(..., description="List of changes")


class WhatsAppWebhookPayload(BaseModel):
    """Model for incoming WhatsApp webhook payload."""
    
    object: str = Field(..., description="Webhook object type")
    entry: List[WhatsAppEntry] = Field(..., description="List of entries")
