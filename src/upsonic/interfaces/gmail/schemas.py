"""
Schemas for Gmail Interface Integration.

This module contains Pydantic models for Gmail API responses.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class CheckEmailsResponse(BaseModel):
    """Response model for the check emails endpoint."""
    status: str = Field(..., description="Status of the operation")
    processed_count: int = Field(..., description="Number of emails processed")
    message_ids: List[str] = Field(..., description="List of processed message IDs")


class AgentEmailResponse(BaseModel):
    """Structured response from the Agent for email processing."""
    action: Literal["reply", "ignore"] = Field(..., description="Action to take: 'reply' or 'ignore'")
    reply_body: str = Field(..., description="The body of the reply email (required if action is 'reply')")
    reasoning: str = Field(..., description="Brief reasoning for the decision")
