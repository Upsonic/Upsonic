"""
Schemas for Slack Interface Integration.

This module contains Pydantic models for Slack event processing and responses.
"""

from pydantic import BaseModel, Field


class SlackEventResponse(BaseModel):
    """Response model for Slack event processing."""
    status: str = Field(default="ok", description="Processing status")


class SlackChallengeResponse(BaseModel):
    """Response model for Slack URL verification challenge."""
    challenge: str = Field(description="Challenge string to echo back to Slack")
