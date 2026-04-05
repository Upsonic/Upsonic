"""
Schemas for Mail (SMTP/IMAP) Interface Integration.

This module contains Pydantic models for the generic mail interface.
"""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class AttachmentInfo(BaseModel):
    """Metadata for an email attachment."""
    filename: str = Field(..., description="Name of the attached file")
    content_type: str = Field(..., description="MIME type of the attachment")
    size: int = Field(..., description="Size in bytes")


class EmailSummary(BaseModel):
    """Summary of a single email message."""
    uid: str = Field(..., description="Unique identifier of the email")
    message_id: str = Field(default="", description="Email Message-ID header")
    sender: str = Field(default="", alias="from", description="Sender address")
    to: str = Field(default="", description="Recipient address")
    cc: str = Field(default="", description="CC addresses")
    subject: str = Field(default="", description="Email subject")
    date: str = Field(default="", description="Date the email was sent")
    body: str = Field(default="", description="Email body text")
    in_reply_to: str = Field(default="", description="In-Reply-To header")
    references: str = Field(default="", description="References header")
    attachments: List[AttachmentInfo] = Field(default_factory=list, description="List of attachments")

    model_config = {"populate_by_name": True}


class CheckEmailsResponse(BaseModel):
    """Response model for the check emails endpoint."""
    status: str = Field(..., description="Status of the operation")
    processed_count: int = Field(..., description="Number of emails processed")
    email_uids: List[str] = Field(..., description="List of processed email UIDs")


class EmailListResponse(BaseModel):
    """Response model for listing emails."""
    status: str = Field(default="success", description="Status of the operation")
    count: int = Field(..., description="Number of emails returned")
    emails: List[EmailSummary] = Field(..., description="List of emails")


class SendEmailRequest(BaseModel):
    """Request model for sending a new email."""
    to: Union[str, List[str]] = Field(..., description="Recipient email address or list of addresses")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")
    cc: Optional[Union[str, List[str]]] = Field(default=None, description="CC recipient(s)")
    bcc: Optional[Union[str, List[str]]] = Field(default=None, description="BCC recipient(s)")
    html: bool = Field(default=False, description="Send as HTML email")


class SearchEmailRequest(BaseModel):
    """Request model for searching emails."""
    query: str = Field(..., description='IMAP search query (e.g., \'FROM "user@example.com"\')')
    count: int = Field(default=10, ge=1, description="Maximum number of results")
    mailbox: str = Field(default="INBOX", description="Mailbox to search")


class MailboxStatusResponse(BaseModel):
    """Response model for mailbox status."""
    mailbox: str = Field(..., description="Mailbox name")
    total: int = Field(..., description="Total messages")
    unseen: int = Field(..., description="Unseen messages")
    recent: int = Field(..., description="Recent messages")


class AgentEmailResponse(BaseModel):
    """Structured response from the Agent for email processing."""
    action: Literal["reply", "ignore"] = Field(
        ..., description="Action to take: 'reply' or 'ignore'"
    )
    reply_body: str = Field(
        ..., description="The body of the reply email (required if action is 'reply')"
    )
    reasoning: str = Field(
        ..., description="Brief reasoning for the decision"
    )
