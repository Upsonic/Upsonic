"""
Generic Mail (SMTP/IMAP) Interface for the Upsonic Framework.

This module provides a universal email interface that works with any
mail provider supporting standard SMTP and IMAP protocols. It supports:
- Fetching and listing emails via IMAP
- Sending emails and replies via SMTP
- Attachment handling (incoming and outgoing)
- Task and Chat modes
- Whitelist-based access control
- Heartbeat auto-poll for AutonomousAgent
- Event deduplication
- Configurable mailbox/folder

Works with: Gmail, Outlook, Yahoo, Zoho, self-hosted mail servers, etc.
"""

import asyncio
import os
import re
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, HTTPException, Header, Query, status

from upsonic.interfaces.base import Interface
from upsonic.interfaces.schemas import InterfaceMode
from upsonic.interfaces.mail.schemas import (
    AgentEmailResponse,
    CheckEmailsResponse,
    EmailListResponse,
    EmailSummary,
    MailboxStatusResponse,
    SearchEmailRequest,
    SendEmailRequest,
)
from upsonic.tools.custom_tools.mail import MailTools
from upsonic.utils.printing import debug_log, error_log, info_log

if TYPE_CHECKING:
    from upsonic.agent import Agent
    from upsonic.storage.base import Storage


class MailInterface(Interface):
    """
    Generic Mail interface for the Upsonic framework.

    This interface enables an Agent to:
    - Poll for unread emails via IMAP
    - List, search, and manage emails
    - Process emails through the agent (task or chat mode)
    - Reply to emails via SMTP (with optional attachments)
    - Handle incoming attachments (passed to agent as temp files)
    - Auto-poll on interval via heartbeat (for AutonomousAgent)

    Supports two operating modes:
    - TASK: Each email is processed as an independent task (default)
    - CHAT: Emails from the same sender continue a conversation session.
            Sending "/reset" in an email body resets the conversation.

    Supports whitelist-based access control:
    - Only emails from allowed_emails can interact with the agent
    - Unauthorized senders are silently skipped

    Attributes:
        agent: The AI agent that processes emails
        mail_tools: The MailTools instance for SMTP/IMAP operations
        api_secret: Secret token to protect API endpoints
        mode: Operating mode (TASK or CHAT)
        allowed_emails: Set of allowed email addresses (whitelist)
        mailbox: IMAP mailbox/folder to poll (default: INBOX)
    """

    def __init__(
        self,
        agent: "Agent",
        name: str = "Mail",
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        imap_host: Optional[str] = None,
        imap_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        from_address: Optional[str] = None,
        api_secret: Optional[str] = None,
        mode: Union[InterfaceMode, str] = InterfaceMode.TASK,
        reset_command: Optional[str] = "/reset",
        storage: Optional["Storage"] = None,
        allowed_emails: Optional[List[str]] = None,
        mailbox: str = "INBOX",
    ):
        """
        Initialize the Mail interface.

        Args:
            agent: The AI agent to process emails
            name: Interface name (defaults to "Mail")
            smtp_host: SMTP server hostname (or MAIL_SMTP_HOST env var)
            smtp_port: SMTP server port (or MAIL_SMTP_PORT env var, default: 587)
            imap_host: IMAP server hostname (or MAIL_IMAP_HOST env var)
            imap_port: IMAP server port (or MAIL_IMAP_PORT env var, default: 993)
            username: Email account username (or MAIL_USERNAME env var)
            password: Email account password (or MAIL_PASSWORD env var)
            use_ssl: Use SSL for SMTP instead of STARTTLS (default: False)
            from_address: Sender address (defaults to username)
            api_secret: Secret token for API authentication (or MAIL_API_SECRET env var)
            mode: Operating mode - TASK or CHAT
            reset_command: Command to reset chat session (CHAT mode only). Default: "/reset"
            storage: Optional storage backend for chat sessions.
            allowed_emails: List of allowed sender email addresses. If provided, only
                           emails from these addresses will be processed. If None, all
                           emails are processed.
            mailbox: IMAP mailbox/folder to poll (default: INBOX)
        """
        super().__init__(
            agent=agent,
            name=name,
            mode=mode,
            reset_command=reset_command,
            storage=storage,
        )

        # Initialize mail tools
        self.mail_tools = MailTools(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            imap_host=imap_host,
            imap_port=imap_port,
            username=username,
            password=password,
            use_ssl=use_ssl,
            from_address=from_address,
        )

        # API Secret for endpoint protection
        self.api_secret = api_secret or os.getenv("MAIL_API_SECRET")
        if not self.api_secret:
            debug_log(
                "MAIL_API_SECRET not set. API endpoints will not be protected. "
                "Please set the MAIL_API_SECRET environment variable for security."
            )

        # Mailbox to poll
        self.mailbox = mailbox

        # Whitelist: allowed email addresses (normalized to lowercase)
        self._allowed_emails: Optional[Set[str]] = None
        if allowed_emails is not None:
            self._allowed_emails = {e.lower().strip() for e in allowed_emails}
            info_log(f"Mail whitelist enabled with {len(self._allowed_emails)} allowed email(s)")

        # Event deduplication: UID -> timestamp
        self._processed_emails: Dict[str, float] = {}
        self._dedup_window: int = 300  # 5 minutes

        # Heartbeat (auto-poll) state
        self._heartbeat_task: Optional[asyncio.Task] = None

        info_log(f"Mail interface initialized: mode={self.mode.value}, agent={agent}")

    # ── Access Control ───────────────────────────────────────────────

    def is_email_allowed(self, sender: str) -> bool:
        """
        Check if a sender email address is allowed to interact with the agent.

        Args:
            sender: Sender string (can be in "Name <email>" format)

        Returns:
            bool: True if allowed or no whitelist configured, False otherwise
        """
        if self._allowed_emails is None:
            return True
        normalized = self._extract_sender_id(sender)
        return normalized in self._allowed_emails

    # ── Deduplication ────────────────────────────────────────────────

    def _is_duplicate(self, uid: str) -> bool:
        """Check if an email UID has already been processed recently."""
        if uid in self._processed_emails:
            debug_log(f"Duplicate email received: {uid}")
            return True
        return False

    def _mark_processed(self, uid: str) -> None:
        """Mark an email UID as processed."""
        self._processed_emails[uid] = time.time()
        # Cleanup old entries when cache gets large
        if len(self._processed_emails) > 1000:
            self._cleanup_processed_emails()

    def _cleanup_processed_emails(self) -> None:
        """Remove expired entries from the dedup cache."""
        now = time.time()
        expired = [
            uid for uid, ts in self._processed_emails.items()
            if now - ts > self._dedup_window
        ]
        for uid in expired:
            del self._processed_emails[uid]

    # ── API Secret Verification ──────────────────────────────────────

    def _verify_secret(self, secret: Optional[str]) -> None:
        """Verify API secret if configured. Raises HTTPException on failure."""
        if self.api_secret:
            if not secret or secret != self.api_secret:
                error_log("Mail API authentication failed: Invalid secret")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication secret",
                )

    # ── Routes ───────────────────────────────────────────────────────

    def attach_routes(self) -> APIRouter:
        """
        Create and attach mail routes to the FastAPI application.

        Routes:
            POST /mail/check          - Process unread emails through the agent
            GET  /mail/inbox          - List recent emails
            GET  /mail/unread         - List unread emails
            POST /mail/send           - Send a new email
            POST /mail/search         - Search emails
            GET  /mail/folders        - List mailboxes/folders
            GET  /mail/status         - Get mailbox status (counts)
            POST /mail/{uid}/read     - Mark email as read
            POST /mail/{uid}/unread   - Mark email as unread
            POST /mail/{uid}/delete   - Delete an email
            POST /mail/{uid}/move     - Move email to another folder
            GET  /mail/health         - Health check

        Returns:
            APIRouter: Router with mail endpoints
        """
        router = APIRouter(prefix="/mail", tags=["Mail"])

        # ── Process unread emails through agent ──────────────────────

        @router.post("/check", response_model=CheckEmailsResponse, summary="Check and Process Emails")
        async def check_emails(
            count: int = Query(10, ge=1, description="Maximum number of emails to process"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Trigger the agent to check for unread emails and process them."""
            self._verify_secret(x_upsonic_mail_secret)
            return await self.check_and_process_emails(count)

        # ── List emails ──────────────────────────────────────────────

        @router.get("/inbox", response_model=EmailListResponse, summary="List Recent Emails")
        async def list_inbox(
            count: int = Query(20, ge=1, le=100, description="Number of emails to return"),
            mailbox: str = Query("INBOX", description="Mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """List the most recent emails (read and unread)."""
            self._verify_secret(x_upsonic_mail_secret)
            emails = await self.mail_tools.aget_latest_emails(count, mailbox)
            return EmailListResponse(
                count=len(emails),
                emails=[EmailSummary(**self._email_to_summary(e)) for e in emails],
            )

        @router.get("/unread", response_model=EmailListResponse, summary="List Unread Emails")
        async def list_unread(
            count: int = Query(20, ge=1, le=100, description="Number of emails to return"),
            mailbox: str = Query("INBOX", description="Mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """List unread emails only."""
            self._verify_secret(x_upsonic_mail_secret)
            emails = await self.mail_tools.aget_unread_emails(count, mailbox)
            return EmailListResponse(
                count=len(emails),
                emails=[EmailSummary(**self._email_to_summary(e)) for e in emails],
            )

        # ── Send email ───────────────────────────────────────────────

        @router.post("/send", summary="Send Email")
        async def send_email(
            request: SendEmailRequest,
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Send a new email."""
            self._verify_secret(x_upsonic_mail_secret)
            success = await self.mail_tools.asend_email(
                to=request.to,
                subject=request.subject,
                body=request.body,
                cc=request.cc,
                bcc=request.bcc,
                html=request.html,
            )
            if success:
                return {"status": "success", "message": f"Email sent to {request.to}"}
            raise HTTPException(status_code=500, detail="Failed to send email")

        # ── Search emails ────────────────────────────────────────────

        @router.post("/search", response_model=EmailListResponse, summary="Search Emails")
        async def search_emails(
            request: SearchEmailRequest,
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Search emails using IMAP search criteria."""
            self._verify_secret(x_upsonic_mail_secret)
            emails = await self.mail_tools.asearch_emails(
                query=request.query, count=request.count, mailbox=request.mailbox
            )
            return EmailListResponse(
                count=len(emails),
                emails=[EmailSummary(**self._email_to_summary(e)) for e in emails],
            )

        # ── List folders ─────────────────────────────────────────────

        @router.get("/folders", summary="List Mailbox Folders")
        async def list_folders(
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """List all available mailboxes/folders."""
            self._verify_secret(x_upsonic_mail_secret)
            folders = await self.mail_tools.alist_mailboxes()
            return {"status": "success", "folders": folders}

        # ── Mailbox status ───────────────────────────────────────────

        @router.get("/status", response_model=MailboxStatusResponse, summary="Mailbox Status")
        async def mailbox_status(
            mailbox: str = Query("INBOX", description="Mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Get the status of a mailbox (total, unseen, recent counts)."""
            self._verify_secret(x_upsonic_mail_secret)
            stats = await self.mail_tools.aget_mailbox_status(mailbox)
            return MailboxStatusResponse(mailbox=mailbox, **stats)

        # ── Per-email actions ────────────────────────────────────────

        @router.post("/{uid}/read", summary="Mark Email as Read")
        async def mark_read(
            uid: str,
            mailbox: str = Query("INBOX", description="Mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Mark an email as read by its UID."""
            self._verify_secret(x_upsonic_mail_secret)
            success = await self.mail_tools.amark_email_as_read(uid, mailbox)
            if success:
                return {"status": "success", "uid": uid, "action": "marked_read"}
            raise HTTPException(status_code=500, detail="Failed to mark email as read")

        @router.post("/{uid}/unread", summary="Mark Email as Unread")
        async def mark_unread(
            uid: str,
            mailbox: str = Query("INBOX", description="Mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Mark an email as unread by its UID."""
            self._verify_secret(x_upsonic_mail_secret)
            success = await self.mail_tools.amark_email_as_unread(uid, mailbox)
            if success:
                return {"status": "success", "uid": uid, "action": "marked_unread"}
            raise HTTPException(status_code=500, detail="Failed to mark email as unread")

        @router.post("/{uid}/delete", summary="Delete Email")
        async def delete_email(
            uid: str,
            mailbox: str = Query("INBOX", description="Mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Delete an email by its UID."""
            self._verify_secret(x_upsonic_mail_secret)
            success = await self.mail_tools.adelete_email(uid, mailbox)
            if success:
                return {"status": "success", "uid": uid, "action": "deleted"}
            raise HTTPException(status_code=500, detail="Failed to delete email")

        @router.post("/{uid}/move", summary="Move Email")
        async def move_email(
            uid: str,
            destination: str = Query(..., description="Destination mailbox/folder"),
            source: str = Query("INBOX", description="Source mailbox/folder"),
            x_upsonic_mail_secret: Optional[str] = Header(None, alias="X-Upsonic-Mail-Secret"),
        ):
            """Move an email to a different mailbox/folder."""
            self._verify_secret(x_upsonic_mail_secret)
            success = await self.mail_tools.amove_email(uid, destination, source)
            if success:
                return {"status": "success", "uid": uid, "action": "moved", "destination": destination}
            raise HTTPException(status_code=500, detail="Failed to move email")

        # ── Health check ─────────────────────────────────────────────

        @router.get("/health", summary="Health Check")
        async def health_check_endpoint():
            """Health check endpoint for Mail interface."""
            return await self.health_check()

        # ── Startup: heartbeat ───────────────────────────────────────

        @router.on_event("startup")
        async def start_heartbeat() -> None:
            self._start_heartbeat()

        info_log("Mail routes attached with prefix: /mail")
        return router

    # ── Health Check ─────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the mail interface."""
        base_health = await super().health_check()

        imap_connected = False
        try:
            conn = await asyncio.to_thread(self.mail_tools._get_imap_connection)
            conn.logout()
            imap_connected = True
        except Exception:
            pass

        base_health["configuration"] = {
            "imap_connected": imap_connected,
            "smtp_host": self.mail_tools.smtp_host,
            "imap_host": self.mail_tools.imap_host,
            "from_address": self.mail_tools.from_address,
            "mailbox": self.mailbox,
            "auth_configured": bool(self.api_secret),
            "mode": self.mode.value,
            "reset_command": self._reset_command.command if self._reset_enabled else None,
            "active_chat_sessions": len(self._chat_sessions) if self.is_chat_mode() else 0,
            "whitelist_enabled": self._allowed_emails is not None,
            "allowed_emails_count": len(self._allowed_emails) if self._allowed_emails else 0,
            "heartbeat_active": self._heartbeat_task is not None and not self._heartbeat_task.done(),
            "dedup_cache_size": len(self._processed_emails),
        }
        return base_health

    # ── Heartbeat (Auto-Poll) ────────────────────────────────────────

    def _start_heartbeat(self) -> None:
        """
        Start the heartbeat background task if the agent is AutonomousAgent
        with heartbeat enabled. The heartbeat auto-polls for new emails on
        a configurable interval.

        Safe to call multiple times — will not create duplicate tasks.
        """
        from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent

        if not isinstance(self.agent, AutonomousAgent):
            return
        if not self.agent.heartbeat:
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        info_log(
            f"Mail heartbeat started: period={self.agent.heartbeat_period}min, "
            f"mailbox={self.mailbox}"
        )

    async def _heartbeat_loop(self) -> None:
        """
        Background coroutine that periodically checks for new unread emails
        and processes them through the agent.

        For mail, the heartbeat serves as an auto-poll mechanism — checking
        the IMAP mailbox on each tick and processing any new emails found.
        """
        from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent

        if not isinstance(self.agent, AutonomousAgent):
            return
        if not self.agent.heartbeat:
            return

        period_seconds: int = self.agent.heartbeat_period * 60

        while True:
            await asyncio.sleep(period_seconds)

            try:
                result = await self.check_and_process_emails(count=10)
                if result.processed_count > 0:
                    info_log(
                        f"Mail heartbeat: processed {result.processed_count} email(s)"
                    )
                else:
                    debug_log("Mail heartbeat: no new emails")
            except Exception as exc:
                error_log(f"Mail heartbeat error: {exc}")

    # ── Core Processing ──────────────────────────────────────────────

    async def check_and_process_emails(self, count: int = 10) -> CheckEmailsResponse:
        """
        Fetch unread emails and process them according to the interface mode.

        In TASK mode: Each email is processed as an independent task.
        In CHAT mode: Emails from the same sender continue a conversation session.

        Includes deduplication to prevent processing the same email twice.

        Args:
            count: Number of emails to fetch

        Returns:
            CheckEmailsResponse: Summary of processed emails
        """
        info_log(f"Checking for up to {count} unread emails (mode={self.mode.value})...")

        try:
            messages = await self.mail_tools.aget_unread_emails(count, self.mailbox)
        except Exception as e:
            error_log(f"Failed to fetch unread emails: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch emails: {str(e)}",
            )

        if not messages:
            info_log("No unread emails found.")
            return CheckEmailsResponse(
                status="success",
                processed_count=0,
                email_uids=[],
            )

        processed_uids: List[str] = []

        for msg_data in messages:
            try:
                uid = msg_data.get("uid", "")
                sender = msg_data.get("from", "")
                subject = msg_data.get("subject", "")
                body = msg_data.get("body", "")

                # Deduplication check
                if self._is_duplicate(uid):
                    continue

                info_log(f"Processing email {uid} from {sender}: {subject}")

                # Check whitelist
                if not self.is_email_allowed(sender):
                    info_log(self.get_unauthorized_message())
                    await self.mail_tools.amark_email_as_read(uid, self.mailbox)
                    self._mark_processed(uid)
                    continue

                # Check for reset command in CHAT mode
                if self.is_chat_mode() and self.is_reset_command(body.strip()):
                    await self._handle_reset_command(msg_data)
                    self._mark_processed(uid)
                    processed_uids.append(uid)
                    continue

                # Process based on mode
                if self.is_task_mode():
                    await self._process_email_task_mode(msg_data)
                else:
                    await self._process_email_chat_mode(msg_data)

                # Mark as read after processing
                await self.mail_tools.amark_email_as_read(uid, self.mailbox)
                self._mark_processed(uid)
                processed_uids.append(uid)

            except Exception as e:
                error_log(f"Error processing email {msg_data.get('uid')}: {e}")
                continue

        return CheckEmailsResponse(
            status="success",
            processed_count=len(processed_uids),
            email_uids=processed_uids,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _email_to_summary(email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal email dict to schema-compatible dict."""
        return {
            "uid": email_data.get("uid", ""),
            "message_id": email_data.get("message_id", ""),
            "from": email_data.get("from", ""),
            "to": email_data.get("to", ""),
            "cc": email_data.get("cc", ""),
            "subject": email_data.get("subject", ""),
            "date": email_data.get("date", ""),
            "body": email_data.get("body", ""),
            "in_reply_to": email_data.get("in_reply_to", ""),
            "references": email_data.get("references", ""),
            "attachments": email_data.get("attachments", []),
        }

    def _extract_sender_id(self, sender: str) -> str:
        """
        Extract a normalized email address from the sender field.

        Args:
            sender: The 'From' field, e.g., 'John Doe <john@example.com>'

        Returns:
            str: Normalized email address
        """
        match = re.search(r"<([^>]+)>", sender)
        if match:
            return match.group(1).lower().strip()
        return sender.lower().strip()

    async def _send_reply(self, email_data: Dict[str, Any], reply_text: str) -> None:
        """
        Send a reply to an email using SMTP.

        Args:
            email_data: The original email dictionary
            reply_text: The body of the reply
        """
        try:
            sender = email_data.get("from", "")
            to_address = self._extract_sender_id(sender)
            await self.mail_tools.asend_reply(
                to=to_address,
                subject=email_data.get("subject", ""),
                body=reply_text,
                message_id=email_data.get("message_id", ""),
                references=email_data.get("references", ""),
            )
            info_log(f"Sent reply to {to_address}")
        except Exception as e:
            error_log(f"Failed to send reply: {e}")

    async def _handle_reset_command(self, msg_data: Dict[str, Any]) -> None:
        """Handle a reset command email in CHAT mode."""
        uid = msg_data.get("uid", "")
        sender = msg_data.get("from", "")
        user_id = self._extract_sender_id(sender)

        info_log(f"Reset command received from {sender} for email {uid}")

        was_reset = await self.areset_chat_session(user_id)

        if was_reset:
            reply_text = (
                "Your conversation has been reset. "
                "I'm ready to start fresh! How can I help you?"
            )
        else:
            reply_text = (
                "No active conversation found to reset. "
                "Send me a new message to start a conversation!"
            )

        await self._send_reply(msg_data, reply_text)
        await self.mail_tools.amark_email_as_read(uid, self.mailbox)

        info_log(f"Reset command processed for user {user_id}")

    # ── Attachment Handling ──────────────────────────────────────────

    async def _download_attachments_to_temp(self, msg_data: Dict[str, Any]) -> List[str]:
        """
        Download email attachments to temporary files for agent processing.

        Args:
            msg_data: Email data dictionary with 'uid' field

        Returns:
            List of temporary file paths. Caller is responsible for cleanup.
        """
        uid = msg_data.get("uid", "")
        attachments_meta = msg_data.get("attachments", [])
        if not attachments_meta:
            return []

        try:
            raw_attachments = await self.mail_tools.aget_raw_attachments(uid, self.mailbox)
        except Exception as e:
            error_log(f"Failed to download attachments for email {uid}: {e}")
            return []

        temp_files = []
        for filename, content_type, data in raw_attachments:
            try:
                # Determine extension from filename or content type
                _, ext = os.path.splitext(filename)
                if not ext:
                    import mimetypes
                    ext = mimetypes.guess_extension(content_type) or ""

                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=ext,
                    prefix="mail_attachment_",
                )
                tmp.write(data)
                tmp.close()
                temp_files.append(tmp.name)
            except Exception as e:
                error_log(f"Failed to save attachment {filename}: {e}")
                continue

        return temp_files

    @staticmethod
    def _cleanup_temp_files(temp_files: List[str]) -> None:
        """Remove temporary files created for attachment processing."""
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass

    # ── Task Mode Processing ─────────────────────────────────────────

    async def _process_email_task_mode(self, msg_data: Dict[str, Any]) -> None:
        """
        Process an email in TASK mode (independent task per email).

        Creates a structured Task for the agent to decide whether to reply or ignore.
        Handles attachments by downloading to temp files and passing to the agent.
        """
        from upsonic.tasks.tasks import Task

        uid = msg_data.get("uid", "")
        sender = msg_data.get("from", "")
        subject = msg_data.get("subject", "")
        body = msg_data.get("body", "")
        attachments_meta = msg_data.get("attachments", [])

        # Build attachment info string
        attachment_info = ""
        if attachments_meta:
            attachment_list = ", ".join(
                f"{a['filename']} ({a['content_type']}, {a['size']} bytes)"
                for a in attachments_meta
            )
            attachment_info = f"\nAttachments: {attachment_list}\n"

        task_description = (
            f"You are an email assistant. You have received a new email.\n\n"
            f"EMAIL:\n"
            f"From: {sender}\n"
            f"Subject: {subject}\n"
            f"{attachment_info}"
            f"Content:\n{body}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Decide whether to 'reply' or 'ignore' (ignore spam, automated notifications, "
            f"or no-action-needed emails).\n"
            f"2. If you reply, the reply_body must contain ONLY the direct response to the email. "
            f"Do NOT include greetings like 'Dear X', do NOT include subject lines, do NOT include "
            f"signatures, do NOT include any metadata or formatting beyond the actual response text. "
            f"Just write the reply as you would in a natural, concise email response.\n"
            f"3. Provide a brief reasoning for your decision."
        )

        # Download attachments to temp files
        temp_files = await self._download_attachments_to_temp(msg_data)

        try:
            if temp_files:
                task = Task(task_description, response_format=AgentEmailResponse, attachments=temp_files)
            else:
                task = Task(task_description, response_format=AgentEmailResponse)

            await self.agent.do_async(task)

            run_result = self.agent.get_run_output()

            if run_result and run_result.output:
                response: AgentEmailResponse = run_result.output

                info_log(f"Agent decision for {uid}: {response.action} (Reason: {response.reasoning})")

                if response.action == "reply":
                    if response.reply_body:
                        await self._send_reply(msg_data, response.reply_body)
                    else:
                        error_log(f"Agent decided to reply but provided no body for email {uid}")
                else:
                    info_log(f"Skipping reply for email {uid}")
        finally:
            self._cleanup_temp_files(temp_files)

    # ── Chat Mode Processing ─────────────────────────────────────────

    async def _process_email_chat_mode(self, msg_data: Dict[str, Any]) -> None:
        """
        Process an email in CHAT mode (conversation session per sender).

        Uses the chat session to maintain conversation history.
        Handles attachments by downloading to temp files.
        """
        uid = msg_data.get("uid", "")
        sender = msg_data.get("from", "")
        subject = msg_data.get("subject", "")
        body = msg_data.get("body", "")
        attachments_meta = msg_data.get("attachments", [])

        user_id = self._extract_sender_id(sender)
        chat = await self.aget_chat_session(user_id)

        chat_message = (
            f"You received an email. Respond with ONLY the direct reply text. "
            f"No greetings, no signatures, no metadata, no subject lines — just the response.\n\n"
            f"From: {sender}\n"
            f"Subject: {subject}\n"
        )
        if attachments_meta:
            attachment_list = ", ".join(a["filename"] for a in attachments_meta)
            chat_message += f"Attachments: {attachment_list}\n"
        chat_message += f"\n{body}"

        info_log(f"Processing email {uid} in CHAT mode for user {user_id}")

        temp_files = await self._download_attachments_to_temp(msg_data)

        try:
            # Pass attachments to chat if available
            if temp_files:
                response_text = await chat.invoke(chat_message, attachments=temp_files)
            else:
                response_text = await chat.invoke(chat_message)

            if response_text:
                await self._send_reply(msg_data, response_text)
                info_log(f"Sent chat response to {sender} for email {uid}")
            else:
                debug_log(f"No response generated for email {uid}")

        except Exception as e:
            error_log(f"Error in chat mode processing for email {uid}: {e}")
            error_reply = (
                "I apologize, but I encountered an error processing your message. "
                "Please try again or send '/reset' to start a new conversation."
            )
            await self._send_reply(msg_data, error_reply)
        finally:
            self._cleanup_temp_files(temp_files)
