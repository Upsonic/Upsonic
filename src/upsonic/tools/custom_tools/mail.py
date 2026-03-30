"""
Generic Mail Toolkit for SMTP (send) and IMAP (receive) operations.

Works with any mail provider that supports standard SMTP/IMAP protocols
(Gmail, Outlook, Yahoo, self-hosted, etc.).

Required Configuration:
----------------------
- SMTP host, port, username, password (for sending)
- IMAP host, port, username, password (for receiving)

Environment Variables (optional, can also pass directly):
- MAIL_SMTP_HOST: SMTP server hostname
- MAIL_SMTP_PORT: SMTP server port (default: 587)
- MAIL_IMAP_HOST: IMAP server hostname
- MAIL_IMAP_PORT: IMAP server port (default: 993)
- MAIL_USERNAME: Email account username
- MAIL_PASSWORD: Email account password (or app password)
- MAIL_USE_SSL: Use SSL for SMTP (default: false, uses STARTTLS)
"""

import asyncio
import email
import email.utils
import imaplib
import mimetypes
import os
import re
import smtplib
import ssl
from email.header import decode_header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import getenv
from typing import Any, Dict, List, Optional, Tuple, Union

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool


def _decode_header_value(value) -> str:
    """Decode an email header value that may be encoded."""
    if not value:
        return ""
    # Handle email.header.Header objects
    if hasattr(value, '__str__') and not isinstance(value, str):
        value = str(value)
    decoded_parts = decode_header(value)
    result = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            # Some emails have invalid charset names like 'unknown-8bit'
            try:
                result.append(part.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                result.append(part.decode("utf-8", errors="replace"))
        else:
            result.append(part)
    return "".join(result)


def _extract_body(msg: email.message.Message) -> str:
    """Extract the plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        # Fallback to HTML if no plain text
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/html" and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def _extract_attachments_metadata(msg: email.message.Message) -> List[Dict[str, Any]]:
    """Extract attachment metadata (filename, size, content_type) from an email."""
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        content_disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in content_disposition or "inline" in content_disposition:
            filename = part.get_filename()
            if filename:
                filename = _decode_header_value(filename)
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            size = len(payload) if payload else 0
            attachments.append({
                "filename": filename or "unknown",
                "content_type": content_type,
                "size": size,
            })
    return attachments


def _extract_attachment_bytes(msg: email.message.Message) -> List[Tuple[str, str, bytes]]:
    """Extract attachment data (filename, content_type, bytes) from an email."""
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        content_disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in content_disposition or "inline" in content_disposition:
            filename = part.get_filename()
            if filename:
                filename = _decode_header_value(filename)
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            if payload:
                attachments.append((filename or "unknown", content_type, payload))
    return attachments


class MailTools(ToolKit):
    """
    Generic Mail toolkit providing SMTP (send) and IMAP (receive) tools.

    Works with any standard mail provider. Configure SMTP for sending
    and IMAP for receiving emails.
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        imap_host: Optional[str] = None,
        imap_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        from_address: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MailTools with SMTP and IMAP configuration.

        Args:
            smtp_host: SMTP server hostname (or MAIL_SMTP_HOST env var)
            smtp_port: SMTP server port (or MAIL_SMTP_PORT env var, default: 587)
            imap_host: IMAP server hostname (or MAIL_IMAP_HOST env var)
            imap_port: IMAP server port (or MAIL_IMAP_PORT env var, default: 993)
            username: Email account username (or MAIL_USERNAME env var)
            password: Email account password (or MAIL_PASSWORD env var)
            use_ssl: Use SSL for SMTP instead of STARTTLS (default: False)
            from_address: Sender address (defaults to username)
        """
        super().__init__(**kwargs)

        self.smtp_host = smtp_host or getenv("MAIL_SMTP_HOST", "")
        self.smtp_port = smtp_port or int(getenv("MAIL_SMTP_PORT", "587"))
        self.imap_host = imap_host or getenv("MAIL_IMAP_HOST", "")
        self.imap_port = imap_port or int(getenv("MAIL_IMAP_PORT", "993"))
        self.username = username or getenv("MAIL_USERNAME", "")
        self.password = password or getenv("MAIL_PASSWORD", "")
        self.use_ssl = use_ssl or getenv("MAIL_USE_SSL", "false").lower() == "true"
        self.from_address = from_address or self.username

    # ── IMAP helpers (internal) ──────────────────────────────────────

    def _get_imap_connection(self) -> imaplib.IMAP4_SSL:
        """Create and return an authenticated IMAP connection."""
        conn = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
        conn.login(self.username, self.password)
        return conn

    def _parse_email(self, raw_email: bytes, uid: str) -> Dict[str, Any]:
        """Parse a raw email into a structured dictionary including attachment metadata."""
        msg = email.message_from_bytes(raw_email)
        return {
            "uid": uid,
            "message_id": msg.get("Message-ID", ""),
            "from": _decode_header_value(msg.get("From", "")),
            "to": _decode_header_value(msg.get("To", "")),
            "cc": _decode_header_value(msg.get("Cc", "")),
            "subject": _decode_header_value(msg.get("Subject", "")),
            "date": msg.get("Date", ""),
            "body": _extract_body(msg),
            "in_reply_to": msg.get("In-Reply-To", ""),
            "references": msg.get("References", ""),
            "attachments": _extract_attachments_metadata(msg),
        }

    def _fetch_uids(
        self,
        conn: imaplib.IMAP4_SSL,
        search_criteria: str,
        count: int,
    ) -> List[Dict[str, Any]]:
        """Fetch and parse emails by IMAP search criteria (internal helper)."""
        status, data = conn.uid("search", None, search_criteria)
        if status != "OK" or not data[0]:
            return []

        uids = data[0].split()
        uids = uids[-count:]

        results = []
        for uid in uids:
            uid_str = uid.decode() if isinstance(uid, bytes) else uid
            status, msg_data = conn.uid("fetch", uid_str, "(RFC822)")
            if status == "OK" and msg_data[0]:
                raw_email = msg_data[0][1]
                parsed = self._parse_email(raw_email, uid_str)
                results.append(parsed)

        return results

    # ── SMTP helpers (internal) ──────────────────────────────────────

    def _get_smtp_connection(self) -> smtplib.SMTP:
        """Create and return an authenticated SMTP connection."""
        if self.use_ssl:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context)
        else:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
        server.login(self.username, self.password)
        return server

    def _build_message_with_attachments(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html: bool = False,
        attachment_paths: Optional[List[str]] = None,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
    ) -> MIMEMultipart:
        """Build a MIME message with optional attachments, CC/BCC, and reply headers."""
        msg = MIMEMultipart("mixed")
        msg["From"] = self.from_address
        # Handle single or multiple recipients
        if isinstance(to, list):
            msg["To"] = ", ".join(to)
        else:
            msg["To"] = to
        if cc:
            msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
        msg["Subject"] = subject

        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            if references:
                msg["References"] = f"{references} {in_reply_to}"
            else:
                msg["References"] = in_reply_to

        content_type = "html" if html else "plain"
        msg.attach(MIMEText(body, content_type, "utf-8"))

        for file_path in (attachment_paths or []):
            if not os.path.isfile(file_path):
                continue
            filename = os.path.basename(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"

            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=filename)
            part["Content-Disposition"] = f'attachment; filename="{filename}"'
            msg.attach(part)

        return msg

    def _send_message(
        self,
        msg: MIMEMultipart,
        to: Union[str, List[str]],
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
    ) -> bool:
        """Send a prepared MIME message via SMTP."""
        # Build full recipient list: to + cc + bcc
        recipients: List[str] = []
        if isinstance(to, list):
            recipients.extend(to)
        else:
            recipients.append(to)
        if cc:
            recipients.extend(cc if isinstance(cc, list) else [cc])
        if bcc:
            recipients.extend(bcc if isinstance(bcc, list) else [bcc])

        server = self._get_smtp_connection()
        try:
            server.sendmail(self.from_address, recipients, msg.as_string())
            return True
        finally:
            server.quit()

    # ── IMAP Tools (exposed to agent) ────────────────────────────────

    @tool
    def get_unread_emails(self, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Fetch unread emails from the mailbox.

        Args:
            count: Maximum number of unread emails to fetch (default: 10)
            mailbox: Mailbox/folder to check (default: INBOX)

        Returns:
            List of email dictionaries with uid, from, to, subject, date, body, attachments fields.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox, readonly=False)
            return self._fetch_uids(conn, "UNSEEN", count)
        finally:
            conn.logout()

    async def aget_unread_emails(self, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """Async version of get_unread_emails."""
        return await asyncio.to_thread(self.get_unread_emails, count, mailbox)

    @tool
    def get_latest_emails(self, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Fetch the latest emails from the mailbox (read or unread).

        Args:
            count: Maximum number of emails to fetch (default: 10)
            mailbox: Mailbox/folder to check (default: INBOX)

        Returns:
            List of email dictionaries with uid, from, to, subject, date, body, attachments fields.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox, readonly=True)
            return self._fetch_uids(conn, "ALL", count)
        finally:
            conn.logout()

    async def aget_latest_emails(self, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """Async version of get_latest_emails."""
        return await asyncio.to_thread(self.get_latest_emails, count, mailbox)

    @tool
    def get_emails_from_sender(self, sender_email: str, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Fetch emails from a specific sender.

        Args:
            sender_email: The sender's email address to filter by
            count: Maximum number of emails to fetch (default: 10)
            mailbox: Mailbox/folder to search (default: INBOX)

        Returns:
            List of email dictionaries from the specified sender.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox, readonly=True)
            return self._fetch_uids(conn, f'FROM "{sender_email}"', count)
        finally:
            conn.logout()

    async def aget_emails_from_sender(self, sender_email: str, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """Async version of get_emails_from_sender."""
        return await asyncio.to_thread(self.get_emails_from_sender, sender_email, count, mailbox)

    @tool
    def search_emails(self, query: str, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Search emails using IMAP search criteria.

        Args:
            query: IMAP search query (e.g., 'FROM "user@example.com"', 'SUBJECT "hello"',
                   'SINCE "01-Jan-2024"', 'TEXT "keyword"')
            count: Maximum number of results (default: 10)
            mailbox: Mailbox/folder to search (default: INBOX)

        Returns:
            List of email dictionaries matching the search criteria.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox, readonly=True)
            return self._fetch_uids(conn, query, count)
        finally:
            conn.logout()

    async def asearch_emails(self, query: str, count: int = 10, mailbox: str = "INBOX") -> List[Dict[str, Any]]:
        """Async version of search_emails."""
        return await asyncio.to_thread(self.search_emails, query, count, mailbox)

    @tool
    def mark_email_as_read(self, uid: str, mailbox: str = "INBOX") -> bool:
        """
        Mark an email as read (seen) by its UID.

        Args:
            uid: The UID of the email to mark as read
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox)
            status, _ = conn.uid("store", uid, "+FLAGS", "\\Seen")
            return status == "OK"
        finally:
            conn.logout()

    async def amark_email_as_read(self, uid: str, mailbox: str = "INBOX") -> bool:
        """Async version of mark_email_as_read."""
        return await asyncio.to_thread(self.mark_email_as_read, uid, mailbox)

    @tool
    def mark_email_as_unread(self, uid: str, mailbox: str = "INBOX") -> bool:
        """
        Mark an email as unread (unseen) by its UID.

        Args:
            uid: The UID of the email to mark as unread
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox)
            status, _ = conn.uid("store", uid, "-FLAGS", "\\Seen")
            return status == "OK"
        finally:
            conn.logout()

    async def amark_email_as_unread(self, uid: str, mailbox: str = "INBOX") -> bool:
        """Async version of mark_email_as_unread."""
        return await asyncio.to_thread(self.mark_email_as_unread, uid, mailbox)

    @tool
    def flag_email(self, uid: str, mailbox: str = "INBOX") -> bool:
        """
        Flag/star an email by its UID.

        Args:
            uid: The UID of the email to flag
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox)
            status, _ = conn.uid("store", uid, "+FLAGS", "\\Flagged")
            return status == "OK"
        finally:
            conn.logout()

    async def aflag_email(self, uid: str, mailbox: str = "INBOX") -> bool:
        """Async version of flag_email."""
        return await asyncio.to_thread(self.flag_email, uid, mailbox)

    @tool
    def unflag_email(self, uid: str, mailbox: str = "INBOX") -> bool:
        """
        Remove flag/star from an email by its UID.

        Args:
            uid: The UID of the email to unflag
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox)
            status, _ = conn.uid("store", uid, "-FLAGS", "\\Flagged")
            return status == "OK"
        finally:
            conn.logout()

    async def aunflag_email(self, uid: str, mailbox: str = "INBOX") -> bool:
        """Async version of unflag_email."""
        return await asyncio.to_thread(self.unflag_email, uid, mailbox)

    @tool
    def delete_email(self, uid: str, mailbox: str = "INBOX") -> bool:
        """
        Delete an email by its UID. Marks it with the Deleted flag and expunges.

        Args:
            uid: The UID of the email to delete
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox)
            status, _ = conn.uid("store", uid, "+FLAGS", "\\Deleted")
            if status == "OK":
                conn.expunge()
                return True
            return False
        finally:
            conn.logout()

    async def adelete_email(self, uid: str, mailbox: str = "INBOX") -> bool:
        """Async version of delete_email."""
        return await asyncio.to_thread(self.delete_email, uid, mailbox)

    @tool
    def move_email(self, uid: str, destination: str, source: str = "INBOX") -> bool:
        """
        Move an email to a different mailbox/folder.

        Args:
            uid: The UID of the email to move
            destination: Target mailbox/folder name (e.g., "Archive", "Trash", "Spam")
            source: Source mailbox/folder (default: INBOX)

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(source)
            # Copy to destination, then delete from source
            status, _ = conn.uid("copy", uid, destination)
            if status == "OK":
                conn.uid("store", uid, "+FLAGS", "\\Deleted")
                conn.expunge()
                return True
            return False
        finally:
            conn.logout()

    async def amove_email(self, uid: str, destination: str, source: str = "INBOX") -> bool:
        """Async version of move_email."""
        return await asyncio.to_thread(self.move_email, uid, destination, source)

    @tool
    def download_attachments(self, uid: str, save_dir: str, mailbox: str = "INBOX") -> List[str]:
        """
        Download all attachments from an email and save them to a directory.

        Args:
            uid: The UID of the email
            save_dir: Directory path to save attachments
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            List of saved file paths.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox, readonly=True)
            status, msg_data = conn.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not msg_data[0]:
                return []

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            attachment_data = _extract_attachment_bytes(msg)

            os.makedirs(save_dir, exist_ok=True)
            saved_paths = []
            for filename, content_type, data in attachment_data:
                file_path = os.path.join(save_dir, filename)
                # Avoid overwriting: append counter if file exists
                base, ext = os.path.splitext(file_path)
                counter = 1
                while os.path.exists(file_path):
                    file_path = f"{base}_{counter}{ext}"
                    counter += 1
                with open(file_path, "wb") as f:
                    f.write(data)
                saved_paths.append(file_path)

            return saved_paths
        finally:
            conn.logout()

    async def adownload_attachments(self, uid: str, save_dir: str, mailbox: str = "INBOX") -> List[str]:
        """Async version of download_attachments."""
        return await asyncio.to_thread(self.download_attachments, uid, save_dir, mailbox)

    def get_raw_attachments(self, uid: str, mailbox: str = "INBOX") -> List[Tuple[str, str, bytes]]:
        """
        Get raw attachment data (filename, content_type, bytes) from an email.
        Internal method — not exposed as agent tool.

        Args:
            uid: The UID of the email
            mailbox: Mailbox/folder (default: INBOX)

        Returns:
            List of (filename, content_type, bytes) tuples.
        """
        conn = self._get_imap_connection()
        try:
            conn.select(mailbox, readonly=True)
            status, msg_data = conn.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not msg_data[0]:
                return []
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            return _extract_attachment_bytes(msg)
        finally:
            conn.logout()

    async def aget_raw_attachments(self, uid: str, mailbox: str = "INBOX") -> List[Tuple[str, str, bytes]]:
        """Async version of get_raw_attachments."""
        return await asyncio.to_thread(self.get_raw_attachments, uid, mailbox)

    # ── SMTP Tools (exposed to agent) ────────────────────────────────

    @tool
    def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        html: bool = False,
    ) -> bool:
        """
        Send an email via SMTP to one or multiple recipients.

        Args:
            to: Recipient email address or list of addresses
            subject: Email subject line
            body: Email body content
            cc: CC recipient(s) - single address or list of addresses (optional)
            bcc: BCC recipient(s) - single address or list of addresses (optional)
            html: If True, send as HTML email; otherwise plain text (default: False)

        Returns:
            True if the email was sent successfully.
        """
        msg = self._build_message_with_attachments(to, subject, body, html=html, cc=cc, bcc=bcc)
        return self._send_message(msg, to, cc=cc, bcc=bcc)

    async def asend_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        html: bool = False,
    ) -> bool:
        """Async version of send_email."""
        return await asyncio.to_thread(self.send_email, to, subject, body, cc, bcc, html)

    @tool
    def send_email_with_attachments(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        attachment_paths: List[str],
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        html: bool = False,
    ) -> bool:
        """
        Send an email with file attachments via SMTP to one or multiple recipients.

        Args:
            to: Recipient email address or list of addresses
            subject: Email subject line
            body: Email body content
            attachment_paths: List of file paths to attach
            cc: CC recipient(s) - single address or list of addresses (optional)
            bcc: BCC recipient(s) - single address or list of addresses (optional)
            html: If True, send body as HTML; otherwise plain text (default: False)

        Returns:
            True if the email was sent successfully.
        """
        msg = self._build_message_with_attachments(
            to, subject, body, html=html, attachment_paths=attachment_paths, cc=cc, bcc=bcc
        )
        return self._send_message(msg, to, cc=cc, bcc=bcc)

    async def asend_email_with_attachments(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        attachment_paths: List[str],
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        html: bool = False,
    ) -> bool:
        """Async version of send_email_with_attachments."""
        return await asyncio.to_thread(
            self.send_email_with_attachments, to, subject, body, attachment_paths, cc, bcc, html
        )

    @tool
    def send_reply(
        self,
        to: str,
        subject: str,
        body: str,
        message_id: str,
        references: str = "",
        html: bool = False,
    ) -> bool:
        """
        Send a reply to an existing email thread via SMTP.

        Args:
            to: Recipient email address
            subject: Email subject (will auto-prepend 'Re:' if not present)
            body: Reply body content
            message_id: The Message-ID of the email being replied to
            references: The References header chain (for threading)
            html: If True, send as HTML; otherwise plain text (default: False)

        Returns:
            True if the reply was sent successfully.
        """
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        msg = self._build_message_with_attachments(
            to, subject, body, html=html,
            in_reply_to=message_id, references=references,
        )
        return self._send_message(msg, to)

    async def asend_reply(
        self,
        to: str,
        subject: str,
        body: str,
        message_id: str,
        references: str = "",
        html: bool = False,
    ) -> bool:
        """Async version of send_reply."""
        return await asyncio.to_thread(
            self.send_reply, to, subject, body, message_id, references, html
        )

    @tool
    def send_reply_with_attachments(
        self,
        to: str,
        subject: str,
        body: str,
        message_id: str,
        attachment_paths: List[str],
        references: str = "",
        html: bool = False,
    ) -> bool:
        """
        Send a reply with file attachments to an existing email thread.

        Args:
            to: Recipient email address
            subject: Email subject (will auto-prepend 'Re:' if not present)
            body: Reply body content
            message_id: The Message-ID of the email being replied to
            attachment_paths: List of file paths to attach
            references: The References header chain (for threading)
            html: If True, send body as HTML; otherwise plain text (default: False)

        Returns:
            True if the reply was sent successfully.
        """
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        msg = self._build_message_with_attachments(
            to, subject, body, html=html,
            attachment_paths=attachment_paths,
            in_reply_to=message_id, references=references,
        )
        return self._send_message(msg, to)

    async def asend_reply_with_attachments(
        self,
        to: str,
        subject: str,
        body: str,
        message_id: str,
        attachment_paths: List[str],
        references: str = "",
        html: bool = False,
    ) -> bool:
        """Async version of send_reply_with_attachments."""
        return await asyncio.to_thread(
            self.send_reply_with_attachments,
            to, subject, body, message_id, attachment_paths, references, html,
        )

    @tool
    def list_mailboxes(self) -> List[str]:
        """
        List all available mailboxes/folders on the IMAP server.

        Returns:
            List of mailbox names.
        """
        conn = self._get_imap_connection()
        try:
            status, mailboxes = conn.list()
            if status != "OK":
                return []
            result = []
            for mb in mailboxes:
                if isinstance(mb, bytes):
                    match = re.search(rb'"([^"]*)"$|(\S+)$', mb)
                    if match:
                        name = (match.group(1) or match.group(2)).decode("utf-8", errors="replace")
                        result.append(name)
            return result
        finally:
            conn.logout()

    async def alist_mailboxes(self) -> List[str]:
        """Async version of list_mailboxes."""
        return await asyncio.to_thread(self.list_mailboxes)

    @tool
    def get_mailbox_status(self, mailbox: str = "INBOX") -> Dict[str, int]:
        """
        Get the status of a mailbox (total messages, unseen, recent).

        Args:
            mailbox: Mailbox/folder to check (default: INBOX)

        Returns:
            Dictionary with 'total', 'unseen', and 'recent' counts.
        """
        conn = self._get_imap_connection()
        try:
            status, data = conn.status(mailbox, "(MESSAGES UNSEEN RECENT)")
            if status != "OK":
                return {"total": 0, "unseen": 0, "recent": 0}
            # Parse response like: "INBOX" (MESSAGES 5 UNSEEN 2 RECENT 1)
            response = data[0].decode() if isinstance(data[0], bytes) else data[0]
            total = int(re.search(r"MESSAGES\s+(\d+)", response).group(1)) if re.search(r"MESSAGES\s+(\d+)", response) else 0
            unseen = int(re.search(r"UNSEEN\s+(\d+)", response).group(1)) if re.search(r"UNSEEN\s+(\d+)", response) else 0
            recent = int(re.search(r"RECENT\s+(\d+)", response).group(1)) if re.search(r"RECENT\s+(\d+)", response) else 0
            return {"total": total, "unseen": unseen, "recent": recent}
        finally:
            conn.logout()

    async def aget_mailbox_status(self, mailbox: str = "INBOX") -> Dict[str, int]:
        """Async version of get_mailbox_status."""
        return await asyncio.to_thread(self.get_mailbox_status, mailbox)
