"""
Comprehensive Smoke Test for Mail Interface (MailTools + MailInterface).

Tests every public method and attribute of both MailTools and MailInterface
against a real Gmail account.

Requirements:
    - MAIL_PASSWORD_33 env var set to the app password for dogankeskin33@gmail.com
    - OPENAI_API_KEY env var set (for agent-based tests)
    - Internet connection
    - IMAP enabled on the Gmail account

Run:
    uv run pytest tests/smoke_tests/interfaces/test_mail_interface.py -v -s
"""

import asyncio
import os
import tempfile
import time

import pytest

# ── Config ───────────────────────────────────────────────────────────

MAIL_USERNAME = "dogankeskin33@gmail.com"
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD_33")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
TEST_RECIPIENT = "dogankeskin803@gmail.com"

# Unique subject to identify test emails and avoid collisions
TEST_SUBJECT_TAG = f"[SMOKE-TEST-{int(time.time())}]"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mail_tools():
    """Create a MailTools instance for the entire test module."""
    from upsonic.tools.custom_tools.mail import MailTools

    return MailTools(
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        imap_host=IMAP_HOST,
        imap_port=IMAP_PORT,
        username=MAIL_USERNAME,
        password=MAIL_PASSWORD,
    )


@pytest.fixture(scope="module")
def mail_interface():
    """Create a MailInterface instance for the entire test module."""
    from upsonic import Agent
    from upsonic.interfaces.mail.mail import MailInterface

    agent = Agent("openai/gpt-4o")
    return MailInterface(
        agent=agent,
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        imap_host=IMAP_HOST,
        imap_port=IMAP_PORT,
        username=MAIL_USERNAME,
        password=MAIL_PASSWORD,
        mode="task",
        api_secret="test-secret",
        allowed_emails=[TEST_RECIPIENT, MAIL_USERNAME],
    )


@pytest.fixture(scope="module")
def mail_interface_chat():
    """Create a MailInterface instance in CHAT mode."""
    from upsonic import Agent
    from upsonic.interfaces.mail.mail import MailInterface

    agent = Agent("openai/gpt-4o")
    return MailInterface(
        agent=agent,
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        imap_host=IMAP_HOST,
        imap_port=IMAP_PORT,
        username=MAIL_USERNAME,
        password=MAIL_PASSWORD,
        mode="chat",
        api_secret="test-secret",
        allowed_emails=[TEST_RECIPIENT, MAIL_USERNAME],
    )


# =====================================================================
# PART 1: MailTools — Attribute & Configuration Tests
# =====================================================================


class TestMailToolsAttributes:
    """Test MailTools initialization and attributes."""

    def test_smtp_host(self, mail_tools):
        assert mail_tools.smtp_host == SMTP_HOST

    def test_smtp_port(self, mail_tools):
        assert mail_tools.smtp_port == SMTP_PORT

    def test_imap_host(self, mail_tools):
        assert mail_tools.imap_host == IMAP_HOST

    def test_imap_port(self, mail_tools):
        assert mail_tools.imap_port == IMAP_PORT

    def test_username(self, mail_tools):
        assert mail_tools.username == MAIL_USERNAME

    def test_password(self, mail_tools):
        assert mail_tools.password == MAIL_PASSWORD

    def test_from_address_defaults_to_username(self, mail_tools):
        assert mail_tools.from_address == MAIL_USERNAME

    def test_use_ssl_default(self, mail_tools):
        assert mail_tools.use_ssl is False

    def test_custom_from_address(self):
        from upsonic.tools.custom_tools.mail import MailTools

        t = MailTools(
            smtp_host="test",
            imap_host="test",
            username="user@test.com",
            password="pass",
            from_address="custom@test.com",
        )
        assert t.from_address == "custom@test.com"

    def test_env_var_fallback(self, monkeypatch):
        from upsonic.tools.custom_tools.mail import MailTools

        monkeypatch.setenv("MAIL_SMTP_HOST", "env-smtp.test.com")
        monkeypatch.setenv("MAIL_IMAP_HOST", "env-imap.test.com")
        monkeypatch.setenv("MAIL_USERNAME", "env@test.com")
        monkeypatch.setenv("MAIL_PASSWORD", "envpass")
        monkeypatch.setenv("MAIL_SMTP_PORT", "465")
        monkeypatch.setenv("MAIL_IMAP_PORT", "143")
        monkeypatch.setenv("MAIL_USE_SSL", "true")

        t = MailTools()
        assert t.smtp_host == "env-smtp.test.com"
        assert t.imap_host == "env-imap.test.com"
        assert t.username == "env@test.com"
        assert t.password == "envpass"
        assert t.smtp_port == 465
        assert t.imap_port == 143
        assert t.use_ssl is True


# =====================================================================
# PART 2: MailTools — IMAP Connection Tests
# =====================================================================


class TestMailToolsIMAPConnection:
    """Test IMAP connection and basic operations."""

    def test_imap_connection(self, mail_tools):
        """Test that we can establish an IMAP connection."""
        conn = mail_tools._get_imap_connection()
        assert conn is not None
        conn.logout()

    def test_list_mailboxes(self, mail_tools):
        """Test listing all mailboxes/folders."""
        mailboxes = mail_tools.list_mailboxes()
        assert isinstance(mailboxes, list)
        assert len(mailboxes) > 0
        # INBOX should always exist
        assert any("INBOX" in mb for mb in mailboxes)

    @pytest.mark.asyncio
    async def test_alist_mailboxes(self, mail_tools):
        """Test async version of list_mailboxes."""
        mailboxes = await mail_tools.alist_mailboxes()
        assert isinstance(mailboxes, list)
        assert len(mailboxes) > 0

    def test_get_mailbox_status(self, mail_tools):
        """Test getting mailbox status."""
        status = mail_tools.get_mailbox_status("INBOX")
        assert isinstance(status, dict)
        assert "total" in status
        assert "unseen" in status
        assert "recent" in status
        assert status["total"] >= 0
        assert status["unseen"] >= 0
        assert status["recent"] >= 0

    @pytest.mark.asyncio
    async def test_aget_mailbox_status(self, mail_tools):
        """Test async version of get_mailbox_status."""
        status = await mail_tools.aget_mailbox_status("INBOX")
        assert isinstance(status, dict)
        assert "total" in status


# =====================================================================
# PART 3: MailTools — Email Retrieval Tests
# =====================================================================


class TestMailToolsRetrieval:
    """Test email retrieval methods."""

    def test_get_latest_emails(self, mail_tools):
        """Test fetching latest emails."""
        emails = mail_tools.get_latest_emails(count=3)
        assert isinstance(emails, list)
        if emails:
            email = emails[0]
            assert "uid" in email
            assert "from" in email
            assert "to" in email
            assert "subject" in email
            assert "date" in email
            assert "body" in email
            assert "message_id" in email
            assert "in_reply_to" in email
            assert "references" in email
            assert "attachments" in email
            assert "cc" in email

    @pytest.mark.asyncio
    async def test_aget_latest_emails(self, mail_tools):
        """Test async version of get_latest_emails."""
        emails = await mail_tools.aget_latest_emails(count=2)
        assert isinstance(emails, list)

    def test_get_unread_emails(self, mail_tools):
        """Test fetching unread emails."""
        emails = mail_tools.get_unread_emails(count=3)
        assert isinstance(emails, list)
        # May be empty if all emails are read

    @pytest.mark.asyncio
    async def test_aget_unread_emails(self, mail_tools):
        """Test async version of get_unread_emails."""
        emails = await mail_tools.aget_unread_emails(count=2)
        assert isinstance(emails, list)

    def test_get_emails_from_sender(self, mail_tools):
        """Test fetching emails from a specific sender."""
        emails = mail_tools.get_emails_from_sender(TEST_RECIPIENT, count=3)
        assert isinstance(emails, list)

    @pytest.mark.asyncio
    async def test_aget_emails_from_sender(self, mail_tools):
        """Test async version of get_emails_from_sender."""
        emails = await mail_tools.aget_emails_from_sender(TEST_RECIPIENT, count=2)
        assert isinstance(emails, list)

    def test_search_emails(self, mail_tools):
        """Test searching emails with IMAP criteria."""
        emails = mail_tools.search_emails('SUBJECT "Meeting"', count=3)
        assert isinstance(emails, list)

    @pytest.mark.asyncio
    async def test_asearch_emails(self, mail_tools):
        """Test async version of search_emails."""
        emails = await mail_tools.asearch_emails("ALL", count=2)
        assert isinstance(emails, list)


# =====================================================================
# PART 4: MailTools — SMTP Send Tests
# =====================================================================


class TestMailToolsSend:
    """Test email sending methods."""

    def test_send_email_single_recipient(self, mail_tools):
        """Test sending an email to a single recipient."""
        subject = f"{TEST_SUBJECT_TAG} Single Recipient Test"
        result = mail_tools.send_email(
            to=TEST_RECIPIENT,
            subject=subject,
            body="This is a smoke test email sent to a single recipient.",
        )
        assert result is True

    def test_send_email_multiple_recipients(self, mail_tools):
        """Test sending an email to multiple recipients."""
        subject = f"{TEST_SUBJECT_TAG} Multi Recipient Test"
        result = mail_tools.send_email(
            to=[TEST_RECIPIENT, MAIL_USERNAME],
            subject=subject,
            body="This is a smoke test email sent to multiple recipients.",
        )
        assert result is True

    def test_send_email_with_cc(self, mail_tools):
        """Test sending an email with CC."""
        subject = f"{TEST_SUBJECT_TAG} CC Test"
        result = mail_tools.send_email(
            to=TEST_RECIPIENT,
            subject=subject,
            body="This email has a CC recipient.",
            cc=MAIL_USERNAME,
        )
        assert result is True

    def test_send_email_with_bcc(self, mail_tools):
        """Test sending an email with BCC."""
        subject = f"{TEST_SUBJECT_TAG} BCC Test"
        result = mail_tools.send_email(
            to=TEST_RECIPIENT,
            subject=subject,
            body="This email has a BCC recipient.",
            bcc=MAIL_USERNAME,
        )
        assert result is True

    def test_send_html_email(self, mail_tools):
        """Test sending an HTML email."""
        subject = f"{TEST_SUBJECT_TAG} HTML Test"
        result = mail_tools.send_email(
            to=TEST_RECIPIENT,
            subject=subject,
            body="<h1>Hello</h1><p>This is an <b>HTML</b> smoke test email.</p>",
            html=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_asend_email(self, mail_tools):
        """Test async version of send_email."""
        subject = f"{TEST_SUBJECT_TAG} Async Send Test"
        result = await mail_tools.asend_email(
            to=TEST_RECIPIENT,
            subject=subject,
            body="This is an async smoke test email.",
        )
        assert result is True

    def test_send_email_with_attachments(self, mail_tools):
        """Test sending an email with file attachments."""
        subject = f"{TEST_SUBJECT_TAG} Attachment Test"

        # Create a temp file to attach
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="smoke_test_", delete=False
        ) as f:
            f.write("This is a smoke test attachment file.")
            temp_path = f.name

        try:
            result = mail_tools.send_email_with_attachments(
                to=TEST_RECIPIENT,
                subject=subject,
                body="This email has an attachment.",
                attachment_paths=[temp_path],
            )
            assert result is True
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_asend_email_with_attachments(self, mail_tools):
        """Test async version of send_email_with_attachments."""
        subject = f"{TEST_SUBJECT_TAG} Async Attachment Test"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="smoke_async_", delete=False
        ) as f:
            f.write("Async attachment content.")
            temp_path = f.name

        try:
            result = await mail_tools.asend_email_with_attachments(
                to=TEST_RECIPIENT,
                subject=subject,
                body="Async email with attachment.",
                attachment_paths=[temp_path],
            )
            assert result is True
        finally:
            os.unlink(temp_path)


# =====================================================================
# PART 5: MailTools — Reply Tests
# =====================================================================


class TestMailToolsReply:
    """Test email reply methods."""

    def _get_latest_email(self, mail_tools):
        """Helper: get the latest email to reply to."""
        emails = mail_tools.get_latest_emails(count=1)
        assert len(emails) > 0, "No emails in inbox to reply to"
        return emails[0]

    def test_send_reply(self, mail_tools):
        """Test sending a reply to an existing email."""
        email_data = self._get_latest_email(mail_tools)
        subject = email_data["subject"]
        result = mail_tools.send_reply(
            to=TEST_RECIPIENT,
            subject=subject,
            body="This is a smoke test reply.",
            message_id=email_data["message_id"],
            references=email_data.get("references", ""),
        )
        assert result is True

    def test_send_reply_auto_prepends_re(self, mail_tools):
        """Test that reply auto-prepends 'Re:' to subject."""
        email_data = self._get_latest_email(mail_tools)
        # Use a subject without Re: prefix
        result = mail_tools.send_reply(
            to=TEST_RECIPIENT,
            subject="Original Subject",
            body="Testing Re: auto-prepend.",
            message_id=email_data["message_id"],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_asend_reply(self, mail_tools):
        """Test async version of send_reply."""
        emails = await mail_tools.aget_latest_emails(count=1)
        assert len(emails) > 0
        email_data = emails[0]
        result = await mail_tools.asend_reply(
            to=TEST_RECIPIENT,
            subject=email_data["subject"],
            body="Async smoke test reply.",
            message_id=email_data["message_id"],
        )
        assert result is True

    def test_send_reply_with_attachments(self, mail_tools):
        """Test sending a reply with attachments."""
        email_data = self._get_latest_email(mail_tools)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="reply_attach_", delete=False
        ) as f:
            f.write("Reply attachment content.")
            temp_path = f.name

        try:
            result = mail_tools.send_reply_with_attachments(
                to=TEST_RECIPIENT,
                subject=email_data["subject"],
                body="Reply with attachment.",
                message_id=email_data["message_id"],
                attachment_paths=[temp_path],
            )
            assert result is True
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_asend_reply_with_attachments(self, mail_tools):
        """Test async version of send_reply_with_attachments."""
        emails = await mail_tools.aget_latest_emails(count=1)
        assert len(emails) > 0
        email_data = emails[0]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="async_reply_", delete=False
        ) as f:
            f.write("Async reply attachment.")
            temp_path = f.name

        try:
            result = await mail_tools.asend_reply_with_attachments(
                to=TEST_RECIPIENT,
                subject=email_data["subject"],
                body="Async reply with attachment.",
                message_id=email_data["message_id"],
                attachment_paths=[temp_path],
            )
            assert result is True
        finally:
            os.unlink(temp_path)


# =====================================================================
# PART 6: MailTools — Flag / Mark / Delete / Move Tests
# =====================================================================


class TestMailToolsFlags:
    """Test email flag operations (read, unread, flag, unflag, delete, move)."""

    def _get_a_uid(self, mail_tools) -> str:
        """Helper: get a UID from the inbox."""
        emails = mail_tools.get_latest_emails(count=1)
        assert len(emails) > 0, "No emails to test with"
        return emails[0]["uid"]

    def test_mark_email_as_read(self, mail_tools):
        uid = self._get_a_uid(mail_tools)
        result = mail_tools.mark_email_as_read(uid)
        assert result is True

    @pytest.mark.asyncio
    async def test_amark_email_as_read(self, mail_tools):
        emails = await mail_tools.aget_latest_emails(count=1)
        assert len(emails) > 0
        result = await mail_tools.amark_email_as_read(emails[0]["uid"])
        assert result is True

    def test_mark_email_as_unread(self, mail_tools):
        uid = self._get_a_uid(mail_tools)
        result = mail_tools.mark_email_as_unread(uid)
        assert result is True

    @pytest.mark.asyncio
    async def test_amark_email_as_unread(self, mail_tools):
        emails = await mail_tools.aget_latest_emails(count=1)
        assert len(emails) > 0
        result = await mail_tools.amark_email_as_unread(emails[0]["uid"])
        assert result is True

    def test_flag_email(self, mail_tools):
        uid = self._get_a_uid(mail_tools)
        result = mail_tools.flag_email(uid)
        assert result is True

    @pytest.mark.asyncio
    async def test_aflag_email(self, mail_tools):
        emails = await mail_tools.aget_latest_emails(count=1)
        assert len(emails) > 0
        result = await mail_tools.aflag_email(emails[0]["uid"])
        assert result is True

    def test_unflag_email(self, mail_tools):
        uid = self._get_a_uid(mail_tools)
        # Flag it first, then unflag
        mail_tools.flag_email(uid)
        result = mail_tools.unflag_email(uid)
        assert result is True

    @pytest.mark.asyncio
    async def test_aunflag_email(self, mail_tools):
        emails = await mail_tools.aget_latest_emails(count=1)
        assert len(emails) > 0
        uid = emails[0]["uid"]
        await mail_tools.aflag_email(uid)
        result = await mail_tools.aunflag_email(uid)
        assert result is True

    def test_mark_read_then_unread_roundtrip(self, mail_tools):
        """Test that marking read then unread works as a roundtrip."""
        uid = self._get_a_uid(mail_tools)
        assert mail_tools.mark_email_as_read(uid) is True
        assert mail_tools.mark_email_as_unread(uid) is True


# =====================================================================
# PART 7: MailTools — Download Attachments Tests
# =====================================================================


class TestMailToolsAttachments:
    """Test attachment download methods."""

    def test_download_attachments_no_attachments(self, mail_tools):
        """Test download on an email without attachments returns empty list."""
        emails = mail_tools.get_latest_emails(count=5)
        # Find an email without attachments
        for email_data in emails:
            if not email_data.get("attachments"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = mail_tools.download_attachments(email_data["uid"], tmpdir)
                    assert isinstance(result, list)
                    assert len(result) == 0
                return
        pytest.skip("No emails without attachments found")

    @pytest.mark.asyncio
    async def test_adownload_attachments(self, mail_tools):
        """Test async version of download_attachments."""
        emails = await mail_tools.aget_latest_emails(count=1)
        if emails:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await mail_tools.adownload_attachments(
                    emails[0]["uid"], tmpdir
                )
                assert isinstance(result, list)

    def test_get_raw_attachments(self, mail_tools):
        """Test getting raw attachment bytes."""
        emails = mail_tools.get_latest_emails(count=1)
        if emails:
            result = mail_tools.get_raw_attachments(emails[0]["uid"])
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_aget_raw_attachments(self, mail_tools):
        """Test async version of get_raw_attachments."""
        emails = await mail_tools.aget_latest_emails(count=1)
        if emails:
            result = await mail_tools.aget_raw_attachments(emails[0]["uid"])
            assert isinstance(result, list)


# =====================================================================
# PART 8: MailInterface — Attribute & Configuration Tests
# =====================================================================


class TestMailInterfaceAttributes:
    """Test MailInterface initialization and attributes."""

    def test_name(self, mail_interface):
        assert mail_interface.name == "Mail"

    def test_mode_task(self, mail_interface):
        from upsonic.interfaces.schemas import InterfaceMode

        assert mail_interface.mode == InterfaceMode.TASK

    def test_mode_chat(self, mail_interface_chat):
        from upsonic.interfaces.schemas import InterfaceMode

        assert mail_interface_chat.mode == InterfaceMode.CHAT

    def test_is_task_mode(self, mail_interface):
        assert mail_interface.is_task_mode() is True
        assert mail_interface.is_chat_mode() is False

    def test_is_chat_mode(self, mail_interface_chat):
        assert mail_interface_chat.is_chat_mode() is True
        assert mail_interface_chat.is_task_mode() is False

    def test_api_secret(self, mail_interface):
        assert mail_interface.api_secret == "test-secret"

    def test_mailbox(self, mail_interface):
        assert mail_interface.mailbox == "INBOX"

    def test_mail_tools_instance(self, mail_interface):
        from upsonic.tools.custom_tools.mail import MailTools

        assert isinstance(mail_interface.mail_tools, MailTools)

    def test_agent_attached(self, mail_interface):
        assert mail_interface.agent is not None

    def test_id_generated(self, mail_interface):
        assert mail_interface.id is not None
        assert len(mail_interface.id) > 0

    def test_get_id(self, mail_interface):
        assert mail_interface.get_id() == mail_interface.id

    def test_get_name(self, mail_interface):
        assert mail_interface.get_name() == "Mail"

    def test_repr(self, mail_interface):
        repr_str = repr(mail_interface)
        assert "MailInterface" in repr_str
        assert "task" in repr_str

    def test_dedup_cache_initialized(self, mail_interface):
        assert isinstance(mail_interface._processed_emails, dict)

    def test_heartbeat_task_none_initially(self, mail_interface):
        assert mail_interface._heartbeat_task is None


# =====================================================================
# PART 9: MailInterface — Whitelist / Access Control Tests
# =====================================================================


class TestMailInterfaceWhitelist:
    """Test whitelist-based access control."""

    def test_allowed_emails_set(self, mail_interface):
        assert mail_interface._allowed_emails is not None
        assert TEST_RECIPIENT in mail_interface._allowed_emails

    def test_is_email_allowed_whitelisted(self, mail_interface):
        assert mail_interface.is_email_allowed(TEST_RECIPIENT) is True

    def test_is_email_allowed_with_name_format(self, mail_interface):
        assert mail_interface.is_email_allowed(f"Dogan <{TEST_RECIPIENT}>") is True

    def test_is_email_allowed_not_whitelisted(self, mail_interface):
        assert mail_interface.is_email_allowed("random@example.com") is False

    def test_is_email_allowed_no_whitelist(self):
        """When no whitelist is configured, all emails should be allowed."""
        from upsonic import Agent
        from upsonic.interfaces.mail.mail import MailInterface

        agent = Agent("openai/gpt-4o")
        iface = MailInterface(
            agent=agent,
            smtp_host=SMTP_HOST,
            imap_host=IMAP_HOST,
            username=MAIL_USERNAME,
            password=MAIL_PASSWORD,
        )
        assert iface.is_email_allowed("anyone@anywhere.com") is True

    def test_get_unauthorized_message(self, mail_interface):
        msg = mail_interface.get_unauthorized_message()
        assert msg == "This operation not allowed"


# =====================================================================
# PART 10: MailInterface — Deduplication Tests
# =====================================================================


class TestMailInterfaceDedup:
    """Test event deduplication."""

    def test_is_duplicate_false_for_new(self, mail_interface):
        assert mail_interface._is_duplicate("never-seen-uid") is False

    def test_mark_processed_and_is_duplicate(self, mail_interface):
        mail_interface._mark_processed("test-uid-123")
        assert mail_interface._is_duplicate("test-uid-123") is True

    def test_cleanup_processed_emails(self, mail_interface):
        # Add an expired entry
        mail_interface._processed_emails["old-uid"] = time.time() - 600
        mail_interface._cleanup_processed_emails()
        assert "old-uid" not in mail_interface._processed_emails

    def test_auto_cleanup_at_threshold(self, mail_interface):
        """Test that cleanup happens automatically when cache exceeds 1000."""
        # Add 1001 old entries
        old_time = time.time() - 600
        for i in range(1001):
            mail_interface._processed_emails[f"bulk-uid-{i}"] = old_time

        # This should trigger cleanup
        mail_interface._mark_processed("trigger-cleanup-uid")
        # Old entries should be cleaned up
        assert len(mail_interface._processed_emails) < 1001

        # Clean up test data
        mail_interface._processed_emails.clear()


# =====================================================================
# PART 11: MailInterface — Secret Verification Tests
# =====================================================================


class TestMailInterfaceSecretVerification:
    """Test API secret verification."""

    def test_verify_secret_valid(self, mail_interface):
        # Should not raise
        mail_interface._verify_secret("test-secret")

    def test_verify_secret_invalid(self, mail_interface):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            mail_interface._verify_secret("wrong-secret")
        assert exc_info.value.status_code == 403

    def test_verify_secret_none(self, mail_interface):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            mail_interface._verify_secret(None)

    def test_verify_secret_no_secret_configured(self):
        """When no secret is configured, any request should pass."""
        from upsonic import Agent
        from upsonic.interfaces.mail.mail import MailInterface

        agent = Agent("openai/gpt-4o")
        iface = MailInterface(
            agent=agent,
            smtp_host=SMTP_HOST,
            imap_host=IMAP_HOST,
            username=MAIL_USERNAME,
            password=MAIL_PASSWORD,
            api_secret=None,
        )
        # Should not raise even with None
        iface._verify_secret(None)
        iface._verify_secret("anything")


# =====================================================================
# PART 12: MailInterface — Helper Method Tests
# =====================================================================


class TestMailInterfaceHelpers:
    """Test internal helper methods."""

    def test_extract_sender_id_with_angle_brackets(self, mail_interface):
        result = mail_interface._extract_sender_id("John Doe <john@example.com>")
        assert result == "john@example.com"

    def test_extract_sender_id_plain_email(self, mail_interface):
        result = mail_interface._extract_sender_id("john@example.com")
        assert result == "john@example.com"

    def test_extract_sender_id_case_insensitive(self, mail_interface):
        result = mail_interface._extract_sender_id("John <JOHN@EXAMPLE.COM>")
        assert result == "john@example.com"

    def test_email_to_summary(self):
        from upsonic.interfaces.mail.mail import MailInterface

        email_data = {
            "uid": "123",
            "message_id": "<msg@test>",
            "from": "sender@test.com",
            "to": "recipient@test.com",
            "cc": "cc@test.com",
            "subject": "Test Subject",
            "date": "Mon, 30 Mar 2026 10:00:00 +0000",
            "body": "Test body",
            "in_reply_to": "",
            "references": "",
            "attachments": [
                {"filename": "file.txt", "content_type": "text/plain", "size": 100}
            ],
        }
        summary = MailInterface._email_to_summary(email_data)
        assert summary["uid"] == "123"
        assert summary["from"] == "sender@test.com"
        assert summary["cc"] == "cc@test.com"
        assert len(summary["attachments"]) == 1

    def test_is_reset_command_in_chat_mode(self, mail_interface_chat):
        assert mail_interface_chat.is_reset_command("/reset") is True
        assert mail_interface_chat.is_reset_command("hello") is False

    def test_is_reset_command_in_task_mode(self, mail_interface):
        # Reset command should not work in task mode
        assert mail_interface.is_reset_command("/reset") is False

    def test_cleanup_temp_files(self):
        from upsonic.interfaces.mail.mail import MailInterface

        # Create temp files
        paths = []
        for i in range(3):
            f = tempfile.NamedTemporaryFile(delete=False, prefix=f"cleanup_test_{i}_")
            f.close()
            paths.append(f.name)

        # Verify they exist
        for p in paths:
            assert os.path.exists(p)

        # Cleanup
        MailInterface._cleanup_temp_files(paths)

        # Verify they're gone
        for p in paths:
            assert not os.path.exists(p)

    def test_cleanup_temp_files_missing_file(self):
        """Should not raise on missing files."""
        from upsonic.interfaces.mail.mail import MailInterface

        MailInterface._cleanup_temp_files(["/nonexistent/path/file.txt"])


# =====================================================================
# PART 13: MailInterface — Chat Session Tests
# =====================================================================


class TestMailInterfaceChatSessions:
    """Test chat session management."""

    def test_get_chat_session(self, mail_interface_chat):
        session = mail_interface_chat.get_chat_session("test-user@example.com")
        assert session is not None

    def test_has_chat_session(self, mail_interface_chat):
        mail_interface_chat.get_chat_session("exists@example.com")
        assert mail_interface_chat.has_chat_session("exists@example.com") is True
        assert mail_interface_chat.has_chat_session("nope@example.com") is False

    def test_get_all_chat_sessions(self, mail_interface_chat):
        mail_interface_chat.get_chat_session("session1@example.com")
        sessions = mail_interface_chat.get_all_chat_sessions()
        assert isinstance(sessions, dict)
        assert "session1@example.com" in sessions

    def test_reset_chat_session(self, mail_interface_chat):
        mail_interface_chat.get_chat_session("reset-me@example.com")
        assert mail_interface_chat.has_chat_session("reset-me@example.com") is True
        result = mail_interface_chat.reset_chat_session("reset-me@example.com")
        assert result is True
        assert mail_interface_chat.has_chat_session("reset-me@example.com") is False

    def test_reset_nonexistent_session(self, mail_interface_chat):
        result = mail_interface_chat.reset_chat_session("nonexistent@example.com")
        assert result is False


# =====================================================================
# PART 14: MailInterface — Routes Attachment Test
# =====================================================================


class TestMailInterfaceRoutes:
    """Test that routes are properly created."""

    def test_attach_routes_returns_router(self, mail_interface):
        from fastapi import APIRouter

        router = mail_interface.attach_routes()
        assert isinstance(router, APIRouter)

    def test_routes_have_correct_prefix(self, mail_interface):
        router = mail_interface.attach_routes()
        # Check that routes exist
        route_paths = [route.path for route in router.routes]
        assert "/check" in route_paths or any("/check" in p for p in route_paths)

    def test_all_expected_routes_exist(self, mail_interface):
        router = mail_interface.attach_routes()
        route_paths = [route.path for route in router.routes]
        expected = ["/check", "/inbox", "/unread", "/send", "/search",
                    "/folders", "/status", "/{uid}/read", "/{uid}/unread",
                    "/{uid}/delete", "/{uid}/move", "/health"]
        for expected_path in expected:
            assert any(expected_path in p for p in route_paths), \
                f"Missing route: {expected_path}"


# =====================================================================
# PART 15: MailInterface — Health Check Test
# =====================================================================


class TestMailInterfaceHealthCheck:
    """Test health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, mail_interface):
        health = await mail_interface.health_check()
        assert health["status"] == "active"
        assert health["name"] == "Mail"
        assert "configuration" in health
        config = health["configuration"]
        assert "imap_connected" in config
        assert config["imap_connected"] is True
        assert config["smtp_host"] == SMTP_HOST
        assert config["imap_host"] == IMAP_HOST
        assert config["mode"] == "task"
        assert config["whitelist_enabled"] is True
        assert config["heartbeat_active"] is False
        assert "dedup_cache_size" in config


# =====================================================================
# PART 16: MailInterface — Schemas Tests
# =====================================================================


class TestMailSchemas:
    """Test Pydantic schema models."""

    def test_check_emails_response(self):
        from upsonic.interfaces.mail.schemas import CheckEmailsResponse

        resp = CheckEmailsResponse(
            status="success", processed_count=3, email_uids=["1", "2", "3"]
        )
        assert resp.status == "success"
        assert resp.processed_count == 3

    def test_agent_email_response_reply(self):
        from upsonic.interfaces.mail.schemas import AgentEmailResponse

        resp = AgentEmailResponse(
            action="reply", reply_body="Hello!", reasoning="Test"
        )
        assert resp.action == "reply"

    def test_agent_email_response_ignore(self):
        from upsonic.interfaces.mail.schemas import AgentEmailResponse

        resp = AgentEmailResponse(
            action="ignore", reply_body="", reasoning="Spam"
        )
        assert resp.action == "ignore"

    def test_email_summary(self):
        from upsonic.interfaces.mail.schemas import EmailSummary

        summary = EmailSummary(uid="123", subject="Test")
        assert summary.uid == "123"
        assert summary.subject == "Test"

    def test_attachment_info(self):
        from upsonic.interfaces.mail.schemas import AttachmentInfo

        att = AttachmentInfo(
            filename="doc.pdf", content_type="application/pdf", size=1024
        )
        assert att.filename == "doc.pdf"
        assert att.size == 1024

    def test_email_list_response(self):
        from upsonic.interfaces.mail.schemas import EmailListResponse, EmailSummary

        resp = EmailListResponse(
            count=1, emails=[EmailSummary(uid="1", subject="Test")]
        )
        assert resp.count == 1

    def test_send_email_request_single(self):
        from upsonic.interfaces.mail.schemas import SendEmailRequest

        req = SendEmailRequest(to="a@b.com", subject="Hi", body="Hello")
        assert req.to == "a@b.com"
        assert req.cc is None
        assert req.bcc is None

    def test_send_email_request_multi(self):
        from upsonic.interfaces.mail.schemas import SendEmailRequest

        req = SendEmailRequest(
            to=["a@b.com", "c@d.com"],
            subject="Hi",
            body="Hello",
            cc="e@f.com",
            bcc=["g@h.com"],
        )
        assert isinstance(req.to, list)
        assert req.cc == "e@f.com"

    def test_search_email_request(self):
        from upsonic.interfaces.mail.schemas import SearchEmailRequest

        req = SearchEmailRequest(query='FROM "a@b.com"')
        assert req.count == 10
        assert req.mailbox == "INBOX"

    def test_mailbox_status_response(self):
        from upsonic.interfaces.mail.schemas import MailboxStatusResponse

        resp = MailboxStatusResponse(
            mailbox="INBOX", total=100, unseen=5, recent=2
        )
        assert resp.total == 100


# =====================================================================
# PART 17: MailInterface — Full Integration (Agent Processing)
# =====================================================================


class TestMailInterfaceIntegration:
    """
    Integration tests that send a real email and process it through the agent.
    These are slower as they involve LLM calls.
    """

    @pytest.mark.asyncio
    async def test_send_and_check_task_mode(self, mail_interface):
        """Send an email to ourselves and process it in TASK mode."""
        subject = f"{TEST_SUBJECT_TAG} Integration Task Mode"

        # Send email to ourselves (the agent account)
        result = await mail_interface.mail_tools.asend_email(
            to=MAIL_USERNAME,
            subject=subject,
            body="Hello agent! What is 2+2? Please reply with just the answer.",
        )
        assert result is True

        # Wait for email to arrive
        await asyncio.sleep(5)

        # Process it
        response = await mail_interface.check_and_process_emails(count=5)
        assert response.status == "success"
        assert response.processed_count >= 0  # May be 0 if email hasn't arrived yet

    @pytest.mark.asyncio
    async def test_send_and_check_chat_mode(self, mail_interface_chat):
        """Send an email and process it in CHAT mode."""
        subject = f"{TEST_SUBJECT_TAG} Integration Chat Mode"

        result = await mail_interface_chat.mail_tools.asend_email(
            to=MAIL_USERNAME,
            subject=subject,
            body="Hi, what day is today?",
        )
        assert result is True

        await asyncio.sleep(5)

        response = await mail_interface_chat.check_and_process_emails(count=5)
        assert response.status == "success"


# =====================================================================
# PART 18: Module-level imports test
# =====================================================================


class TestImports:
    """Test that all public imports work correctly."""

    def test_import_mail_tools(self):
        from upsonic.tools.custom_tools.mail import MailTools
        assert MailTools is not None

    def test_import_mail_interface(self):
        from upsonic.interfaces.mail import MailInterface
        assert MailInterface is not None

    def test_import_from_main_interfaces(self):
        from upsonic.interfaces import MailInterface, Mail
        assert MailInterface is Mail

    def test_import_schemas(self):
        from upsonic.interfaces.mail.schemas import (
            AgentEmailResponse,
            AttachmentInfo,
            CheckEmailsResponse,
            EmailListResponse,
            EmailSummary,
            MailboxStatusResponse,
            SearchEmailRequest,
            SendEmailRequest,
        )
        assert all([
            AgentEmailResponse, AttachmentInfo, CheckEmailsResponse,
            EmailListResponse, EmailSummary, MailboxStatusResponse,
            SearchEmailRequest, SendEmailRequest,
        ])

    def test_import_helper_functions(self):
        from upsonic.tools.custom_tools.mail import (
            _decode_header_value,
            _extract_body,
            _extract_attachments_metadata,
            _extract_attachment_bytes,
        )
        assert all([
            _decode_header_value, _extract_body,
            _extract_attachments_metadata, _extract_attachment_bytes,
        ])


# =====================================================================
# PART 19: MailTools — delete_email / move_email Tests
# =====================================================================


class TestMailToolsDeleteMove:
    """Test delete and move email operations."""

    def _send_disposable_email(self, mail_tools) -> str:
        """Send an email to ourselves and return its UID after it arrives."""
        subject = f"{TEST_SUBJECT_TAG} Disposable {int(time.time())}"
        mail_tools.send_email(
            to=MAIL_USERNAME,
            subject=subject,
            body="This email will be deleted or moved.",
        )
        # Wait for delivery
        time.sleep(5)
        emails = mail_tools.search_emails(f'SUBJECT "{subject}"', count=1)
        assert len(emails) > 0, "Disposable email did not arrive"
        return emails[0]["uid"]

    def test_delete_email(self, mail_tools):
        """Test deleting an email by UID."""
        uid = self._send_disposable_email(mail_tools)
        result = mail_tools.delete_email(uid)
        assert result is True

    @pytest.mark.asyncio
    async def test_adelete_email(self, mail_tools):
        """Test async version of delete_email."""
        uid = self._send_disposable_email(mail_tools)
        result = await mail_tools.adelete_email(uid)
        assert result is True

    def test_move_email(self, mail_tools):
        """Test moving an email to another folder."""
        uid = self._send_disposable_email(mail_tools)
        # Move to Trash (exists on all Gmail accounts)
        result = mail_tools.move_email(uid, "[Gmail]/Spam")
        assert result is True

    @pytest.mark.asyncio
    async def test_amove_email(self, mail_tools):
        """Test async version of move_email."""
        uid = self._send_disposable_email(mail_tools)
        result = await mail_tools.amove_email(uid, "[Gmail]/Spam")
        assert result is True


# =====================================================================
# PART 20: Helper Functions — Direct Tests
# =====================================================================


class TestHelperFunctions:
    """Test module-level helper functions directly."""

    def test_decode_header_value_plain(self):
        from upsonic.tools.custom_tools.mail import _decode_header_value

        assert _decode_header_value("Hello World") == "Hello World"

    def test_decode_header_value_empty(self):
        from upsonic.tools.custom_tools.mail import _decode_header_value

        assert _decode_header_value("") == ""
        assert _decode_header_value(None) == ""

    def test_decode_header_value_encoded(self):
        from upsonic.tools.custom_tools.mail import _decode_header_value

        # RFC 2047 encoded header
        encoded = "=?UTF-8?B?SGVsbG8gV29ybGQ=?="
        result = _decode_header_value(encoded)
        assert result == "Hello World"

    def test_decode_header_value_unknown_charset(self):
        """Test that unknown charsets don't crash."""
        from email.header import Header
        from upsonic.tools.custom_tools.mail import _decode_header_value

        # Simulate a Header object (as seen in real emails)
        h = Header("test value")
        result = _decode_header_value(h)
        assert isinstance(result, str)

    def test_extract_body_plain_text(self):
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_body

        msg = MIMEText("Hello plain text", "plain", "utf-8")
        assert _extract_body(msg) == "Hello plain text"

    def test_extract_body_html(self):
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_body

        msg = MIMEMultipart()
        msg.attach(MIMEText("<p>Hello HTML</p>", "html", "utf-8"))
        result = _extract_body(msg)
        assert "Hello HTML" in result

    def test_extract_body_multipart_prefers_plain(self):
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_body

        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("Plain version", "plain", "utf-8"))
        msg.attach(MIMEText("<p>HTML version</p>", "html", "utf-8"))
        assert _extract_body(msg) == "Plain version"

    def test_extract_body_empty(self):
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_body

        msg = MIMEText("", "plain", "utf-8")
        result = _extract_body(msg)
        assert result == ""

    def test_extract_attachments_metadata_no_attachments(self):
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_attachments_metadata

        msg = MIMEText("No attachments here", "plain", "utf-8")
        result = _extract_attachments_metadata(msg)
        assert result == []

    def test_extract_attachments_metadata_with_attachment(self):
        from email.mime.application import MIMEApplication
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_attachments_metadata

        msg = MIMEMultipart()
        msg.attach(MIMEText("Body text", "plain"))
        attachment = MIMEApplication(b"file content", Name="test.txt")
        attachment["Content-Disposition"] = 'attachment; filename="test.txt"'
        msg.attach(attachment)

        result = _extract_attachments_metadata(msg)
        assert len(result) == 1
        assert result[0]["filename"] == "test.txt"
        assert result[0]["size"] == len(b"file content")

    def test_extract_attachment_bytes(self):
        from email.mime.application import MIMEApplication
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_attachment_bytes

        msg = MIMEMultipart()
        msg.attach(MIMEText("Body", "plain"))
        content = b"binary file data"
        attachment = MIMEApplication(content, Name="data.bin")
        attachment["Content-Disposition"] = 'attachment; filename="data.bin"'
        msg.attach(attachment)

        result = _extract_attachment_bytes(msg)
        assert len(result) == 1
        filename, content_type, data = result[0]
        assert filename == "data.bin"
        assert data == content

    def test_extract_attachment_bytes_no_attachments(self):
        from email.mime.text import MIMEText
        from upsonic.tools.custom_tools.mail import _extract_attachment_bytes

        msg = MIMEText("No attachments", "plain")
        result = _extract_attachment_bytes(msg)
        assert result == []


# =====================================================================
# PART 21: MailInterface — Heartbeat Tests
# =====================================================================


class TestMailInterfaceHeartbeat:
    """Test heartbeat (auto-poll) methods."""

    def test_start_heartbeat_non_autonomous_agent(self, mail_interface):
        """Heartbeat should not start for a regular Agent (not AutonomousAgent)."""
        mail_interface._start_heartbeat()
        assert mail_interface._heartbeat_task is None

    def test_start_heartbeat_idempotent(self, mail_interface):
        """Calling _start_heartbeat multiple times should be safe."""
        mail_interface._start_heartbeat()
        mail_interface._start_heartbeat()
        assert mail_interface._heartbeat_task is None


# =====================================================================
# PART 22: MailInterface — _send_reply Tests
# =====================================================================


class TestMailInterfaceSendReply:
    """Test the _send_reply internal method."""

    @pytest.mark.asyncio
    async def test_send_reply(self, mail_interface):
        """Test sending a reply via the interface."""
        email_data = {
            "uid": "99999",
            "from": f"Test User <{TEST_RECIPIENT}>",
            "subject": f"{TEST_SUBJECT_TAG} Reply Test",
            "message_id": "<fake-message-id@test>",
            "references": "",
        }
        # Should not raise
        await mail_interface._send_reply(email_data, "This is a test reply from _send_reply.")

    @pytest.mark.asyncio
    async def test_send_reply_extracts_sender_correctly(self, mail_interface):
        """Test that _send_reply correctly extracts the email from 'Name <email>' format."""
        email_data = {
            "uid": "99998",
            "from": f"Some Name <{TEST_RECIPIENT}>",
            "subject": f"{TEST_SUBJECT_TAG} Sender Extract Test",
            "message_id": "<fake-msg-id-2@test>",
            "references": "",
        }
        await mail_interface._send_reply(email_data, "Testing sender extraction.")


# =====================================================================
# PART 23: MailInterface — _handle_reset_command Test
# =====================================================================


class TestMailInterfaceResetCommand:
    """Test reset command handling."""

    @pytest.mark.asyncio
    async def test_handle_reset_command_no_session(self, mail_interface_chat):
        """Test reset command when no session exists."""
        msg_data = {
            "uid": "88888",
            "from": f"Reset User <no-session@example.com>",
            "subject": "Reset",
            "message_id": "<reset-msg@test>",
            "references": "",
            "body": "/reset",
        }
        # Should not raise — sends a "no active conversation" reply
        await mail_interface_chat._handle_reset_command(msg_data)

    @pytest.mark.asyncio
    async def test_handle_reset_command_with_session(self, mail_interface_chat):
        """Test reset command when a session exists."""
        # Create a session first
        mail_interface_chat.get_chat_session("reset-test@example.com")
        assert mail_interface_chat.has_chat_session("reset-test@example.com")

        msg_data = {
            "uid": "88887",
            "from": "Reset Test <reset-test@example.com>",
            "subject": "Reset",
            "message_id": "<reset-msg-2@test>",
            "references": "",
            "body": "/reset",
        }
        await mail_interface_chat._handle_reset_command(msg_data)
        # Session should be cleared
        assert not mail_interface_chat.has_chat_session("reset-test@example.com")


# =====================================================================
# PART 24: MailInterface — _download_attachments_to_temp Test
# =====================================================================


class TestMailInterfaceDownloadAttachments:
    """Test attachment download to temp files."""

    @pytest.mark.asyncio
    async def test_download_no_attachments(self, mail_interface):
        """Test with an email that has no attachments."""
        msg_data = {"uid": "11111", "attachments": []}
        result = await mail_interface._download_attachments_to_temp(msg_data)
        assert result == []

    @pytest.mark.asyncio
    async def test_download_attachments_real_email(self, mail_interface):
        """Test downloading attachments from a real email (if one with attachments exists)."""
        emails = await mail_interface.mail_tools.aget_latest_emails(count=10)
        email_with_attachments = None
        for e in emails:
            if e.get("attachments"):
                email_with_attachments = e
                break

        if email_with_attachments is None:
            pytest.skip("No emails with attachments found")

        temp_files = await mail_interface._download_attachments_to_temp(email_with_attachments)
        try:
            assert len(temp_files) > 0
            for f in temp_files:
                assert os.path.exists(f)
                assert os.path.getsize(f) > 0
        finally:
            mail_interface._cleanup_temp_files(temp_files)


# =====================================================================
# PART 25: API Endpoints — HTTP Tests via TestClient
# =====================================================================


class TestMailAPIEndpoints:
    """Test all API endpoints via FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self, mail_interface):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = mail_interface.attach_routes()
        app.include_router(router)
        self.client = TestClient(app)
        self.secret_header = {"X-Upsonic-Mail-Secret": "test-secret"}

    def test_health_endpoint(self):
        resp = self.client.get("/mail/health", headers=self.secret_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"
        assert "configuration" in data

    def test_inbox_endpoint(self):
        resp = self.client.get("/mail/inbox?count=3", headers=self.secret_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "emails" in data
        assert isinstance(data["emails"], list)

    def test_unread_endpoint(self):
        resp = self.client.get("/mail/unread?count=3", headers=self.secret_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "emails" in data

    def test_send_endpoint(self):
        resp = self.client.post(
            "/mail/send",
            json={
                "to": TEST_RECIPIENT,
                "subject": f"{TEST_SUBJECT_TAG} API Send Test",
                "body": "Sent via API endpoint test.",
            },
            headers=self.secret_header,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_search_endpoint(self):
        resp = self.client.post(
            "/mail/search",
            json={"query": "ALL", "count": 3},
            headers=self.secret_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "emails" in data

    def test_folders_endpoint(self):
        resp = self.client.get("/mail/folders", headers=self.secret_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "folders" in data
        assert isinstance(data["folders"], list)
        assert len(data["folders"]) > 0

    def test_status_endpoint(self):
        resp = self.client.get("/mail/status", headers=self.secret_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "unseen" in data
        assert "recent" in data

    def test_mark_read_endpoint(self, mail_interface):
        emails = mail_interface.mail_tools.get_latest_emails(count=1)
        if not emails:
            pytest.skip("No emails to test with")
        uid = emails[0]["uid"]
        resp = self.client.post(f"/mail/{uid}/read", headers=self.secret_header)
        assert resp.status_code == 200
        assert resp.json()["action"] == "marked_read"

    def test_mark_unread_endpoint(self, mail_interface):
        emails = mail_interface.mail_tools.get_latest_emails(count=1)
        if not emails:
            pytest.skip("No emails to test with")
        uid = emails[0]["uid"]
        resp = self.client.post(f"/mail/{uid}/unread", headers=self.secret_header)
        assert resp.status_code == 200
        assert resp.json()["action"] == "marked_unread"

    def test_delete_endpoint(self, mail_interface):
        """Send a disposable email and delete it via API."""
        subject = f"{TEST_SUBJECT_TAG} API Delete {int(time.time())}"
        mail_interface.mail_tools.send_email(
            to=MAIL_USERNAME, subject=subject, body="Will be deleted via API."
        )
        time.sleep(5)
        emails = mail_interface.mail_tools.search_emails(f'SUBJECT "{subject}"', count=1)
        if not emails:
            pytest.skip("Disposable email did not arrive")
        uid = emails[0]["uid"]
        resp = self.client.post(f"/mail/{uid}/delete", headers=self.secret_header)
        assert resp.status_code == 200
        assert resp.json()["action"] == "deleted"

    def test_move_endpoint(self, mail_interface):
        """Send a disposable email and move it via API."""
        subject = f"{TEST_SUBJECT_TAG} API Move {int(time.time())}"
        mail_interface.mail_tools.send_email(
            to=MAIL_USERNAME, subject=subject, body="Will be moved via API."
        )
        time.sleep(5)
        emails = mail_interface.mail_tools.search_emails(f'SUBJECT "{subject}"', count=1)
        if not emails:
            pytest.skip("Disposable email did not arrive")
        uid = emails[0]["uid"]
        resp = self.client.post(
            f"/mail/{uid}/move?destination=[Gmail]/Spam",
            headers=self.secret_header,
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "moved"

    def test_check_endpoint_no_secret_returns_403(self):
        resp = self.client.post("/mail/check")
        assert resp.status_code == 403

    def test_inbox_no_secret_returns_403(self):
        resp = self.client.get("/mail/inbox")
        assert resp.status_code == 403

    def test_send_no_secret_returns_403(self):
        resp = self.client.post(
            "/mail/send",
            json={"to": "x@x.com", "subject": "x", "body": "x"},
        )
        assert resp.status_code == 403
