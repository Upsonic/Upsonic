import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, HTTPException, Header, Query, status

from upsonic.interfaces.base import Interface
from upsonic.interfaces.schemas import InterfaceMode
from upsonic.interfaces.gmail.schemas import CheckEmailsResponse, AgentEmailResponse
from upsonic.tools.custom_tools.gmail import GmailTools
from upsonic.utils.printing import debug_log, error_log, info_log

if TYPE_CHECKING:
    from upsonic.agent import Agent
    from upsonic.storage.base import Storage


class GmailInterface(Interface):
    """
    Gmail API interface for the Upsonic framework.

    This interface enables an Agent to:
    - Read unread emails
    - Reply to emails using Gmail tools
    - Manage labels and organization
    - Act as an automated email assistant
    
    Supports two operating modes:
    - TASK: Each email is processed as an independent task (default)
    - CHAT: Emails from the same sender continue a conversation session.
            Sending "/reset" in an email body resets the conversation.
    
    Supports whitelist-based access control:
    - Only emails from allowed_emails can interact with the agent
    - Unauthorized senders receive "This operation not allowed"

    Attributes:
        agent: The AI agent that processes emails
        gmail_tools: The Gmail toolkit instance
        api_secret: Secret token to protect the check endpoint
        mode: Operating mode (TASK or CHAT)
        allowed_emails: Set of allowed email addresses (whitelist)
    """

    def __init__(
        self,
        agent: "Agent",
        name: str = "Gmail",
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
        api_secret: Optional[str] = None,
        mode: Union[InterfaceMode, str] = InterfaceMode.TASK,
        reset_command: Optional[str] = "/reset",
        storage: Optional["Storage"] = None,
        allowed_emails: Optional[List[str]] = None,
    ):
        """
        Initialize the Gmail interface.

        Args:
            agent: The AI agent to process emails
            name: Interface name (defaults to "Gmail")
            credentials_path: Path to credentials.json
            token_path: Path to token.json
            api_secret: Secret token for API authentication (or set GMAIL_API_SECRET)
            mode: Operating mode - TASK for independent tasks, CHAT for conversation sessions.
                  Can be InterfaceMode enum or string ("task" or "chat").
            reset_command: Command string to reset chat session (only applies in CHAT mode).
                          Set to None to disable reset command. Default: "/reset"
            storage: Optional storage backend for chat sessions.
            allowed_emails: List of allowed email addresses. If provided, only emails from
                           these addresses will be processed. Others receive "This operation not allowed".
                           If None, all emails are processed.
        """
        super().__init__(
            agent=agent,
            name=name,
            mode=mode,
            reset_command=reset_command,
            storage=storage,
        )

        # Initialize Gmail tools
        self.gmail_tools = GmailTools(
            credentials_path=credentials_path,
            token_path=token_path
        )

        # API Secret for endpoint protection
        self.api_secret = api_secret or os.getenv("GMAIL_API_SECRET")
        if not self.api_secret:
            debug_log(
                "GMAIL_API_SECRET not set. The /check endpoint will not be protected. "
                "Please set the GMAIL_API_SECRET environment variable for security."
            )
        
        # Whitelist: allowed email addresses (normalized to lowercase)
        self._allowed_emails: Optional[Set[str]] = None
        if allowed_emails is not None:
            self._allowed_emails = {email.lower().strip() for email in allowed_emails}
            info_log(f"Gmail whitelist enabled with {len(self._allowed_emails)} allowed email(s)")

        info_log(f"Gmail interface initialized: mode={self.mode.value}, agent={agent}")
    
    def is_email_allowed(self, email: str) -> bool:
        """
        Check if an email address is allowed to interact with the agent.
        
        Args:
            email: Email address to check (can be in "Name <email>" format)
            
        Returns:
            bool: True if allowed or no whitelist configured, False otherwise
        """
        if self._allowed_emails is None:
            return True
        
        # Extract email address from "Name <email@domain.com>" format
        normalized_email = self._extract_sender_id(email)
        return normalized_email in self._allowed_emails

    def attach_routes(self) -> APIRouter:
        """
        Create and attach Gmail routes to the FastAPI application.

        Routes:
            POST /check - Manually trigger a check for new unread emails
            GET /health - Health check endpoint

        Returns:
            APIRouter: Router with Gmail endpoints
        """
        router = APIRouter(prefix="/gmail", tags=["Gmail"])

        @router.post("/check", response_model=CheckEmailsResponse, summary="Check and Process Emails")
        async def check_emails(
            count: int = Query(3, ge=1, description="Maximum number of emails to process"),
            x_upsonic_gmail_secret: Optional[str] = Header(None, alias="X-Upsonic-Gmail-Secret")
        ):
            """
            Trigger the agent to check for unread emails and process them.
            
            Args:
                count: Maximum number of emails to process (default: 10)
                x_upsonic_gmail_secret: Secret token for authentication
            """
            # Verify Secret if configured
            if self.api_secret:
                if not x_upsonic_gmail_secret or x_upsonic_gmail_secret != self.api_secret:
                    error_log("Gmail API authentication failed: Invalid secret")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid authentication secret"
                    )
            
            return await self.check_and_process_emails(count)

        @router.get("/health", summary="Health Check")
        async def health_check_endpoint():
            """Health check endpoint for Gmail interface."""
            return await self.health_check()

        info_log("Gmail routes attached with prefix: /gmail")
        return router

    async def health_check(self) -> Dict[str, Any]:
        """Check health status."""
        base_health = await super().health_check()
        
        # Check if we can access the service (implies valid auth)
        is_connected = False
        try:
            # Lightweight check: just verify service object exists
            if self.gmail_tools.service:
                is_connected = True
        except Exception:
            is_connected = False

        base_health["configuration"] = {
            "connected": is_connected,
            "tools_enabled": len(self.gmail_tools.functions()),
            "auth_configured": bool(self.api_secret),
            "mode": self.mode.value,
            "reset_command": self._reset_command.command if self._reset_enabled else None,
            "active_chat_sessions": len(self._chat_sessions) if self.is_chat_mode() else 0,
            "whitelist_enabled": self._allowed_emails is not None,
            "allowed_emails_count": len(self._allowed_emails) if self._allowed_emails else 0,
        }
        return base_health

    async def _send_reply(self, email_data: Dict, reply_text: str):
        """
        Send a reply to an email using the Gmail tools.
        
        Args:
            email_data: The original email dictionary
            reply_text: The body of the reply
        """
        try:
            await asyncio.to_thread(
                self.gmail_tools.send_email_reply,
                thread_id=email_data.get("thread_id"),
                message_id=email_data.get("id"),
                to=email_data.get("from"),
                subject=email_data.get("subject"),
                body=reply_text
            )
            info_log(f"Sent reply to {email_data.get('from')}")
        except Exception as e:
            error_log(f"Failed to send reply: {e}")

    async def check_and_process_emails(self, count: int = 10) -> CheckEmailsResponse:
        """
        Fetch unread emails and process them according to the interface mode.

        In TASK mode: Each email is processed as an independent task.
        In CHAT mode: Emails from the same sender continue a conversation session.

        Args:
            count: Number of emails to fetch

        Returns:
            CheckEmailsResponse: Summary of processed emails
        """
        info_log(f"Checking for up to {count} unread emails (mode={self.mode.value})...")

        try:
            # Run blocking Gmail API call in thread
            messages = await asyncio.to_thread(
                self.gmail_tools.get_unread_messages_raw, count
            )
        except Exception as e:
            error_log(f"Failed to fetch unread emails: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch emails: {str(e)}"
            )

        if not messages:
            info_log("No unread emails found.")
            return CheckEmailsResponse(
                status="success",
                processed_count=0,
                message_ids=[]
            )

        processed_ids = []

        for msg_data in messages:
            try:
                msg_id = msg_data.get("id")
                sender = msg_data.get("from")
                subject = msg_data.get("subject")
                body = msg_data.get("body", "")

                info_log(f"Processing email {msg_id} from {sender}: {subject}")
                
                # Check whitelist - if sender is not allowed, skip processing
                if not self.is_email_allowed(sender):
                    info_log(self.get_unauthorized_message())
                    # Mark as read to avoid reprocessing
                    await asyncio.to_thread(
                        self.gmail_tools.mark_email_as_read, msg_id
                    )
                    continue

                # Check for reset command in CHAT mode
                if self.is_chat_mode() and self.is_reset_command(body):
                    await self._handle_reset_command(msg_data)
                    processed_ids.append(msg_id)
                    continue

                # Process based on mode
                if self.is_task_mode():
                    await self._process_email_task_mode(msg_data)
                else:
                    await self._process_email_chat_mode(msg_data)
                
                # Mark as read AFTER processing
                await asyncio.to_thread(
                    self.gmail_tools.mark_email_as_read, msg_id
                )
                
                processed_ids.append(msg_id)

            except Exception as e:
                error_log(f"Error processing email {msg_data.get('id')}: {e}")
                # Continue to next email even if one fails
                continue

        return CheckEmailsResponse(
            status="success",
            processed_count=len(processed_ids),
            message_ids=processed_ids
        )
    
    def _extract_sender_id(self, sender: str) -> str:
        """
        Extract a unique user ID from the sender email address.
        
        Args:
            sender: The 'From' field, e.g., 'John Doe <john@example.com>'
            
        Returns:
            str: Normalized email address as user ID
        """
        import re
        # Extract email from "Name <email@domain.com>" format
        match = re.search(r'<([^>]+)>', sender)
        if match:
            return match.group(1).lower().strip()
        # If no angle brackets, assume it's just the email
        return sender.lower().strip()
    
    async def _handle_reset_command(self, msg_data: Dict[str, Any]) -> None:
        """
        Handle a reset command email in CHAT mode.
        
        Args:
            msg_data: Email data dictionary
        """
        msg_id = msg_data.get("id")
        sender = msg_data.get("from", "")
        user_id = self._extract_sender_id(sender)
        
        info_log(f"Reset command received from {sender} for email {msg_id}")
        
        # Reset the chat session
        was_reset = await self.areset_chat_session(user_id)
        
        # Send confirmation reply
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
        
        # Mark as read
        await asyncio.to_thread(
            self.gmail_tools.mark_email_as_read, msg_id
        )
        
        info_log(f"Reset command processed for user {user_id}")
    
    async def _process_email_task_mode(self, msg_data: Dict[str, Any]) -> None:
        """
        Process an email in TASK mode (independent task per email).
        
        Args:
            msg_data: Email data dictionary
        """
        from upsonic.tasks.tasks import Task
        
        msg_id = msg_data.get("id")
        sender = msg_data.get("from")
        subject = msg_data.get("subject")
        body = msg_data.get("body")

        # Enhanced Task Description with structured output requirement
        task_description = (
            f"You are an intelligent email assistant. You have received a new email to process.\n\n"
            f"EMAIL CONTEXT:\n"
            f"--------------------------------------------------\n"
            f"From: {sender}\n"
            f"Subject: {subject}\n"
            f"Content:\n{body}\n"
            f"--------------------------------------------------\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Analyze the email content, sender, and intent carefully.\n"
            f"2. Decide whether to 'reply' or 'ignore' (e.g., for spam, automated notifications, or no-action-needed emails).\n"
            f"3. If you decide to reply, draft a professional, helpful, and context-aware response.\n"
            f"4. Provide a brief reasoning for your decision."
        )

        # Create task with specific response format (Pydantic)
        task = Task(task_description, response_format=AgentEmailResponse)

        # Execute agent
        await self.agent.do_async(task)

        # Get structured result
        run_result = self.agent.get_run_output()
        
        if run_result and run_result.output:
            # The output is already an instance of AgentEmailResponse thanks to response_format
            response: AgentEmailResponse = run_result.output
            
            info_log(f"Agent decision for {msg_id}: {response.action} (Reason: {response.reasoning})")
            
            if response.action == "reply":
                if response.reply_body:
                    await self._send_reply(msg_data, response.reply_body)
                else:
                    error_log(f"Agent decided to reply but provided no body for email {msg_id}")
            else:
                info_log(f"Skipping reply for email {msg_id}")
    
    async def _process_email_chat_mode(self, msg_data: Dict[str, Any]) -> None:
        """
        Process an email in CHAT mode (conversation session per sender).
        
        In CHAT mode, each email continues the conversation with the sender.
        The agent has access to the full conversation history.
        
        Args:
            msg_data: Email data dictionary
        """
        msg_id = msg_data.get("id")
        sender = msg_data.get("from", "")
        subject = msg_data.get("subject", "")
        body = msg_data.get("body", "")
        
        # Get user ID from sender email
        user_id = self._extract_sender_id(sender)
        
        # Get or create chat session for this sender
        chat = await self.aget_chat_session(user_id)
        
        # Format the email content as a chat message
        # Include subject only if it's a new thread (first message or subject changed)
        chat_message = f"[Email from: {sender}]\n"
        if subject:
            chat_message += f"Subject: {subject}\n\n"
        chat_message += body
        
        info_log(f"Processing email {msg_id} in CHAT mode for user {user_id}")
        
        try:
            # Use the chat session to process the message
            # The chat maintains conversation history automatically
            response_text = await chat.invoke(chat_message)
            
            if response_text:
                # Send the response as a reply
                await self._send_reply(msg_data, response_text)
                info_log(f"Sent chat response to {sender} for email {msg_id}")
            else:
                debug_log(f"No response generated for email {msg_id}")
                
        except Exception as e:
            error_log(f"Error in chat mode processing for email {msg_id}: {e}")
            # Send a generic error response
            error_reply = (
                "I apologize, but I encountered an error processing your message. "
                "Please try again or send '/reset' to start a new conversation."
            )
            await self._send_reply(msg_data, error_reply)
