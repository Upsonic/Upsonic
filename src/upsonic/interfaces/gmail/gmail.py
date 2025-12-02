import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Header, status

from upsonic.interfaces.base import Interface
from upsonic.interfaces.gmail.schemas import CheckEmailsResponse, AgentEmailResponse
from upsonic.tools.custom_tools.gmail import GmailTools
from upsonic.utils.printing import debug_log, error_log, info_log

if TYPE_CHECKING:
    from upsonic.agent import Agent


class GmailInterface(Interface):
    """
    Gmail API interface for the Upsonic framework.

    This interface enables an Agent to:
    - Read unread emails
    - Reply to emails using Gmail tools
    - Manage labels and organization
    - Act as an automated email assistant

    Attributes:
        agent: The AI agent that processes emails
        gmail_tools: The Gmail toolkit instance
        api_secret: Secret token to protect the check endpoint
    """

    def __init__(
        self,
        agent: "Agent",
        name: str = "Gmail",
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize the Gmail interface.

        Args:
            agent: The AI agent to process emails
            name: Interface name (defaults to "Gmail")
            credentials_path: Path to credentials.json
            token_path: Path to token.json
            api_secret: Secret token for API authentication (or set GMAIL_API_SECRET)
        """
        super().__init__(agent, name)

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

        info_log(f"Gmail interface initialized with agent: {agent}")

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
            count: int = 10,
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
            "auth_configured": bool(self.api_secret)
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
        Fetch unread emails and task the agent to process them.

        Args:
            count: Number of emails to fetch

        Returns:
            CheckEmailsResponse: Summary of processed emails
        """
        info_log(f"Checking for up to {count} unread emails...")

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
        
        # Import Task here to avoid circular imports if any
        from upsonic.tasks.tasks import Task

        for msg_data in messages:
            try:
                msg_id = msg_data.get("id")
                sender = msg_data.get("from")
                subject = msg_data.get("subject")
                body = msg_data.get("body")

                info_log(f"Processing email {msg_id} from {sender}: {subject}")

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
                run_result = self.agent.get_run_result()
                
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
