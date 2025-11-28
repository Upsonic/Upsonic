import hashlib
import hmac
import json
import re
import time
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from upsonic.interfaces.base import Interface
from upsonic.interfaces.slack.schemas import SlackEventResponse, SlackChallengeResponse
from upsonic.tools.custom_tools.slack import SlackTools
from upsonic.utils.printing import debug_log, error_log, info_log
from upsonic.tasks.tasks import Task

if TYPE_CHECKING:
    from upsonic.agent import Agent


class SlackInterface(Interface):
    """
    Slack interface for the Upsonic framework.

    This interface handles:
    - Slack event verification
    - Incoming message processing (app_mention, message)
    - Outgoing message sending
    - Agent integration for automatic responses
    - Event deduplication
    """

    def __init__(
        self,
        agent: "Agent",
        signing_secret: Optional[str] = None,
        verification_token: Optional[str] = None,
        name: str = "Slack",
        reply_to_mentions_only: bool = True,
    ):
        """
        Initialize the Slack interface.

        Args:
            agent: The AI agent to process messages
            signing_secret: Slack signing secret (or set SLACK_SIGNING_SECRET)
            verification_token: Slack verification token (or set SLACK_VERIFICATION_TOKEN)
            name: Interface name (defaults to "Slack")
            reply_to_mentions_only: Whether to only reply to mentions (default: True)
        """
        super().__init__(agent, name)

        self.signing_secret = signing_secret or os.getenv("SLACK_SIGNING_SECRET")
        if not self.signing_secret:
            debug_log(
                "SLACK_SIGNING_SECRET not set. Signature verification might fail. "
                "Please set the SLACK_SIGNING_SECRET environment variable for security."
            )

        self.verification_token = verification_token or os.getenv("SLACK_VERIFICATION_TOKEN")
        self.reply_to_mentions_only = reply_to_mentions_only

        # Initialize Slack tools for sending messages
        self.slack_tools = SlackTools()
        
        # Event deduplication cache: event_id -> timestamp
        self._processed_events: Dict[str, float] = {}
        self._dedup_window = 300  # Keep event IDs for 5 minutes
        
        info_log(f"Slack interface initialized with agent: {agent}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the Slack interface.
        
        Returns:
            Dict[str, Any]: Health status
        """
        status_data = {
            "status": "active",
            "name": self.name,
            "id": self.id,
            "configuration": {
                "signing_secret_configured": bool(self.signing_secret),
                "verification_token_configured": bool(self.verification_token),
                "reply_to_mentions_only": self.reply_to_mentions_only,
            },
            "tools_initialized": self.slack_tools.client is not None
        }
        
        if not self.signing_secret:
            status_data["status"] = "degraded"
            status_data["issues"] = ["SLACK_SIGNING_SECRET is missing"]
            
        return status_data

    def _verify_slack_signature(self, body: bytes, timestamp: str, signature: str) -> bool:
        """
        Verify the Slack request signature.

        Args:
            body: Raw request body bytes
            timestamp: Request timestamp
            signature: X-Slack-Signature header value

        Returns:
            bool: True if signature is valid, False otherwise
        """
        if not self.signing_secret:
            error_log("SLACK_SIGNING_SECRET not configured, cannot verify signature")
            return False

        # Ensure the request timestamp is recent (prevent replay attacks)
        if abs(time.time() - int(timestamp)) > 60 * 5:
            error_log(f"Request timestamp expired: {timestamp}")
            return False

        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        my_signature = (
            "v0="
            + hmac.new(
                self.signing_secret.encode("utf-8"),
                sig_basestring.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(my_signature, signature)

    async def _send_slack_message(
        self, channel: str, thread_ts: Optional[str], message: str, italics: bool = False
    ):
        """
        Send a message to Slack, handling long messages and formatting.

        Args:
            channel: Channel ID
            thread_ts: Thread timestamp to reply to (None to post directly in channel)
            message: Message content
            italics: Whether to italicize the message (e.g. for reasoning)
        """
        if not message:
            return

        # Check message length limit (Slack is approx 40000 chars, but safer to stay lower)
        limit = 4000
        
        if len(message) <= limit:
            text_to_send = message
            if italics:
                # Handle multi-line messages by making each line italic
                text_to_send = "\n".join([f"_{line}_" for line in message.split("\n")])
            
            # If thread_ts is None, post directly to channel
            if thread_ts:
                self.slack_tools.send_message_thread(
                    channel=channel, text=text_to_send, thread_ts=thread_ts
                )
            else:
                self.slack_tools.send_message(
                    channel=channel, text=text_to_send
                )
            return

        # Split message into batches
        message_batches = [message[i : i + limit] for i in range(0, len(message), limit)]

        for i, batch in enumerate(message_batches, 1):
            batch_message = f"[{i}/{len(message_batches)}] {batch}"
            if italics:
                batch_message = "\n".join([f"_{line}_" for line in batch_message.split("\n")])
            
            if thread_ts:
                self.slack_tools.send_message_thread(
                    channel=channel, text=batch_message, thread_ts=thread_ts
                )
            else:
                self.slack_tools.send_message(
                    channel=channel, text=batch_message
                )

    def _cleanup_processed_events(self):
        """Remove old events from the deduplication cache."""
        current_time = time.time()
        expired_events = [
            eid for eid, ts in self._processed_events.items() 
            if current_time - ts > self._dedup_window
        ]
        for eid in expired_events:
            del self._processed_events[eid]

    async def _process_slack_event(self, event: Dict[str, Any]):
        """
        Process a Slack event (message or app_mention).

        Args:
            event: Event data from Slack
        """
        try:
            # Deduplication check
            event_id = event.get("event_ts")  # Using event_ts as ID
            if not event_id:
                # Fallback to generating one if missing (unlikely for valid events)
                event_id = str(time.time())
                
            if event_id in self._processed_events:
                debug_log(f"Duplicate event received: {event_id}")
                return
            
            self._processed_events[event_id] = time.time()
            
            # Occasional cleanup
            if len(self._processed_events) > 1000:
                self._cleanup_processed_events()

            event_type = event.get("type")
            channel_type = event.get("channel_type", "")
            
            # Only handle app_mention and message events
            if event_type not in ("app_mention", "message"):
                return

            # Handle duplicate replies / bot messages
            if event.get("bot_id") or event.get("subtype") == "bot_message":
                return

            # Filter based on configuration
            if not self.reply_to_mentions_only and event_type == "app_mention":
                # If we reply to everything, app_mention is just one type, proceed
                pass
            elif self.reply_to_mentions_only:
                 # If reply_to_mentions_only is True:
                 # 1. Accept app_mention
                 # 2. Accept message only if it is a DM (channel_type == 'im')
                if event_type == "message" and channel_type != "im":
                    return

            user = event.get("user")
            text = event.get("text", "")
            channel = event.get("channel", "")
            
            # For @mentions, remove the bot mention from the text
            if event_type == "app_mention":
                # Remove <@BOT_ID> from the message
                text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
            
            # Don't reply in thread - reply directly in channel/DM
            # Only use thread_ts if the original message was in a thread
            ts = event.get("thread_ts")  # Will be None if not in a thread

            info_log(f"Processing Slack event from {user} in {channel}: {text[:50]}...")

            # Execute agent
            task = Task(text)
            await self.agent.do_async(task)

            # Get result
            run_result = self.agent.get_run_result()
            if not run_result:
                error_log("No run result from agent")
                return

            model_response = run_result.get_last_model_response()
            
            if model_response:
                # Send reasoning if available
                if hasattr(model_response, "thinking") and model_response.thinking:
                    await self._send_slack_message(
                        channel=channel,
                        thread_ts=ts,
                        message=f"Reasoning: \n{model_response.thinking}",
                        italics=True,
                    )
                
                # Send content
                content = model_response.text
                if content:
                    await self._send_slack_message(
                        channel=channel,
                        thread_ts=ts,
                        message=content,
                    )
            elif run_result.output:
                # Fallback to generic output
                await self._send_slack_message(
                    channel=channel,
                    thread_ts=ts,
                    message=str(run_result.output),
                )

        except Exception as e:
            import traceback
            error_log(f"Error processing Slack event: {e}\n{traceback.format_exc()}")

    def attach_routes(self) -> APIRouter:
        """
        Create and attach Slack routes to the FastAPI application.

        Returns:
            APIRouter: Router with Slack endpoints
        """
        router = APIRouter(prefix="/slack", tags=["Slack"])

        @router.post(
            "/events",
            response_model=Union[SlackChallengeResponse, SlackEventResponse],
            response_model_exclude_none=True,
            status_code=status.HTTP_200_OK,
        )
        async def slack_events(request: Request, background_tasks: BackgroundTasks):
            """
            Handle incoming Slack events.
            """
            try:
                body = await request.body()
                
                # Check headers
                timestamp = request.headers.get("X-Slack-Request-Timestamp")
                signature = request.headers.get("X-Slack-Signature")
                
                if not timestamp or not signature:
                     # Fail silently or with 400, but for Slack strictness 400 is good
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Missing Slack headers"
                    )

                # Verify signature
                if not self._verify_slack_signature(body, timestamp, signature):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid signature"
                    )

                # Parse data
                data = await request.json()
                
                # Handle URL Verification
                if data.get("type") == "url_verification":
                    return SlackChallengeResponse(challenge=data.get("challenge"))

                # Handle Events
                if "event" in data:
                    event = data["event"]
                    # Process in background
                    background_tasks.add_task(self._process_slack_event, event)

                return SlackEventResponse(status="ok")
            
            except HTTPException:
                raise
            except Exception as e:
                error_log(f"Error handling Slack request: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )

        @router.get("/health", summary="Health Check")
        async def health_check_endpoint():
            """Health check endpoint for Slack interface."""
            return await self.health_check()

        info_log("Slack routes attached with prefix: /slack")
        return router
