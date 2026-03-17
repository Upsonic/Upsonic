"""
Discord Bot API Interface for the Upsonic Framework.

This module provides a comprehensive Discord Bot integration with support for:
- All message types (text, attachments, embeds)
- Message components (buttons, select menus) for HITL confirmations
- Typing indicators
- File downloads and uploads
- Gateway WebSocket-based event handling
- Task and Chat modes
- Whitelist-based access control

Based on the official Discord API: https://discord.com/developers/docs
"""

import asyncio
import json
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Literal, Optional, Set, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from upsonic.interfaces.base import Interface
from upsonic.interfaces.schemas import InterfaceMode
from upsonic.interfaces.discord.schemas import (
    DiscordGatewayPayload,
    DiscordMessage,
    DiscordInteraction,
)
from upsonic.tools.custom_tools.discord import DiscordTools
from upsonic.utils.printing import debug_log, error_log, info_log

if TYPE_CHECKING:
    from upsonic.agent import Agent
    from upsonic.storage.base import Storage


# Discord Gateway opcodes
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_RESUME = 6
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

# Discord intents
INTENT_GUILDS = 1 << 0
INTENT_GUILD_MESSAGES = 1 << 9
INTENT_GUILD_MESSAGE_REACTIONS = 1 << 10
INTENT_DIRECT_MESSAGES = 1 << 12
INTENT_MESSAGE_CONTENT = 1 << 15

DEFAULT_INTENTS = INTENT_GUILDS | INTENT_GUILD_MESSAGES | INTENT_DIRECT_MESSAGES | INTENT_MESSAGE_CONTENT


def _format_confirmation_message(tool_name: str, tool_args: Optional[Dict[str, Any]]) -> str:
    """Build a short human-readable line for a tool requiring confirmation."""
    args_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in (tool_args or {}).items())[:120]
    return f"Tool {tool_name}({args_str}) requires confirmation."


class DiscordInterface(Interface):
    """
    Discord Bot API interface for the Upsonic framework.

    This interface provides comprehensive Discord Bot integration:
    - Gateway WebSocket-based event handling for receiving messages
    - All message types (text, attachments, embeds)
    - Message components (buttons) for HITL confirmations
    - Typing indicators
    - File downloads from Discord CDN
    - Guild and DM message handling

    Supports two operating modes:
    - TASK: Each message is processed as an independent task (default)
    - CHAT: Messages from the same user continue a conversation session.
            Sending "/reset" resets the conversation.

    Supports whitelist-based access control:
    - Only messages from allowed_user_ids can interact with the agent
    - Unauthorized users are silently ignored (logged only)

    Attributes:
        agent: The AI agent that processes messages
        discord_tools: The Discord toolkit instance for API calls
        mode: Operating mode (TASK or CHAT)
        allowed_user_ids: Set of allowed Discord user IDs (whitelist)
    """

    def __init__(
        self,
        agent: "Agent",
        bot_token: Optional[str] = None,
        name: str = "Discord",
        mode: Union[InterfaceMode, str] = InterfaceMode.TASK,
        reset_command: Optional[str] = "/reset",
        storage: Optional["Storage"] = None,
        allowed_user_ids: Optional[List[str]] = None,
        allowed_channel_ids: Optional[List[str]] = None,
        allowed_guild_ids: Optional[List[str]] = None,
        intents: int = DEFAULT_INTENTS,
        typing_indicator: bool = True,
        stream: bool = False,
        max_message_length: int = 2000,
        process_dm: bool = True,
        process_guild_messages: bool = True,
        heartbeat_channel_id: Optional[str] = None,
    ):
        """
        Initialize the Discord interface.

        Args:
            agent: The AI agent to process messages
            bot_token: Discord Bot API token (or set DISCORD_BOT_TOKEN env var)
            name: Interface name (defaults to "Discord")
            mode: Operating mode - TASK for independent tasks, CHAT for conversation sessions.
            reset_command: Command to reset chat session (only applies in CHAT mode).
            storage: Optional storage backend for chat sessions.
            allowed_user_ids: List of allowed Discord user IDs. If None, all users are allowed.
            allowed_channel_ids: List of allowed channel IDs. If None, all channels are allowed.
            allowed_guild_ids: List of allowed guild IDs. If None, all guilds are allowed.
            intents: Discord Gateway intents bitmask. Default includes GUILDS, GUILD_MESSAGES,
                    DIRECT_MESSAGES, and MESSAGE_CONTENT.
            typing_indicator: Whether to send typing indicator before responding (default: True).
            stream: Whether to stream agent responses by progressively editing the message.
            max_message_length: Maximum message length before splitting (default: 2000).
            process_dm: Whether to process direct messages (default: True).
            process_guild_messages: Whether to process guild messages (default: True).
            heartbeat_channel_id: Discord channel ID to send heartbeat responses to.
        """
        super().__init__(
            agent=agent,
            name=name,
            mode=mode,
            reset_command=reset_command,
            storage=storage,
        )

        self._bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")

        # Initialize Discord tools
        self.discord_tools = DiscordTools(
            bot_token=self._bot_token,
            max_message_length=max_message_length,
        )

        # Whitelist: allowed Discord user IDs (snowflakes)
        self._allowed_user_ids: Optional[Set[str]] = None
        if allowed_user_ids is not None:
            self._allowed_user_ids = set(allowed_user_ids)
            info_log(f"Discord whitelist enabled with {len(self._allowed_user_ids)} allowed user(s)")

        self._allowed_channel_ids: Optional[Set[str]] = None
        if allowed_channel_ids is not None:
            self._allowed_channel_ids = set(allowed_channel_ids)

        self._allowed_guild_ids: Optional[Set[str]] = None
        if allowed_guild_ids is not None:
            self._allowed_guild_ids = set(allowed_guild_ids)

        # Gateway settings
        self._intents = intents
        self._gateway_ws: Any = None
        self._gateway_task: Optional[asyncio.Task] = None
        self._heartbeat_task_gw: Optional[asyncio.Task] = None
        self._session_id: Optional[str] = None
        self._sequence: Optional[int] = None
        self._resume_gateway_url: Optional[str] = None
        self._bot_user_id: Optional[str] = None
        self._application_id: Optional[str] = None

        # Behavior options
        self.typing_indicator: bool = typing_indicator
        self.stream: bool = stream
        self.process_dm: bool = process_dm
        self.process_guild_messages: bool = process_guild_messages
        self.heartbeat_channel_id: Optional[str] = heartbeat_channel_id
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._auto_heartbeat_channel_id: Optional[str] = None

        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}

        # Typing indicator management
        self._typing_tasks: Dict[str, asyncio.Task] = {}

        info_log(f"Discord interface initialized: mode={self.mode.value}, stream={self.stream}, agent={agent}")

    def is_user_allowed(self, user_id: str) -> bool:
        """Check if a Discord user ID is allowed to interact with the agent."""
        if self._allowed_user_ids is None:
            return True
        return user_id in self._allowed_user_ids

    def is_channel_allowed(self, channel_id: str) -> bool:
        """Check if a Discord channel ID is allowed."""
        if self._allowed_channel_ids is None:
            return True
        return channel_id in self._allowed_channel_ids

    def is_guild_allowed(self, guild_id: Optional[str]) -> bool:
        """Check if a Discord guild ID is allowed."""
        if self._allowed_guild_ids is None:
            return True
        if guild_id is None:
            return True  # DMs have no guild
        return guild_id in self._allowed_guild_ids

    # ─── HITL Confirmation ───────────────────────────────────────────────

    async def _send_confirmation_and_store(
        self,
        output: Any,
        channel_id: str,
        user_id: str,
        mode: Literal["task", "chat"],
        reply_to: Optional[str] = None,
    ) -> None:
        """Send a confirmation message with Confirm/Reject buttons and store pending state."""
        active = getattr(output, "active_requirements", None) or []
        first_req = next((r for r in active if getattr(r, "needs_confirmation", False)), None)
        if not first_req:
            return
        tool_exec = getattr(first_req, "tool_execution", None)
        tool_name = getattr(tool_exec, "tool_name", "tool") if tool_exec else "tool"
        tool_args = getattr(tool_exec, "tool_args", None) or {}
        text = _format_confirmation_message(tool_name, tool_args)
        pending_key = uuid.uuid4().hex[:12]
        run_id = getattr(output, "run_id", None) or ""
        self._pending_confirmations[pending_key] = {
            "run_id": run_id,
            "output": output,
            "channel_id": channel_id,
            "user_id": user_id,
            "mode": mode,
        }
        components = [
            {
                "type": 1,  # ACTION_ROW
                "components": [
                    {
                        "type": 2,  # BUTTON
                        "style": 3,  # SUCCESS (green)
                        "label": "Confirm",
                        "custom_id": f"cfm:{pending_key}:0:y",
                    },
                    {
                        "type": 2,  # BUTTON
                        "style": 4,  # DANGER (red)
                        "label": "Reject",
                        "custom_id": f"cfm:{pending_key}:0:n",
                    },
                ],
            }
        ]
        await self.discord_tools.asend_message(
            channel_id=channel_id,
            content=text,
            components=components,
            reply_to=reply_to,
        )

    # ─── Health Check ────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the Discord interface."""
        base_health = await super().health_check()

        bot_info = None
        is_connected = False
        try:
            bot_info = await self.discord_tools.aget_me()
            is_connected = bot_info is not None
        except Exception as e:
            debug_log(f"Bot connectivity check failed: {e}")

        base_health["configuration"] = {
            "bot_token_configured": bool(self._bot_token),
            "mode": self.mode.value,
            "reset_command": self._reset_command.command if self._reset_enabled else None,
            "active_chat_sessions": len(self._chat_sessions) if self.is_chat_mode() else 0,
            "whitelist_enabled": self._allowed_user_ids is not None,
            "allowed_user_ids_count": len(self._allowed_user_ids) if self._allowed_user_ids else 0,
            "process_dm": self.process_dm,
            "process_guild_messages": self.process_guild_messages,
            "gateway_connected": self._gateway_ws is not None,
        }

        if bot_info:
            base_health["bot"] = {
                "connected": is_connected,
                "id": bot_info.get("id"),
                "username": bot_info.get("username"),
                "global_name": bot_info.get("global_name"),
            }
        else:
            base_health["bot"] = {"connected": False}

        if not self._bot_token:
            base_health["status"] = "degraded"
            base_health["issues"] = ["DISCORD_BOT_TOKEN is missing"]

        return base_health

    # ─── FastAPI Routes ──────────────────────────────────────────────────

    def attach_routes(self) -> APIRouter:
        """
        Create and attach Discord routes to the FastAPI application.

        Routes:
            POST /interaction - Discord Interactions endpoint (optional, for slash commands)
            GET /health - Health check endpoint

        The primary message handling is done via the Gateway WebSocket,
        which is started as a background task on FastAPI startup.

        Returns:
            APIRouter: Router with Discord endpoints
        """
        router = APIRouter(prefix="/discord", tags=["Discord"])

        @router.post("/interaction", status_code=status.HTTP_200_OK)
        async def interaction_endpoint(request: Request, background_tasks: BackgroundTasks):
            """
            Discord Interactions endpoint for slash commands and component interactions.

            This endpoint handles interactions sent via HTTP (when configured as
            the Interactions Endpoint URL in the Discord Developer Portal).
            """
            try:
                data = await request.json()

                # Handle Discord's verification ping
                if data.get("type") == 1:
                    return {"type": 1}

                interaction = DiscordInteraction(**data)
                background_tasks.add_task(self._process_interaction, interaction)

                # Acknowledge the interaction (type 5 = deferred channel message)
                return {"type": 5}

            except Exception as e:
                error_log(f"Discord interaction error: {e}")
                return {"type": 1}

        @router.get("/health", summary="Health Check")
        async def health_check_endpoint():
            """Health check endpoint for Discord interface."""
            return await self.health_check()

        @router.on_event("startup")
        async def start_gateway():
            """Start the Discord Gateway WebSocket connection on startup."""
            await self._start_gateway()

        @router.on_event("startup")
        async def start_heartbeat() -> None:
            self._start_heartbeat()

        @router.on_event("shutdown")
        async def stop_gateway():
            """Gracefully close the Gateway connection on shutdown."""
            await self._stop_gateway()

        info_log("Discord routes attached with prefix: /discord")
        return router

    # ─── Gateway WebSocket ───────────────────────────────────────────────

    async def _start_gateway(self) -> None:
        """Start the Gateway WebSocket connection as a background task."""
        if not self._bot_token:
            error_log("Cannot start Discord Gateway: bot token not configured")
            return

        if self._gateway_task is not None and not self._gateway_task.done():
            return

        self._gateway_task = asyncio.create_task(self._gateway_connect())
        info_log("Discord Gateway connection started")

    async def _stop_gateway(self) -> None:
        """Stop the Gateway WebSocket connection."""
        if self._heartbeat_task_gw and not self._heartbeat_task_gw.done():
            self._heartbeat_task_gw.cancel()

        if self._gateway_ws:
            try:
                await self._gateway_ws.close()
            except Exception:
                pass
            self._gateway_ws = None

        if self._gateway_task and not self._gateway_task.done():
            self._gateway_task.cancel()

        info_log("Discord Gateway connection stopped")

    async def _get_gateway_url(self) -> str:
        """Fetch the Gateway URL from Discord's REST API."""
        default_url = "wss://gateway.discord.gg/?v=10&encoding=json"
        try:
            result = await self.discord_tools._api_request("GET", "/gateway/bot")
            if result and isinstance(result, dict) and "url" in result:
                url = result["url"]
                return f"{url}/?v=10&encoding=json"
        except Exception as e:
            debug_log(f"Failed to fetch Gateway URL, using default: {e}")
        return default_url

    async def _gateway_connect(self) -> None:
        """Establish and maintain the Gateway WebSocket connection."""
        try:
            import websockets
        except ImportError:
            error_log(
                "websockets package is required for Discord Gateway. "
                "Install it with: pip install websockets"
            )
            return

        gateway_url = await self._get_gateway_url()
        info_log(f"Discord Gateway URL: {gateway_url}")

        while True:
            try:
                resume_url = self._resume_gateway_url or gateway_url

                async with websockets.connect(
                    resume_url,
                    open_timeout=30,
                    close_timeout=10,
                ) as ws:
                    self._gateway_ws = ws

                    # Wait for Hello (op 10)
                    hello_raw = await ws.recv()
                    hello = json.loads(hello_raw)

                    if hello.get("op") != OP_HELLO:
                        error_log(f"Expected Hello (op 10), got op {hello.get('op')}")
                        continue

                    heartbeat_interval = hello["d"]["heartbeat_interval"] / 1000.0
                    info_log(f"Discord Gateway Hello received, heartbeat interval: {heartbeat_interval}s")

                    # Start heartbeat loop
                    self._heartbeat_task_gw = asyncio.create_task(
                        self._gateway_heartbeat_loop(ws, heartbeat_interval)
                    )

                    # Identify or Resume
                    if self._session_id and self._sequence is not None:
                        # Resume
                        await ws.send(json.dumps({
                            "op": OP_RESUME,
                            "d": {
                                "token": self._bot_token,
                                "session_id": self._session_id,
                                "seq": self._sequence,
                            },
                        }))
                        info_log("Discord Gateway: Resuming session")
                    else:
                        # Identify
                        await ws.send(json.dumps({
                            "op": OP_IDENTIFY,
                            "d": {
                                "token": self._bot_token,
                                "intents": self._intents,
                                "properties": {
                                    "os": "linux",
                                    "browser": "upsonic",
                                    "device": "upsonic",
                                },
                            },
                        }))
                        info_log("Discord Gateway: Identifying")

                    # Listen for events
                    await self._gateway_listen(ws)

            except asyncio.CancelledError:
                info_log("Discord Gateway task cancelled")
                return
            except Exception as e:
                error_log(f"Discord Gateway error: {e}")
                # Clean up
                if self._heartbeat_task_gw and not self._heartbeat_task_gw.done():
                    self._heartbeat_task_gw.cancel()
                self._gateway_ws = None

                # Wait before reconnecting
                await asyncio.sleep(5)
                info_log("Discord Gateway: Attempting reconnection...")

    async def _gateway_heartbeat_loop(self, ws: Any, interval: float) -> None:
        """Send periodic heartbeats to the Gateway."""
        try:
            while True:
                await asyncio.sleep(interval)
                await ws.send(json.dumps({
                    "op": OP_HEARTBEAT,
                    "d": self._sequence,
                }))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            debug_log(f"Gateway heartbeat error: {e}")

    async def _gateway_listen(self, ws: Any) -> None:
        """Listen for events from the Gateway WebSocket."""
        async for raw_msg in ws:
            try:
                payload = json.loads(raw_msg)
                op = payload.get("op")
                data = payload.get("d")
                seq = payload.get("s")
                event_name = payload.get("t")

                # Update sequence number
                if seq is not None:
                    self._sequence = seq

                if op == OP_DISPATCH:
                    await self._process_gateway_event(event_name, data)

                elif op == OP_HEARTBEAT:
                    # Discord is requesting an immediate heartbeat
                    await ws.send(json.dumps({
                        "op": OP_HEARTBEAT,
                        "d": self._sequence,
                    }))

                elif op == OP_RECONNECT:
                    info_log("Discord Gateway: Reconnect requested")
                    return  # Will reconnect in the outer loop

                elif op == OP_INVALID_SESSION:
                    resumable = data if isinstance(data, bool) else False
                    if not resumable:
                        self._session_id = None
                        self._sequence = None
                    info_log(f"Discord Gateway: Invalid session (resumable={resumable})")
                    await asyncio.sleep(3)
                    return  # Will reconnect

                elif op == OP_HEARTBEAT_ACK:
                    pass  # All good

            except Exception as e:
                error_log(f"Error processing Gateway message: {e}")

    async def _process_gateway_event(self, event_name: Optional[str], data: Any) -> None:
        """Process a dispatched Gateway event."""
        if not event_name:
            return

        if event_name == "READY":
            self._session_id = data.get("session_id")
            self._resume_gateway_url = data.get("resume_gateway_url")
            user = data.get("user", {})
            self._bot_user_id = user.get("id")
            self._application_id = data.get("application", {}).get("id")
            info_log(f"Discord Gateway: Ready as {user.get('username')}#{user.get('discriminator', '0')}")

        elif event_name == "RESUMED":
            info_log("Discord Gateway: Session resumed")

        elif event_name == "MESSAGE_CREATE":
            try:
                message = DiscordMessage(**data)
                # Ignore messages from the bot itself
                if message.author.bot and message.author.id == self._bot_user_id:
                    return
                asyncio.create_task(self._process_message(message))
            except Exception as e:
                error_log(f"Error parsing MESSAGE_CREATE: {e}")

        elif event_name == "INTERACTION_CREATE":
            try:
                interaction = DiscordInteraction(**data)
                asyncio.create_task(self._process_interaction(interaction))
            except Exception as e:
                error_log(f"Error parsing INTERACTION_CREATE: {e}")

    # ─── Message Processing ──────────────────────────────────────────────

    async def _process_message(self, message: DiscordMessage) -> None:
        """Process an incoming Discord message."""
        try:
            user = message.author
            user_id = user.id
            channel_id = message.channel_id
            guild_id = message.guild_id

            # Ignore bot messages
            if user.bot:
                return

            # Check if it's a DM (no guild_id) or guild message
            is_dm = guild_id is None
            if is_dm and not self.process_dm:
                debug_log("Skipping DM message (process_dm=False)")
                return
            if not is_dm and not self.process_guild_messages:
                debug_log("Skipping guild message (process_guild_messages=False)")
                return

            # Check whitelists
            if not self.is_user_allowed(user_id):
                info_log(self.get_unauthorized_message())
                return
            if not self.is_channel_allowed(channel_id):
                debug_log(f"Skipping message from non-allowed channel {channel_id}")
                return
            if not self.is_guild_allowed(guild_id):
                debug_log(f"Skipping message from non-allowed guild {guild_id}")
                return

            text = message.content or ""

            info_log(f"Processing Discord message from {user_id} in {channel_id} (mode={self.mode.value}): {text[:50]}...")

            if self._auto_heartbeat_channel_id is None:
                self._auto_heartbeat_channel_id = channel_id

            # Check for reset command in CHAT mode
            if self.is_chat_mode() and self.is_reset_command(text):
                await self._handle_reset_command(channel_id, user_id)
                return

            # Send typing indicator
            if self.typing_indicator:
                self._start_typing_indicator(channel_id)

            # Process based on content type
            if message.attachments:
                await self._process_attachment_message(message, user_id, channel_id)
            elif text:
                await self._process_text_message(message, user_id, channel_id)
            else:
                debug_log(f"Unsupported message type from {user_id}")

        except Exception as e:
            import traceback
            error_log(f"Error processing Discord message: {e}\n{traceback.format_exc()}")
        finally:
            # Stop typing indicator
            self._stop_typing_indicator(message.channel_id)

    def _start_typing_indicator(self, channel_id: str) -> None:
        """Start a repeating typing indicator for a channel."""
        async def _typing_loop():
            try:
                while True:
                    await self.discord_tools.atrigger_typing(channel_id)
                    await asyncio.sleep(8)  # Refresh before 10s expiry
            except asyncio.CancelledError:
                pass

        # Cancel any existing typing task for this channel
        self._stop_typing_indicator(channel_id)
        self._typing_tasks[channel_id] = asyncio.create_task(_typing_loop())

    def _stop_typing_indicator(self, channel_id: str) -> None:
        """Stop the typing indicator for a channel."""
        task = self._typing_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()

    async def _handle_reset_command(self, channel_id: str, user_id: str) -> None:
        """Handle reset command in CHAT mode."""
        info_log(f"Reset command received from user {user_id}")

        was_reset = await self.areset_chat_session(user_id)

        if was_reset:
            if self.agent.workspace:
                greeting_result = await self.agent.execute_workspace_greeting_async()
                if greeting_result:
                    reply_text = str(greeting_result)
                else:
                    reply_text = "Your conversation has been reset. I'm ready to start fresh!"
            else:
                reply_text = "Your conversation has been reset. I'm ready to start fresh!"
        else:
            reply_text = "No active conversation found. Send me a message to start!"

        await self.discord_tools.asend_message(
            channel_id=channel_id,
            content=reply_text,
        )

    # ─── Streaming ───────────────────────────────────────────────────────

    async def _stream_to_discord(
        self,
        channel_id: str,
        stream_iterator: AsyncIterator[str],
        reply_to: Optional[str] = None,
    ) -> None:
        """
        Stream agent response to Discord by sending an initial message and
        progressively editing it with accumulated text chunks.

        Discord allows ~5 edits per 5 seconds per message, so updates are
        throttled to ~2 second intervals.
        """
        accumulated_text: str = ""
        sent_message_id: Optional[str] = None
        last_update_time: float = 0.0
        update_interval: float = 2.0

        async for chunk in stream_iterator:
            if not chunk:
                continue

            accumulated_text += chunk

            now = time.monotonic()

            if sent_message_id is None:
                result = await self.discord_tools.asend_message(
                    channel_id=channel_id,
                    content=accumulated_text,
                    reply_to=reply_to,
                )
                if result and isinstance(result, dict):
                    sent_message_id = result.get("id")
                last_update_time = now
            elif now - last_update_time >= update_interval:
                if sent_message_id:
                    await self.discord_tools.aedit_message(
                        channel_id=channel_id,
                        message_id=sent_message_id,
                        content=accumulated_text,
                    )
                last_update_time = now

        # Final update with complete text
        if sent_message_id and accumulated_text:
            await self.discord_tools.aedit_message(
                channel_id=channel_id,
                message_id=sent_message_id,
                content=accumulated_text,
            )
        elif not sent_message_id and accumulated_text:
            await self.discord_tools.asend_message(
                channel_id=channel_id,
                content=accumulated_text,
                reply_to=reply_to,
            )

    # ─── Text Message Processing ─────────────────────────────────────────

    async def _process_text_message(
        self,
        message: DiscordMessage,
        user_id: str,
        channel_id: str,
    ) -> None:
        """Process a text message."""
        text = message.content or ""

        # Skip bot commands that aren't the reset command
        if text.startswith("/") and not self.is_reset_command(text):
            debug_log(f"Received command: {text}")

        if self.is_task_mode():
            await self._process_task_mode(text, user_id, channel_id, message)
        else:
            await self._process_chat_mode(text, user_id, channel_id, message)

    async def _process_task_mode(
        self,
        text: str,
        user_id: str,
        channel_id: str,
        message: DiscordMessage,
    ) -> None:
        """Process message in TASK mode."""
        from upsonic.tasks.tasks import Task

        task = Task(text)

        if self.stream:
            stream_iterator: AsyncIterator[str] = await self.agent.astream(task, events=False)
            await self._stream_to_discord(
                channel_id=channel_id,
                stream_iterator=stream_iterator,
                reply_to=message.id,
            )
            return

        output = await self.agent.do_async(task, return_output=True)
        if output.is_paused and getattr(output, "pause_reason", None) == "confirmation":
            await self._send_confirmation_and_store(
                output, channel_id, user_id, "task", reply_to=message.id,
            )
            return

        response_text: Optional[str] = None
        model_response = output.get_last_model_response() if hasattr(output, "get_last_model_response") else None
        if model_response:
            if hasattr(model_response, "thinking") and model_response.thinking:
                pass
            response_text = getattr(model_response, "text", None)
        if response_text is None and getattr(output, "output", None):
            response_text = str(output.output)
        if response_text:
            await self.discord_tools.asend_message(
                channel_id=channel_id,
                content=response_text,
                reply_to=message.id,
            )

    async def _process_chat_mode(
        self,
        text: str,
        user_id: str,
        channel_id: str,
        message: DiscordMessage,
    ) -> None:
        """Process message in CHAT mode."""
        try:
            chat = await self.aget_chat_session(user_id)

            if self.stream:
                stream_iterator: AsyncIterator[str] = chat.stream(text, events=False)
                await self._stream_to_discord(
                    channel_id=channel_id,
                    stream_iterator=stream_iterator,
                    reply_to=message.id,
                )
                return

            result = await chat.invoke(text, return_run_output=True)
            text_attr = getattr(result, "text", result) if not isinstance(result, str) else result
            run_output = getattr(result, "run_output", None) if not isinstance(result, str) else None
            if (
                run_output is not None
                and getattr(run_output, "is_paused", False)
                and getattr(run_output, "pause_reason", None) == "confirmation"
            ):
                await self._send_confirmation_and_store(
                    run_output, channel_id, user_id, "chat", reply_to=message.id,
                )
                return
            reply_text = text_attr if isinstance(text_attr, str) else str(text_attr)
            if reply_text:
                await self.discord_tools.asend_message(
                    channel_id=channel_id,
                    content=reply_text,
                    reply_to=message.id,
                )
        except Exception as e:
            error_log(f"Error in chat mode: {e}")
            await self.discord_tools.asend_message(
                channel_id=channel_id,
                content="Sorry, I encountered an error. Please try again.",
                reply_to=message.id,
            )

    # ─── Attachment Processing ───────────────────────────────────────────

    IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"}
    AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav", "audio/ogg", "audio/flac", "audio/aac", "audio/m4a"}
    VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/quicktime", "video/webm", "video/x-msvideo"}
    DOCUMENT_TYPES = {
        "application/pdf", "text/plain", "text/csv", "text/html", "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/msword", "application/vnd.ms-excel",
    }

    def _get_media_category(self, mime_type: str) -> str:
        """Get the category of a media type."""
        if not mime_type:
            return "unknown"
        mime_lower = mime_type.lower()
        if mime_lower in self.IMAGE_TYPES or mime_lower.startswith("image/"):
            return "image"
        if mime_lower in self.AUDIO_TYPES or mime_lower.startswith("audio/"):
            return "audio"
        if mime_lower in self.VIDEO_TYPES or mime_lower.startswith("video/"):
            return "video"
        if mime_lower in self.DOCUMENT_TYPES or mime_lower.startswith("application/") or mime_lower.startswith("text/"):
            return "document"
        return "unknown"

    def _get_format_error_message(self, mime_type: str, error: Exception) -> str:
        """Generate a user-friendly error message based on the media type and error."""
        category = self._get_media_category(mime_type)
        error_str = str(error).lower()

        if "wav" in error_str and "mp3" in error_str:
            return (
                f"The audio format '{mime_type}' is not supported by the current AI model.\n\n"
                "This model only accepts WAV and MP3 audio files.\n"
                "Please convert your audio and try again."
            )

        if category == "video":
            return (
                "Video files are not supported by the current AI model.\n\n"
                "Most AI models cannot process video content directly.\n"
                "You can try:\n"
                "- Extracting frames as images\n"
                "- Extracting audio as MP3/WAV\n"
                "- Describing the video content in text"
            )

        if category == "audio":
            return (
                f"Could not process this audio file ({mime_type}).\n\n"
                "The AI model may not support this audio format.\n"
                "Try converting to WAV or MP3 format."
            )

        if category == "image":
            return (
                f"Could not process this image ({mime_type}).\n\n"
                "Try using a common format like JPEG or PNG."
            )

        if category == "document":
            return (
                f"Could not process this document ({mime_type}).\n\n"
                "Try using PDF or plain text format."
            )

        return (
            f"Could not process this file ({mime_type}).\n\n"
            f"Error: {str(error)[:200]}"
        )

    async def _process_attachment_message(
        self,
        message: DiscordMessage,
        user_id: str,
        channel_id: str,
    ) -> None:
        """Process a message with attachments."""
        if not message.attachments:
            return

        caption = message.content or "Process this file"

        for attachment in message.attachments:
            mime_type = attachment.content_type or "application/octet-stream"

            # Download the attachment
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(attachment.url)
                    response.raise_for_status()
                    file_bytes = response.content
            except Exception as e:
                error_log(f"Error downloading attachment: {e}")
                await self.discord_tools.asend_message(
                    channel_id=channel_id,
                    content=f"Failed to download attachment: {attachment.filename}",
                    reply_to=message.id,
                )
                continue

            await self._process_media_with_agent(
                media_bytes=file_bytes,
                mime_type=mime_type,
                caption=caption,
                user_id=user_id,
                channel_id=channel_id,
                message=message,
            )

    async def _process_media_with_agent(
        self,
        media_bytes: bytes,
        mime_type: str,
        caption: str,
        user_id: str,
        channel_id: str,
        message: DiscordMessage,
    ) -> None:
        """Process media content with the agent."""
        import tempfile
        import mimetypes as mimetypes_module
        from upsonic.tasks.tasks import Task

        category = self._get_media_category(mime_type)
        info_log(f"Processing {category} media: {mime_type}")

        extension = mimetypes_module.guess_extension(mime_type) or ".bin"

        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
                prefix="discord_media_",
            )
            temp_file.write(media_bytes)
            temp_file.close()
            temp_path = temp_file.name

            task = Task(
                description=caption,
                attachments=[temp_path],
            )

            if self.is_task_mode():
                try:
                    output = await self.agent.do_async(task, return_output=True)
                    if output.is_paused and getattr(output, "pause_reason", None) == "confirmation":
                        await self._send_confirmation_and_store(
                            output, channel_id, user_id, "task", reply_to=message.id,
                        )
                    else:
                        response_text = None
                        model_response = output.get_last_model_response()
                        if model_response:
                            response_text = model_response.text
                        elif output.output:
                            response_text = str(output.output)
                        if response_text:
                            await self.discord_tools.asend_message(
                                channel_id=channel_id,
                                content=response_text,
                                reply_to=message.id,
                            )
                except Exception as e:
                    error_log(f"Error processing {category} in task mode: {e}")
                    error_message = self._get_format_error_message(mime_type, e)
                    await self.discord_tools.asend_message(
                        channel_id=channel_id,
                        content=error_message,
                        reply_to=message.id,
                    )
            else:
                try:
                    chat = await self.aget_chat_session(user_id)
                    result = await chat.invoke(task, return_run_output=True)
                    text_attr = getattr(result, "text", result) if not isinstance(result, str) else result
                    run_output = getattr(result, "run_output", None) if not isinstance(result, str) else None
                    if (
                        run_output is not None
                        and getattr(run_output, "is_paused", False)
                        and getattr(run_output, "pause_reason", None) == "confirmation"
                    ):
                        await self._send_confirmation_and_store(
                            run_output, channel_id, user_id, "chat", reply_to=message.id,
                        )
                    else:
                        reply_text = text_attr if isinstance(text_attr, str) else str(text_attr)
                        if reply_text:
                            await self.discord_tools.asend_message(
                                channel_id=channel_id,
                                content=reply_text,
                                reply_to=message.id,
                            )
                except Exception as e:
                    error_log(f"Error processing {category} in chat mode: {e}")
                    error_message = self._get_format_error_message(mime_type, e)
                    await self.discord_tools.asend_message(
                        channel_id=channel_id,
                        content=error_message,
                        reply_to=message.id,
                    )
        finally:
            if temp_file and temp_file.name:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    # ─── Interaction Processing ──────────────────────────────────────────

    async def _process_interaction(self, interaction: DiscordInteraction) -> None:
        """Process a Discord interaction (button click, slash command, etc.)."""
        try:
            # Get user info
            user = interaction.user or (interaction.member.user if interaction.member else None)
            if not user:
                return

            user_id = user.id

            if not self.is_user_allowed(user_id):
                info_log(self.get_unauthorized_message())
                # Acknowledge with ephemeral message
                await self.discord_tools.acreate_interaction_response(
                    interaction.id, interaction.token, 4,
                    {"content": self.get_unauthorized_message(), "flags": 64},
                )
                return

            # Handle component interactions (buttons, select menus)
            if interaction.type == 3 and interaction.data:
                custom_id = interaction.data.custom_id or ""

                if custom_id.startswith("cfm:"):
                    # Acknowledge the interaction
                    await self.discord_tools.acreate_interaction_response(
                        interaction.id, interaction.token, 6,  # DEFERRED_UPDATE_MESSAGE
                    )

                    parts = custom_id.split(":")
                    if len(parts) >= 4:
                        pending_key = parts[1]
                        confirmed = parts[3].lower() == "y"
                        state = self._pending_confirmations.pop(pending_key, None)
                        if state is None:
                            channel_id = interaction.channel_id
                            if channel_id:
                                await self.discord_tools.asend_message(
                                    channel_id=channel_id,
                                    content="Confirmation expired. Please start again.",
                                )
                            return

                        run_id = state["run_id"]
                        output = state["output"]
                        channel_id = state["channel_id"]
                        mode = state.get("mode", "task")

                        active = getattr(output, "active_requirements", None) or []
                        first_req = next((r for r in active if getattr(r, "needs_confirmation", False)), None)
                        if first_req:
                            if confirmed:
                                first_req.confirm()
                            else:
                                first_req.reject()

                        requirements = getattr(output, "requirements", None) or []
                        result = await self.agent.continue_run_async(
                            run_id=run_id,
                            requirements=requirements,
                            return_output=True,
                        )

                        if result.is_paused and getattr(result, "pause_reason", None) == "confirmation":
                            await self._send_confirmation_and_store(
                                result, channel_id, user_id, mode,
                            )
                        else:
                            response_text = None
                            model_response = result.get_last_model_response() if hasattr(result, "get_last_model_response") else None
                            if model_response:
                                response_text = getattr(model_response, "text", None)
                            if response_text is None and getattr(result, "output", None):
                                response_text = str(result.output)
                            if response_text:
                                await self.discord_tools.asend_message(
                                    channel_id=channel_id,
                                    content=response_text,
                                )
                    return

                # Generic component interaction
                await self.discord_tools.acreate_interaction_response(
                    interaction.id, interaction.token, 6,
                )

        except Exception as e:
            import traceback
            error_log(f"Error processing Discord interaction: {e}\n{traceback.format_exc()}")

    # ─── Heartbeat (Autonomous Agent) ────────────────────────────────────

    def _resolve_heartbeat_channel_id(self) -> Optional[str]:
        """Resolve the Discord channel ID for heartbeat delivery."""
        if self.heartbeat_channel_id is not None:
            return self.heartbeat_channel_id
        return self._auto_heartbeat_channel_id

    async def _heartbeat_loop(self) -> None:
        """Background coroutine that periodically executes the agent's heartbeat."""
        from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent

        if not isinstance(self.agent, AutonomousAgent):
            return
        if not self.agent.heartbeat:
            return

        period_seconds: int = self.agent.heartbeat_period * 60

        while True:
            await asyncio.sleep(period_seconds)

            target_channel_id: Optional[str] = self._resolve_heartbeat_channel_id()
            if target_channel_id is None:
                debug_log("Heartbeat tick skipped: no target channel_id known yet")
                continue

            try:
                result: Optional[str] = await self.agent.aexecute_heartbeat()
                if result:
                    await self.discord_tools.asend_message(
                        channel_id=target_channel_id,
                        content=result,
                    )
                    info_log(f"Heartbeat response sent to Discord channel {target_channel_id}")
            except Exception as exc:
                error_log(f"Discord heartbeat error: {exc}")

    def _start_heartbeat(self) -> None:
        """Start the heartbeat background task if conditions are met."""
        from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent

        if not isinstance(self.agent, AutonomousAgent):
            return
        if not self.agent.heartbeat:
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        info_log(
            f"Discord heartbeat started: period={self.agent.heartbeat_period}min, "
            f"channel_id={self.heartbeat_channel_id or '(auto-detect)'}"
        )
