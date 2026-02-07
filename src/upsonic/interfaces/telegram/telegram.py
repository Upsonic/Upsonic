"""
Telegram Bot API Interface for the Upsonic Framework.

This module provides a comprehensive Telegram Bot integration with support for:
- All message types (text, photo, video, audio, document, voice, sticker, etc.)
- Inline keyboards and callback queries
- Reply keyboards
- Typing indicators (chat actions)
- File downloads and uploads
- Webhook-based message handling
- Task and Chat modes
- Whitelist-based access control
- Bot commands

Based on the official Telegram Bot API: https://core.telegram.org/bots/api
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from upsonic.interfaces.base import Interface
from upsonic.interfaces.schemas import InterfaceMode
from upsonic.interfaces.telegram.schemas import (
    TelegramWebhookPayload,
    TelegramMessage,
    TelegramCallbackQuery,
)
from upsonic.tools.custom_tools.telegram import TelegramTools
from upsonic.utils.printing import debug_log, error_log, info_log

if TYPE_CHECKING:
    from upsonic.agent import Agent
    from upsonic.storage.base import Storage


class TelegramInterface(Interface):
    """
    Telegram Bot API interface for the Upsonic framework.
    
    This interface provides comprehensive Telegram Bot integration:
    - Webhook-based message handling
    - All message types (text, photo, video, audio, document, voice, sticker, location, etc.)
    - Inline keyboards with callback query support
    - Reply keyboards
    - Typing indicators (chat actions)
    - File downloads from Telegram servers
    - Bot commands support
    - Group and channel message handling
    
    Supports two operating modes:
    - TASK: Each message is processed as an independent task (default)
    - CHAT: Messages from the same user continue a conversation session.
            Sending "/reset" resets the conversation.
    
    Supports whitelist-based access control:
    - Only messages from allowed_user_ids can interact with the agent
    - Unauthorized users are silently ignored (logged only)
    
    Attributes:
        agent: The AI agent that processes messages
        telegram_tools: The Telegram toolkit instance for API calls
        mode: Operating mode (TASK or CHAT)
        allowed_user_ids: Set of allowed Telegram user IDs (whitelist)
        webhook_secret: Optional secret token for webhook validation
    """
    
    def __init__(
        self,
        agent: "Agent",
        bot_token: Optional[str] = None,
        name: str = "Telegram",
        mode: Union[InterfaceMode, str] = InterfaceMode.TASK,
        reset_command: Optional[str] = "/reset",
        storage: Optional["Storage"] = None,
        allowed_user_ids: Optional[List[int]] = None,
        webhook_secret: Optional[str] = None,
        webhook_url: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
        disable_web_page_preview: bool = False,
        disable_notification: bool = False,
        protect_content: bool = False,
        reply_in_groups: bool = True,
        reply_in_channels: bool = False,
        process_edited_messages: bool = False,
        process_callback_queries: bool = True,
        typing_indicator: bool = True,
        max_message_length: int = 4096,
    ):
        """
        Initialize the Telegram interface.
        
        Args:
            agent: The AI agent to process messages
            bot_token: Telegram Bot API token (or set TELEGRAM_BOT_TOKEN env var)
            name: Interface name (defaults to "Telegram")
            mode: Operating mode - TASK for independent tasks, CHAT for conversation sessions.
                  Can be InterfaceMode enum or string ("task" or "chat").
            reset_command: Command to reset chat session (only applies in CHAT mode).
                          Set to None to disable. Default: "/reset"
            storage: Optional storage backend for chat sessions.
            allowed_user_ids: List of allowed Telegram user IDs. If provided, only messages
                             from these users will be processed. Others are silently ignored.
                             If None, all users are allowed.
            webhook_secret: Optional secret token for webhook validation (X-Telegram-Bot-Api-Secret-Token).
                           If set, webhook requests without valid token are rejected.
            webhook_url: Base URL for the webhook (e.g., "https://your-domain.ngrok-free.app").
                        If provided, webhook will be automatically set when attach_routes is called.
                        The full webhook URL will be: {webhook_url}/telegram/webhook
            parse_mode: Default parse mode for messages ("HTML", "Markdown", "MarkdownV2", or None).
            disable_web_page_preview: Disable link previews in messages by default.
            disable_notification: Send messages silently by default.
            protect_content: Protect messages from forwarding and saving by default.
            reply_in_groups: Whether to process messages in groups (default: True).
            reply_in_channels: Whether to process messages in channels (default: False).
            process_edited_messages: Whether to process edited messages (default: False).
            process_callback_queries: Whether to process inline keyboard callbacks (default: True).
            typing_indicator: Whether to send typing indicator before responding (default: True).
            max_message_length: Maximum message length before splitting (default: 4096).
        """
        super().__init__(
            agent=agent,
            name=name,
            mode=mode,
            reset_command=reset_command,
            storage=storage,
        )
        
        # Initialize Telegram tools
        self.telegram_tools = TelegramTools(
            bot_token=bot_token,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            disable_notification=disable_notification,
            protect_content=protect_content,
            max_message_length=max_message_length,
        )
        
        # Webhook secret for validation
        self.webhook_secret = webhook_secret or os.getenv("TELEGRAM_WEBHOOK_SECRET")
        
        # Webhook URL for automatic setup
        self._webhook_url = webhook_url or os.getenv("TELEGRAM_WEBHOOK_URL")
        self._webhook_set = False
        
        # Whitelist: allowed Telegram user IDs
        self._allowed_user_ids: Optional[Set[int]] = None
        if allowed_user_ids is not None:
            self._allowed_user_ids = set(allowed_user_ids)
            info_log(f"Telegram whitelist enabled with {len(self._allowed_user_ids)} allowed user(s)")
        
        # Behavior options
        self.reply_in_groups = reply_in_groups
        self.reply_in_channels = reply_in_channels
        self.process_edited_messages = process_edited_messages
        self.process_callback_queries = process_callback_queries
        self.typing_indicator = typing_indicator
        
        info_log(f"Telegram interface initialized: mode={self.mode.value}, agent={agent}")
    
    def is_user_allowed(self, user_id: int) -> bool:
        """
        Check if a Telegram user ID is allowed to interact with the agent.
        
        Args:
            user_id: Telegram user ID to check
            
        Returns:
            bool: True if allowed or no whitelist configured, False otherwise
        """
        if self._allowed_user_ids is None:
            return True
        return user_id in self._allowed_user_ids
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the Telegram interface."""
        base_health = await super().health_check()
        
        # Check bot connectivity
        bot_info = None
        is_connected = False
        try:
            bot_info = await self.telegram_tools.get_me()
            is_connected = bot_info is not None
        except Exception as e:
            debug_log(f"Bot connectivity check failed: {e}")
        
        base_health["configuration"] = {
            "bot_token_configured": bool(self.telegram_tools.bot_token),
            "webhook_secret_configured": bool(self.webhook_secret),
            "mode": self.mode.value,
            "reset_command": self._reset_command.command if self._reset_enabled else None,
            "active_chat_sessions": len(self._chat_sessions) if self.is_chat_mode() else 0,
            "whitelist_enabled": self._allowed_user_ids is not None,
            "allowed_user_ids_count": len(self._allowed_user_ids) if self._allowed_user_ids else 0,
            "parse_mode": self.telegram_tools.parse_mode,
            "reply_in_groups": self.reply_in_groups,
            "reply_in_channels": self.reply_in_channels,
            "process_edited_messages": self.process_edited_messages,
            "process_callback_queries": self.process_callback_queries,
        }
        
        if bot_info:
            base_health["bot"] = {
                "connected": is_connected,
                "id": bot_info.get("id"),
                "username": bot_info.get("username"),
                "first_name": bot_info.get("first_name"),
                "can_join_groups": bot_info.get("can_join_groups"),
                "can_read_all_group_messages": bot_info.get("can_read_all_group_messages"),
                "supports_inline_queries": bot_info.get("supports_inline_queries"),
            }
        else:
            base_health["bot"] = {"connected": False}
        
        if not self.telegram_tools.bot_token:
            base_health["status"] = "degraded"
            base_health["issues"] = ["TELEGRAM_BOT_TOKEN is missing"]
        
        return base_health
    
    def attach_routes(self) -> APIRouter:
        """
        Create and attach Telegram routes to the FastAPI application.
        
        Routes:
            POST /webhook - Telegram webhook endpoint
            POST /set-webhook - Set webhook URL
            POST /delete-webhook - Delete webhook
            GET /webhook-info - Get webhook info
            GET /health - Health check endpoint
            
        Returns:
            APIRouter: Router with Telegram endpoints
        """
        router = APIRouter(prefix="/telegram", tags=["Telegram"])
        
        @router.post("/webhook", status_code=status.HTTP_200_OK)
        async def webhook(request: Request, background_tasks: BackgroundTasks):
            """
            Telegram webhook endpoint.
            
            Receives updates from Telegram and processes them in the background.
            """
            # Validate webhook secret if configured
            if self.webhook_secret:
                secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
                if secret != self.webhook_secret:
                    error_log("Telegram webhook: Invalid secret token")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid secret token"
                    )
            
            try:
                data = await request.json()
                update = TelegramWebhookPayload(**data)
                
                # Process update in background
                background_tasks.add_task(self._process_update, update)
                
                return {"ok": True}
                
            except Exception as e:
                error_log(f"Telegram webhook error: {e}")
                return {"ok": True}  # Always return OK to Telegram
        
        @router.post("/set-webhook", summary="Set Webhook")
        async def set_webhook_endpoint(
            url: str,
            secret_token: Optional[str] = None,
            drop_pending_updates: bool = False,
        ):
            """Set the webhook URL for this bot."""
            success = await self.telegram_tools.set_webhook(
                url=url,
                secret_token=secret_token or self.webhook_secret,
                drop_pending_updates=drop_pending_updates,
            )
            return {"success": success}
        
        @router.post("/delete-webhook", summary="Delete Webhook")
        async def delete_webhook_endpoint(drop_pending_updates: bool = False):
            """Delete the current webhook."""
            success = await self.telegram_tools.delete_webhook(drop_pending_updates=drop_pending_updates)
            return {"success": success}
        
        @router.get("/webhook-info", summary="Get Webhook Info")
        async def webhook_info_endpoint():
            """Get current webhook status."""
            return await self.telegram_tools.get_webhook_info()
        
        @router.get("/health", summary="Health Check")
        async def health_check_endpoint():
            """Health check endpoint for Telegram interface."""
            return await self.health_check()
        
        # Add startup event to auto-set webhook if configured
        @router.on_event("startup")
        async def auto_set_webhook():
            """Automatically set webhook on startup if webhook_url is configured."""
            await self._auto_set_webhook()
        
        info_log("Telegram routes attached with prefix: /telegram")
        return router
    
    async def _auto_set_webhook(self) -> None:
        """Automatically set the webhook if webhook_url is configured."""
        if self._webhook_url and not self._webhook_set:
            # Ensure URL ends properly
            base_url = self._webhook_url.rstrip("/")
            full_webhook_url = f"{base_url}/telegram/webhook"
            
            info_log(f"Auto-setting webhook to: {full_webhook_url}")
            
            try:
                success = await self.telegram_tools.set_webhook(
                    url=full_webhook_url,
                    secret_token=self.webhook_secret,
                )
                if success:
                    self._webhook_set = True
                    info_log(f"Webhook automatically set to: {full_webhook_url}")
                else:
                    error_log("Failed to auto-set webhook")
            except Exception as e:
                error_log(f"Error auto-setting webhook: {e}")
    
    
    async def _process_update(self, update: TelegramWebhookPayload) -> None:
        """Process an incoming Telegram update."""
        try:
            # Handle message updates
            if update.message:
                await self._process_message(update.message)
            
            # Handle edited messages if enabled
            elif update.edited_message and self.process_edited_messages:
                await self._process_message(update.edited_message, is_edited=True)
            
            # Handle channel posts if enabled
            elif update.channel_post and self.reply_in_channels:
                await self._process_message(update.channel_post)
            
            # Handle edited channel posts if enabled
            elif update.edited_channel_post and self.process_edited_messages and self.reply_in_channels:
                await self._process_message(update.edited_channel_post, is_edited=True)
            
            # Handle callback queries if enabled
            elif update.callback_query and self.process_callback_queries:
                await self._process_callback_query(update.callback_query)
                
        except Exception as e:
            import traceback
            error_log(f"Error processing Telegram update: {e}\n{traceback.format_exc()}")
    
    async def _process_message(
        self,
        message: TelegramMessage,
        is_edited: bool = False,
    ) -> None:
        """Process an incoming Telegram message."""
        chat = message.chat
        user = message.from_user
        
        if not user:
            debug_log("Message without user, skipping")
            return
        
        user_id = user.id
        chat_id = chat.id
        chat_type = chat.type
        
        # Check if we should process this chat type
        if chat_type in ("group", "supergroup") and not self.reply_in_groups:
            debug_log("Skipping group message (reply_in_groups=False)")
            return
        
        if chat_type == "channel" and not self.reply_in_channels:
            debug_log("Skipping channel message (reply_in_channels=False)")
            return
        
        # Check whitelist
        if not self.is_user_allowed(user_id):
            info_log(self.get_unauthorized_message())
            return
        
        # Get message text or caption
        text = message.text or message.caption or ""
        
        info_log(f"Processing Telegram message from {user_id} in {chat_id} (mode={self.mode.value}): {text[:50]}...")
        
        # Check for reset command in CHAT mode
        if self.is_chat_mode() and self.is_reset_command(text):
            await self._handle_reset_command(chat_id, user_id, message.message_thread_id)
            return
        
        # Send typing indicator
        if self.typing_indicator:
            await self.telegram_tools.send_chat_action(chat_id, "typing", message.message_thread_id)
        
        # Determine message type and process accordingly
        if message.text:
            await self._process_text_message(message, user_id, chat_id)
        elif message.photo:
            await self._process_photo_message(message, user_id, chat_id)
        elif message.document:
            await self._process_document_message(message, user_id, chat_id)
        elif message.voice:
            await self._process_voice_message(message, user_id, chat_id)
        elif message.audio:
            await self._process_audio_message(message, user_id, chat_id)
        elif message.video:
            await self._process_video_message(message, user_id, chat_id)
        elif message.video_note:
            await self._process_video_note_message(message, user_id, chat_id)
        elif message.sticker:
            await self._process_sticker_message(message, user_id, chat_id)
        elif message.location:
            await self._process_location_message(message, user_id, chat_id)
        elif message.venue:
            await self._process_venue_message(message, user_id, chat_id)
        elif message.contact:
            await self._process_contact_message(message, user_id, chat_id)
        elif message.poll:
            await self._process_poll_message(message, user_id, chat_id)
        else:
            debug_log(f"Unsupported message type from {user_id}")
    
    async def _handle_reset_command(
        self,
        chat_id: int,
        user_id: int,
        message_thread_id: Optional[int] = None,
    ) -> None:
        """Handle reset command in CHAT mode."""
        info_log(f"Reset command received from user {user_id}")
        
        was_reset = await self.areset_chat_session(str(user_id))
        
        if was_reset:
            # Check if agent has workspace and execute greeting
            if self.agent.workspace:
                greeting_result = await self.agent.execute_workspace_greeting_async()
                if greeting_result:
                    reply_text = str(greeting_result)
                else:
                    reply_text = "✅ Your conversation has been reset. I'm ready to start fresh!"
            else:
                reply_text = "✅ Your conversation has been reset. I'm ready to start fresh!"
        else:
            reply_text = "No active conversation found. Send me a message to start!"
        
        await self.telegram_tools.send_message(
            chat_id=chat_id,
            text=reply_text,
            message_thread_id=message_thread_id,
        )
    
    async def _process_text_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a text message."""
        text = message.text or ""
        
        # Skip bot commands that aren't the reset command
        if text.startswith("/") and not self.is_reset_command(text):
            # You can add custom command handling here
            debug_log(f"Received command: {text}")
        
        # Process based on mode
        if self.is_task_mode():
            await self._process_task_mode(text, user_id, chat_id, message)
        else:
            await self._process_chat_mode(text, user_id, chat_id, message)
    
    async def _process_task_mode(
        self,
        text: str,
        user_id: int,
        chat_id: int,
        message: TelegramMessage,
    ) -> None:
        """Process message in TASK mode."""
        from upsonic.tasks.tasks import Task
        
        task = Task(text)
        await self.agent.do_async(task)
        
        run_result = self.agent.get_run_output()
        if not run_result:
            error_log("No run result from agent")
            return
        
        response_text = None
        model_response = run_result.get_last_model_response()
        
        if model_response:
            if hasattr(model_response, "thinking") and model_response.thinking:
                # Optionally send reasoning
                pass
            response_text = model_response.text
        elif run_result.output:
            response_text = str(run_result.output)
        
        if response_text:
            await self.telegram_tools.send_message(
                chat_id=chat_id,
                text=response_text,
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
    
    async def _process_chat_mode(
        self,
        text: str,
        user_id: int,
        chat_id: int,
        message: TelegramMessage,
    ) -> None:
        """Process message in CHAT mode."""
        try:
            chat = await self.aget_chat_session(str(user_id))
            response_text = await chat.invoke(text)
            
            if response_text:
                await self.telegram_tools.send_message(
                    chat_id=chat_id,
                    text=response_text,
                    reply_to_message_id=message.message_id,
                    message_thread_id=message.message_thread_id,
                )
        except Exception as e:
            error_log(f"Error in chat mode: {e}")
            await self.telegram_tools.send_message(
                chat_id=chat_id,
                text="Sorry, I encountered an error. Please try again.",
                reply_to_message_id=message.message_id,
            )
    
    async def _process_photo_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a photo message."""
        if not message.photo:
            return
        
        # Get the largest photo
        photo = message.photo[-1]
        caption = message.caption or "Describe this image"
        
        # Download the photo
        file_info = await self.telegram_tools.get_file(photo.file_id)
        if not file_info or "file_path" not in file_info:
            return
        
        photo_bytes = await self.telegram_tools.download_file(file_info["file_path"])
        if not photo_bytes:
            return
        
        await self._process_media_with_agent(
            media_bytes=photo_bytes,
            mime_type="image/jpeg",
            caption=caption,
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
    
    async def _process_document_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a document message."""
        if not message.document:
            return
        
        doc = message.document
        caption = message.caption or "Process this document"
        mime_type = doc.mime_type or "application/octet-stream"
        
        file_info = await self.telegram_tools.get_file(doc.file_id)
        if not file_info or "file_path" not in file_info:
            return
        
        doc_bytes = await self.telegram_tools.download_file(file_info["file_path"])
        if not doc_bytes:
            return
        
        await self._process_media_with_agent(
            media_bytes=doc_bytes,
            mime_type=mime_type,
            caption=caption,
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
    
    async def _process_voice_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a voice message.
        
        Note: Telegram voice messages are typically in OGG format.
        The agent will attempt to process it, and if the model doesn't support
        OGG, a helpful error message will be shown to the user.
        """
        if not message.voice:
            return
        
        voice = message.voice
        mime_type = voice.mime_type or "audio/ogg"
        
        file_info = await self.telegram_tools.get_file(voice.file_id)
        if not file_info or "file_path" not in file_info:
            return
        
        voice_bytes = await self.telegram_tools.download_file(file_info["file_path"])
        if not voice_bytes:
            return
        
        await self._process_media_with_agent(
            media_bytes=voice_bytes,
            mime_type=mime_type,
            caption="Transcribe and respond to this voice message",
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
    
    async def _process_audio_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process an audio message."""
        if not message.audio:
            return
        
        audio = message.audio
        caption = message.caption or "Process this audio file"
        mime_type = audio.mime_type or "audio/mpeg"
        
        file_info = await self.telegram_tools.get_file(audio.file_id)
        if not file_info or "file_path" not in file_info:
            return
        
        audio_bytes = await self.telegram_tools.download_file(file_info["file_path"])
        if not audio_bytes:
            return
        
        await self._process_media_with_agent(
            media_bytes=audio_bytes,
            mime_type=mime_type,
            caption=caption,
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
    
    async def _process_video_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a video message."""
        if not message.video:
            return
        
        video = message.video
        caption = message.caption or "Describe this video"
        mime_type = video.mime_type or "video/mp4"
        
        file_info = await self.telegram_tools.get_file(video.file_id)
        if not file_info or "file_path" not in file_info:
            return
        
        video_bytes = await self.telegram_tools.download_file(file_info["file_path"])
        if not video_bytes:
            return
        
        await self._process_media_with_agent(
            media_bytes=video_bytes,
            mime_type=mime_type,
            caption=caption,
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
    
    async def _process_video_note_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a video note (round video) message."""
        if not message.video_note:
            return
        
        video_note = message.video_note
        
        file_info = await self.telegram_tools.get_file(video_note.file_id)
        if not file_info or "file_path" not in file_info:
            return
        
        video_bytes = await self.telegram_tools.download_file(file_info["file_path"])
        if not video_bytes:
            return
        
        await self._process_media_with_agent(
            media_bytes=video_bytes,
            mime_type="video/mp4",
            caption="Describe this video message",
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
    
    async def _process_sticker_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a sticker message."""
        if not message.sticker:
            return
        
        sticker = message.sticker
        emoji = sticker.emoji or "a sticker"
        
        # Process as text with sticker info
        text = f"User sent a sticker: {emoji}"
        
        if self.is_task_mode():
            await self._process_task_mode(text, user_id, chat_id, message)
        else:
            await self._process_chat_mode(text, user_id, chat_id, message)
    
    async def _process_location_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a location message."""
        if not message.location:
            return
        
        loc = message.location
        text = f"User shared a location: Latitude {loc.latitude}, Longitude {loc.longitude}"
        
        if self.is_task_mode():
            await self._process_task_mode(text, user_id, chat_id, message)
        else:
            await self._process_chat_mode(text, user_id, chat_id, message)
    
    async def _process_venue_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a venue message."""
        if not message.venue:
            return
        
        venue = message.venue
        loc = venue.location
        text = (
            f"User shared a venue: {venue.title}\n"
            f"Address: {venue.address}\n"
            f"Location: Latitude {loc.latitude}, Longitude {loc.longitude}"
        )
        
        if self.is_task_mode():
            await self._process_task_mode(text, user_id, chat_id, message)
        else:
            await self._process_chat_mode(text, user_id, chat_id, message)
    
    async def _process_contact_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a contact message."""
        if not message.contact:
            return
        
        contact = message.contact
        text = f"User shared a contact: {contact.first_name} {contact.last_name or ''} ({contact.phone_number})"
        
        if self.is_task_mode():
            await self._process_task_mode(text, user_id, chat_id, message)
        else:
            await self._process_chat_mode(text, user_id, chat_id, message)
    
    async def _process_poll_message(
        self,
        message: TelegramMessage,
        user_id: int,
        chat_id: int,
    ) -> None:
        """Process a poll message with full poll options."""
        if not message.poll:
            return
        
        poll = message.poll
        poll_type = "quiz" if poll.type == "quiz" else "regular poll"
        status = "closed" if poll.is_closed else "open"
        
        # Build options text with indices for agent to reference
        options_text = ""
        if poll.options:
            options_lines = []
            for i, option in enumerate(poll.options):
                vote_info = f" ({option.voter_count} votes)" if option.voter_count > 0 else ""
                options_lines.append(f"  {i}. {option.text}{vote_info}")
            options_text = "\nOptions:\n" + "\n".join(options_lines)
        
        text = (
            f"User shared a {poll_type}: {poll.question}{options_text}\n"
            f"Status: {status}\n"
            f"Total voters: {poll.total_voter_count}\n\n"
            f"Note: If asked to pick an option, respond with your choice. "
            f"Reference the option by its number (0-indexed) or text."
        )
        
        if self.is_task_mode():
            await self._process_task_mode(text, user_id, chat_id, message)
        else:
            await self._process_chat_mode(text, user_id, chat_id, message)
    
    # Known media type categories (for logging/info purposes)
    # Note: Actual support depends on the LLM being used
    IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"}
    AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav", "audio/ogg", "audio/flac", "audio/aac", "audio/m4a"}
    VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/quicktime", "video/webm", "video/x-msvideo"}
    DOCUMENT_TYPES = {
        "application/pdf", "text/plain", "text/csv", "text/html", "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/msword", "application/vnd.ms-excel"
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
        
        # Check for specific format errors
        if "wav" in error_str and "mp3" in error_str:
            return (
                f"⚠️ The audio format '{mime_type}' is not supported by the current AI model.\n\n"
                "This model only accepts WAV and MP3 audio files.\n"
                "Please convert your audio and try again."
            )
        
        if category == "video":
            return (
                "⚠️ Video files are not supported by the current AI model.\n\n"
                "Most AI models cannot process video content directly.\n"
                "You can try:\n"
                "• Extracting frames as images\n"
                "• Extracting audio as MP3/WAV\n"
                "• Describing the video content in text"
            )
        
        if category == "audio":
            return (
                f"⚠️ Could not process this audio file ({mime_type}).\n\n"
                "The AI model may not support this audio format.\n"
                "Try converting to WAV or MP3 format."
            )
        
        if category == "image":
            return (
                f"⚠️ Could not process this image ({mime_type}).\n\n"
                "Try using a common format like JPEG or PNG."
            )
        
        if category == "document":
            return (
                f"⚠️ Could not process this document ({mime_type}).\n\n"
                "Try using PDF or plain text format."
            )
        
        # Generic fallback
        return (
            f"⚠️ Could not process this file ({mime_type}).\n\n"
            f"Error: {str(error)[:200]}"
        )
    
    async def _process_media_with_agent(
        self,
        media_bytes: bytes,
        mime_type: str,
        caption: str,
        user_id: int,
        chat_id: int,
        message: TelegramMessage,
    ) -> None:
        """
        Process media content with the agent.
        
        Uses temporary file approach to pass binary content to the agent,
        which works with both Task and Chat modes.
        
        Attempts to process all media types and catches errors gracefully,
        providing user-friendly error messages when the model doesn't support
        a particular format.
        """
        import tempfile
        import mimetypes as mimetypes_module
        from upsonic.tasks.tasks import Task
        
        category = self._get_media_category(mime_type)
        info_log(f"Processing {category} media: {mime_type}")
        
        # Determine file extension from mime type
        extension = mimetypes_module.guess_extension(mime_type) or ".bin"
        
        # Create a temporary file to store the media
        temp_file = None
        temp_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
                prefix="telegram_media_"
            )
            temp_file.write(media_bytes)
            temp_file.close()
            temp_path = temp_file.name
            
            # Create task with caption as description and temp file as attachment
            task = Task(
                description=caption,
                attachments=[temp_path]
            )
            
            if self.is_task_mode():
                try:
                    await self.agent.do_async(task)
                    
                    run_result = self.agent.get_run_output()
                    if run_result:
                        response_text = None
                        model_response = run_result.get_last_model_response()
                        
                        if model_response:
                            response_text = model_response.text
                        elif run_result.output:
                            response_text = str(run_result.output)
                        
                        if response_text:
                            await self.telegram_tools.send_message(
                                chat_id=chat_id,
                                text=response_text,
                                reply_to_message_id=message.message_id,
                                message_thread_id=message.message_thread_id,
                            )
                except Exception as e:
                    error_log(f"Error processing {category} in task mode: {e}")
                    error_message = self._get_format_error_message(mime_type, e)
                    await self.telegram_tools.send_message(
                        chat_id=chat_id,
                        text=error_message,
                        reply_to_message_id=message.message_id,
                        message_thread_id=message.message_thread_id,
                    )
            else:
                try:
                    chat = await self.aget_chat_session(str(user_id))
                    response_text = await chat.invoke(task)
                    
                    if response_text:
                        await self.telegram_tools.send_message(
                            chat_id=chat_id,
                            text=response_text,
                            reply_to_message_id=message.message_id,
                            message_thread_id=message.message_thread_id,
                        )
                except Exception as e:
                    error_log(f"Error processing {category} in chat mode: {e}")
                    error_message = self._get_format_error_message(mime_type, e)
                    await self.telegram_tools.send_message(
                        chat_id=chat_id,
                        text=error_message,
                        reply_to_message_id=message.message_id,
                        message_thread_id=message.message_thread_id,
                    )
        finally:
            # Clean up the temporary file
            if temp_file and temp_file.name:
                import os
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
    
    async def _process_callback_query(self, callback_query: TelegramCallbackQuery) -> None:
        """Process a callback query from an inline keyboard."""
        user = callback_query.from_user
        user_id = user.id
        
        # Check whitelist
        if not self.is_user_allowed(user_id):
            info_log(self.get_unauthorized_message())
            await self.telegram_tools.answer_callback_query(callback_query.id)
            return
        
        callback_data = callback_query.data or ""
        info_log(f"Processing callback query from {user_id}: {callback_data}")
        
        # Answer the callback query to remove loading state
        await self.telegram_tools.answer_callback_query(callback_query.id)
        
        # Process callback data with agent
        chat_id = callback_query.message.chat.id if callback_query.message else None
        
        if chat_id:
            text = f"User clicked button with data: {callback_data}"
            
            if self.is_task_mode():
                from upsonic.tasks.tasks import Task
                task = Task(text)
                await self.agent.do_async(task)
                
                run_result = self.agent.get_run_output()
                if run_result and run_result.output:
                    await self.telegram_tools.send_message(chat_id=chat_id, text=str(run_result.output))
            else:
                try:
                    chat = await self.aget_chat_session(str(user_id))
                    response_text = await chat.invoke(text)
                    if response_text:
                        await self.telegram_tools.send_message(chat_id=chat_id, text=response_text)
                except Exception as e:
                    error_log(f"Error processing callback in chat mode: {e}")
