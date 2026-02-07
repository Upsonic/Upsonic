import base64
import binascii
import hashlib
import hmac
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse

from upsonic.interfaces.base import Interface
from upsonic.interfaces.schemas import InterfaceMode
from upsonic.interfaces.whatsapp.schemas import WhatsAppWebhookPayload
from upsonic.utils.printing import debug_log, error_log, info_log
from upsonic.utils.integrations.whatsapp import (
    get_media_async,
    send_image_message_async,
    typing_indicator_async,
    upload_media_async,
)
from upsonic.tools.custom_tools.whatsapp import WhatsAppTools

if TYPE_CHECKING:
    from upsonic.agent import Agent
    from upsonic.storage.base import Storage


class WhatsAppInterface(Interface):
    """
    WhatsApp Business API interface for the Upsonic framework.
    
    This interface handles:
    - Webhook verification for Meta/Facebook
    - Incoming message processing (text, image, audio, video, document)
    - Outgoing message sending with image output support
    - Agent integration for automatic responses
    - Security validation (webhook signatures)
    - Full support for multiple image outputs from agent
    
    Supports two operating modes:
    - TASK: Each message is processed as an independent task (default)
    - CHAT: Messages from the same sender continue a conversation session.
            Sending "/reset" resets the conversation.
    
    Supports whitelist-based access control:
    - Only messages from allowed_numbers can interact with the agent
    - Unauthorized numbers receive "This operation not allowed"
    
    Attributes:
        agent: The AI agent that processes messages
        whatsapp_tools: WhatsApp API toolkit for sending messages
        verify_token: Token for webhook verification
        app_secret: App secret for webhook signature validation
        mode: Operating mode (TASK or CHAT)
        allowed_numbers: Set of allowed phone numbers (whitelist)
    """
    
    def __init__(
        self,
        agent: "Agent",
        verify_token: Optional[str] = None,
        app_secret: Optional[str] = None,
        name: str = "WhatsApp",
        mode: Union[InterfaceMode, str] = InterfaceMode.TASK,
        reset_command: Optional[str] = "/reset",
        storage: Optional["Storage"] = None,
        allowed_numbers: Optional[List[str]] = None,
    ):
        """
        Initialize the WhatsApp interface.
        
        Args:
            agent: The AI agent to process messages
            verify_token: WhatsApp webhook verification token (or set WHATSAPP_VERIFY_TOKEN)
            app_secret: WhatsApp app secret for signature validation (or set WHATSAPP_APP_SECRET)
            name: Interface name (defaults to "WhatsApp")
            mode: Operating mode - TASK for independent tasks, CHAT for conversation sessions.
                  Can be InterfaceMode enum or string ("task" or "chat").
            reset_command: Command string to reset chat session (only applies in CHAT mode).
                          Set to None to disable reset command. Default: "/reset"
            storage: Optional storage backend for chat sessions.
            allowed_numbers: List of allowed phone numbers. If provided, only messages from
                            these numbers will be processed. Others receive "This operation not allowed".
                            Numbers should be in international format (e.g., "905551234567").
                            If None, all numbers are allowed.
        """
        super().__init__(
            agent=agent,
            name=name,
            mode=mode,
            reset_command=reset_command,
            storage=storage,
        )
        
        # Initialize WhatsApp tools for sending messages
        self.whatsapp_tools = WhatsAppTools()
        
        # Webhook verification token
        self.verify_token = verify_token or os.getenv("WHATSAPP_VERIFY_TOKEN")
        if not self.verify_token:
            error_log(
                "WHATSAPP_VERIFY_TOKEN not set. Webhook verification will fail. "
                "Please set the WHATSAPP_VERIFY_TOKEN environment variable."
            )
        
        # App secret for webhook signature validation
        self.app_secret = app_secret or os.getenv("WHATSAPP_APP_SECRET")
        if not self.app_secret:
            debug_log(
                "WHATSAPP_APP_SECRET not set. Webhook signature validation will be skipped. "
                "Set the WHATSAPP_APP_SECRET environment variable for production security."
            )
        
        # Whitelist: allowed phone numbers (normalized - only digits)
        self._allowed_numbers: Optional[Set[str]] = None
        if allowed_numbers is not None:
            self._allowed_numbers = {self._normalize_phone_number(num) for num in allowed_numbers}
            info_log(f"WhatsApp whitelist enabled with {len(self._allowed_numbers)} allowed number(s)")
        
        info_log(f"WhatsApp interface initialized: mode={self.mode.value}, agent={agent}")
    
    def _normalize_phone_number(self, phone: str) -> str:
        """
        Normalize a phone number by removing all non-digit characters.
        
        Args:
            phone: Phone number in any format
            
        Returns:
            str: Phone number with only digits
        """
        return re.sub(r'\D', '', phone)
    
    def is_number_allowed(self, phone: str) -> bool:
        """
        Check if a phone number is allowed to interact with the agent.
        
        Args:
            phone: Phone number to check (WhatsApp ID format)
            
        Returns:
            bool: True if allowed or no whitelist configured, False otherwise
        """
        if self._allowed_numbers is None:
            return True
        
        normalized = self._normalize_phone_number(phone)
        return normalized in self._allowed_numbers
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the WhatsApp interface.
        
        Returns:
            Dict[str, Any]: Health status
        """
        status_data = {
            "status": "active",
            "name": self.name,
            "id": self.id,
            "configuration": {
                "verify_token_configured": bool(self.verify_token),
                "app_secret_configured": bool(self.app_secret),
                "mode": self.mode.value,
                "reset_command": self._reset_command.command if self._reset_enabled else None,
                "active_chat_sessions": len(self._chat_sessions) if self.is_chat_mode() else 0,
                "whitelist_enabled": self._allowed_numbers is not None,
                "allowed_numbers_count": len(self._allowed_numbers) if self._allowed_numbers else 0,
            }
        }
        
        if not self.verify_token:
            status_data["status"] = "degraded"
            status_data["issues"] = ["WHATSAPP_VERIFY_TOKEN is missing"]
            
        return status_data
    
    def _validate_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Validate the webhook signature from Meta/Facebook.
        
        Args:
            payload: Raw request body bytes
            signature: X-Hub-Signature-256 header value
            
        Returns:
            bool: True if signature is valid or validation is disabled, False otherwise
        """
        # Skip validation if app secret is not configured
        if not self.app_secret:
            debug_log("Webhook signature validation skipped (no app secret configured)")
            return True
        
        # Signature should start with "sha256="
        if not signature.startswith("sha256="):
            error_log("Invalid signature format (missing sha256= prefix)")
            return False
        
        # Extract the signature hash
        signature_hash = signature[7:]  # Remove "sha256=" prefix
        
        # Calculate expected signature
        expected_signature = hmac.new(
            key=self.app_secret.encode(),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison to prevent timing attacks
        is_valid = hmac.compare_digest(signature_hash, expected_signature)
        
        if not is_valid:
            error_log("Webhook signature validation failed")
        
        return is_valid
    
    async def _send_whatsapp_message(
        self,
        recipient: str,
        message: str,
        italics: bool = False
    ):
        """
        Send a WhatsApp text message with support for long messages and formatting.
        
        Args:
            recipient: Recipient's WhatsApp ID
            message: Message text to send
            italics: Whether to format message in italics
        """
        if len(message) <= 4096:
            if italics:
                # Handle multi-line messages by making each line italic
                formatted_message = "\n".join([f"_{line}_" for line in message.split("\n")])
                await self.whatsapp_tools.send_text_message(
                    recipient=recipient,
                    text=formatted_message
                )
            else:
                await self.whatsapp_tools.send_text_message(
                    recipient=recipient,
                    text=message
                )
            return
        
        # Split message into batches of 4000 characters (WhatsApp message limit is 4096)
        message_batches = [message[i : i + 4000] for i in range(0, len(message), 4000)]
        
        # Add a prefix with the batch number
        for i, batch in enumerate(message_batches, 1):
            batch_message = f"[{i}/{len(message_batches)}] {batch}"
            if italics:
                # Handle multi-line messages by making each line italic
                formatted_batch = "\n".join([f"_{line}_" for line in batch_message.split("\n")])
                await self.whatsapp_tools.send_text_message(
                    recipient=recipient,
                    text=formatted_batch
                )
            else:
                await self.whatsapp_tools.send_text_message(
                    recipient=recipient,
                    text=batch_message
                )
    
    async def _process_image_outputs(
        self,
        recipient: str
    ):
        """
        Process and send all image outputs from agent response using run_result.
        
        Args:
            recipient: Recipient's WhatsApp ID
        """
        # Get ModelResponse from run_result only
        run_result = self.agent.get_run_output()
        if not run_result:
            return False
        
        # Get the last ModelResponse from run_result
        model_response = run_result.get_last_model_response()
        
        if not model_response or not hasattr(model_response, 'images'):
            return False
        
        images = model_response.images
        if not images:
            return False
        
        number_of_images = len(images)
        info_log(f"Processing {number_of_images} generated image(s) for {recipient}")
        
        for i in range(number_of_images):
            image_content = images[i].data
            
            # Convert image content to bytes
            image_bytes = None
            if isinstance(image_content, bytes):
                try:
                    # Try to decode as base64 string first
                    decoded_string = image_content.decode("utf-8")
                    image_bytes = base64.b64decode(decoded_string)
                except (UnicodeDecodeError, binascii.Error):
                    # If decoding fails, use bytes directly
                    image_bytes = image_content
            elif isinstance(image_content, str):
                # Assume it's base64 encoded string
                image_bytes = base64.b64decode(image_content)
            else:
                error_log(
                    f"Unexpected image content type: {type(image_content)} for user {recipient}"
                )
                continue
            
            if image_bytes:
                try:
                    # Upload image to WhatsApp
                    media_id = await upload_media_async(
                        media_data=image_bytes,
                        mime_type="image/png",
                        filename="image.png"
                    )
                    
                    # Send image (text content will be sent separately after all images)
                    await send_image_message_async(
                        media_id=media_id,
                        recipient=recipient,
                        text=None  # Text sent separately
                    )
                except Exception as e:
                    error_log(f"Failed to upload/send image {i+1}/{number_of_images}: {e}")
                    # Continue with next image
                    continue
            else:
                error_log(
                    f"Could not process image content for user {recipient}. Type: {type(image_content)}"
                )
        
        return True
    
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
    
    async def _process_message_with_agent(
        self,
        message_text: str,
        sender: str,
        media_attachments: Optional[Dict[str, Any]] = None
    ):
        """
        Process a message with the agent and handle response (text and images).
        
        Uses temporary file approach for media attachments to work with Task API.
        Attempts to process all media types and catches errors gracefully,
        providing user-friendly error messages when the model doesn't support
        a particular format.
        
        Args:
            message_text: Text content of the message
            sender: Sender's WhatsApp ID
            media_attachments: Optional dict with media attachments (image, video, audio, document)
        """
        import tempfile
        import mimetypes
        import os as os_module
        
        temp_files: List[str] = []
        media_mime_types: List[str] = []
        
        try:
            from upsonic.tasks.tasks import Task
            
            # Add media attachments if provided (save to temp files)
            if media_attachments:
                for media_type, media_data in media_attachments.items():
                    if media_data:
                        media_bytes = media_data.get("bytes")
                        mime_type = media_data.get("mime_type", f"{media_type}/mp4")
                        
                        if media_bytes:
                            category = self._get_media_category(mime_type)
                            info_log(f"Processing {category} media from {sender}: {mime_type}")
                            media_mime_types.append(mime_type)
                            
                            # Determine file extension from mime type
                            extension = mimetypes.guess_extension(mime_type) or ".bin"
                            
                            # Create temp file
                            temp_file = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=extension,
                                prefix="whatsapp_media_"
                            )
                            temp_file.write(media_bytes)
                            temp_file.close()
                            temp_files.append(temp_file.name)
            
            # Create task with message text and temp file attachments
            task = Task(
                description=message_text,
                attachments=temp_files if temp_files else None
            )
            
            # Execute with agent
            try:
                await self.agent.do_async(task)
            except Exception as agent_error:
                # If there was media, provide a specific error message
                if media_mime_types:
                    error_message = self._get_format_error_message(media_mime_types[0], agent_error)
                    await self._send_whatsapp_message(recipient=sender, message=error_message)
                    return
                raise
            
            # Get ModelResponse from run_result only
            run_result = self.agent.get_run_output()
            if not run_result:
                error_log("No run_result available after agent execution")
                return
            
            # Get the last ModelResponse from run_result
            model_response = run_result.get_last_model_response()
            
            if not model_response:
                error_log("No ModelResponse found in run_result messages")
                # Fallback to output if available
                if run_result.output:
                    await self._send_whatsapp_message(
                        recipient=sender,
                        message=str(run_result.output)
                    )
                return
            
            # Handle reasoning content if available (from ModelResponse)
            if hasattr(model_response, 'thinking') and model_response.thinking:
                await self._send_whatsapp_message(
                    recipient=sender,
                    message=f"Reasoning: \n{model_response.thinking}",
                    italics=True
                )
            
            # Process and send images if any
            has_images = await self._process_image_outputs(sender)
            
            # Get text content from ModelResponse
            text_content = None
            if hasattr(model_response, 'text') and model_response.text:
                text_content = model_response.text
            elif run_result.output:
                # Fallback to run_result output
                text_content = str(run_result.output)
            
            # Send text content (after images if any)
            if text_content:
                await self._send_whatsapp_message(recipient=sender, message=text_content)
            
        except Exception as e:
            import traceback
            error_log(f"Error processing message with agent: {e}\n{traceback.format_exc()}")
            raise
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                try:
                    os_module.unlink(temp_path)
                except OSError:
                    pass
    
    async def _process_text_message(
        self,
        message: Dict[str, Any],
        sender: str,
        message_id: str
    ):
        """
        Process incoming text message and send agent response.
        
        Handles both TASK and CHAT modes:
        - TASK: Each message is an independent task
        - CHAT: Messages continue a conversation session
        
        Args:
            message: Message data from webhook
            sender: Sender's WhatsApp ID
            message_id: Message ID
        """
        try:
            message_text = message.get("text", {}).get("body", "")
            if not message_text:
                debug_log(f"Empty text message from {sender}")
                return
            
            info_log(f"Processing text message from {sender} (mode={self.mode.value}): {message_text[:100]}...")
            
            # Check whitelist - if sender is not allowed, skip processing
            if not self.is_number_allowed(sender):
                info_log(self.get_unauthorized_message())
                return
            
            # Check for reset command in CHAT mode
            if self.is_chat_mode() and self.is_reset_command(message_text):
                await self._handle_reset_command(sender)
                return
            
            # Show typing indicator
            await typing_indicator_async(message_id)
            
            # Process based on mode
            if self.is_task_mode():
                await self._process_message_with_agent(
                    message_text=message_text,
                    sender=sender
                )
            else:
                await self._process_message_chat_mode(
                    message_text=message_text,
                    sender=sender
                )
            
        except Exception as e:
            import traceback
            error_log(f"Error processing text message from {sender}: {e}\n{traceback.format_exc()}")
            
            # Send error message to user
            try:
                error_msg = "Sorry, there was an error processing your message. Please try again later."
                if self.is_chat_mode():
                    error_msg += " Send '/reset' to start a new conversation."
                await self._send_whatsapp_message(
                    recipient=sender,
                    message=error_msg
                )
            except Exception as send_error:
                error_log(f"Error sending error message: {str(send_error)}")
    
    async def _handle_reset_command(self, sender: str) -> None:
        """
        Handle a reset command in CHAT mode.
        
        Args:
            sender: Sender's WhatsApp ID
        """
        info_log(f"Reset command received from {sender}")
        
        # Reset the chat session
        was_reset = await self.areset_chat_session(sender)
        
        # Send confirmation
        if was_reset:
            # Check if agent has workspace and execute greeting
            if self.agent.workspace:
                greeting_result = await self.agent.execute_workspace_greeting_async()
                if greeting_result:
                    reply_text = str(greeting_result)
                else:
                    reply_text = (
                        "✅ Your conversation has been reset. "
                        "I'm ready to start fresh! How can I help you?"
                    )
            else:
                reply_text = (
                    "✅ Your conversation has been reset. "
                    "I'm ready to start fresh! How can I help you?"
                )
        else:
            reply_text = (
                "No active conversation found to reset. "
                "Send me a message to start a new conversation!"
            )
        
        await self._send_whatsapp_message(recipient=sender, message=reply_text)
        info_log(f"Reset command processed for user {sender}")
    
    async def _process_message_chat_mode(
        self,
        message_text: str,
        sender: str,
        media_attachments: Optional[Dict[str, Any]] = None
    ):
        """
        Process a message in CHAT mode (conversation session).
        
        In CHAT mode, each sender has a persistent conversation session.
        Messages are accumulated and the agent has access to the full history.
        
        Uses temporary file approach for media attachments to work with Task API.
        Validates media types before processing to prevent unsupported content errors.
        
        Args:
            message_text: Text content of the message
            sender: Sender's WhatsApp ID (used as user_id)
            media_attachments: Optional dict with media attachments
        """
        import tempfile
        import mimetypes
        import os
        
        temp_files: List[str] = []
        media_mime_types: List[str] = []
        
        try:
            # Get or create chat session for this sender
            chat = await self.aget_chat_session(sender)
            
            info_log(f"Processing message in CHAT mode for user {sender}")
            
            # If there are media attachments, save to temp files and use as attachments
            if media_attachments:
                from upsonic.tasks.tasks import Task
                
                for media_type, media_data in media_attachments.items():
                    if media_data:
                        media_bytes = media_data.get("bytes")
                        mime_type = media_data.get("mime_type", f"{media_type}/mp4")
                        
                        if media_bytes:
                            category = self._get_media_category(mime_type)
                            info_log(f"Processing {category} media in chat mode: {mime_type}")
                            media_mime_types.append(mime_type)
                            
                            # Determine file extension from mime type
                            extension = mimetypes.guess_extension(mime_type) or ".bin"
                            
                            # Create temp file
                            temp_file = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=extension,
                                prefix="whatsapp_media_"
                            )
                            temp_file.write(media_bytes)
                            temp_file.close()
                            temp_files.append(temp_file.name)
                
                # Create task with message text and temp file attachments
                task = Task(
                    description=message_text,
                    attachments=temp_files if temp_files else None
                )
                try:
                    response_text = await chat.invoke(task)
                except Exception as agent_error:
                    # If there was media, provide a specific error message
                    if media_mime_types:
                        error_message = self._get_format_error_message(media_mime_types[0], agent_error)
                        await self._send_whatsapp_message(recipient=sender, message=error_message)
                        return
                    raise
            else:
                # Simple text message
                response_text = await chat.invoke(message_text)
            
            if response_text:
                await self._send_whatsapp_message(recipient=sender, message=response_text)
                info_log(f"Sent chat response to {sender}")
            else:
                debug_log(f"No response generated for {sender}")
                
        except Exception as e:
            error_log(f"Error in chat mode processing for {sender}: {e}")
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    async def _process_media_message(
        self,
        message: Dict[str, Any],
        sender: str,
        message_id: str,
        media_type_key: str
    ):
        """
        Generic handler for all media types (image, audio, video, document) using BinaryContent.
        
        Handles both TASK and CHAT modes.
        
        Args:
            message: Message data from webhook
            sender: Sender's WhatsApp ID
            message_id: Message ID
            media_type_key: Key for media type ("image", "audio", "video", "document")
        """
        try:
            # Check whitelist - if sender is not allowed, skip processing
            if not self.is_number_allowed(sender):
                info_log(self.get_unauthorized_message())
                return
            
            media_data = message.get(media_type_key, {})
            media_id = media_data.get("id")
            
            if not media_id:
                error_log(f"No media_id in {media_type_key} message from {sender}")
                return
            
            info_log(f"Processing {media_type_key} message from {sender} (mode={self.mode.value}, media_id: {media_id})")
            
            # Show typing indicator
            await typing_indicator_async(message_id)
            
            # Download the media
            media_bytes = await get_media_async(media_id)
            
            if isinstance(media_bytes, dict) and "error" in media_bytes:
                error_log(f"Failed to download {media_type_key}: {media_bytes['error']}")
                await self._send_whatsapp_message(
                    recipient=sender,
                    message=f"Sorry, I couldn't download your {media_type_key}. Please try again."
                )
                return
            
            # Get caption or default message
            try:
                message_text = media_data.get("caption", "")
            except Exception:
                message_text = ""
            
            # Set default message based on media type
            if not message_text:
                if media_type_key == "image":
                    message_text = "Describe the image"
                elif media_type_key == "video":
                    message_text = "Describe the video"
                elif media_type_key == "audio":
                    message_text = "Reply to audio"
                elif media_type_key == "document":
                    message_text = "Process the document"
                else:
                    message_text = f"Analyze this {media_type_key}"
            
            # Get MIME type
            mime_type = media_data.get("mime_type", f"{media_type_key}/mp4")
            if media_type_key == "image":
                mime_type = media_data.get("mime_type", "image/jpeg")
            elif media_type_key == "audio":
                mime_type = media_data.get("mime_type", "audio/mpeg")
            elif media_type_key == "document":
                mime_type = media_data.get("mime_type", "application/pdf")
            
            media_attachments = {
                media_type_key: {
                    "bytes": media_bytes,
                    "mime_type": mime_type
                }
            }
            
            # Process based on mode
            if self.is_task_mode():
                await self._process_message_with_agent(
                    message_text=message_text,
                    sender=sender,
                    media_attachments=media_attachments
                )
            else:
                await self._process_message_chat_mode(
                    message_text=message_text,
                    sender=sender,
                    media_attachments=media_attachments
                )
            
        except Exception as e:
            import traceback
            error_log(f"Error processing {media_type_key} message from {sender}: {e}\n{traceback.format_exc()}")
            try:
                error_msg = f"Sorry, there was an error processing your {media_type_key}."
                if self.is_chat_mode():
                    error_msg += " Send '/reset' to start a new conversation."
                await self._send_whatsapp_message(
                    recipient=sender,
                    message=error_msg
                )
            except Exception as send_error:
                error_log(f"Error sending error message: {str(send_error)}")
    
    async def _process_message(self, message: Dict[str, Any]):
        """
        Route message to appropriate handler based on type.
        
        Args:
            message: Message data from webhook
        """
        message_type = message.get("type")
        sender = message.get("from")
        message_id = message.get("id")
        
        if not sender or not message_id:
            error_log(f"Invalid message structure: {message}")
            return
        
        debug_log(f"Processing {message_type} message from {sender}")
        
        # Route to appropriate handler
        if message_type == "text":
            await self._process_text_message(message, sender, message_id)
        elif message_type in ["image", "audio", "video", "document"]:
            await self._process_media_message(message, sender, message_id, message_type)
        else:
            debug_log(f"Unsupported message type: {message_type}")
    
    def attach_routes(self) -> APIRouter:
        """
        Create and attach WhatsApp routes to the FastAPI application.
        
        Routes:
            GET /webhook - Webhook verification endpoint
            POST /webhook - Incoming message webhook
            GET /health - Health check endpoint
            
        Returns:
            APIRouter: Router with WhatsApp endpoints
        """
        router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])
        
        @router.get("/webhook", summary="Webhook Verification")
        async def verify_webhook(
            mode: str = Query(..., alias="hub.mode", description="Verification mode"),
            token: str = Query(..., alias="hub.verify_token", description="Verification token"),
            challenge: str = Query(..., alias="hub.challenge", description="Challenge string"),
        ):
            """
            Webhook verification endpoint for Meta/Facebook.
            
            This endpoint is called by Meta to verify the webhook URL.
            It validates the verify token and returns the challenge string.
            
            Query Parameters:
                - hub.mode: Should be "subscribe"
                - hub.verify_token: Must match WHATSAPP_VERIFY_TOKEN
                - hub.challenge: Challenge string to return
            """
            debug_log(f"Webhook verification request - mode: {mode}, token: {token[:10] if len(token) >= 10 else '***'}...")
            
            # Check if verify token is configured
            if not self.verify_token:
                error_log("WHATSAPP_VERIFY_TOKEN is not configured")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="WHATSAPP_VERIFY_TOKEN is not set"
                )
            
            # Validate mode and token
            if mode == "subscribe" and token == self.verify_token:
                if not challenge:
                    error_log("No challenge received in verification request")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No challenge received"
                    )
                
                info_log("Webhook verification successful")
                return PlainTextResponse(content=challenge)
            else:
                error_log(f"Webhook verification failed - invalid token or mode (mode: {mode})")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid verify token or mode"
                )
        
        @router.post("/webhook", summary="Incoming Messages Webhook")
        async def webhook(request: Request, background_tasks: BackgroundTasks):
            """
            Webhook endpoint for incoming WhatsApp messages.
            
            This endpoint receives all incoming messages, statuses, and notifications
            from the WhatsApp Business API. Messages are processed in the background
            to ensure fast webhook response times.
            
            Returns immediately with a 200 status to acknowledge receipt to Meta.
            """
            try:
                # Get raw body for signature validation
                body = await request.body()
                
                # Validate webhook signature if app secret is configured
                signature = request.headers.get("X-Hub-Signature-256", "")
                if not self._validate_webhook_signature(body, signature):
                    error_log("Webhook signature validation failed")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid signature"
                    )
                
                # Parse the JSON payload
                import json
                data = json.loads(body)
                
                # Validate webhook object type
                if data.get("object") != "whatsapp_business_account":
                    debug_log(f"Received non-WhatsApp webhook object: {data.get('object')}")
                    return {"status": "ignored"}
                
                debug_log(f"Received webhook data: {data}")
                
                # Validate and extract webhook data
                payload = WhatsAppWebhookPayload(**data)
                
                # Process messages in background to return quickly
                for entry in payload.entry:
                    for change in entry.changes:
                        value = change.value
                        
                        # Process messages in background
                        if value.messages:
                            for message in value.messages:
                                background_tasks.add_task(
                                    self._process_message,
                                    message
                                )
                        
                        # Log status updates (delivery, read receipts, etc.)
                        if value.statuses:
                            for status_update in value.statuses:
                                debug_log(f"Status update: {status_update}")
                
                # Return immediately to acknowledge receipt
                return {"status": "processing"}
                
            except json.JSONDecodeError as e:
                error_log(f"Failed to parse webhook payload: {e}")
                # Return 200 to prevent Meta from retrying invalid payloads
                return {"status": "invalid_payload"}
            except HTTPException:
                # Re-raise HTTP exceptions (signature validation failures)
                raise
            except Exception as e:
                import traceback
                error_log(f"Error processing webhook: {e}\n{traceback.format_exc()}")
                # Return 200 to prevent Meta from retrying on errors
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
        
        @router.get("/health", summary="Health Check")
        async def health_check_endpoint():
            """Health check endpoint for WhatsApp interface."""
            return await self.health_check()
        
        info_log(f"WhatsApp routes attached with prefix: /whatsapp")
        return router
