"""
Telegram Bot API Toolkit for Upsonic Framework.

This module provides comprehensive Telegram Bot API integration with support for:
- Sending text messages with formatting
- Sending photos, documents, audio, video
- Inline keyboards and reply keyboards
- Typing indicators (chat actions)
- File downloads from Telegram servers
- Webhook management
- Message editing and deletion

Required Environment Variables:
-----------------------------
- TELEGRAM_BOT_TOKEN: Telegram Bot API token from @BotFather
- TELEGRAM_WEBHOOK_SECRET: (Optional) Secret token for webhook validation

How to Get Bot Token:
-------------------
1. Open Telegram and search for @BotFather
2. Send /newbot command
3. Follow the instructions to create a new bot
4. Copy the bot token provided

Example Usage:
    ```python
    from upsonic.tools.custom_tools.telegram import TelegramTools
    
    # Initialize with bot token
    tools = TelegramTools(bot_token="YOUR_BOT_TOKEN")
    
    # Send a message
    await tools.send_message(chat_id=123456789, text="Hello!")
    
    # Send with inline keyboard
    keyboard = {
        "inline_keyboard": [[
            {"text": "Button 1", "callback_data": "btn1"},
            {"text": "Button 2", "callback_data": "btn2"}
        ]]
    }
    await tools.send_message(chat_id=123456789, text="Choose:", reply_markup=keyboard)
    ```
"""

from os import getenv
from typing import Any, Dict, List, Optional, Union

import httpx

from upsonic.utils.printing import error_log


class TelegramTools:
    """
    Telegram Bot API toolkit for sending messages and managing bot operations.
    
    This toolkit provides methods for:
    - Sending text messages with HTML/Markdown formatting
    - Sending media (photos, documents, audio, video, voice)
    - Managing inline and reply keyboards
    - Sending chat actions (typing indicator, etc.)
    - Answering callback queries
    - Downloading files from Telegram servers
    - Managing webhooks
    - Editing and deleting messages
    
    Attributes:
        bot_token: Telegram Bot API token
        parse_mode: Default parse mode for messages
        disable_web_page_preview: Default setting for link previews
        disable_notification: Default setting for silent messages
        protect_content: Default setting for content protection
        max_message_length: Maximum message length before splitting
    """
    
    # Telegram Bot API base URLs
    API_BASE_URL = "https://api.telegram.org/bot{token}"
    FILE_API_URL = "https://api.telegram.org/file/bot{token}"
    
    # Supported chat actions
    CHAT_ACTIONS = [
        "typing", "upload_photo", "record_video", "upload_video",
        "record_voice", "upload_voice", "upload_document", "choose_sticker",
        "find_location", "record_video_note", "upload_video_note"
    ]
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
        disable_web_page_preview: bool = False,
        disable_notification: bool = False,
        protect_content: bool = False,
        max_message_length: int = 4096,
        timeout: float = 30.0,
    ):
        """
        Initialize the Telegram toolkit.
        
        Args:
            bot_token: Telegram Bot API token. If not provided, reads from
                      TELEGRAM_BOT_TOKEN environment variable.
            parse_mode: Default parse mode for messages ("HTML", "Markdown", 
                       "MarkdownV2", or None).
            disable_web_page_preview: Disable link previews by default.
            disable_notification: Send messages silently by default.
            protect_content: Protect messages from forwarding/saving by default.
            max_message_length: Maximum message length before splitting (default: 4096).
            timeout: HTTP request timeout in seconds (default: 30.0).
        """
        self.bot_token = bot_token or getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            error_log(
                "TELEGRAM_BOT_TOKEN not set. Please set the TELEGRAM_BOT_TOKEN "
                "environment variable or pass bot_token to the constructor."
            )
        
        self.parse_mode = parse_mode
        self.disable_web_page_preview = disable_web_page_preview
        self.disable_notification = disable_notification
        self.protect_content = protect_content
        self.max_message_length = max_message_length
        self.timeout = timeout
        
        # HTTP client (lazy initialization)
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Bot info cache
        self._bot_info: Optional[Dict[str, Any]] = None
    
    @property
    def api_url(self) -> str:
        """Get the Telegram Bot API URL."""
        return self.API_BASE_URL.format(token=self.bot_token)
    
    @property
    def file_url(self) -> str:
        """Get the Telegram File API URL."""
        return self.FILE_API_URL.format(token=self.bot_token)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
    
    async def _api_request(
        self,
        method: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to the Telegram Bot API.
        
        Args:
            method: API method name (e.g., "sendMessage")
            data: Request data
            files: Files to upload
            
        Returns:
            API response data or None on error
        """
        if not self.bot_token:
            error_log("Cannot make API request: bot token not configured")
            return None
        
        url = f"{self.api_url}/{method}"
        client = await self._get_client()
        
        try:
            if files:
                response = await client.post(url, data=data, files=files)
            else:
                response = await client.post(url, json=data)
            
            response.raise_for_status()
            result = response.json()
            
            if not result.get("ok"):
                error_log(f"Telegram API error: {result.get('description')}")
                return None
            
            return result.get("result")
            
        except httpx.HTTPStatusError as e:
            error_log(f"Telegram API HTTP error: {e}")
            return None
        except Exception as e:
            error_log(f"Telegram API request failed: {e}")
            return None
    

    
    async def get_me(self) -> Optional[Dict[str, Any]]:
        """
        Get basic information about the bot.
        
        Returns:
            Bot information including id, username, first_name, etc.
        """
        if self._bot_info is None:
            self._bot_info = await self._api_request("getMe")
        return self._bot_info
    

    
    async def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = None,
        disable_web_page_preview: Optional[bool] = None,
        disable_notification: Optional[bool] = None,
        protect_content: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a text message to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat or username.
            text: Text of the message to send.
            parse_mode: Mode for parsing entities ("HTML", "Markdown", "MarkdownV2").
            disable_web_page_preview: Disables link previews.
            disable_notification: Sends message silently.
            protect_content: Protects message from forwarding/saving.
            reply_to_message_id: ID of the original message to reply to.
            reply_markup: Inline keyboard, reply keyboard, or other markup.
            message_thread_id: Unique identifier for the message thread (forum topics).
            
        Returns:
            The sent Message on success, None on failure.
        """
        # Handle long messages by splitting
        if len(text) > self.max_message_length:
            return await self._send_long_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                disable_notification=disable_notification,
                protect_content=protect_content,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            )
        
        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        
        # Apply defaults or overrides
        if parse_mode is not None:
            data["parse_mode"] = parse_mode
        elif self.parse_mode:
            data["parse_mode"] = self.parse_mode
        
        if disable_web_page_preview is not None:
            data["disable_web_page_preview"] = disable_web_page_preview
        elif self.disable_web_page_preview:
            data["disable_web_page_preview"] = True
        
        if disable_notification is not None:
            data["disable_notification"] = disable_notification
        elif self.disable_notification:
            data["disable_notification"] = True
        
        if protect_content is not None:
            data["protect_content"] = protect_content
        elif self.protect_content:
            data["protect_content"] = True
        
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        if reply_markup:
            data["reply_markup"] = reply_markup
        
        return await self._api_request("sendMessage", data)
    
    async def _send_long_message(
        self,
        chat_id: Union[int, str],
        text: str,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Send a long message by splitting it into chunks."""
        chunks: List[str] = []
        remaining = text
        
        while remaining:
            if len(remaining) <= self.max_message_length:
                chunks.append(remaining)
                break
            
            # Find a good split point
            split_point = remaining.rfind("\n", 0, self.max_message_length)
            if split_point == -1:
                split_point = remaining.rfind(" ", 0, self.max_message_length)
            if split_point == -1:
                split_point = self.max_message_length
            
            chunks.append(remaining[:split_point])
            remaining = remaining[split_point:].lstrip()
        
        last_result = None
        for i, chunk in enumerate(chunks):
            prefix = f"[{i+1}/{len(chunks)}] " if len(chunks) > 1 else ""
            result = await self._api_request("sendMessage", {
                "chat_id": chat_id,
                "text": prefix + chunk,
                **{k: v for k, v in kwargs.items() if v is not None}
            })
            if result:
                last_result = result
        
        return last_result
    
    async def send_photo(
        self,
        chat_id: Union[int, str],
        photo: Union[str, bytes],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a photo to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            photo: Photo to send (file_id, URL, or bytes).
            caption: Photo caption.
            parse_mode: Mode for parsing entities in the caption.
            reply_to_message_id: ID of the original message to reply to.
            reply_markup: Inline keyboard or other markup.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message on success.
        """
        data: Dict[str, Any] = {"chat_id": chat_id}
        
        if caption:
            data["caption"] = caption
        if parse_mode or self.parse_mode:
            data["parse_mode"] = parse_mode or self.parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            data["reply_markup"] = reply_markup
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        if isinstance(photo, str):
            data["photo"] = photo
            return await self._api_request("sendPhoto", data)
        else:
            files = {"photo": ("photo.jpg", photo, "image/jpeg")}
            return await self._api_request("sendPhoto", data, files=files)
    
    async def send_document(
        self,
        chat_id: Union[int, str],
        document: Union[str, bytes],
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a document to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            document: Document to send (file_id, URL, or bytes).
            filename: Filename for the document when sending bytes.
            caption: Document caption.
            parse_mode: Mode for parsing entities in the caption.
            reply_to_message_id: ID of the original message to reply to.
            reply_markup: Inline keyboard or other markup.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message on success.
        """
        data: Dict[str, Any] = {"chat_id": chat_id}
        
        if caption:
            data["caption"] = caption
        if parse_mode or self.parse_mode:
            data["parse_mode"] = parse_mode or self.parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            data["reply_markup"] = reply_markup
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        if isinstance(document, str):
            data["document"] = document
            return await self._api_request("sendDocument", data)
        else:
            fname = filename or "document"
            files = {"document": (fname, document, "application/octet-stream")}
            return await self._api_request("sendDocument", data, files=files)
    
    async def send_audio(
        self,
        chat_id: Union[int, str],
        audio: Union[str, bytes],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        duration: Optional[int] = None,
        performer: Optional[str] = None,
        title: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send an audio file to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            audio: Audio file to send (file_id, URL, or bytes).
            caption: Audio caption.
            parse_mode: Mode for parsing entities in the caption.
            duration: Duration of the audio in seconds.
            performer: Performer of the audio.
            title: Title of the audio.
            reply_to_message_id: ID of the original message to reply to.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message on success.
        """
        data: Dict[str, Any] = {"chat_id": chat_id}
        
        if caption:
            data["caption"] = caption
        if parse_mode or self.parse_mode:
            data["parse_mode"] = parse_mode or self.parse_mode
        if duration:
            data["duration"] = duration
        if performer:
            data["performer"] = performer
        if title:
            data["title"] = title
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        if isinstance(audio, str):
            data["audio"] = audio
            return await self._api_request("sendAudio", data)
        else:
            files = {"audio": ("audio.mp3", audio, "audio/mpeg")}
            return await self._api_request("sendAudio", data, files=files)
    
    async def send_video(
        self,
        chat_id: Union[int, str],
        video: Union[str, bytes],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        duration: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a video to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            video: Video to send (file_id, URL, or bytes).
            caption: Video caption.
            parse_mode: Mode for parsing entities in the caption.
            duration: Duration of the video in seconds.
            width: Video width.
            height: Video height.
            reply_to_message_id: ID of the original message to reply to.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message on success.
        """
        data: Dict[str, Any] = {"chat_id": chat_id}
        
        if caption:
            data["caption"] = caption
        if parse_mode or self.parse_mode:
            data["parse_mode"] = parse_mode or self.parse_mode
        if duration:
            data["duration"] = duration
        if width:
            data["width"] = width
        if height:
            data["height"] = height
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        if isinstance(video, str):
            data["video"] = video
            return await self._api_request("sendVideo", data)
        else:
            files = {"video": ("video.mp4", video, "video/mp4")}
            return await self._api_request("sendVideo", data, files=files)
    
    async def send_voice(
        self,
        chat_id: Union[int, str],
        voice: Union[str, bytes],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        duration: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a voice message to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            voice: Voice message to send (file_id, URL, or bytes).
            caption: Voice message caption.
            parse_mode: Mode for parsing entities in the caption.
            duration: Duration of the voice message in seconds.
            reply_to_message_id: ID of the original message to reply to.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message on success.
        """
        data: Dict[str, Any] = {"chat_id": chat_id}
        
        if caption:
            data["caption"] = caption
        if parse_mode or self.parse_mode:
            data["parse_mode"] = parse_mode or self.parse_mode
        if duration:
            data["duration"] = duration
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        if isinstance(voice, str):
            data["voice"] = voice
            return await self._api_request("sendVoice", data)
        else:
            files = {"voice": ("voice.ogg", voice, "audio/ogg")}
            return await self._api_request("sendVoice", data, files=files)
    
    async def send_location(
        self,
        chat_id: Union[int, str],
        latitude: float,
        longitude: float,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a location to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            latitude: Latitude of the location.
            longitude: Longitude of the location.
            reply_to_message_id: ID of the original message to reply to.
            reply_markup: Inline keyboard or other markup.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message on success.
        """
        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "latitude": latitude,
            "longitude": longitude,
        }
        
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            data["reply_markup"] = reply_markup
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        return await self._api_request("sendLocation", data)
    
    
    async def send_poll(
        self,
        chat_id: Union[int, str],
        question: str,
        options: List[str],
        is_anonymous: bool = True,
        poll_type: str = "regular",
        allows_multiple_answers: bool = False,
        correct_option_id: Optional[int] = None,
        explanation: Optional[str] = None,
        explanation_parse_mode: Optional[str] = None,
        open_period: Optional[int] = None,
        close_date: Optional[int] = None,
        is_closed: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a poll to a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            question: Poll question (1-300 characters).
            options: List of answer options (2-10 strings, 1-100 characters each).
            is_anonymous: True if the poll needs to be anonymous.
            poll_type: Poll type: "quiz" or "regular".
            allows_multiple_answers: True if the poll allows multiple answers (ignored for quizzes).
            correct_option_id: 0-based index of the correct answer option (required for quizzes).
            explanation: Text shown when user selects wrong answer or taps the lamp icon (0-200 chars).
            explanation_parse_mode: Mode for parsing entities in the explanation.
            open_period: Amount of time in seconds the poll will be active (5-600).
            close_date: Unix timestamp when the poll will be automatically closed.
            is_closed: True if the poll needs to be immediately closed.
            reply_to_message_id: ID of the original message to reply to.
            reply_markup: Additional interface options (inline keyboard).
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            The sent Message containing the poll on success.
            
        Example:
            # Create a regular poll
            await tools.send_poll(
                chat_id=123456789,
                question="What's your favorite color?",
                options=["Red", "Blue", "Green", "Yellow"]
            )
            
            # Create a quiz
            await tools.send_poll(
                chat_id=123456789,
                question="What is 2 + 2?",
                options=["3", "4", "5", "6"],
                poll_type="quiz",
                correct_option_id=1,
                explanation="Basic arithmetic: 2 + 2 = 4"
            )
        """
        if len(options) < 2 or len(options) > 10:
            error_log("Poll must have between 2 and 10 options")
            return None
        
        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "question": question,
            "options": options,
            "is_anonymous": is_anonymous,
            "type": poll_type,
        }
        
        if poll_type == "regular":
            data["allows_multiple_answers"] = allows_multiple_answers
        elif poll_type == "quiz":
            if correct_option_id is not None:
                data["correct_option_id"] = correct_option_id
            if explanation:
                data["explanation"] = explanation
                if explanation_parse_mode or self.parse_mode:
                    data["explanation_parse_mode"] = explanation_parse_mode or self.parse_mode
        
        if open_period:
            data["open_period"] = open_period
        if close_date:
            data["close_date"] = close_date
        if is_closed:
            data["is_closed"] = True
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            data["reply_markup"] = reply_markup
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        return await self._api_request("sendPoll", data)
    
    async def stop_poll(
        self,
        chat_id: Union[int, str],
        message_id: int,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Stop a poll which was sent by the bot.
        
        Args:
            chat_id: Unique identifier for the target chat.
            message_id: Identifier of the original message with the poll.
            reply_markup: New inline keyboard to replace the poll (optional).
            
        Returns:
            The stopped Poll object on success.
        """
        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        
        if reply_markup:
            data["reply_markup"] = reply_markup
        
        return await self._api_request("stopPoll", data)

    
    async def send_chat_action(
        self,
        chat_id: Union[int, str],
        action: str = "typing",
        message_thread_id: Optional[int] = None,
    ) -> bool:
        """
        Send a chat action (typing indicator, etc.).
        
        Args:
            chat_id: Unique identifier for the target chat.
            action: Type of action to broadcast. One of: typing, upload_photo,
                   record_video, upload_video, record_voice, upload_voice,
                   upload_document, choose_sticker, find_location,
                   record_video_note, upload_video_note.
            message_thread_id: Unique identifier for the message thread.
            
        Returns:
            True on success.
        """
        if action not in self.CHAT_ACTIONS:
            action = "typing"
        
        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "action": action,
        }
        
        if message_thread_id:
            data["message_thread_id"] = message_thread_id
        
        result = await self._api_request("sendChatAction", data)
        return result is True
    

    
    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
        url: Optional[str] = None,
        cache_time: int = 0,
    ) -> bool:
        """
        Answer a callback query from an inline keyboard.
        
        Args:
            callback_query_id: Unique identifier for the query.
            text: Text of the notification (0-200 characters).
            show_alert: If True, show an alert instead of a notification.
            url: URL to be opened by the user's client.
            cache_time: Maximum time in seconds to cache the result.
            
        Returns:
            True on success.
        """
        data: Dict[str, Any] = {
            "callback_query_id": callback_query_id,
        }
        
        if text:
            data["text"] = text
        if show_alert:
            data["show_alert"] = True
        if url:
            data["url"] = url
        if cache_time:
            data["cache_time"] = cache_time
        
        result = await self._api_request("answerCallbackQuery", data)
        return result is True
    
    
    async def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basic info about a file and prepare it for downloading.
        
        Args:
            file_id: File identifier to get info about.
            
        Returns:
            File object with file_path for downloading.
        """
        return await self._api_request("getFile", {"file_id": file_id})
    
    async def download_file(self, file_path: str) -> Optional[bytes]:
        """
        Download a file from Telegram servers.
        
        Args:
            file_path: File path from getFile response.
            
        Returns:
            File content as bytes.
        """
        if not self.bot_token:
            return None
        
        url = f"{self.file_url}/{file_path}"
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            error_log(f"Failed to download file: {e}")
            return None
    

    
    async def edit_message_text(
        self,
        text: str,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
        disable_web_page_preview: Optional[bool] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Edit text of a message.
        
        Args:
            text: New text of the message.
            chat_id: Required if inline_message_id is not specified.
            message_id: Required if inline_message_id is not specified.
            inline_message_id: Required if chat_id and message_id are not specified.
            parse_mode: Mode for parsing entities.
            disable_web_page_preview: Disables link previews.
            reply_markup: Inline keyboard.
            
        Returns:
            The edited Message on success.
        """
        data: Dict[str, Any] = {"text": text}
        
        if chat_id and message_id:
            data["chat_id"] = chat_id
            data["message_id"] = message_id
        elif inline_message_id:
            data["inline_message_id"] = inline_message_id
        else:
            error_log("Either chat_id+message_id or inline_message_id is required")
            return None
        
        if parse_mode or self.parse_mode:
            data["parse_mode"] = parse_mode or self.parse_mode
        if disable_web_page_preview is not None:
            data["disable_web_page_preview"] = disable_web_page_preview
        elif self.disable_web_page_preview:
            data["disable_web_page_preview"] = True
        if reply_markup:
            data["reply_markup"] = reply_markup
        
        return await self._api_request("editMessageText", data)
    
    async def edit_message_reply_markup(
        self,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Edit the reply markup of a message.
        
        Args:
            chat_id: Required if inline_message_id is not specified.
            message_id: Required if inline_message_id is not specified.
            inline_message_id: Required if chat_id and message_id are not specified.
            reply_markup: New inline keyboard.
            
        Returns:
            The edited Message on success.
        """
        data: Dict[str, Any] = {}
        
        if chat_id and message_id:
            data["chat_id"] = chat_id
            data["message_id"] = message_id
        elif inline_message_id:
            data["inline_message_id"] = inline_message_id
        else:
            error_log("Either chat_id+message_id or inline_message_id is required")
            return None
        
        if reply_markup:
            data["reply_markup"] = reply_markup
        
        return await self._api_request("editMessageReplyMarkup", data)
    
    async def delete_message(
        self,
        chat_id: Union[int, str],
        message_id: int,
    ) -> bool:
        """
        Delete a message.
        
        Args:
            chat_id: Unique identifier for the target chat.
            message_id: Identifier of the message to delete.
            
        Returns:
            True on success.
        """
        result = await self._api_request("deleteMessage", {
            "chat_id": chat_id,
            "message_id": message_id,
        })
        return result is True
    

    
    async def set_webhook(
        self,
        url: str,
        secret_token: Optional[str] = None,
        max_connections: int = 40,
        allowed_updates: Optional[List[str]] = None,
        drop_pending_updates: bool = False,
        certificate: Optional[bytes] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Set the webhook URL for receiving updates.
        
        Args:
            url: HTTPS URL to send updates to.
            secret_token: Secret token for X-Telegram-Bot-Api-Secret-Token header.
            max_connections: Maximum allowed simultaneous connections (1-100).
            allowed_updates: List of update types to receive.
            drop_pending_updates: Drop all pending updates.
            certificate: Public key certificate (for self-signed).
            ip_address: Fixed IP address for webhook requests.
            
        Returns:
            True on success.
        """
        data: Dict[str, Any] = {
            "url": url,
            "max_connections": max_connections,
        }
        
        if secret_token:
            data["secret_token"] = secret_token
        if allowed_updates:
            data["allowed_updates"] = allowed_updates
        if drop_pending_updates:
            data["drop_pending_updates"] = True
        if ip_address:
            data["ip_address"] = ip_address
        
        if certificate:
            files = {"certificate": ("certificate.pem", certificate, "application/x-pem-file")}
            result = await self._api_request("setWebhook", data, files=files)
        else:
            result = await self._api_request("setWebhook", data)
        
        # Telegram API returns True on success, or None on error
        # result can be True (success) or None (error from _api_request)
        return result is True
    
    async def delete_webhook(self, drop_pending_updates: bool = False) -> bool:
        """
        Delete the webhook.
        
        Args:
            drop_pending_updates: Drop all pending updates.
            
        Returns:
            True on success.
        """
        result = await self._api_request("deleteWebhook", {
            "drop_pending_updates": drop_pending_updates,
        })
        return result is True
    
    async def get_webhook_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current webhook status.
        
        Returns:
            WebhookInfo object with webhook details.
        """
        return await self._api_request("getWebhookInfo")
    
    
    async def get_chat(self, chat_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get up-to-date information about a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            
        Returns:
            Chat object.
        """
        return await self._api_request("getChat", {"chat_id": chat_id})
    
    async def get_chat_member(
        self,
        chat_id: Union[int, str],
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a chat member.
        
        Args:
            chat_id: Unique identifier for the target chat.
            user_id: Unique identifier of the target user.
            
        Returns:
            ChatMember object.
        """
        return await self._api_request("getChatMember", {
            "chat_id": chat_id,
            "user_id": user_id,
        })
    
    async def get_chat_member_count(self, chat_id: Union[int, str]) -> Optional[int]:
        """
        Get the number of members in a chat.
        
        Args:
            chat_id: Unique identifier for the target chat.
            
        Returns:
            Number of members in the chat.
        """
        return await self._api_request("getChatMemberCount", {"chat_id": chat_id})
    

    
    def get_poll_tools(self, chat_id: Union[int, str]) -> List:
        """
        Get poll-related tool functions for the agent to use.
        
        This returns a list of functions that can be added to an agent's tools
        to enable poll creation capabilities.
        
        Args:
            chat_id: The chat ID where polls will be created.
            
        Returns:
            List of tool functions (create_poll, stop_poll).
            
        Usage:
            ```python
            tools = telegram_tools.get_poll_tools(chat_id=123456789)
            agent = Agent(model="openai/gpt-4o", tools=tools)
            ```
        """

        
        async def create_poll(
            question: str,
            options: List[str],
            is_anonymous: bool = True,
            poll_type: str = "regular",
            allows_multiple_answers: bool = False,
        ) -> str:
            """
            Create a new poll in the Telegram chat.
            
            Args:
                question: The poll question (1-300 characters).
                options: List of answer options (2-10 options, each 1-100 characters).
                is_anonymous: Whether the poll is anonymous. Default: True.
                poll_type: Type of poll - "regular" or "quiz". Default: "regular".
                allows_multiple_answers: Allow multiple answers for regular polls. Default: False.
            
            Returns:
                A message indicating the poll was created successfully with the message ID.
            """
            result = await self.send_poll(
                chat_id=chat_id,
                question=question,
                options=options,
                is_anonymous=is_anonymous,
                poll_type=poll_type,
                allows_multiple_answers=allows_multiple_answers,
            )
            if result:
                message_id = result.get("message_id", "unknown")
                return f"Poll created successfully! Message ID: {message_id}"
            return "Failed to create poll."
        
        return [create_poll]