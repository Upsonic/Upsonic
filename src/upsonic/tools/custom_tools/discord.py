"""
Discord Bot API Toolkit for Upsonic Framework.

This module provides comprehensive Discord Bot API integration with support for:
- Sending text messages with markdown formatting
- Sending embeds (rich content)
- Sending files and attachments
- Message editing and deletion
- Reactions
- Typing indicators
- Thread creation
- DM channel creation
- Pin/unpin messages
- Guild and channel information

Required Environment Variables:
-----------------------------
- DISCORD_BOT_TOKEN: Discord Bot token from Discord Developer Portal

How to Get Bot Token:
-------------------
1. Go to https://discord.com/developers/applications
2. Create a new application
3. Go to the Bot section
4. Click "Add Bot" and copy the token
5. Enable required intents (MESSAGE CONTENT for reading messages)

Example Usage:
    ```python
    from upsonic.tools.custom_tools.discord import DiscordTools

    # Initialize with bot token
    tools = DiscordTools(bot_token="YOUR_BOT_TOKEN")

    # Send a message
    await tools.send_message(channel_id="123456789", content="Hello!")

    # Send with embed
    embeds = [{"title": "Hello", "description": "World", "color": 5814783}]
    await tools.send_message(channel_id="123456789", embeds=embeds)
    ```
"""

import asyncio
import json
from os import getenv
from typing import Any, Dict, List, Optional, Union

import httpx

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool
from upsonic.utils.async_utils import run_async
from upsonic.utils.integrations.discord import sanitize_text_for_discord
from upsonic.utils.printing import error_log


class DiscordTools(ToolKit):
    """
    Discord Bot API toolkit for sending messages and managing bot operations.

    This toolkit provides methods for:
    - Sending text messages with markdown formatting
    - Sending embeds (rich content)
    - Sending files and attachments
    - Typing indicators
    - Reactions (add/remove)
    - Message editing and deletion
    - Thread creation
    - DM channel management
    - Pin/unpin messages
    - Guild and channel information

    Attributes:
        bot_token: Discord Bot API token
        max_message_length: Maximum message length before splitting (2000)
    """

    # Discord API base URLs
    API_BASE_URL = "https://discord.com/api/v10"
    CDN_BASE_URL = "https://cdn.discordapp.com"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        max_message_length: int = 2000,
        http_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Discord toolkit.

        Args:
            bot_token: Discord Bot API token. If not provided, reads from
                      DISCORD_BOT_TOKEN environment variable.
            max_message_length: Maximum message length before splitting.
            http_timeout: HTTP request timeout in seconds.
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)

        self.bot_token: Optional[str] = bot_token or getenv("DISCORD_BOT_TOKEN")
        if not self.bot_token:
            error_log(
                "DISCORD_BOT_TOKEN not set. Please set the DISCORD_BOT_TOKEN "
                "environment variable or pass bot_token to the constructor."
            )

        self.max_message_length: int = max_message_length
        self.http_timeout: float = http_timeout

        # HTTP client (lazy initialization)
        self._http_client: Optional[httpx.AsyncClient] = None

        # Bot info cache
        self._bot_info: Optional[Dict[str, Any]] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with Discord authorization header."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self.http_timeout,
                headers={
                    "Authorization": f"Bot {self.bot_token}",
                    "User-Agent": "DiscordBot (upsonic, 1.0)",
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _api_request(
        self,
        http_method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        raise_on_client_error: bool = False,
    ) -> Optional[Any]:
        """
        Make a request to the Discord REST API.

        Args:
            http_method: HTTP method ("GET", "POST", "PATCH", "PUT", "DELETE")
            endpoint: API endpoint path (e.g., "/channels/123/messages")
            data: Request JSON body
            files: Files to upload (multipart)
            raise_on_client_error: If True, re-raise HTTP 4xx errors so callers
                can implement retry/fallback logic.

        Returns:
            API response data, or None on error. DELETE requests may return True.
        """
        if not self.bot_token:
            error_log("Cannot make API request: bot token not configured")
            return None

        url = f"{self.API_BASE_URL}{endpoint}"
        client = await self._get_client()

        try:
            if files:
                # Multipart form upload
                payload_json = json.dumps(data) if data else None
                form_data: Dict[str, Any] = {}
                if payload_json:
                    form_data["payload_json"] = payload_json
                response = await client.request(
                    http_method, url, data=form_data, files=files,
                )
            elif data and http_method.upper() in ("POST", "PATCH", "PUT"):
                response = await client.request(http_method, url, json=data)
            else:
                response = await client.request(http_method, url, params=data if http_method.upper() == "GET" else None)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.json().get("retry_after", 1.0)
                from upsonic.utils.printing import debug_log
                debug_log(f"Discord rate limited, retrying after {retry_after}s")
                await asyncio.sleep(float(retry_after))
                return await self._api_request(http_method, endpoint, data, files, raise_on_client_error)

            response.raise_for_status()

            # Some endpoints return 204 No Content
            if response.status_code == 204:
                return True

            return response.json()

        except httpx.HTTPStatusError as e:
            if raise_on_client_error and 400 <= e.response.status_code < 500:
                raise
            error_log(f"Discord API HTTP error: {e}")
            return None
        except Exception as e:
            error_log(f"Discord API request failed: {e}")
            return None

    # ─── Bot Information ─────────────────────────────────────────────────

    @tool
    def get_me(self) -> Optional[Dict[str, Any]]:
        """Get basic information about the bot.

        Returns:
            Bot user information including id, username, avatar, etc.
        """
        return run_async(self.aget_me())

    async def aget_me(self) -> Optional[Dict[str, Any]]:
        if self._bot_info is None:
            self._bot_info = await self._api_request("GET", "/users/@me")
        return self._bot_info

    # ─── Message Sending ─────────────────────────────────────────────────

    @tool
    def send_message(
        self,
        channel_id: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        reply_to: Optional[str] = None,
        tts: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to a Discord channel.

        Supports text content, embeds, and message components (buttons, select menus).
        Long messages are automatically split into chunks.

        Args:
            channel_id: ID of the target channel.
            content: Text content of the message (supports Discord markdown).
            embeds: List of embed objects for rich content.
            components: List of component objects (action rows with buttons, etc.).
            reply_to: Message ID to reply to.
            tts: Whether to send as text-to-speech.

        Returns:
            The sent Message object on success, None on failure.
        """
        return run_async(self.asend_message(
            channel_id=channel_id,
            content=content,
            embeds=embeds,
            components=components,
            reply_to=reply_to,
            tts=tts,
        ))

    async def asend_message(
        self,
        channel_id: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        reply_to: Optional[str] = None,
        tts: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not content and not embeds:
            error_log("send_message called with no content or embeds, skipping")
            return None

        # Handle long messages by splitting
        if content and len(content) > self.max_message_length:
            return await self._send_long_message(
                channel_id=channel_id,
                content=content,
                embeds=embeds,
                components=components,
                reply_to=reply_to,
                tts=tts,
            )

        data: Dict[str, Any] = {}
        if content:
            data["content"] = content
        if embeds:
            data["embeds"] = embeds
        if components:
            data["components"] = components
        if tts:
            data["tts"] = True
        if reply_to:
            data["message_reference"] = {"message_id": reply_to}

        try:
            result = await self._api_request(
                "POST", f"/channels/{channel_id}/messages", data, raise_on_client_error=True,
            )
            return result
        except httpx.HTTPStatusError:
            pass

        # Retry with sanitized text
        if content:
            sanitized_content = sanitize_text_for_discord(content)
            data["content"] = sanitized_content
            from upsonic.utils.printing import debug_log
            debug_log("Discord sendMessage retry: sanitized text")

        return await self._api_request("POST", f"/channels/{channel_id}/messages", data)

    async def _send_long_message(
        self,
        channel_id: str,
        content: str,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Send a long message by splitting it into chunks."""
        chunks: List[str] = []
        remaining = content

        while remaining:
            if len(remaining) <= self.max_message_length:
                chunks.append(remaining)
                break

            split_point = remaining.rfind("\n", 0, self.max_message_length)
            if split_point == -1:
                split_point = remaining.rfind(" ", 0, self.max_message_length)
            if split_point == -1:
                split_point = self.max_message_length

            chunks.append(remaining[:split_point])
            remaining = remaining[split_point:].lstrip()

        last_result = None
        # Only send embeds/components with the last chunk
        embeds = kwargs.pop("embeds", None)
        components = kwargs.pop("components", None)
        reply_to = kwargs.pop("reply_to", None)
        tts = kwargs.pop("tts", False)

        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            prefix = f"[{i+1}/{len(chunks)}] " if len(chunks) > 1 else ""

            data: Dict[str, Any] = {"content": prefix + chunk}
            if tts:
                data["tts"] = True
            if reply_to and i == 0:
                data["message_reference"] = {"message_id": reply_to}
            if is_last and embeds:
                data["embeds"] = embeds
            if is_last and components:
                data["components"] = components

            result = await self._api_request("POST", f"/channels/{channel_id}/messages", data)
            if result:
                last_result = result

        return last_result

    # ─── Message Editing ─────────────────────────────────────────────────

    @tool
    def edit_message(
        self,
        channel_id: str,
        message_id: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Edit a previously sent message.

        Args:
            channel_id: ID of the channel containing the message.
            message_id: ID of the message to edit.
            content: New text content.
            embeds: New embed objects.
            components: New component objects.

        Returns:
            The edited Message object on success, None on failure.
        """
        return run_async(self.aedit_message(
            channel_id=channel_id,
            message_id=message_id,
            content=content,
            embeds=embeds,
            components=components,
        ))

    async def aedit_message(
        self,
        channel_id: str,
        message_id: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        data: Dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if embeds is not None:
            data["embeds"] = embeds
        if components is not None:
            data["components"] = components

        try:
            return await self._api_request(
                "PATCH", f"/channels/{channel_id}/messages/{message_id}",
                data, raise_on_client_error=True,
            )
        except httpx.HTTPStatusError:
            if content:
                data["content"] = sanitize_text_for_discord(content)
            return await self._api_request(
                "PATCH", f"/channels/{channel_id}/messages/{message_id}", data,
            )

    # ─── Message Deletion ────────────────────────────────────────────────

    @tool
    def delete_message(
        self,
        channel_id: str,
        message_id: str,
    ) -> bool:
        """
        Delete a message from a channel.

        Args:
            channel_id: ID of the channel containing the message.
            message_id: ID of the message to delete.

        Returns:
            True on success, False on failure.
        """
        return run_async(self.adelete_message(channel_id=channel_id, message_id=message_id))

    async def adelete_message(self, channel_id: str, message_id: str) -> bool:
        result = await self._api_request("DELETE", f"/channels/{channel_id}/messages/{message_id}")
        return result is not None

    # ─── File Sending ────────────────────────────────────────────────────

    @tool
    def send_file(
        self,
        channel_id: str,
        file_data: bytes,
        filename: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        reply_to: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a file to a Discord channel.

        Args:
            channel_id: ID of the target channel.
            file_data: File content as bytes.
            filename: Name for the file.
            content: Optional text content alongside the file.
            embeds: Optional embed objects.
            reply_to: Optional message ID to reply to.

        Returns:
            The sent Message object on success, None on failure.
        """
        return run_async(self.asend_file(
            channel_id=channel_id,
            file_data=file_data,
            filename=filename,
            content=content,
            embeds=embeds,
            reply_to=reply_to,
        ))

    async def asend_file(
        self,
        channel_id: str,
        file_data: bytes,
        filename: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        reply_to: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        data: Dict[str, Any] = {}
        if content:
            data["content"] = content
        if embeds:
            data["embeds"] = embeds
        if reply_to:
            data["message_reference"] = {"message_id": reply_to}

        files = {"files[0]": (filename, file_data)}
        return await self._api_request(
            "POST", f"/channels/{channel_id}/messages", data=data, files=files,
        )

    # ─── Reactions ───────────────────────────────────────────────────────

    @tool
    def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """
        Add a reaction to a message.

        Args:
            channel_id: ID of the channel containing the message.
            message_id: ID of the message to react to.
            emoji: Emoji to react with. For standard unicode emoji, use the
                   character directly (e.g., "👍"). For custom emoji, use
                   "name:id" format (e.g., "custom_emoji:123456789").

        Returns:
            True on success, False on failure.
        """
        return run_async(self.aadd_reaction(
            channel_id=channel_id, message_id=message_id, emoji=emoji,
        ))

    async def aadd_reaction(self, channel_id: str, message_id: str, emoji: str) -> bool:
        # URL-encode the emoji for the path
        import urllib.parse
        encoded_emoji = urllib.parse.quote(emoji)
        result = await self._api_request(
            "PUT",
            f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded_emoji}/@me",
        )
        return result is not None

    @tool
    def remove_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """
        Remove the bot's reaction from a message.

        Args:
            channel_id: ID of the channel containing the message.
            message_id: ID of the message.
            emoji: Emoji to remove.

        Returns:
            True on success, False on failure.
        """
        return run_async(self.aremove_reaction(
            channel_id=channel_id, message_id=message_id, emoji=emoji,
        ))

    async def aremove_reaction(self, channel_id: str, message_id: str, emoji: str) -> bool:
        import urllib.parse
        encoded_emoji = urllib.parse.quote(emoji)
        result = await self._api_request(
            "DELETE",
            f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded_emoji}/@me",
        )
        return result is not None

    # ─── Typing Indicator ────────────────────────────────────────────────

    @tool
    def trigger_typing(
        self,
        channel_id: str,
    ) -> bool:
        """
        Send a typing indicator to a channel.

        The typing indicator lasts for approximately 10 seconds or until
        a message is sent, whichever comes first. Call repeatedly for
        longer operations.

        Args:
            channel_id: ID of the target channel.

        Returns:
            True on success, False on failure.
        """
        return run_async(self.atrigger_typing(channel_id=channel_id))

    async def atrigger_typing(self, channel_id: str) -> bool:
        result = await self._api_request("POST", f"/channels/{channel_id}/typing")
        return result is not None

    # ─── Channel Information ─────────────────────────────────────────────

    @tool
    def get_channel(
        self,
        channel_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a channel.

        Args:
            channel_id: ID of the channel.

        Returns:
            Channel object on success, None on failure.
        """
        return run_async(self.aget_channel(channel_id=channel_id))

    async def aget_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        return await self._api_request("GET", f"/channels/{channel_id}")

    # ─── Guild Information ───────────────────────────────────────────────

    @tool
    def get_guild(
        self,
        guild_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a guild (server).

        Args:
            guild_id: ID of the guild.

        Returns:
            Guild object on success, None on failure.
        """
        return run_async(self.aget_guild(guild_id=guild_id))

    async def aget_guild(self, guild_id: str) -> Optional[Dict[str, Any]]:
        return await self._api_request("GET", f"/guilds/{guild_id}")

    @tool
    def get_guild_member(
        self,
        guild_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a guild member.

        Args:
            guild_id: ID of the guild.
            user_id: ID of the user.

        Returns:
            Guild member object on success, None on failure.
        """
        return run_async(self.aget_guild_member(guild_id=guild_id, user_id=user_id))

    async def aget_guild_member(self, guild_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._api_request("GET", f"/guilds/{guild_id}/members/{user_id}")

    @tool
    def get_guild_channels(
        self,
        guild_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get all channels in a guild.

        Args:
            guild_id: ID of the guild.

        Returns:
            List of channel objects on success, None on failure.
        """
        return run_async(self.aget_guild_channels(guild_id=guild_id))

    async def aget_guild_channels(self, guild_id: str) -> Optional[List[Dict[str, Any]]]:
        return await self._api_request("GET", f"/guilds/{guild_id}/channels")

    # ─── Message Retrieval ───────────────────────────────────────────────

    @tool
    def get_message(
        self,
        channel_id: str,
        message_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific message from a channel.

        Args:
            channel_id: ID of the channel.
            message_id: ID of the message.

        Returns:
            Message object on success, None on failure.
        """
        return run_async(self.aget_message(channel_id=channel_id, message_id=message_id))

    async def aget_message(self, channel_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        return await self._api_request("GET", f"/channels/{channel_id}/messages/{message_id}")

    @tool
    def get_channel_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: Optional[str] = None,
        after: Optional[str] = None,
        around: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get messages from a channel.

        Args:
            channel_id: ID of the channel.
            limit: Maximum number of messages to return (1-100, default 50).
            before: Get messages before this message ID.
            after: Get messages after this message ID.
            around: Get messages around this message ID.

        Returns:
            List of message objects on success, None on failure.
        """
        return run_async(self.aget_channel_messages(
            channel_id=channel_id, limit=limit,
            before=before, after=after, around=around,
        ))

    async def aget_channel_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: Optional[str] = None,
        after: Optional[str] = None,
        around: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, Any] = {"limit": min(limit, 100)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        if around:
            params["around"] = around
        return await self._api_request("GET", f"/channels/{channel_id}/messages", data=params)

    # ─── Thread Management ───────────────────────────────────────────────

    @tool
    def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None,
        auto_archive_duration: int = 1440,
        thread_type: int = 11,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new thread in a channel.

        Args:
            channel_id: ID of the channel to create the thread in.
            name: Name of the thread (1-100 characters).
            message_id: If provided, creates a thread from this message.
            auto_archive_duration: Duration in minutes to auto-archive (60, 1440, 4320, 10080).
            thread_type: Type of thread (10=announcement, 11=public, 12=private).

        Returns:
            The created Channel (thread) object on success, None on failure.
        """
        return run_async(self.acreate_thread(
            channel_id=channel_id, name=name, message_id=message_id,
            auto_archive_duration=auto_archive_duration, thread_type=thread_type,
        ))

    async def acreate_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None,
        auto_archive_duration: int = 1440,
        thread_type: int = 11,
    ) -> Optional[Dict[str, Any]]:
        data: Dict[str, Any] = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
        }

        if message_id:
            # Create thread from a message
            return await self._api_request(
                "POST", f"/channels/{channel_id}/messages/{message_id}/threads", data,
            )
        else:
            # Create a standalone thread
            data["type"] = thread_type
            return await self._api_request(
                "POST", f"/channels/{channel_id}/threads", data,
            )

    # ─── DM Channel ─────────────────────────────────────────────────────

    @tool
    def create_dm(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a DM channel with a user.

        Must be called before sending a DM to a user. Returns the DM channel
        object which contains the channel_id needed for send_message.

        Args:
            user_id: ID of the user to create a DM channel with.

        Returns:
            DM Channel object on success, None on failure.
        """
        return run_async(self.acreate_dm(user_id=user_id))

    async def acreate_dm(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._api_request("POST", "/users/@me/channels", {"recipient_id": user_id})

    # ─── Pin/Unpin Messages ──────────────────────────────────────────────

    @tool
    def pin_message(
        self,
        channel_id: str,
        message_id: str,
    ) -> bool:
        """
        Pin a message in a channel.

        Args:
            channel_id: ID of the channel containing the message.
            message_id: ID of the message to pin.

        Returns:
            True on success, False on failure.
        """
        return run_async(self.apin_message(channel_id=channel_id, message_id=message_id))

    async def apin_message(self, channel_id: str, message_id: str) -> bool:
        result = await self._api_request("PUT", f"/channels/{channel_id}/pins/{message_id}")
        return result is not None

    @tool
    def unpin_message(
        self,
        channel_id: str,
        message_id: str,
    ) -> bool:
        """
        Unpin a message from a channel.

        Args:
            channel_id: ID of the channel containing the message.
            message_id: ID of the message to unpin.

        Returns:
            True on success, False on failure.
        """
        return run_async(self.aunpin_message(channel_id=channel_id, message_id=message_id))

    async def aunpin_message(self, channel_id: str, message_id: str) -> bool:
        result = await self._api_request("DELETE", f"/channels/{channel_id}/pins/{message_id}")
        return result is not None

    # ─── Interaction Responses ───────────────────────────────────────────

    async def acreate_interaction_response(
        self,
        interaction_id: str,
        interaction_token: str,
        response_type: int,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Respond to a Discord interaction (button click, slash command, etc.).

        Args:
            interaction_id: ID of the interaction.
            interaction_token: Token of the interaction.
            response_type: Type of response (4=message, 6=deferred update, 7=update message).
            data: Response data (content, embeds, components, etc.).

        Returns:
            Response data on success, None on failure.
        """
        payload: Dict[str, Any] = {"type": response_type}
        if data:
            payload["data"] = data
        return await self._api_request(
            "POST",
            f"/interactions/{interaction_id}/{interaction_token}/callback",
            payload,
        )

    async def aedit_interaction_response(
        self,
        application_id: str,
        interaction_token: str,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Edit the original interaction response.

        Args:
            application_id: ID of the application.
            interaction_token: Token of the interaction.
            content: New text content.
            embeds: New embed objects.
            components: New component objects.

        Returns:
            The edited Message object on success, None on failure.
        """
        data: Dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if embeds is not None:
            data["embeds"] = embeds
        if components is not None:
            data["components"] = components
        return await self._api_request(
            "PATCH",
            f"/webhooks/{application_id}/{interaction_token}/messages/@original",
            data,
        )
