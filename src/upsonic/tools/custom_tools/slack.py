import json
from os import getenv
from typing import Any, Dict, List, Optional

from upsonic.utils.printing import error_log

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    _SLACK_SDK_AVAILABLE = True
except ImportError:
    WebClient = None
    SlackApiError = None
    _SLACK_SDK_AVAILABLE = False


class SlackTools:
    """Comprehensive Slack integration toolkit."""

    def __init__(
        self,
        token: Optional[str] = None,
        markdown: bool = True,
        enable_send_message: bool = True,
        enable_send_message_thread: bool = True,
        enable_list_channels: bool = True,
        enable_get_channel_history: bool = True,
        all: bool = False,
    ):
        """
        Initialize the SlackTools class.
        
        Args:
            token: The Slack API token. Defaults to the SLACK_TOKEN environment variable.
            markdown: Whether to enable Slack markdown formatting. Defaults to True.
            enable_send_message: Whether to enable the send_message tool. Defaults to True.
            enable_send_message_thread: Whether to enable the send_message_thread tool. Defaults to True.
            enable_list_channels: Whether to enable the list_channels tool. Defaults to True.
            enable_get_channel_history: Whether to enable the get_channel_history tool. Defaults to True.
            all: Whether to enable all tools. Defaults to False.
        """
        if not _SLACK_SDK_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="slack_sdk",
                install_command='pip install slack-sdk',
                feature_name="Slack tools"
            )

        self.token: Optional[str] = token or getenv("SLACK_TOKEN")
        if self.token is None or self.token == "":
            raise ValueError("SLACK_TOKEN is not set")
        self.client = WebClient(token=self.token)
        self.markdown = markdown

        self._tools: List[Any] = []
        if enable_send_message or all:
            self._tools.append(self.send_message)
        if enable_send_message_thread or all:
            self._tools.append(self.send_message_thread)
        if enable_list_channels or all:
            self._tools.append(self.list_channels)
        if enable_get_channel_history or all:
            self._tools.append(self.get_channel_history)

    def send_message(self, channel: str, text: str) -> str:
        """
        Send a message to a Slack channel.

        Args:
            channel (str): The channel ID or name to send the message to.
            text (str): The text of the message to send.

        Returns:
            str: A JSON string containing the response from the Slack API.
        """
        try:
            response = self.client.chat_postMessage(channel=channel, text=text, mrkdwn=self.markdown)
            return json.dumps(response.data)
        except SlackApiError as e:
            error_log(f"Error sending message: {e}")
            return json.dumps({"error": str(e)})

    def send_message_thread(self, channel: str, text: str, thread_ts: str) -> str:
        """
        Send a message to a Slack channel.

        Args:
            channel (str): The channel ID or name to send the message to.
            text (str): The text of the message to send.
            thread_ts (ts): The thread to reply to.

        Returns:
            str: A JSON string containing the response from the Slack API.
        """
        try:
            response = self.client.chat_postMessage(
                channel=channel, text=text, thread_ts=thread_ts, mrkdwn=self.markdown
            )
            return json.dumps(response.data)
        except SlackApiError as e:
            error_log(f"Error sending message: {e}")
            return json.dumps({"error": str(e)})

    def list_channels(self) -> str:
        """
        List all channels in the Slack workspace.

        Returns:
            str: A JSON string containing the list of channels.
        """
        try:
            response = self.client.conversations_list()
            channels = [{"id": channel["id"], "name": channel["name"]} for channel in response["channels"]]
            return json.dumps(channels)
        except SlackApiError as e:
            error_log(f"Error listing channels: {e}")
            return json.dumps({"error": str(e)})

    def get_channel_history(self, channel: str, limit: int = 100) -> str:
        """
        Get the message history of a Slack channel.

        Args:
            channel (str): The channel ID to fetch history from.
            limit (int): The maximum number of messages to fetch. Defaults to 100.

        Returns:
            str: A JSON string containing the channel's message history.
        """
        try:
            response = self.client.conversations_history(channel=channel, limit=limit)
            messages: List[Dict[str, Any]] = [  # type: ignore
                {
                    "text": msg.get("text", ""),
                    "user": "webhook" if msg.get("subtype") == "bot_message" else msg.get("user", "unknown"),
                    "ts": msg.get("ts", ""),
                    "sub_type": msg.get("subtype", "unknown"),
                    "attachments": msg.get("attachments", []) if msg.get("subtype") == "bot_message" else "n/a",
                }
                for msg in response.get("messages", [])
            ]
            return json.dumps(messages)
        except SlackApiError as e:
            error_log(f"Error getting channel history: {e}")
            return json.dumps({"error": str(e)})

    def functions(self) -> List:
        """Return the list of tool functions to be used by the agent."""
        return self._tools

    def enable_all_tools(self) -> None:
        """Enable all available Slack tools."""
        self._tools = [
            self.send_message,
            self.send_message_thread,
            self.list_channels,
            self.get_channel_history,
        ]