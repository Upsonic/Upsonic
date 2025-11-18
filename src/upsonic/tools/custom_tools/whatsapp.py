from os import getenv
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

import httpx

from upsonic.tools.base import ToolKit
from upsonic.tools import tool
from upsonic.utils.printing import error_log, debug_log

load_dotenv()

class WhatsAppTools(ToolKit):
    """WhatsApp Business API toolkit for sending messages."""

    base_url = "https://graph.facebook.com"

    def __init__(
        self,
        access_token: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        version: Optional[str] = None,
        recipient_waid: Optional[str] = None,
    ):
        """Initialize WhatsApp toolkit.

        Args:
            access_token: WhatsApp Business API access token
            phone_number_id: WhatsApp Business Account phone number ID
            version: API version to use
            recipient_waid: Default recipient WhatsApp ID (optional)
        """
        # Core credentials
        self.access_token = access_token or getenv("WHATSAPP_ACCESS_TOKEN")
        if not self.access_token:
            error_log("WHATSAPP_ACCESS_TOKEN not set. Please set the WHATSAPP_ACCESS_TOKEN environment variable.")

        self.phone_number_id = phone_number_id or getenv("WHATSAPP_PHONE_NUMBER_ID")
        if not self.phone_number_id:
            error_log(
                "WHATSAPP_PHONE_NUMBER_ID not set. Please set the WHATSAPP_PHONE_NUMBER_ID environment variable."
            )

        # Optional default recipient
        self.default_recipient = recipient_waid or getenv("WHATSAPP_RECIPIENT_WAID")

        # API version
        self.version = version or getenv("WHATSAPP_VERSION", "v22.0")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

    def _get_messages_url(self) -> str:
        """Get the messages endpoint URL."""
        return f"{self.base_url}/{self.version}/{self.phone_number_id}/messages"

    async def _send_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message asynchronously using the WhatsApp API.

        Args:
            data: Message data to send

        Returns:
            API response as dictionary
        """
        url = self._get_messages_url()
        headers = self._get_headers()

        debug_log(f"Sending WhatsApp request to URL: {url}")

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)

            response.raise_for_status()
            return response.json()

    @tool
    async def send_text_message(
        self,
        text: str = "",
        recipient: Optional[str] = None,
        preview_url: bool = False,
        recipient_type: str = "individual",
    ) -> str:
        """Send a text message to a WhatsApp user.

        Args:
            text: The text message to send
            recipient: Recipient's WhatsApp ID or phone number (e.g., "+1234567890"). If not provided, uses default_recipient
            preview_url: Whether to generate previews for links in the message
            recipient_type: Type of recipient, defaults to "individual"

        Returns:
            Success message with message ID
        """
        # Use default recipient if none provided
        if recipient is None:
            if not self.default_recipient:
                raise ValueError("No recipient provided and no default recipient set")
            recipient = self.default_recipient
            debug_log(f"Using default recipient: {recipient}")

        debug_log(f"Sending WhatsApp message to {recipient}: {text}")
        debug_log(f"Current config - Phone Number ID: {self.phone_number_id}, Version: {self.version}")

        data = {
            "messaging_product": "whatsapp",
            "recipient_type": recipient_type,
            "to": recipient,
            "type": "text",
            "text": {"preview_url": preview_url, "body": text},
        }

        try:
            response = await self._send_message(data)
            message_id = response.get("messages", [{}])[0].get("id", "unknown")
            return f"Message sent successfully! Message ID: {message_id}"
        except httpx.HTTPStatusError as e:
            error_log(f"Failed to send WhatsApp message: {e}")
            error_log(f"Error response: {e.response.text if hasattr(e, 'response') else 'No response text'}")
            raise
        except Exception as e:
            error_log(f"Unexpected error sending WhatsApp message: {str(e)}")
            raise

    @tool
    async def send_template_message(
        self,
        recipient: Optional[str] = None,
        template_name: str = "",
        language_code: str = "en_US",
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Send a template message to a WhatsApp user.

        Args:
            recipient: Recipient's WhatsApp ID or phone number (e.g., "+1234567890"). If not provided, uses default_recipient
            template_name: Name of the template to use
            language_code: Language code for the template (e.g., "en_US")
            components: Optional list of template components (header, body, buttons)

        Returns:
            Success message with message ID
        """
        # Use default recipient if none provided
        if recipient is None:
            if not self.default_recipient:
                raise ValueError("No recipient provided and no default recipient set")
            recipient = self.default_recipient

        debug_log(f"Sending WhatsApp template message to {recipient}: {template_name}")

        data = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "template",
            "template": {"name": template_name, "language": {"code": language_code}},
        }

        if components:
            data["template"]["components"] = components  # type: ignore[index]

        try:
            response = await self._send_message(data)
            message_id = response.get("messages", [{}])[0].get("id", "unknown")
            return f"Template message sent successfully! Message ID: {message_id}"
        except httpx.HTTPStatusError as e:
            error_log(f"Failed to send WhatsApp template message: {e}")
            raise
        except Exception as e:
            error_log(f"Unexpected error sending WhatsApp template message: {str(e)}")
            raise