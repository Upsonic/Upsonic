"""
Upsonic Interfaces Module

This module provides a comprehensive interface system for integrating AI agents
with external communication platforms like WhatsApp, Slack, Gmail, and more.

Public API:
    - Interface: Base class for custom interfaces
    - InterfaceManager: Central manager for orchestrating interfaces
    - InterfaceMode: Operating mode enum (TASK or CHAT)
    - WhatsAppInterface: WhatsApp Business API integration
    - SlackInterface: Slack API integration
    - GmailInterface: Gmail API integration
    - TelegramInterface: Telegram Bot API integration
    - InterfaceSettings: Configuration settings
    - WebSocketManager: WebSocket connection manager

Operating Modes:
    - TASK: Each message is processed as an independent task (stateless)
    - CHAT: Messages are accumulated in a conversation session (stateful)
            Users can send "/reset" to clear their conversation history.

Example - Task Mode (default):
    ```python
    from upsonic import Agent
    from upsonic.interfaces import InterfaceManager, WhatsAppInterface
    
    agent = Agent("openai/gpt-4o")
    
    # Task mode - each message is independent
    whatsapp = WhatsAppInterface(agent=agent, mode="task")
    
    manager = InterfaceManager(interfaces=[whatsapp])
    manager.serve(port=8000)
    ```

Example - Chat Mode:
    ```python
    from upsonic import Agent
    from upsonic.interfaces import InterfaceManager, GmailInterface, InterfaceMode
    
    agent = Agent("openai/gpt-4o")
    
    # Chat mode - conversations persist per user
    gmail = GmailInterface(
        agent=agent,
        mode=InterfaceMode.CHAT,
        reset_command="/reset"  # Users can reset by sending this
    )
    
    manager = InterfaceManager(interfaces=[gmail])
    manager.serve(port=8000)
    ```
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Interface
    from .manager import InterfaceManager
    from .whatsapp.whatsapp import WhatsAppInterface
    from .slack.slack import SlackInterface
    from .gmail.gmail import GmailInterface
    from .telegram.telegram import TelegramInterface
    from .settings import InterfaceSettings
    from .websocket_manager import WebSocketManager, WebSocketConnection
    from .auth import (
        get_authentication_dependency,
        validate_websocket_token,
    )
    from .schemas import (
        InterfaceMode,
        InterfaceResetCommand,
        HealthCheckResponse,
        ErrorResponse,
        WebSocketMessage,
        WebSocketConnectionInfo,
        WebSocketStatusResponse,
    )
    from .whatsapp.schemas import WhatsAppWebhookPayload
    from .telegram.schemas import TelegramWebhookPayload

def _get_interface_classes():
    """Lazy import of interface classes."""
    from .base import Interface
    from .manager import InterfaceManager
    from .whatsapp.whatsapp import WhatsAppInterface
    from .slack.slack import SlackInterface
    from .gmail.gmail import GmailInterface
    from .telegram.telegram import TelegramInterface
    from .settings import InterfaceSettings
    from .websocket_manager import WebSocketManager, WebSocketConnection
    
    # Aliases for convenience
    Whatsapp = WhatsAppInterface  # Shortened alias
    Slack = SlackInterface
    Gmail = GmailInterface
    Telegram = TelegramInterface
    
    return {
        'Interface': Interface,
        'InterfaceManager': InterfaceManager,
        'WhatsAppInterface': WhatsAppInterface,
        'Whatsapp': Whatsapp,
        'SlackInterface': SlackInterface,
        'Slack': Slack,
        'GmailInterface': GmailInterface,
        'Gmail': Gmail,
        'TelegramInterface': TelegramInterface,
        'Telegram': Telegram,
        'InterfaceSettings': InterfaceSettings,
        'WebSocketManager': WebSocketManager,
        'WebSocketConnection': WebSocketConnection,
    }

def _get_auth_functions():
    """Lazy import of authentication functions."""
    from .auth import (
        get_authentication_dependency,
        validate_websocket_token,
    )
    
    return {
        'get_authentication_dependency': get_authentication_dependency,
        'validate_websocket_token': validate_websocket_token,
    }

def _get_schema_classes():
    """Lazy import of schema classes."""
    from .schemas import (
        InterfaceMode,
        InterfaceResetCommand,
        HealthCheckResponse,
        ErrorResponse,
        WebSocketMessage,
        WebSocketConnectionInfo,
        WebSocketStatusResponse,
    )
    from .whatsapp.schemas import WhatsAppWebhookPayload
    from .telegram.schemas import TelegramWebhookPayload
    
    return {
        'InterfaceMode': InterfaceMode,
        'InterfaceResetCommand': InterfaceResetCommand,
        'HealthCheckResponse': HealthCheckResponse,
        'ErrorResponse': ErrorResponse,
        'WebSocketMessage': WebSocketMessage,
        'WebSocketConnectionInfo': WebSocketConnectionInfo,
        'WebSocketStatusResponse': WebSocketStatusResponse,
        'WhatsAppWebhookPayload': WhatsAppWebhookPayload,
        'TelegramWebhookPayload': TelegramWebhookPayload,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Interface classes
    interface_classes = _get_interface_classes()
    if name in interface_classes:
        return interface_classes[name]
    
    # Auth functions
    auth_functions = _get_auth_functions()
    if name in auth_functions:
        return auth_functions[name]
    
    # Schema classes
    schema_classes = _get_schema_classes()
    if name in schema_classes:
        return schema_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    # Core classes
    "Interface",
    "InterfaceManager",
    "InterfaceSettings",
    
    # Mode configuration
    "InterfaceMode",
    "InterfaceResetCommand",
    
    # Interface implementations
    "WhatsAppInterface",
    "Whatsapp",  # Alias
    "SlackInterface",
    "Slack",
    "GmailInterface",
    "Gmail",
    "TelegramInterface",
    "Telegram",  # Alias
    
    # WebSocket
    "WebSocketManager",
    "WebSocketConnection",
    
    # Authentication
    "get_authentication_dependency",
    "validate_websocket_token",
    
    # Schemas
    "HealthCheckResponse",
    "ErrorResponse",
    "WhatsAppWebhookPayload",
    "TelegramWebhookPayload",
    "WebSocketMessage",
    "WebSocketConnectionInfo",
    "WebSocketStatusResponse",
]

__version__ = "1.0.0"
