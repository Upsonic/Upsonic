"""
Upsonic Interfaces Module

This module provides a comprehensive interface system for integrating AI agents
with external communication platforms like WhatsApp, Slack, and more.

Public API:
    - Interface: Base class for custom interfaces
    - InterfaceManager: Central manager for orchestrating interfaces
    - WhatsAppInterface: WhatsApp Business API integration
    - GmailInterface: Gmail API integration
    - InterfaceSettings: Configuration settings
    - WebSocketManager: WebSocket connection manager

Example:
    ```python
    from upsonic import Agent
    from upsonic.interfaces import InterfaceManager, WhatsAppInterface
    
    # Create an agent
    agent = Agent("openai/gpt-4o")
    
    # Create WhatsApp interface
    whatsapp = WhatsAppInterface(agent=agent)
    
    # Create and start the interface manager
    manager = InterfaceManager(interfaces=[whatsapp])
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
    from .settings import InterfaceSettings
    from .websocket_manager import WebSocketManager, WebSocketConnection
    from .auth import (
        get_authentication_dependency,
        validate_websocket_token,
    )
    from .schemas import (
        HealthCheckResponse,
        ErrorResponse,
        WebSocketMessage,
        WebSocketConnectionInfo,
        WebSocketStatusResponse,
    )
    from .whatsapp.schemas import WhatsAppWebhookPayload

def _get_interface_classes():
    """Lazy import of interface classes."""
    from .base import Interface
    from .manager import InterfaceManager
    from .whatsapp.whatsapp import WhatsAppInterface
    from .slack.slack import SlackInterface
    from .gmail.gmail import GmailInterface
    from .settings import InterfaceSettings
    from .websocket_manager import WebSocketManager, WebSocketConnection
    
    # Aliases for convenience
    Whatsapp = WhatsAppInterface  # Shortened alias
    Slack = SlackInterface
    Gmail = GmailInterface
    
    return {
        'Interface': Interface,
        'InterfaceManager': InterfaceManager,
        'WhatsAppInterface': WhatsAppInterface,
        'Whatsapp': Whatsapp,
        'SlackInterface': SlackInterface,
        'Slack': Slack,
        'GmailInterface': GmailInterface,
        'Gmail': Gmail,
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
        HealthCheckResponse,
        ErrorResponse,
        WebSocketMessage,
        WebSocketConnectionInfo,
        WebSocketStatusResponse,
    )
    from .whatsapp.schemas import WhatsAppWebhookPayload
    
    return {
        'HealthCheckResponse': HealthCheckResponse,
        'ErrorResponse': ErrorResponse,
        'WebSocketMessage': WebSocketMessage,
        'WebSocketConnectionInfo': WebSocketConnectionInfo,
        'WebSocketStatusResponse': WebSocketStatusResponse,
        'WhatsAppWebhookPayload': WhatsAppWebhookPayload,
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
    
    # Interface implementations
    "WhatsAppInterface",
    "Whatsapp",  # Alias
    "SlackInterface",
    "Slack",
    "GmailInterface",
    "Gmail",
    
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
    "WebSocketMessage",
    "WebSocketConnectionInfo",
    "WebSocketStatusResponse",
]

__version__ = "1.0.0"
