"""
Upsonic Interfaces Module

This module provides a comprehensive interface system for integrating AI agents
with external communication platforms like WhatsApp, Slack, and more.

Public API:
    - Interface: Base class for custom interfaces
    - InterfaceManager: Central manager for orchestrating interfaces
    - WhatsAppInterface: WhatsApp Business API integration
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

from upsonic.interfaces.base import Interface
from upsonic.interfaces.manager import InterfaceManager
from upsonic.interfaces.whatsapp.interface import WhatsAppInterface
from upsonic.interfaces.slack import SlackInterface
from upsonic.interfaces.settings import InterfaceSettings
from upsonic.interfaces.websocket_manager import WebSocketManager, WebSocketConnection
from upsonic.interfaces.auth import (
    get_authentication_dependency,
    validate_websocket_token,
)
from upsonic.interfaces.schemas import (
    # Common schemas
    HealthCheckResponse,
    ErrorResponse,
    
    # WebSocket schemas
    WebSocketMessage,
    WebSocketConnectionInfo,
    WebSocketStatusResponse,
)

# Import WhatsApp-specific schemas from their module
from upsonic.interfaces.whatsapp.schemas import WhatsAppWebhookPayload

# Aliases for convenience
Whatsapp = WhatsAppInterface  # Shortened alias
Slack = SlackInterface

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

