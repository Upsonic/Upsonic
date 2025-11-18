"""
WebSocket Manager for Interface Manager.

This module provides a robust WebSocket connection manager that handles
multiple concurrent connections, broadcasts, and connection lifecycle management.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import WebSocket, status
from upsonic.utils.printing import debug_log, error_log


class WebSocketConnection:
    """
    Represents a single WebSocket connection with metadata.
    
    Attributes:
        id: Unique identifier (UUID) for this connection instance
        name: Human-readable name for this connection (defaults to connection_id)
        websocket: The WebSocket instance
        connection_id: Client-provided identifier for this connection (from URL path)
        connected_at: Timestamp when the connection was established
        metadata: Optional metadata associated with this connection
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ):
        """
        Initialize a WebSocket connection wrapper.
        
        Args:
            websocket: The WebSocket instance
            connection_id: Client-provided identifier for this connection (from URL path)
            metadata: Optional metadata (e.g., user_id, interface_name)
            name: Optional human-readable name (defaults to connection_id)
            id: Optional unique identifier (UUID string). If not provided, a new UUID will be generated.
        """
        self.id = id or str(uuid.uuid4())
        self.name = name or connection_id
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.utcnow()
        self.metadata = metadata or {}
        
    async def send_text(self, data: str) -> bool:
        """
        Send text data through this connection.
        
        Args:
            data: Text data to send
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            await self.websocket.send_text(data)
            return True
        except Exception as e:
            error_log(f"Failed to send text to {self.connection_id}: {e}")
            return False
    
    async def send_json(self, data: Dict[str, Any]) -> bool:
        """
        Send JSON data through this connection.
        
        Args:
            data: Dictionary to send as JSON
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            await self.websocket.send_json(data)
            return True
        except Exception as e:
            error_log(f"Failed to send JSON to {self.connection_id}: {e}")
            return False
    
    async def close(self, code: int = status.WS_1000_NORMAL_CLOSURE, reason: str = ""):
        """
        Close the WebSocket connection.
        
        Args:
            code: WebSocket close code
            reason: Optional reason for closure
        """
        try:
            await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            error_log(f"Error closing connection {self.connection_id}: {e}")
    
    def get_id(self) -> str:
        """
        Get the unique identifier of this connection.
        
        Returns:
            str: The connection UUID
        """
        return self.id
    
    def get_name(self) -> str:
        """
        Get the name of this connection.
        
        Returns:
            str: The connection name
        """
        return self.name
    
    def get_connection_id(self) -> str:
        """
        Get the client-provided connection ID (from URL path).
        
        Returns:
            str: The client-provided connection ID
        """
        return self.connection_id
    
    def __repr__(self) -> str:
        return f"WebSocketConnection(id={self.id}, name={self.name}, connection_id={self.connection_id}, connected_at={self.connected_at})"


class WebSocketManager:
    """
    Manages multiple WebSocket connections with support for broadcasting,
    filtering, and connection lifecycle management.
    
    This manager is thread-safe and can handle multiple concurrent connections
    across different interfaces and users.
    
    Attributes:
        connections: Dictionary mapping connection IDs to WebSocketConnection instances
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.connections: Dict[str, WebSocketConnection] = {}
        self.authenticated_connections: Dict[WebSocket, bool] = {}
        self._lock = asyncio.Lock()
        
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        requires_auth: bool = True
    ) -> WebSocketConnection:
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: The WebSocket instance to register
            connection_id: Unique identifier for this connection
            metadata: Optional metadata for this connection
            requires_auth: Whether this connection requires authentication
            
        Returns:
            WebSocketConnection: The registered connection wrapper
            
        Example:
            ```python
            @router.websocket("/ws/{client_id}")
            async def websocket_endpoint(websocket: WebSocket, client_id: str):
                conn = await manager.connect(
                    websocket,
                    client_id,
                    metadata={"interface": "whatsapp"},
                    requires_auth=True
                )
                # ... handle connection
            ```
        """
        await websocket.accept()
        debug_log(f"WebSocket accepted for connection: {connection_id}")
        
        async with self._lock:
            connection = WebSocketConnection(websocket, connection_id, metadata)
            self.connections[connection_id] = connection
            
            # If auth is not required, mark as authenticated immediately
            self.authenticated_connections[websocket] = not requires_auth
        
        # Send connection confirmation with auth requirement info
        await websocket.send_text(
            json.dumps(
                {
                    "event": "connected",
                    "message": (
                        "Connected. Please authenticate to continue."
                        if requires_auth
                        else "Connected. Authentication not required."
                    ),
                    "requires_auth": requires_auth,
                    "connection_id": connection_id,
                }
            )
        )
            
        debug_log(f"WebSocket connected: {connection_id} (total: {len(self.connections)})")
        return connection
    
    async def disconnect(self, connection_id: str):
        """
        Remove a WebSocket connection.
        
        Args:
            connection_id: ID of the connection to remove
        """
        async with self._lock:
            if connection_id in self.connections:
                connection = self.connections.pop(connection_id)
                # Clean up authentication tracking
                if connection.websocket in self.authenticated_connections:
                    del self.authenticated_connections[connection.websocket]
                try:
                    await connection.close()
                except Exception:
                    pass  # Connection may already be closed
                    
        debug_log(f"WebSocket disconnected: {connection_id} (remaining: {len(self.connections)})")
    
    async def authenticate_websocket(self, websocket: WebSocket):
        """
        Mark a WebSocket connection as authenticated and send confirmation.
        
        Args:
            websocket: The WebSocket instance to authenticate
        """
        self.authenticated_connections[websocket] = True
        debug_log("WebSocket authenticated")
        
        # Send authentication confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "event": "authenticated",
                    "message": "Authentication successful. You can now send commands.",
                }
            )
        )
    
    def is_authenticated(self, websocket: WebSocket) -> bool:
        """
        Check if a WebSocket connection is authenticated.
        
        Args:
            websocket: The WebSocket instance to check
            
        Returns:
            bool: True if authenticated, False otherwise
        """
        return self.authenticated_connections.get(websocket, False)
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """
        Remove WebSocket connection and clean up all associated state.
        
        Args:
            websocket: The WebSocket instance to disconnect
        """
        # Remove from authenticated connections
        if websocket in self.authenticated_connections:
            del self.authenticated_connections[websocket]
        
        # Find and remove from connections by websocket instance
        connection_ids_to_remove = [
            conn_id for conn_id, conn in self.connections.items() 
            if conn.websocket == websocket
        ]
        
        async with self._lock:
            for connection_id in connection_ids_to_remove:
                if connection_id in self.connections:
                    connection = self.connections.pop(connection_id)
                    try:
                        await connection.close()
                    except Exception:
                        pass  # Connection may already be closed
        
        if connection_ids_to_remove:
            debug_log(f"WebSocket disconnected and cleaned up: {connection_ids_to_remove}")
        else:
            debug_log("WebSocket disconnected and cleaned up")
    
    def get_all_connections(self) -> List[WebSocketConnection]:
        """
        Get all active connections.
        
        Returns:
            List[WebSocketConnection]: List of all active connections
        """
        return list(self.connections.values())
    
    def get_connection_count(self) -> int:
        """
        Get the total number of active connections.
        
        Returns:
            int: Number of active connections
        """
        return len(self.connections)
    
    async def close_all(self):
        """Close all active connections gracefully."""
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id)
        
        debug_log("All WebSocket connections closed")

