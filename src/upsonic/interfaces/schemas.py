"""
Schemas module for Interface Manager.

This module provides common Pydantic models for API requests and responses
used across all interfaces. Interface-specific schemas are in their respective modules.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from upsonic._utils import now_utc


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    status: str = Field(default="healthy", description="Service health status")
    timestamp: datetime = Field(default_factory=now_utc, description="Current timestamp")
    interfaces: List[Dict[str, Any]] = Field(default_factory=list, description="List of active interfaces with status")
    connections: int = Field(default=0, description="Number of active WebSocket connections")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=now_utc, description="Error timestamp")


class WebSocketMessage(BaseModel):
    """Model for WebSocket messages."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    type: str = Field(..., description="Message type (e.g., 'agent_response', 'status', 'error')")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=now_utc, description="Message timestamp")
    connection_id: Optional[str] = Field(None, description="Connection ID")


class WebSocketConnectionInfo(BaseModel):
    """Model for WebSocket connection information."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    id: str = Field(..., description="Unique identifier (UUID) for this connection")
    name: str = Field(..., description="Human-readable name for this connection")
    connection_id: str = Field(..., description="Client-provided connection ID (from URL path)")
    connected_at: datetime = Field(..., description="Connection timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Connection metadata")


class WebSocketStatusResponse(BaseModel):
    """Response model for WebSocket status endpoint."""
    
    total_connections: int = Field(..., description="Total number of active connections")
    connections: List[WebSocketConnectionInfo] = Field(..., description="List of active connections")
