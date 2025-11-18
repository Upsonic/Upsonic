"""
Settings module for Interface Manager.

This module provides comprehensive configuration settings for the FastAPI application
used by the InterfaceManager. All settings can be configured via environment variables.
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InterfaceSettings(BaseSettings):
    """
    Comprehensive settings for the FastAPI application used by InterfaceManager.
    
    All settings can be configured via environment variables with the prefix 'UPSONIC_INTERFACE_'.
    For example, to set the security key, use UPSONIC_INTERFACE_SECURITY_KEY.
    
    Attributes:
        security_key: Bearer token for API authentication. If not set, authentication is disabled.
        app_title: FastAPI application title
        app_description: FastAPI application description
        app_version: FastAPI application version
        debug: Enable debug mode
        docs_url: Path to Swagger UI docs (None to disable)
        redoc_url: Path to ReDoc docs (None to disable)
        openapi_url: Path to OpenAPI schema (None to disable)
        cors_enabled: Enable CORS middleware
        cors_origins: List of allowed CORS origins
        cors_allow_credentials: Allow credentials in CORS
        cors_allow_methods: Allowed HTTP methods for CORS
        cors_allow_headers: Allowed HTTP headers for CORS
        trusted_hosts: List of trusted host names
        max_upload_size: Maximum upload size in bytes (default 10MB)
        request_timeout: Request timeout in seconds
        websocket_ping_interval: WebSocket ping interval in seconds
        websocket_ping_timeout: WebSocket ping timeout in seconds
        log_level: Logging level
        access_log: Enable access logging
    """
    
    model_config = SettingsConfigDict(
        env_prefix='UPSONIC_INTERFACE_',
        case_sensitive=False,
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    # Security settings
    security_key: Optional[str] = Field(
        default=None,
        description="Bearer token for API authentication. If not set, authentication is disabled."
    )
    
    # FastAPI application settings
    app_title: str = Field(
        default="Upsonic Interface Manager",
        description="FastAPI application title"
    )
    
    app_description: str = Field(
        default="Interface Manager for Upsonic AI Agent Framework",
        description="FastAPI application description"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="FastAPI application version"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Documentation settings
    docs_url: Optional[str] = Field(
        default="/docs",
        description="Path to Swagger UI documentation (None to disable)"
    )
    
    redoc_url: Optional[str] = Field(
        default="/redoc",
        description="Path to ReDoc documentation (None to disable)"
    )
    
    openapi_url: Optional[str] = Field(
        default="/openapi.json",
        description="Path to OpenAPI schema (None to disable)"
    )
    
    # CORS settings
    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins"
    )
    
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    cors_allow_methods: List[str] = Field(
        default=["*"],
        description="Allowed HTTP methods for CORS"
    )
    
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed HTTP headers for CORS"
    )
    
    # Trusted hosts
    trusted_hosts: Optional[List[str]] = Field(
        default=None,
        description="List of trusted host names (None to disable validation)"
    )
    
    # Request limits
    max_upload_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum upload size in bytes"
    )
    
    request_timeout: int = Field(
        default=300,  # 5 minutes
        description="Request timeout in seconds"
    )
    
    # WebSocket settings
    websocket_ping_interval: float = Field(
        default=20.0,
        description="WebSocket ping interval in seconds"
    )
    
    websocket_ping_timeout: float = Field(
        default=20.0,
        description="WebSocket ping timeout in seconds"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    access_log: bool = Field(
        default=False,
        description="Enable access logging"
    )
    
    def is_auth_enabled(self) -> bool:
        """
        Check if authentication is enabled.
        
        Returns:
            bool: True if security_key is set, False otherwise
        """
        return self.security_key is not None and len(self.security_key) > 0

