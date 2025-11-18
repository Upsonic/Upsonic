"""
Authentication module for Interface Manager.

This module provides authentication functionality for both HTTP requests (Bearer token)
and WebSocket connections. Authentication can be disabled by not setting a security key.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from upsonic.interfaces.settings import InterfaceSettings

# Global HTTPBearer instance for dependency injection
security = HTTPBearer(auto_error=False)


def get_authentication_dependency(settings: InterfaceSettings):
    """
    Create an authentication dependency function for FastAPI routes.
    
    This factory function creates a dependency that can be used with FastAPI's
    Depends() to protect routes with Bearer token authentication.
    
    Args:
        settings: The interface settings containing the security key
        
    Returns:
        A dependency function that validates Bearer tokens
        
    Example:
        ```python
        settings = InterfaceSettings()
        auth_dep = get_authentication_dependency(settings)
        
        @router.get("/protected")
        async def protected_route(authorized: bool = Depends(auth_dep)):
            return {"message": "Access granted"}
        ```
    """
    
    async def auth_dependency(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> bool:
        """
        Validate Bearer token authentication.
        
        Args:
            credentials: HTTP Authorization credentials from the request header
            
        Returns:
            bool: True if authentication succeeds or is disabled
            
        Raises:
            HTTPException: If authentication is required but fails
        """
        # If no security key is configured, skip authentication entirely
        if not settings.is_auth_enabled():
            return True
        
        # If security is enabled but no authorization header is provided, reject
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = credentials.credentials
        
        # Verify the token matches the configured security key
        if token != settings.security_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return True
    
    return auth_dependency


def validate_websocket_token(token: Optional[str], settings: InterfaceSettings) -> bool:
    """
    Validate a bearer token for WebSocket authentication.
    
    This function provides a simple way to validate tokens for WebSocket connections,
    which cannot use the standard HTTPBearer dependency.
    
    Args:
        token: The bearer token to validate (without "Bearer " prefix)
        settings: The interface settings containing the security key
        
    Returns:
        bool: True if the token is valid or authentication is disabled, False otherwise
        
    Example:
        ```python
        @router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            # Get token from query parameter or initial message
            token = websocket.query_params.get("token")
            
            if not validate_websocket_token(token, settings):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
                
            # Continue with WebSocket communication
            ...
        ```
    """
    # If no security key is configured, skip authentication entirely
    if not settings.is_auth_enabled():
        return True
    
    # If token is not provided but auth is required, fail
    if not token:
        return False
    
    # Verify the token matches the configured security key
    return token == settings.security_key

