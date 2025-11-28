"""
Interface Manager for Upsonic Framework.

This module provides the InterfaceManager class which orchestrates multiple
interfaces, manages FastAPI application lifecycle, and provides WebSocket support.
"""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from upsonic.interfaces.base import Interface
from upsonic.interfaces.settings import InterfaceSettings
from upsonic.interfaces.websocket_manager import WebSocketManager
from upsonic.interfaces.schemas import (
    ErrorResponse,
    HealthCheckResponse,
    WebSocketStatusResponse,
    WebSocketConnectionInfo,
)
from upsonic._utils import now_utc
from upsonic.utils.printing import debug_log, error_log, info_log

from upsonic.interfaces.auth import (
    get_authentication_dependency,
    validate_websocket_token,
)


class InterfaceManager:
    """
    Central manager for all interfaces in the Upsonic framework.
    
    The InterfaceManager:
    - Manages multiple interface instances (WhatsApp, Slack, etc.)
    - Creates and configures a FastAPI application
    - Handles WebSocket connections
    - Provides health checks and monitoring
    - Manages authentication and security
    - Serves the application with uvicorn
    
    Attributes:
        id: Unique identifier (UUID) for this manager instance
        name: Human-readable name for this manager
        interfaces: List of registered interface instances
        settings: Configuration settings for the FastAPI app
        app: FastAPI application instance
        websocket_manager: WebSocket connection manager
    """
    
    def __init__(
        self,
        interfaces: Optional[List[Interface]] = None,
        settings: Optional[InterfaceSettings] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ):
        """
        Initialize the Interface Manager.
        
        Args:
            interfaces: List of interface instances to manage
            settings: Configuration settings (uses defaults if not provided)
            name: Optional custom name for this manager (defaults to "InterfaceManager")
            id: Optional unique identifier (UUID string). If not provided, a new UUID will be generated.
            
        Example:
            ```python
            from upsonic import Agent
            from upsonic.interfaces import InterfaceManager, WhatsAppInterface
            
            agent = Agent("openai/gpt-4o")
            whatsapp = WhatsAppInterface(agent=agent)
            
            manager = InterfaceManager(interfaces=[whatsapp])
            manager.serve(port=8000)
            ```
        """
        self.id = id or str(uuid.uuid4())
        self.name = name or "InterfaceManager"
        self.interfaces = interfaces or []
        self.settings = settings or InterfaceSettings()
        self.websocket_manager = WebSocketManager()
        
        # Create FastAPI app with lifespan
        self.app = self._create_app()
        
        # Attach all interface routes
        self._attach_interface_routes()
        
        # Add core routes
        self._add_core_routes()
        
        info_log(
            f"InterfaceManager(id={self.id}, name={self.name}) initialized with {len(self.interfaces)} interface(s)"
        )
    
    def get_id(self) -> str:
        """
        Get the unique identifier of this manager.
        
        Returns:
            str: The manager UUID
        """
        return self.id
    
    def get_name(self) -> str:
        """
        Get the name of this manager.
        
        Returns:
            str: The manager name
        """
        return self.name
    
    def _create_app(self) -> FastAPI:
        """
        Create and configure the FastAPI application.
        
        Returns:
            FastAPI: Configured FastAPI application
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan context manager for startup and shutdown events."""
            # Startup
            info_log("Starting Interface Manager...")
            yield
            # Shutdown
            info_log("Shutting down Interface Manager...")
            await self.websocket_manager.close_all()
        
        app = FastAPI(
            title=self.settings.app_title,
            description=self.settings.app_description,
            version=self.settings.app_version,
            debug=self.settings.debug,
            docs_url=self.settings.docs_url,
            redoc_url=self.settings.redoc_url,
            openapi_url=self.settings.openapi_url,
            lifespan=lifespan,
        )
        
        # Add Request ID Middleware
        @app.middleware("http")
        async def request_id_middleware(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        # Add CORS middleware if enabled
        if self.settings.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.settings.cors_origins,
                allow_credentials=self.settings.cors_allow_credentials,
                allow_methods=self.settings.cors_allow_methods,
                allow_headers=self.settings.cors_allow_headers,
            )
            debug_log("CORS middleware enabled")
        
        # Add trusted host middleware if configured
        if self.settings.trusted_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.settings.trusted_hosts
            )
            debug_log(f"Trusted host middleware enabled: {self.settings.trusted_hosts}")
        
        # Add request size and timeout validation middleware
        @app.middleware("http")
        async def validate_request_middleware(request: Request, call_next):
            """
            Middleware to validate request size and add timeout configuration.
            
            Uses settings.max_upload_size and settings.request_timeout
            """
            # Check content length if provided
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    content_length_int = int(content_length)
                    if content_length_int > self.settings.max_upload_size:
                        return JSONResponse(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content=ErrorResponse(
                                error="Request entity too large",
                                detail=f"Maximum upload size is {self.settings.max_upload_size} bytes"
                            ).model_dump()
                        )
                except ValueError:
                    pass  # Invalid content-length, let it proceed
            
            # Process request with timeout awareness
            # Note: Actual timeout enforcement happens at uvicorn/server level
            # This is informational for logging
            import asyncio
            try:
                response = await asyncio.wait_for(
                    call_next(request),
                    timeout=self.settings.request_timeout
                )
                return response
            except asyncio.TimeoutError:
                error_log(f"Request timeout after {self.settings.request_timeout}s: {request.url}")
                return JSONResponse(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    content=ErrorResponse(
                        error="Request timeout",
                        detail=f"Request exceeded timeout of {self.settings.request_timeout} seconds"
                    ).model_dump()
                )
        
        debug_log(f"Request validation middleware enabled (max_size: {self.settings.max_upload_size}, timeout: {self.settings.request_timeout}s)")
        
        # Add global exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            import traceback
            error_log(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(exc) if self.settings.debug else None
                ).model_dump()
            )
        
        return app
    
    def _attach_interface_routes(self):
        """Attach routes from all registered interfaces."""
        for interface in self.interfaces:
            try:
                router = interface.attach_routes()
                self.app.include_router(router)
                info_log(f"Attached routes for interface: {interface.get_name()}")
            except Exception as e:
                error_log(
                    f"Failed to attach routes for interface {interface.get_name()}: {e}",
                    exc_info=True
                )
    
    def _add_core_routes(self):
        """Add core management routes (health, WebSocket, etc.)."""
        auth_dep = get_authentication_dependency(self.settings)
        
        @self.app.get(
            "/health",
            response_model=HealthCheckResponse,
            summary="Health Check",
            tags=["Core"]
        )
        async def health_check():
            """
            Global health check endpoint.
            
            Returns system status and information about active interfaces.
            """
            interface_statuses = []
            for iface in self.interfaces:
                try:
                    status = await iface.health_check()
                    interface_statuses.append(status)
                except Exception as e:
                    interface_statuses.append({
                        "name": iface.get_name(),
                        "status": "error",
                        "error": str(e)
                    })
            
            return HealthCheckResponse(
                status="healthy",
                timestamp=now_utc(),
                interfaces=interface_statuses,
                connections=self.websocket_manager.get_connection_count()
            )
        
        @self.app.get(
            "/",
            summary="Root Endpoint",
            tags=["Core"]
        )
        async def root():
            """Root endpoint with basic information."""
            return {
                "service": self.settings.app_title,
                "version": self.settings.app_version,
                "interfaces": [iface.get_name() for iface in self.interfaces],
                "status": "running"
            }
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            client_id: str
        ):
            """
            WebSocket endpoint for real-time communication with message-based authentication.
            
            Clients can connect using:
            ws://host:port/ws/{client_id}
            
            If authentication is enabled, clients must send an authentication message:
            {"action": "authenticate", "token": "YOUR_TOKEN"}
            
            Supported actions:
            - authenticate: Authenticate the connection with a token
            - ping: Send a ping to check connection (responds with pong)
            - message: Send a message (requires authentication if enabled)
            
            Configuration:
            - Ping interval: Uses settings.websocket_ping_interval
            - Ping timeout: Uses settings.websocket_ping_timeout
            """
            # Determine if auth is required
            requires_auth = self.settings.is_auth_enabled()
            
            # Accept and register connection
            connection = await self.websocket_manager.connect(
                websocket,
                client_id,
                metadata={
                    "ping_interval": self.settings.websocket_ping_interval,
                    "ping_timeout": self.settings.websocket_ping_timeout
                },
                requires_auth=requires_auth
            )
            
            try:
                # Listen for messages with action-based handling
                while True:
                    data = await websocket.receive_text()
                    
                    try:
                        message = json.loads(data)
                        action = message.get("action")
                        
                        # Handle authentication first
                        if action == "authenticate":
                            token = message.get("token")
                            if not token:
                                await websocket.send_text(
                                    json.dumps({
                                        "event": "auth_error",
                                        "error": "Token is required"
                                    })
                                )
                                continue
                            
                            if validate_websocket_token(token, self.settings):
                                await self.websocket_manager.authenticate_websocket(websocket)
                            else:
                                await websocket.send_text(
                                    json.dumps({
                                        "event": "auth_error",
                                        "error": "Invalid token"
                                    })
                                )
                                continue
                        
                        # Check authentication for all other actions (only when required)
                        elif requires_auth and not self.websocket_manager.is_authenticated(websocket):
                            await websocket.send_text(
                                json.dumps({
                                    "event": "auth_required",
                                    "error": "Authentication required. Send authenticate action with valid token."
                                })
                            )
                            continue
                        
                        # Handle authenticated actions
                        elif action == "ping":
                            await websocket.send_text(
                                json.dumps({
                                    "event": "pong",
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                            )
                        
                        elif action == "message":
                            # Echo back the message (can be extended for custom handling)
                            content = message.get("content", "")
                            await connection.send_json({
                                "event": "message",
                                "content": content,
                                "client_id": client_id
                            })
                        
                        else:
                            await websocket.send_text(
                                json.dumps({
                                    "event": "error",
                                    "error": f"Unknown action: {action}"
                                })
                            )
                    
                    except json.JSONDecodeError:
                        await websocket.send_text(
                            json.dumps({
                                "event": "error",
                                "error": "Invalid JSON format"
                            })
                        )
                    
            except WebSocketDisconnect:
                debug_log(f"WebSocket client disconnected: {client_id}")
            except Exception as e:
                # Don't log common disconnect errors
                if "1012" not in str(e) and "1001" not in str(e):
                    error_log(f"WebSocket error for client {client_id}: {e}")
            finally:
                await self.websocket_manager.disconnect_websocket(websocket)
        
        @self.app.get(
            "/ws/status",
            response_model=WebSocketStatusResponse,
            summary="WebSocket Status",
            tags=["WebSocket"]
        )
        async def websocket_status(authorized: bool = Depends(auth_dep)):
            """
            Get information about active WebSocket connections.
            
            Requires authentication if enabled.
            """
            connections = self.websocket_manager.get_all_connections()
            
            connection_info = [
                WebSocketConnectionInfo(
                    id=conn.get_id(),
                    name=conn.get_name(),
                    connection_id=conn.get_connection_id(),
                    connected_at=conn.connected_at,
                    metadata=conn.metadata
                )
                for conn in connections
            ]
            
            return WebSocketStatusResponse(
                total_connections=len(connections),
                connections=connection_info
            )
        
        info_log("Core routes added")
    
    def add_interface(self, interface: Interface):
        """
        Add an interface dynamically after initialization.
        
        Args:
            interface: Interface instance to add
        """
        self.interfaces.append(interface)
        
        # Attach routes
        try:
            router = interface.attach_routes()
            self.app.include_router(router)
            info_log(f"Dynamically added interface: {interface.get_name()}")
        except Exception as e:
            error_log(f"Failed to add interface {interface.get_name()}: {e}")
            self.interfaces.remove(interface)
            raise
    
    def remove_interface(self, interface_name: str) -> bool:
        """
        Remove an interface by name.
        
        Note: Routes cannot be removed from FastAPI after being added,
        so this only removes from the internal list. A restart is
        required to fully remove routes.
        
        Args:
            interface_name: Name of the interface to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        for interface in self.interfaces:
            if interface.get_name() == interface_name:
                self.interfaces.remove(interface)
                info_log(f"Removed interface: {interface_name}")
                return True
        
        return False
    
    def get_interface(self, interface_name: str) -> Optional[Interface]:
        """
        Get an interface by name.
        
        Args:
            interface_name: Name of the interface to retrieve
            
        Returns:
            Optional[Interface]: The interface if found, None otherwise
        """
        for interface in self.interfaces:
            if interface.get_name() == interface_name:
                return interface
        
        return None
    
    def get_app(self) -> FastAPI:
        """
        Get the FastAPI application instance.
        
        Returns:
            FastAPI: The application instance
        """
        return self.app
    
    def serve(
        self,
        app: Optional[Union[str, FastAPI]] = None,
        *,
        host: str = "localhost",
        port: int = 7777,
        reload: bool = False,
        workers: Optional[int] = None,
        access_log: bool = False,
        **kwargs
    ):
        """
        Start serving the FastAPI application with uvicorn.
        
        Args:
            app: FastAPI app instance or string path (e.g., "module:app").
                 If None, uses self.app
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload for development
            workers: Number of worker processes (None for default)
            access_log: Enable access logging
            **kwargs: Additional arguments passed to uvicorn.run()
            
        Example:
            ```python
            # Simple usage
            manager.serve(port=8000)
            
            # Production usage
            manager.serve(
                host="0.0.0.0",
                port=8000,
                workers=4,
                access_log=True
            )
            
            # Development usage
            manager.serve(
                port=8000,
                reload=True
            )
            ```
        """
        # Determine which app to serve
        if app is None:
            app_to_serve = self.app
        elif isinstance(app, str):
            # String path like "module:app"
            app_to_serve = app
        else:
            app_to_serve = app
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app_to_serve,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            access_log=access_log or self.settings.access_log,
            log_level=self.settings.log_level.lower(),
            **kwargs
        )
        
        server = uvicorn.Server(config)
        
        info_log(f"Starting server on {host}:{port}")
        info_log(f"Interfaces: {', '.join(iface.get_name() for iface in self.interfaces)}")
        info_log(f"Authentication: {'Enabled' if self.settings.is_auth_enabled() else 'Disabled'}")
        
        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            info_log("Server stopped by user")
        except Exception as e:
            import traceback
            error_log(f"Server error: {e}\n{traceback.format_exc()}")
            raise
