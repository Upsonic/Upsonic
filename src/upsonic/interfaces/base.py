import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, Any

from fastapi import APIRouter

if TYPE_CHECKING:
    from upsonic.agent import Agent


class Interface(ABC):
    """
    Abstract base class for all custom interfaces and integrations.
    
    Each interface represents a communication channel (e.g., WhatsApp, Slack)
    that can send and receive messages through a connected Agent.
    
    Attributes:
        id: Unique identifier (UUID) for this interface instance
        name: Human-readable name for this interface
        agent: The AI agent that processes messages from this interface
    """
    
    def __init__(self, agent: "Agent", name: Optional[str] = None, id: Optional[str] = None):
        """
        Initialize the interface with an agent.
        
        Args:
            agent: The AI agent that will handle messages
            name: Optional custom name for this interface. If not provided,
                  uses the class name by default.
            id: Optional unique identifier (UUID string). If not provided,
                a new UUID will be generated.
        """
        self.id = id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.agent = agent
        
    @abstractmethod
    def attach_routes(self) -> APIRouter:
        """
        Create and return FastAPI routes for this interface.
        
        This method must be implemented by each concrete interface class.
        It should create an APIRouter with all necessary endpoints for
        the interface to function (e.g., webhook receivers, message senders).
        
        Returns:
            APIRouter: A FastAPI router containing all routes for this interface
            
        Example:
            ```python
            def attach_routes(self) -> APIRouter:
                router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])
                
                @router.post("/webhook")
                async def webhook(data: dict):
                    # Handle incoming messages
                    pass
                    
                return router
            ```
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the interface.
        
        Returns:
            Dict[str, Any]: Dictionary containing status and details.
            Default implementation returns basic status.
        """
        return {
            "status": "active",
            "name": self.name,
            "id": self.id
        }
    
    def get_id(self) -> str:
        """
        Get the unique identifier of this interface.
        
        Returns:
            str: The interface UUID
        """
        return self.id
    
    def get_name(self) -> str:
        """
        Get the name of this interface.
        
        Returns:
            str: The interface name
        """
        return self.name
    
    def __repr__(self) -> str:
        """String representation of the interface."""
        return f"{self.__class__.__name__}(id={self.id}, name={self.name})"
