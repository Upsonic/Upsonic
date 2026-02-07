import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, Any, Union

from fastapi import APIRouter

from upsonic.interfaces.schemas import InterfaceMode, InterfaceResetCommand

if TYPE_CHECKING:
    from upsonic.agent import Agent
    from upsonic.chat import Chat
    from upsonic.storage.base import Storage


# Fixed message for unauthorized users - DO NOT CHANGE
UNAUTHORIZED_MESSAGE: str = "This operation not allowed"


class Interface(ABC):
    """
    Abstract base class for all custom interfaces and integrations.
    
    Each interface represents a communication channel (e.g., WhatsApp, Slack)
    that can send and receive messages through a connected Agent.
    
    Supports two operating modes:
    - TASK: Each message is treated as an independent task (stateless, default)
    - CHAT: Messages are accumulated in a conversation session (stateful)
    
    Supports whitelist-based access control:
    - Only allowed users can interact with the agent
    - Unauthorized users receive a fixed "This operation not allowed" message
    
    Attributes:
        id: Unique identifier (UUID) for this interface instance
        name: Human-readable name for this interface
        agent: The AI agent that processes messages from this interface
        mode: Operating mode (TASK or CHAT)
        reset_command: Command configuration for resetting chat sessions
    """
    
    def __init__(
        self,
        agent: "Agent",
        name: Optional[str] = None,
        id: Optional[str] = None,
        mode: Union[InterfaceMode, str] = InterfaceMode.TASK,
        reset_command: Optional[str] = "/reset",
        storage: Optional["Storage"] = None,
    ):
        """
        Initialize the interface with an agent.
        
        Args:
            agent: The AI agent that will handle messages
            name: Optional custom name for this interface. If not provided,
                  uses the class name by default.
            id: Optional unique identifier (UUID string). If not provided,
                a new UUID will be generated.
            mode: Operating mode - TASK for independent tasks, CHAT for conversation sessions.
                  Can be InterfaceMode enum or string ("task" or "chat").
            reset_command: Command string to reset chat session (only applies in CHAT mode).
                          Set to None to disable reset command. Default: "/reset"
            storage: Optional storage backend for chat sessions. If not provided,
                    uses InMemoryStorage for chat mode.
        """
        self.id = id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.agent = agent
        
        # Normalize mode to InterfaceMode enum
        if isinstance(mode, str):
            mode = InterfaceMode(mode.lower())
        self.mode: InterfaceMode = mode
        
        # Reset command configuration
        self._reset_command = InterfaceResetCommand()
        if reset_command is not None:
            self._reset_command.command = reset_command
        self._reset_enabled = reset_command is not None
        
        # Storage for chat sessions
        self._storage: Optional["Storage"] = storage
        
        # Chat session cache: Maps user_id -> Chat instance
        # Each unique user/sender gets their own conversation session
        self._chat_sessions: Dict[str, "Chat"] = {}
        
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
    
    def get_mode(self) -> InterfaceMode:
        """
        Get the operating mode of this interface.
        
        Returns:
            InterfaceMode: The current mode (TASK or CHAT)
        """
        return self.mode
    
    def is_task_mode(self) -> bool:
        """Check if interface is operating in TASK mode."""
        return self.mode == InterfaceMode.TASK
    
    def is_chat_mode(self) -> bool:
        """Check if interface is operating in CHAT mode."""
        return self.mode == InterfaceMode.CHAT
    
    def is_reset_command(self, text: str) -> bool:
        """
        Check if the given text is a reset command.
        
        Args:
            text: The message text to check
            
        Returns:
            bool: True if the text matches the reset command
        """
        if not self._reset_enabled or not self.is_chat_mode():
            return False
        return self._reset_command.matches(text)
    
    def _get_storage(self) -> "Storage":
        """
        Get the storage backend for chat sessions.
        
        Returns:
            Storage: The storage instance (creates InMemoryStorage if not set)
        """
        if self._storage is None:
            from upsonic.storage.in_memory.in_memory import InMemoryStorage
            self._storage = InMemoryStorage()
        return self._storage
    
    def get_chat_session(self, user_id: str) -> "Chat":
        """
        Get or create a chat session for the given user.
        
        In CHAT mode, each unique user gets a persistent conversation session.
        Sessions are cached in memory and persist their state to storage.
        
        Args:
            user_id: Unique identifier for the user/sender
            
        Returns:
            Chat: The chat session for this user
        """
        from upsonic.chat import Chat
        
        if user_id not in self._chat_sessions:
            # Create session ID from interface name and user ID
            session_id = f"{self.name.lower()}_{user_id}"
            
            self._chat_sessions[user_id] = Chat(
                session_id=session_id,
                user_id=user_id,
                agent=self.agent,
                storage=self._get_storage(),
                full_session_memory=True,
                summary_memory=False,
                user_analysis_memory=False,
                debug=getattr(self.agent, 'debug', False),
            )
        
        return self._chat_sessions[user_id]
    
    async def aget_chat_session(self, user_id: str) -> "Chat":
        """
        Get or create a chat session for the given user (async version).
        
        Args:
            user_id: Unique identifier for the user/sender
            
        Returns:
            Chat: The chat session for this user
        """
        return self.get_chat_session(user_id)
    
    def reset_chat_session(self, user_id: str) -> bool:
        """
        Reset a chat session for the given user.
        
        This clears the conversation history and creates a fresh session.
        
        Args:
            user_id: Unique identifier for the user/sender
            
        Returns:
            bool: True if a session was reset, False if no session existed
        """
        if user_id in self._chat_sessions:
            chat = self._chat_sessions[user_id]
            chat.reset_session()
            del self._chat_sessions[user_id]
            return True
        return False
    
    async def areset_chat_session(self, user_id: str) -> bool:
        """
        Reset a chat session for the given user (async version).
        
        Args:
            user_id: Unique identifier for the user/sender
            
        Returns:
            bool: True if a session was reset, False if no session existed
        """
        if user_id in self._chat_sessions:
            chat = self._chat_sessions[user_id]
            await chat.areset_session()
            del self._chat_sessions[user_id]
            return True
        return False
    
    def has_chat_session(self, user_id: str) -> bool:
        """
        Check if a chat session exists for the given user.
        
        Args:
            user_id: Unique identifier for the user/sender
            
        Returns:
            bool: True if a session exists
        """
        return user_id in self._chat_sessions
    
    def get_all_chat_sessions(self) -> Dict[str, "Chat"]:
        """
        Get all active chat sessions.
        
        Returns:
            Dict[str, Chat]: Dictionary mapping user_id to Chat instances
        """
        return self._chat_sessions.copy()
    
    def get_unauthorized_message(self) -> str:
        """
        Get the fixed unauthorized message.
        
        This message is returned to users who are not in the whitelist.
        The message is fixed and cannot be changed.
        
        Returns:
            str: The unauthorized message "This operation not allowed"
        """
        return UNAUTHORIZED_MESSAGE
    
    def __repr__(self) -> str:
        """String representation of the interface."""
        return f"{self.__class__.__name__}(id={self.id}, name={self.name}, mode={self.mode.value})"
