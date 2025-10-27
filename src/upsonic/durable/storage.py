from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime


class ExecutionState(Dict[str, Any]):
    """
    Represents the state of an execution at a specific point.
    
    This is a dictionary subclass that provides type hints and validation
    for execution state data.
    
    Standard fields:
        execution_id: Unique identifier for this execution
        task_data: Serialized task object
        step_index: Index of the last completed step
        step_name: Name of the last completed step
        context_data: Serialized StepContext
        timestamp: When this state was saved
        status: 'running', 'paused', 'completed', 'failed'
        error: Error message if status is 'failed'
    """
    pass


class DurableExecutionStorage(ABC):
    """
    Abstract base class for durable execution storage backends.
    
    Implementations must provide methods to:
    - Save execution state at checkpoints
    - Load execution state for resumption
    - List all executions
    - Delete completed/failed executions
    
    All methods should be thread-safe for concurrent access.
    """
    
    @abstractmethod
    async def save_state_async(
        self, 
        execution_id: str, 
        state: ExecutionState
    ) -> None:
        """
        Save execution state asynchronously.
        
        Args:
            execution_id: Unique identifier for the execution
            state: ExecutionState containing all checkpoint data
            
        Raises:
            Exception: If save operation fails
        """
        pass
    
    @abstractmethod
    async def load_state_async(
        self, 
        execution_id: str
    ) -> Optional[ExecutionState]:
        """
        Load execution state asynchronously.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            ExecutionState if found, None otherwise
            
        Raises:
            Exception: If load operation fails
        """
        pass
    
    @abstractmethod
    async def delete_state_async(
        self, 
        execution_id: str
    ) -> bool:
        """
        Delete execution state asynchronously.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            Exception: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def list_executions_async(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions asynchronously.
        
        Args:
            status: Filter by status ('running', 'paused', 'completed', 'failed')
            limit: Maximum number of executions to return
            
        Returns:
            List of execution metadata dictionaries
            
        Raises:
            Exception: If list operation fails
        """
        pass
    
    @abstractmethod
    async def cleanup_old_executions_async(
        self,
        older_than_days: int = 7
    ) -> int:
        """
        Cleanup old completed/failed executions asynchronously.
        
        Args:
            older_than_days: Delete executions older than this many days
            
        Returns:
            Number of executions deleted
            
        Raises:
            Exception: If cleanup operation fails
        """
        pass
    
    
    def save_state(
        self, 
        execution_id: str, 
        state: ExecutionState
    ) -> None:
        """Synchronous wrapper for save_state_async."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self.save_state_async(execution_id, state))
        except RuntimeError:
            return asyncio.run(self.save_state_async(execution_id, state))
    
    def load_state(
        self, 
        execution_id: str
    ) -> Optional[ExecutionState]:
        """Synchronous wrapper for load_state_async."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self.load_state_async(execution_id))
        except RuntimeError:
            return asyncio.run(self.load_state_async(execution_id))
    
    def delete_state(
        self, 
        execution_id: str
    ) -> bool:
        """Synchronous wrapper for delete_state_async."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self.delete_state_async(execution_id))
        except RuntimeError:
            return asyncio.run(self.delete_state_async(execution_id))
    
    def list_executions(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for list_executions_async."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self.list_executions_async(status, limit))
        except RuntimeError:
            return asyncio.run(self.list_executions_async(status, limit))
    
    def cleanup_old_executions(
        self,
        older_than_days: int = 7
    ) -> int:
        """Synchronous wrapper for cleanup_old_executions_async."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self.cleanup_old_executions_async(older_than_days))
        except RuntimeError:
            return asyncio.run(self.cleanup_old_executions_async(older_than_days))

