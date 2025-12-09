from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from ..storage import DurableExecutionStorage, ExecutionState


class InMemoryDurableStorage(DurableExecutionStorage):
    """
    In-memory storage backend for durable execution.
    
    This storage backend keeps all execution state in memory (RAM).
    State is lost when the process terminates, so this is mainly useful for:
    - Testing and development
    - Temporary execution tracking
    - High-performance scenarios where persistence isn't required
    
    Thread-safe through use of locks for concurrent access.
    
    Example:
        ```python
        storage = InMemoryDurableStorage()
        durable = DurableExecution(storage=storage)
        ```
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._storage: Dict[str, ExecutionState] = {}
        self._lock = None
        self._initialized = False
    
    def _ensure_lock(self):
        """Ensure lock is initialized (lazy initialization for async safety)."""
        if not self._initialized:
            import asyncio
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                import threading
                self._lock = threading.Lock()
            self._initialized = True
    
    async def save_state_async(
        self, 
        execution_id: str, 
        state: ExecutionState
    ) -> None:
        """
        Save execution state to memory.
        
        Args:
            execution_id: Unique identifier for the execution
            state: ExecutionState containing all checkpoint data
        """
        self._ensure_lock()
        
        state["saved_at"] = datetime.now(timezone.utc).isoformat()
        
        # CRITICAL: Serialize with cloudpickle to create a true copy!
        # Using cloudpickle is consistent with the rest of the durable feature
        # and handles complex objects (functions, closures, etc.) correctly.
        import cloudpickle
        state_copy = cloudpickle.loads(cloudpickle.dumps(state))
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                self._storage[execution_id] = state_copy
        else:
            with self._lock:
                self._storage[execution_id] = state_copy
    
    async def load_state_async(
        self, 
        execution_id: str
    ) -> Optional[ExecutionState]:
        """
        Load execution state from memory.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            ExecutionState if found, None otherwise
        """
        self._ensure_lock()
        
        # CRITICAL: Return a cloudpickle copy to avoid reference issues!
        # Using cloudpickle is consistent with the rest of the durable feature.
        import cloudpickle
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                state = self._storage.get(execution_id)
                return cloudpickle.loads(cloudpickle.dumps(state)) if state else None
        else:
            with self._lock:
                state = self._storage.get(execution_id)
                return cloudpickle.loads(cloudpickle.dumps(state)) if state else None
    
    async def delete_state_async(
        self, 
        execution_id: str
    ) -> bool:
        """
        Delete execution state from memory.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                if execution_id in self._storage:
                    del self._storage[execution_id]
                    return True
                return False
        else:
            with self._lock:
                if execution_id in self._storage:
                    del self._storage[execution_id]
                    return True
                return False
    
    async def list_executions_async(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions in memory.
        
        Args:
            status: Filter by status ('running', 'paused', 'completed', 'failed')
            limit: Maximum number of executions to return
            
        Returns:
            List of execution metadata dictionaries
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                executions = self._list_executions_internal(status, limit)
        else:
            with self._lock:
                executions = self._list_executions_internal(status, limit)
        
        return executions
    
    def _list_executions_internal(
        self,
        status: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Internal method to list executions (must be called with lock held)."""
        result = []
        
        for execution_id, state in self._storage.items():
            # Extract metadata from the state wrapper
            metadata = state.get("metadata", {})
            
            if status and metadata.get("status") != status:
                continue
            
            result.append({
                "execution_id": execution_id,
                "status": metadata.get("status"),
                "step_name": metadata.get("step_name"),
                "step_index": metadata.get("step_index"),
                "timestamp": metadata.get("timestamp"),
                "saved_at": state.get("saved_at"),
                "error": metadata.get("error"),
            })
        
        result.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if limit:
            result = result[:limit]
        
        return result
    
    async def cleanup_old_executions_async(
        self,
        older_than_days: int = 7
    ) -> int:
        """
        Cleanup old completed/failed executions from memory.
        
        Args:
            older_than_days: Delete executions older than this many days
            
        Returns:
            Number of executions deleted
        """
        self._ensure_lock()
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        to_delete = []
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                to_delete = self._find_old_executions(cutoff_date)
                for execution_id in to_delete:
                    del self._storage[execution_id]
        else:
            with self._lock:
                to_delete = self._find_old_executions(cutoff_date)
                for execution_id in to_delete:
                    del self._storage[execution_id]
        
        return len(to_delete)
    
    def _find_old_executions(self, cutoff_date: datetime) -> List[str]:
        """Find old executions to delete (must be called with lock held)."""
        to_delete = []
        
        for execution_id, state in self._storage.items():
            # Extract metadata from the state wrapper
            metadata = state.get("metadata", {})
            
            if metadata.get("status") not in ["completed", "failed"]:
                continue
            
            timestamp_str = metadata.get("timestamp")
            if not timestamp_str:
                continue
            
            try:
                # Parse timestamp and ensure it's timezone-aware
                timestamp_str_fixed = timestamp_str.replace('Z', '+00:00')
                timestamp = datetime.fromisoformat(timestamp_str_fixed)
                # If timestamp is naive, assume it's UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                if timestamp < cutoff_date:
                    to_delete.append(execution_id)
            except (ValueError, AttributeError):
                continue
        
        return to_delete
    
    def clear_all(self) -> int:
        """
        Clear all executions from memory.
        
        Returns:
            Number of executions cleared
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            # Can't use async from sync method, use sync lock
            import threading
            if not isinstance(self._lock, threading.Lock):
                # Create a new threading lock for this sync operation
                lock = threading.Lock()
                with lock:
                    count = len(self._storage)
                    self._storage.clear()
            else:
                with self._lock:
                    count = len(self._storage)
                    self._storage.clear()
        else:
            with self._lock:
                count = len(self._storage)
                self._storage.clear()
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            # Sync access for stats
            import threading
            lock = threading.Lock()
            with lock:
                stats = self._get_stats_internal()
        else:
            with self._lock:
                stats = self._get_stats_internal()
        
        return stats
    
    def _get_stats_internal(self) -> Dict[str, Any]:
        """Internal method to get stats (must be called with lock held)."""
        total = len(self._storage)
        by_status = {}
        
        for exec_id, state in self._storage.items():
            # Extract metadata from the state wrapper
            metadata = state.get("metadata", {})
            status = metadata.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "backend": "memory",
            "total_executions": total,
            "by_status": by_status,
        }

