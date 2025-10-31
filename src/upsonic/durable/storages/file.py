import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ..storage import DurableExecutionStorage, ExecutionState


class FileDurableStorage(DurableExecutionStorage):
    """
    File-based storage backend for durable execution.
    
    This storage backend persists execution state to the filesystem.
    Each execution is stored as a separate JSON file for easy inspection and debugging.
    
    Features:
    - Simple file-per-execution model
    - Human-readable JSON format
    - Automatic directory creation
    - File locking for concurrent safety
    
    Example:
        ```python
        storage = FileDurableStorage(path="./durable_states")
        durable = DurableExecution(storage=storage)
        ```
    """
    
    def __init__(self, path: str = "./durable_states"):
        """
        Initialize file-based storage.
        
        Args:
            path: Directory path to store execution state files
        """
        self.path = Path(path)
        self._lock = None
        self._initialized = False
        
        self.path.mkdir(parents=True, exist_ok=True)
    
    def _ensure_lock(self):
        """Ensure lock is initialized (lazy initialization for async safety)."""
        if not self._initialized:
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                import threading
                self._lock = threading.Lock()
            self._initialized = True
    
    def _get_file_path(self, execution_id: str) -> Path:
        """Get the file path for an execution ID."""
        safe_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in execution_id)
        return self.path / f"{safe_id}.json"
    
    async def save_state_async(
        self, 
        execution_id: str, 
        state: ExecutionState
    ) -> None:
        """
        Save execution state to a file.
        
        Args:
            execution_id: Unique identifier for the execution
            state: ExecutionState containing all checkpoint data
        """
        self._ensure_lock()
        
        state["saved_at"] = datetime.now(timezone.utc).isoformat()
        state["execution_id"] = execution_id
        
        file_path = self._get_file_path(execution_id)
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                await self._write_file_async(file_path, state)
        else:
            with self._lock:
                self._write_file_sync(file_path, state)
    
    async def _write_file_async(self, file_path: Path, state: ExecutionState):
        """Write file asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_file_sync, file_path, state)
    
    def _write_file_sync(self, file_path: Path, state: ExecutionState):
        """Write file synchronously."""
        temp_path = file_path.with_suffix('.tmp')
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        temp_path.replace(file_path)
    
    async def load_state_async(
        self, 
        execution_id: str
    ) -> Optional[ExecutionState]:
        """
        Load execution state from a file.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            ExecutionState if found, None otherwise
        """
        self._ensure_lock()
        
        file_path = self._get_file_path(execution_id)
        
        if not file_path.exists():
            return None
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                state = await self._read_file_async(file_path)
        else:
            with self._lock:
                state = self._read_file_sync(file_path)
        
        return ExecutionState(state) if state else None
    
    async def _read_file_async(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_file_sync, file_path)
    
    def _read_file_sync(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read file synchronously."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not read state file {file_path}: {e}", "FileDurableStorage")
            return None
    
    async def delete_state_async(
        self, 
        execution_id: str
    ) -> bool:
        """
        Delete execution state file.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_lock()
        
        file_path = self._get_file_path(execution_id)
        
        if not file_path.exists():
            return False
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                deleted = await self._delete_file_async(file_path)
        else:
            with self._lock:
                deleted = self._delete_file_sync(file_path)
        
        return deleted
    
    async def _delete_file_async(self, file_path: Path) -> bool:
        """Delete file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_file_sync, file_path)
    
    def _delete_file_sync(self, file_path: Path) -> bool:
        """Delete file synchronously."""
        try:
            file_path.unlink()
            return True
        except OSError:
            return False
    
    async def list_executions_async(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions from files.
        
        Args:
            status: Filter by status ('running', 'paused', 'completed', 'failed')
            limit: Maximum number of executions to return
            
        Returns:
            List of execution metadata dictionaries
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                executions = await self._list_executions_internal_async(status, limit)
        else:
            with self._lock:
                executions = self._list_executions_internal_sync(status, limit)
        
        return executions
    
    async def _list_executions_internal_async(
        self,
        status: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Async internal method to list executions."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_executions_internal_sync, status, limit)
    
    def _list_executions_internal_sync(
        self,
        status: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Sync internal method to list executions."""
        result = []
        
        for file_path in self.path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                if status and state.get("status") != status:
                    continue
                
                result.append({
                    "execution_id": state.get("execution_id", file_path.stem),
                    "status": state.get("status"),
                    "step_name": state.get("step_name"),
                    "step_index": state.get("step_index"),
                    "timestamp": state.get("timestamp"),
                    "saved_at": state.get("saved_at"),
                    "error": state.get("error"),
                })
            except (json.JSONDecodeError, IOError):
                # Skip corrupt files
                continue
        
        result.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if limit:
            result = result[:limit]
        
        return result
    
    async def cleanup_old_executions_async(
        self,
        older_than_days: int = 7
    ) -> int:
        """
        Cleanup old completed/failed execution files.
        
        Args:
            older_than_days: Delete executions older than this many days
            
        Returns:
            Number of executions deleted
        """
        self._ensure_lock()
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                deleted = await self._cleanup_old_executions_internal_async(cutoff_date)
        else:
            with self._lock:
                deleted = self._cleanup_old_executions_internal_sync(cutoff_date)
        
        return deleted
    
    async def _cleanup_old_executions_internal_async(self, cutoff_date: datetime) -> int:
        """Async internal cleanup method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cleanup_old_executions_internal_sync, cutoff_date)
    
    def _cleanup_old_executions_internal_sync(self, cutoff_date: datetime) -> int:
        """Sync internal cleanup method."""
        deleted_count = 0
        
        for file_path in self.path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                if state.get("status") not in ["completed", "failed"]:
                    continue
                
                timestamp_str = state.get("timestamp")
                if not timestamp_str:
                    continue
                
                # Parse timestamp and ensure it's timezone-aware
                timestamp_str_fixed = timestamp_str.replace('Z', '+00:00')
                timestamp = datetime.fromisoformat(timestamp_str_fixed)
                # If timestamp is naive, assume it's UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                if timestamp < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
            except (json.JSONDecodeError, IOError, ValueError):
                # Skip corrupt files or files with invalid timestamps
                continue
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total = 0
        by_status = {}
        
        for file_path in self.path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                total += 1
                status = state.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
            except (json.JSONDecodeError, IOError):
                continue
        
        return {
            "backend": "file",
            "path": str(self.path),
            "total_executions": total,
            "by_status": by_status,
        }

