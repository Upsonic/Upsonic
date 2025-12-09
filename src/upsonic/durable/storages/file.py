import os
import cloudpickle
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ..storage import DurableExecutionStorage, ExecutionState


class FileDurableStorage(DurableExecutionStorage):
    """
    File-based storage backend for durable execution.
    
    This storage backend persists execution state to the filesystem.
    Each execution is stored as a separate cloudpickle file for reliable serialization
    of complex Python objects.
    
    Features:
    - Simple file-per-execution model
    - Binary cloudpickle format for complete object serialization
    - Automatic directory creation
    - File locking for concurrent safety
    - Metadata file for quick inspection
    
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
        return self.path / f"{safe_id}.pkl"
    
    def _get_metadata_path(self, execution_id: str) -> Path:
        """Get the metadata file path for an execution ID."""
        safe_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in execution_id)
        return self.path / f"{safe_id}.meta"
    
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
        metadata_path = self._get_metadata_path(execution_id)
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                await self._write_file_async(file_path, metadata_path, state)
        else:
            with self._lock:
                self._write_file_sync(file_path, metadata_path, state)
    
    async def _write_file_async(self, file_path: Path, metadata_path: Path, state: ExecutionState):
        """Write file asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_file_sync, file_path, metadata_path, state)
    
    def _write_file_sync(self, file_path: Path, metadata_path: Path, state: ExecutionState):
        """Write file synchronously."""
        temp_path = file_path.with_suffix('.tmp')
        
        # Write the cloudpickle file (consistent with serializer.py)
        with open(temp_path, 'wb') as f:
            cloudpickle.dump(state, f)
        
        temp_path.replace(file_path)
        
        # Write metadata file for quick inspection without unpickling
        metadata = state.get("metadata", {})
        if metadata:
            try:
                import json
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "execution_id": state.get("execution_id"),
                        "saved_at": state.get("saved_at"),
                        **metadata
                    }, f, indent=2)
            except Exception:
                # If metadata write fails, it's not critical
                pass
    
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
            with open(file_path, 'rb') as f:
                return cloudpickle.load(f)
        except (IOError, Exception) as e:
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
        metadata_path = self._get_metadata_path(execution_id)
        
        if not file_path.exists():
            return False
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                deleted = await self._delete_file_async(file_path, metadata_path)
        else:
            with self._lock:
                deleted = self._delete_file_sync(file_path, metadata_path)
        
        return deleted
    
    async def _delete_file_async(self, file_path: Path, metadata_path: Path) -> bool:
        """Delete file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_file_sync, file_path, metadata_path)
    
    def _delete_file_sync(self, file_path: Path, metadata_path: Path) -> bool:
        """Delete file synchronously."""
        try:
            file_path.unlink()
            # Also delete metadata file if it exists
            if metadata_path.exists():
                try:
                    metadata_path.unlink()
                except OSError:
                    pass
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
        
        This method reads metadata files for quick listing without unpickling.
        
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
        
        # Try to read metadata files first (faster)
        for meta_path in self.path.glob("*.meta"):
            try:
                import json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if status and metadata.get("status") != status:
                    continue
                
                result.append({
                    "execution_id": metadata.get("execution_id", meta_path.stem),
                    "status": metadata.get("status"),
                    "step_name": metadata.get("step_name"),
                    "step_index": metadata.get("step_index"),
                    "timestamp": metadata.get("timestamp"),
                    "saved_at": metadata.get("saved_at"),
                })
            except (json.JSONDecodeError, IOError):
                # Skip corrupt metadata files
                continue
        
        # If no metadata files found, fall back to reading cloudpickle files
        if not result:
            for file_path in self.path.glob("*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        state = cloudpickle.load(f)
                    
                    # Extract metadata from the state wrapper
                    metadata = state.get("metadata", {})
                    
                    if status and metadata.get("status") != status:
                        continue
                    
                    result.append({
                        "execution_id": state.get("execution_id", file_path.stem),
                        "status": metadata.get("status"),
                        "step_name": metadata.get("step_name"),
                        "step_index": metadata.get("step_index"),
                        "timestamp": metadata.get("timestamp"),
                        "saved_at": state.get("saved_at"),
                    })
                except (IOError, Exception):
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
        
        # Try metadata files first
        for meta_path in self.path.glob("*.meta"):
            try:
                import json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if metadata.get("status") not in ["completed", "failed"]:
                    continue
                
                timestamp_str = metadata.get("timestamp")
                if not timestamp_str:
                    continue
                
                # Parse timestamp and ensure it's timezone-aware
                timestamp_str_fixed = timestamp_str.replace('Z', '+00:00')
                timestamp = datetime.fromisoformat(timestamp_str_fixed)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                if timestamp < cutoff_date:
                    # Delete both cloudpickle and metadata files
                    pkl_path = self.path / f"{meta_path.stem}.pkl"
                    if pkl_path.exists():
                        pkl_path.unlink()
                    meta_path.unlink()
                    deleted_count += 1
            except (json.JSONDecodeError, IOError, ValueError):
                # Skip corrupt files
                continue
        
        # Fallback: check cloudpickle files without metadata
        for file_path in self.path.glob("*.pkl"):
            meta_path = self.path / f"{file_path.stem}.meta"
            if meta_path.exists():
                # Already processed via metadata
                continue
            
            try:
                with open(file_path, 'rb') as f:
                    state = cloudpickle.load(f)
                
                metadata = state.get("metadata", {})
                
                if metadata.get("status") not in ["completed", "failed"]:
                    continue
                
                timestamp_str = metadata.get("timestamp")
                if not timestamp_str:
                    continue
                
                timestamp_str_fixed = timestamp_str.replace('Z', '+00:00')
                timestamp = datetime.fromisoformat(timestamp_str_fixed)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                if timestamp < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
            except (IOError, ValueError, Exception):
                # Skip corrupt files
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
        
        # Count from metadata files (faster)
        for meta_path in self.path.glob("*.meta"):
            try:
                import json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                total += 1
                status = metadata.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
            except (json.JSONDecodeError, IOError):
                # Skip corrupt files
                continue
        
        # Fallback: count cloudpickle files without metadata
        if total == 0:
            for file_path in self.path.glob("*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        state = cloudpickle.load(f)
                    
                    total += 1
                    metadata = state.get("metadata", {})
                    status = metadata.get("status", "unknown")
                    by_status[status] = by_status.get(status, 0) + 1
                except (IOError, Exception):
                    # Skip corrupt files
                    continue
        
        return {
            "backend": "file",
            "path": str(self.path),
            "total_executions": total,
            "by_status": by_status,
        }
