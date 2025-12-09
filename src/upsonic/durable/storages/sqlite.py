import cloudpickle
import sqlite3
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ..storage import DurableExecutionStorage, ExecutionState


class SQLiteDurableStorage(DurableExecutionStorage):
    """
    SQLite-based storage backend for durable execution.
    
    This storage backend uses SQLite for persistent, queryable storage.
    Uses cloudpickle (BLOB) for state serialization to support complex Python objects.
    
    Benefits:
    - ACID transactions for data safety
    - Efficient querying and filtering
    - Lightweight with no external dependencies
    - Suitable for production use
    - Full Python object serialization via cloudpickle
    
    Example:
        ```python
        storage = SQLiteDurableStorage(db_path="./durable.db")
        durable = DurableExecution(storage=storage)
        ```
    """
    
    def __init__(self, db_path: str = "./durable_executions.db"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = None
        self._initialized = False
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema if not exists."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    error TEXT,
                    state_data BLOB NOT NULL
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_status 
                ON executions(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON executions(timestamp)
            ''')
            
            conn.commit()
        finally:
            conn.close()
    
    def _ensure_lock(self):
        """Ensure lock is initialized (lazy initialization for async safety)."""
        if not self._initialized:
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                import threading
                self._lock = threading.Lock()
            self._initialized = True
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn
    
    async def save_state_async(
        self, 
        execution_id: str, 
        state: ExecutionState
    ) -> None:
        """
        Save execution state to SQLite database.
        
        Args:
            execution_id: Unique identifier for the execution
            state: ExecutionState containing all checkpoint data
        """
        self._ensure_lock()
        
        # Add metadata
        state["saved_at"] = datetime.now(timezone.utc).isoformat()
        state["execution_id"] = execution_id
        
        # Extract metadata for indexed columns
        metadata = state.get("metadata", {})
        status = metadata.get("status", "running")
        step_index = metadata.get("step_index", 0)
        step_name = metadata.get("step_name", "unknown")
        timestamp = metadata.get("timestamp", datetime.now(timezone.utc).isoformat())
        error = metadata.get("error")
        
        # Serialize entire state to cloudpickle (consistent with serializer.py)
        state_blob = cloudpickle.dumps(state)
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                await self._save_state_internal_async(
                    execution_id, status, step_index, step_name, 
                    timestamp, state.get("saved_at"), error, state_blob
                )
        else:
            with self._lock:
                self._save_state_internal_sync(
                    execution_id, status, step_index, step_name,
                    timestamp, state.get("saved_at"), error, state_blob
                )
    
    async def _save_state_internal_async(
        self, 
        execution_id: str, 
        status: str,
        step_index: int,
        step_name: str,
        timestamp: str,
        saved_at: str,
        error: Optional[str],
        state_blob: bytes
    ):
        """Async internal save method."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._save_state_internal_sync,
            execution_id, status, step_index, step_name,
            timestamp, saved_at, error, state_blob
        )
    
    def _save_state_internal_sync(
        self, 
        execution_id: str, 
        status: str,
        step_index: int,
        step_name: str,
        timestamp: str,
        saved_at: str,
        error: Optional[str],
        state_blob: bytes
    ):
        """Sync internal save method."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO executions 
                (execution_id, status, step_index, step_name, timestamp, saved_at, error, state_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution_id,
                status,
                step_index,
                step_name,
                timestamp,
                saved_at,
                error,
                state_blob
            ))
            conn.commit()
        finally:
            conn.close()
    
    async def load_state_async(
        self, 
        execution_id: str
    ) -> Optional[ExecutionState]:
        """
        Load execution state from SQLite database.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            ExecutionState if found, None otherwise
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                state = await self._load_state_internal_async(execution_id)
        else:
            with self._lock:
                state = self._load_state_internal_sync(execution_id)
        
        return ExecutionState(state) if state else None
    
    async def _load_state_internal_async(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Async internal load method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_state_internal_sync, execution_id)
    
    def _load_state_internal_sync(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Sync internal load method."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT state_data FROM executions 
                WHERE execution_id = ?
            ''', (execution_id,))
            
            row = cursor.fetchone()
            if row:
                return cloudpickle.loads(row['state_data'])
            return None
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not deserialize state for {execution_id}: {e}", "SQLiteDurableStorage")
            return None
        finally:
            conn.close()
    
    async def delete_state_async(
        self, 
        execution_id: str
    ) -> bool:
        """
        Delete execution state from SQLite database.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_lock()
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                deleted = await self._delete_state_internal_async(execution_id)
        else:
            with self._lock:
                deleted = self._delete_state_internal_sync(execution_id)
        
        return deleted
    
    async def _delete_state_internal_async(self, execution_id: str) -> bool:
        """Async internal delete method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_state_internal_sync, execution_id)
    
    def _delete_state_internal_sync(self, execution_id: str) -> bool:
        """Sync internal delete method."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM executions WHERE execution_id = ?
            ''', (execution_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    async def list_executions_async(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions from SQLite database.
        
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
        """Async internal list method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_executions_internal_sync, status, limit)
    
    def _list_executions_internal_sync(
        self,
        status: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Sync internal list method."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            query = '''
                SELECT execution_id, status, step_index, step_name, 
                       timestamp, saved_at, error
                FROM executions
            '''
            params = []
            
            if status:
                query += ' WHERE status = ?'
                params.append(status)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    async def cleanup_old_executions_async(
        self,
        older_than_days: int = 7
    ) -> int:
        """
        Cleanup old completed/failed executions from SQLite database.
        
        Args:
            older_than_days: Delete executions older than this many days
            
        Returns:
            Number of executions deleted
        """
        self._ensure_lock()
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cutoff_str = cutoff_date.isoformat()
        
        if hasattr(self._lock, '__aenter__'):
            async with self._lock:
                deleted = await self._cleanup_old_executions_internal_async(cutoff_str)
        else:
            with self._lock:
                deleted = self._cleanup_old_executions_internal_sync(cutoff_str)
        
        return deleted
    
    async def _cleanup_old_executions_internal_async(self, cutoff_str: str) -> int:
        """Async internal cleanup method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cleanup_old_executions_internal_sync, cutoff_str)
    
    def _cleanup_old_executions_internal_sync(self, cutoff_str: str) -> int:
        """Sync internal cleanup method."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM executions 
                WHERE status IN ('completed', 'failed')
                AND timestamp < ?
            ''', (cutoff_str,))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) as total FROM executions')
            total = cursor.fetchone()['total']
            
            # Get count by status
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM executions 
                GROUP BY status
            ''')
            by_status = {row['status']: row['count'] for row in cursor.fetchall()}
            
            return {
                "backend": "sqlite",
                "db_path": str(self.db_path),
                "total_executions": total,
                "by_status": by_status,
            }
        finally:
            conn.close()
    
    def vacuum(self):
        """
        Optimize database by running VACUUM.
        This reclaims space from deleted records.
        """
        conn = self._get_connection()
        try:
            conn.execute('VACUUM')
        finally:
            conn.close()
