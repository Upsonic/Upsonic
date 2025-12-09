import cloudpickle
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from ..storage import DurableExecutionStorage, ExecutionState


class RedisDurableStorage(DurableExecutionStorage):
    """
    Redis-based storage backend for durable execution.
    
    This storage backend uses Redis for distributed, scalable storage.
    Uses cloudpickle for state serialization to support complex Python objects.
    
    Benefits:
    - High performance and low latency
    - Distributed storage for multi-instance deployments
    - Built-in TTL support for automatic cleanup
    - Pub/sub capabilities for real-time monitoring
    - Full Python object serialization via cloudpickle
    
    Requires redis-py package: `pip install redis`
    
    Example:
        ```python
        storage = RedisDurableStorage(
            host="localhost",
            port=6379,
            db=0
        )
        durable = DurableExecution(storage=storage)
        ```
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "durable_exec:",
        **redis_kwargs
    ):
        """
        Initialize Redis storage.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            prefix: Key prefix for all execution states
            **redis_kwargs: Additional arguments for redis.Redis()
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis storage requires redis package. "
                "Install it with: pip install redis"
            )
        
        self.prefix = prefix
        # Don't use decode_responses since we're storing binary cloudpickle data
        self._redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # Keep as bytes for cloudpickle
            **redis_kwargs
        )
        
        # Test connection
        try:
            self._redis.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Redis at {host}:{port}: {e}")
    
    def _make_key(self, execution_id: str) -> str:
        """Create Redis key for an execution ID."""
        return f"{self.prefix}{execution_id}"
    
    def _make_metadata_key(self, execution_id: str) -> str:
        """Create Redis key for execution metadata."""
        return f"{self.prefix}meta:{execution_id}"
    
    def _make_index_key(self, status: str) -> str:
        """Create Redis set key for status index."""
        return f"{self.prefix}index:{status}"
    
    async def save_state_async(
        self, 
        execution_id: str, 
        state: ExecutionState
    ) -> None:
        """
        Save execution state to Redis.
        
        Args:
            execution_id: Unique identifier for the execution
            state: ExecutionState containing all checkpoint data
        """
        import asyncio
        
        # Add metadata
        state["saved_at"] = datetime.now(timezone.utc).isoformat()
        state["execution_id"] = execution_id
        
        # Extract metadata for indexing
        metadata = state.get("metadata", {})
        status = metadata.get("status", "running")
        step_index = metadata.get("step_index", 0)
        step_name = metadata.get("step_name", "unknown")
        timestamp = metadata.get("timestamp", datetime.now(timezone.utc).isoformat())
        
        # Serialize state to cloudpickle (consistent with serializer.py)
        state_blob = cloudpickle.dumps(state)
        
        # Redis keys
        key = self._make_key(execution_id)
        metadata_key = self._make_metadata_key(execution_id)
        
        # Run Redis operations in thread pool (they're blocking)
        def _save_to_redis():
            try:
                # Use pipeline for atomic operations
                pipe = self._redis.pipeline()
                
                # Save full state as binary cloudpickle
                pipe.set(key, state_blob)
                
                # Save lightweight metadata for listing
                # Hash fields must be strings for compatibility
                metadata_dict = {
                    b"execution_id": execution_id.encode('utf-8'),
                    b"status": status.encode('utf-8'),
                    b"step_index": str(step_index).encode('utf-8'),
                    b"step_name": step_name.encode('utf-8'),
                    b"timestamp": timestamp.encode('utf-8'),
                    b"saved_at": state.get("saved_at", "").encode('utf-8'),
                }
                pipe.hset(metadata_key, mapping=metadata_dict)
                
                # Update status index
                index_key = self._make_index_key(status)
                pipe.sadd(index_key, execution_id.encode('utf-8'))
                
                pipe.execute()
            except Exception as e:
                import sys
                print(f"ERROR: Redis save failed: {e}", file=sys.stderr)
                raise
        
        await asyncio.to_thread(_save_to_redis)
    
    async def load_state_async(
        self, 
        execution_id: str
    ) -> Optional[ExecutionState]:
        """
        Load execution state from Redis.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            ExecutionState if found, None otherwise
        """
        import asyncio
        
        key = self._make_key(execution_id)
        state_blob = await asyncio.to_thread(self._redis.get, key)
        
        if state_blob is None:
            return None
        
        try:
            state = cloudpickle.loads(state_blob)
            return ExecutionState(state)
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not deserialize state for {execution_id}: {e}", "RedisDurableStorage")
            return None
    
    async def delete_state_async(
        self, 
        execution_id: str
    ) -> bool:
        """
        Delete execution state from Redis.
        
        Args:
            execution_id: Unique identifier for the execution
            
        Returns:
            True if deleted, False if not found
        """
        import asyncio
        
        state = await self.load_state_async(execution_id)
        
        key = self._make_key(execution_id)
        metadata_key = self._make_metadata_key(execution_id)
        
        def _delete_from_redis():
            pipe = self._redis.pipeline()
            
            pipe.delete(key)
            pipe.delete(metadata_key)
            
            if state:
                metadata = state.get("metadata", {})
                status = metadata.get("status", "running")
                index_key = self._make_index_key(status)
                pipe.srem(index_key, execution_id.encode('utf-8'))
            
            results = pipe.execute()
            
            return any(r > 0 for r in results if isinstance(r, int))
        
        return await asyncio.to_thread(_delete_from_redis)
    
    async def list_executions_async(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions from Redis.
        
        Args:
            status: Filter by status ('running', 'paused', 'completed', 'failed')
            limit: Maximum number of executions to return
            
        Returns:
            List of execution metadata dictionaries
        """
        import asyncio
        
        def _list_from_redis():
            result = []
            
            if status:
                # Use index for efficient filtering
                index_key = self._make_index_key(status)
                execution_ids = self._redis.smembers(index_key)
                # Decode bytes to strings
                execution_ids = [eid.decode('utf-8') if isinstance(eid, bytes) else eid for eid in execution_ids]
            else:
                # Scan all metadata keys
                pattern = f"{self.prefix}meta:*"
                execution_ids = set()
                for key in self._redis.scan_iter(match=pattern):
                    # Decode key if it's bytes
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    # Extract execution_id from key
                    exec_id = key[len(f"{self.prefix}meta:"):]
                    execution_ids.add(exec_id)
            
            # Fetch metadata for each execution
            for execution_id in execution_ids:
                metadata_key = self._make_metadata_key(execution_id)
                metadata_raw = self._redis.hgetall(metadata_key)
                
                # Decode metadata (keys and values are bytes)
                metadata = {}
                for k, v in metadata_raw.items():
                    key_str = k.decode('utf-8') if isinstance(k, bytes) else k
                    val_str = v.decode('utf-8') if isinstance(v, bytes) else v
                    metadata[key_str] = val_str
                
                if metadata:
                    step_index_str = metadata.get("step_index", "0")
                    try:
                        step_index = int(step_index_str)
                    except ValueError:
                        step_index = 0
                    
                    result.append({
                        "execution_id": metadata.get("execution_id", execution_id),
                        "status": metadata.get("status"),
                        "step_index": step_index,
                    "step_name": metadata.get("step_name"),
                    "timestamp": metadata.get("timestamp"),
                    "saved_at": metadata.get("saved_at"),
                    "error": metadata.get("error"),
                })
            
            result.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            if limit:
                result = result[:limit]
            
            return result
        
        return await asyncio.to_thread(_list_from_redis)
    
    async def cleanup_old_executions_async(
        self,
        older_than_days: int = 7
    ) -> int:
        """
        Cleanup old completed/failed executions from Redis.
        
        Args:
            older_than_days: Delete executions older than this many days
            
        Returns:
            Number of executions deleted
        """
        import asyncio
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        deleted_count = 0
        
        for status in ["completed", "failed"]:
            index_key = self._make_index_key(status)
            execution_ids_raw = await asyncio.to_thread(self._redis.smembers, index_key)
            execution_ids = [eid.decode('utf-8') if isinstance(eid, bytes) else eid for eid in execution_ids_raw]
            
            for execution_id in execution_ids:
                metadata_key = self._make_metadata_key(execution_id)
                metadata_raw = await asyncio.to_thread(self._redis.hgetall, metadata_key)
                
                # Decode metadata
                metadata = {}
                for k, v in metadata_raw.items():
                    key_str = k.decode('utf-8') if isinstance(k, bytes) else k
                    val_str = v.decode('utf-8') if isinstance(v, bytes) else v
                    metadata[key_str] = val_str
                
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
                        deleted = await self.delete_state_async(execution_id)
                        if deleted:
                            deleted_count += 1
                except (ValueError, AttributeError):
                    # Skip invalid timestamps
                    continue
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        # Count by status using indexes
        by_status = {}
        for status in ["running", "paused", "completed", "failed"]:
            index_key = self._make_index_key(status)
            count = self._redis.scard(index_key)
            if count > 0:
                by_status[status] = count
        
        total = sum(by_status.values())
        
        redis_info = self._redis.info()
        
        return {
            "backend": "redis",
            "host": self._redis.connection_pool.connection_kwargs.get("host"),
            "port": self._redis.connection_pool.connection_kwargs.get("port"),
            "db": self._redis.connection_pool.connection_kwargs.get("db"),
            "total_executions": total,
            "by_status": by_status,
            "redis_version": redis_info.get("redis_version"),
            "used_memory_human": redis_info.get("used_memory_human"),
        }
    
    def set_ttl(self, execution_id: str, ttl_seconds: int):
        """
        Set TTL (time to live) for an execution state.
        
        Args:
            execution_id: Unique identifier for the execution
            ttl_seconds: TTL in seconds (after which Redis will auto-delete)
        """
        key = self._make_key(execution_id)
        metadata_key = self._make_metadata_key(execution_id)
        
        pipe = self._redis.pipeline()
        pipe.expire(key, ttl_seconds)
        pipe.expire(metadata_key, ttl_seconds)
        pipe.execute()
    
    def clear_all_indexes(self):
        """
        Clear all status indexes.
        Useful for cleanup or recovery operations.
        """
        for status in ["running", "paused", "completed", "failed"]:
            index_key = self._make_index_key(status)
            self._redis.delete(index_key)
    
    def rebuild_indexes(self):
        """
        Rebuild status indexes from existing execution states.
        Useful for recovery after index corruption.
        """
        self.clear_all_indexes()
        
        # Scan all execution states
        pattern = f"{self.prefix}meta:*"
        for metadata_key_raw in self._redis.scan_iter(match=pattern):
            metadata_key = metadata_key_raw.decode('utf-8') if isinstance(metadata_key_raw, bytes) else metadata_key_raw
            metadata_raw = self._redis.hgetall(metadata_key)
            
            # Decode metadata
            metadata = {}
            for k, v in metadata_raw.items():
                key_str = k.decode('utf-8') if isinstance(k, bytes) else k
                val_str = v.decode('utf-8') if isinstance(v, bytes) else v
                metadata[key_str] = val_str
            
            status = metadata.get("status")
            execution_id = metadata.get("execution_id")
            
            if status and execution_id:
                index_key = self._make_index_key(status)
                self._redis.sadd(index_key, execution_id.encode('utf-8'))
