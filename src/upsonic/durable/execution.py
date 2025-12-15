import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from .storage import DurableExecutionStorage, ExecutionState
from .serializer import serialize_state, deserialize_state, reconstruct_context


class DurableExecution:
    """
    Manages durable execution for agent tasks.
    
    This class provides the main interface for durable execution features:
    - Automatic state persistence at each pipeline step
    - Recovery from failures and interruptions
    - Resumption from last successful checkpoint
    - Multiple storage backend support
    
    The DurableExecution instance is attached to a Task and coordinates
    with the agent's pipeline to save state after each step.
    
    Example:
        ```python
        from upsonic import Task, Agent
        from upsonic.durable import DurableExecution, FileDurableStorage
        
        # Create durable execution with file storage
        storage = FileDurableStorage(path="./durable_state")
        durable = DurableExecution(storage=storage)
        
        # Create task with durable execution
        task = Task("Process payment", durable_execution=durable)
        agent = Agent("Payment Agent")
        
        # Execute with automatic state persistence
        try:
            result = agent.do(task)
        except Exception:
            # State is automatically saved on error
            pass
        
        # Resume from where it left off
        agent.continue_durable(task.durable_execution_id)
        ```
    
    Attributes:
        storage: The storage backend for persisting execution state
        execution_id: Unique identifier for this execution
        auto_cleanup: Whether to automatically cleanup on success
        debug: Enable debug logging
    """
    
    def __init__(
        self,
        storage: DurableExecutionStorage,
        execution_id: Optional[str] = None,
        auto_cleanup: bool = True,
        debug: bool = False
    ):
        """
        Initialize durable execution manager.
        
        Args:
            storage: Storage backend for persisting state
            execution_id: Optional execution ID (auto-generated if not provided)
            auto_cleanup: Automatically cleanup state on successful completion
            debug: Enable debug logging
        """
        self.storage = storage
        self.execution_id = execution_id or self._generate_execution_id()
        self.auto_cleanup = auto_cleanup
        self.debug = debug
    
    @staticmethod
    def _generate_execution_id() -> str:
        """
        Generate a unique execution ID.
        
        Returns:
            Unique execution ID string
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}-{unique_id}"
    
    async def save_checkpoint_async(
        self,
        task: Any,
        context: Any,
        step_index: int,
        step_name: str,
        status: str = "running",
        error: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a checkpoint asynchronously.
        
        This method is called by the pipeline after each successful step
        to persist the current execution state.
        
        Args:
            task: Task object
            context: StepContext object
            step_index: Index of completed step
            step_name: Name of completed step
            status: Execution status ('running', 'paused', 'completed', 'failed')
            error: Error message if any
            agent_state: Additional agent state to preserve
        """
        # Serialize state using cloudpickle
        state_bytes = serialize_state(
            task=task,
            context=context,
            step_index=step_index,
            step_name=step_name,
            status=status,
            error=error,
            agent_state=agent_state
        )
        
        # Storage expects ExecutionState (dict-like), so we wrap the bytes
        # with metadata for storage layer
        state_wrapper = ExecutionState({
            "pickle_data": state_bytes,  # cloudpickle serialized data
            "metadata": {
                "step_index": step_index,
                "step_name": step_name,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        })
        
        await self.storage.save_state_async(self.execution_id, state_wrapper)
        
        if self.debug:
            from upsonic.utils.printing import info_log
            info_log(
                f"Checkpoint saved: {self.execution_id} at step {step_index} ({step_name})",
                "DurableExecution"
            )
    
    def save_checkpoint(
        self,
        task: Any,
        context: Any,
        step_index: int,
        step_name: str,
        status: str = "running",
        error: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a checkpoint synchronously.
        
        Synchronous wrapper for save_checkpoint_async.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task to run in background
                asyncio.create_task(self.save_checkpoint_async(
                    task, context, step_index, step_name, status, error, agent_state
                ))
            else:
                loop.run_until_complete(self.save_checkpoint_async(
                    task, context, step_index, step_name, status, error, agent_state
                ))
        except RuntimeError:
            asyncio.run(self.save_checkpoint_async(
                task, context, step_index, step_name, status, error, agent_state
            ))
    
    async def load_checkpoint_async(self) -> Optional[Dict[str, Any]]:
        """
        Load the last checkpoint asynchronously.
        
        Returns:
            Dictionary with state components:
                - version: str (serialization version)
                - timestamp: str (ISO format timestamp)
                - task: Task object (fully deserialized)
                - context: StepContext object (fully deserialized)
                - step_index: int
                - step_name: str
                - status: str
                - error: Optional[str]
                - agent_state: Dict
            
            None if no checkpoint found
            
        Note:
            With cloudpickle, both task and context are fully reconstructed.
            The context will need agent/model references updated via
            reconstruct_context() before use.
        """
        state_wrapper = await self.storage.load_state_async(self.execution_id)
        
        if state_wrapper is None:
            return None
        
        # Extract the cloudpickle serialized data from the wrapper
        state_bytes = state_wrapper.get("pickle_data")
        if state_bytes is None:
            # Backward compatibility: if no pickle_data key, might be old JSON format
            from upsonic.utils.printing import warning_log
            warning_log(
                f"Checkpoint {self.execution_id} is in old JSON format, cannot deserialize",
                "DurableExecution"
            )
            return None
        
        # Deserialize using cloudpickle
        deserialized = deserialize_state(state_bytes)
        
        if self.debug:
            from upsonic.utils.printing import info_log
            info_log(
                f"Checkpoint loaded: {self.execution_id} from step {deserialized['step_index']} ({deserialized['step_name']})",
                "DurableExecution"
            )
        
        return deserialized
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the last checkpoint synchronously.
        
        Synchronous wrapper for load_checkpoint_async.
        
        Returns:
            Dictionary with state components (see load_checkpoint_async for details)
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self.load_checkpoint_async())
        except RuntimeError:
            return asyncio.run(self.load_checkpoint_async())
    
    async def mark_completed_async(self) -> None:
        """
        Mark execution as completed asynchronously.
        
        If auto_cleanup is enabled, this will also delete the execution state.
        """
        if self.auto_cleanup:
            await self.storage.delete_state_async(self.execution_id)
            if self.debug:
                from upsonic.utils.printing import info_log
                info_log(
                    f"Execution completed and cleaned up: {self.execution_id}",
                    "DurableExecution"
                )
        else:
            # Just update status to completed
            state = await self.storage.load_state_async(self.execution_id)
            if state:
                # Update status in metadata (where it's actually stored)
                if "metadata" not in state:
                    state["metadata"] = {}
                
                state["metadata"]["status"] = "completed"
                await self.storage.save_state_async(self.execution_id, state)
            
            if self.debug:
                from upsonic.utils.printing import info_log
                info_log(
                    f"Execution marked as completed: {self.execution_id}",
                    "DurableExecution"
                )
    
    def mark_completed(self) -> None:
        """
        Mark execution as completed synchronously.
        
        Synchronous wrapper for mark_completed_async.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.mark_completed_async())
            else:
                loop.run_until_complete(self.mark_completed_async())
        except RuntimeError:
            asyncio.run(self.mark_completed_async())
    
    async def mark_failed_async(self, error: str) -> None:
        """
        Mark execution as failed asynchronously.
        
        Args:
            error: Error message describing the failure
        """
        state = await self.storage.load_state_async(self.execution_id)
        if state:
            # Update status and error in metadata (where it's actually stored)
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["status"] = "failed"
            state["metadata"]["error"] = error
            await self.storage.save_state_async(self.execution_id, state)
        
        if self.debug:
            from upsonic.utils.printing import error_log
            error_log(
                f"Execution failed: {self.execution_id} - {error}",
                "DurableExecution"
            )
    
    def mark_failed(self, error: str) -> None:
        """
        Mark execution as failed synchronously.
        
        Synchronous wrapper for mark_failed_async.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.mark_failed_async(error))
            else:
                loop.run_until_complete(self.mark_failed_async(error))
        except RuntimeError:
            asyncio.run(self.mark_failed_async(error))
    
    def get_execution_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about this execution.
        
        Returns:
            Dictionary with execution metadata, or None if not found
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(self._get_execution_info_async())
        except RuntimeError:
            return asyncio.run(self._get_execution_info_async())
    
    async def _get_execution_info_async(self) -> Optional[Dict[str, Any]]:
        """Async helper for get_execution_info."""
        state_wrapper = await self.storage.load_state_async(self.execution_id)
        if not state_wrapper:
            return None
        
        # Get metadata from the wrapper (this should always exist)
        metadata = state_wrapper.get("metadata")
        if not metadata:
            # Fallback: try to extract from cloudpickle data
            pickle_data = state_wrapper.get("pickle_data")
            if pickle_data:
                try:
                    import cloudpickle
                    actual_state = cloudpickle.loads(pickle_data)
                    # Create metadata from actual state
                    metadata = {
                        "status": actual_state.get("status"),
                        "step_index": actual_state.get("step_index"),
                        "step_name": actual_state.get("step_name"),
                        "timestamp": actual_state.get("timestamp"),
                        "error": actual_state.get("error"),
                    }
                except Exception as e:
                    # If we can't deserialize, return minimal info
                    return {
                        "execution_id": self.execution_id,
                        "status": "unknown",
                        "step_index": None,
                        "step_name": None,
                        "timestamp": state_wrapper.get("saved_at"),
                        "saved_at": state_wrapper.get("saved_at"),
                        "error": f"Could not deserialize: {e}",
                    }
            else:
                # No metadata and no pickle_data
                return {
                    "execution_id": self.execution_id,
                    "status": "unknown",
                    "step_index": None,
                    "step_name": None,
                    "timestamp": state_wrapper.get("saved_at"),
                    "saved_at": state_wrapper.get("saved_at"),
                    "error": "No metadata available",
                }
        
        # Also try to get error from pickle_data if not in metadata
        error_msg = metadata.get("error")
        if not error_msg:
            pickle_data = state_wrapper.get("pickle_data")
            if pickle_data:
                try:
                    import cloudpickle
                    actual_state = cloudpickle.loads(pickle_data)
                    error_msg = actual_state.get("error")
                except:
                    pass
        
        return {
            "execution_id": self.execution_id,
            "status": metadata.get("status", "unknown"),
            "step_index": metadata.get("step_index"),
            "step_name": metadata.get("step_name"),
            "timestamp": metadata.get("timestamp"),
            "saved_at": state_wrapper.get("saved_at"),
            "error": error_msg,
        }
    
    @staticmethod
    def list_all_executions(
        storage: DurableExecutionStorage,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions in storage.
        
        Args:
            storage: Storage backend to query
            status: Filter by status
            limit: Maximum number to return
            
        Returns:
            List of execution metadata dictionaries
        """
        return storage.list_executions(status=status, limit=limit)
    
    @staticmethod
    async def list_all_executions_async(
        storage: DurableExecutionStorage,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all executions in storage asynchronously.
        
        Args:
            storage: Storage backend to query
            status: Filter by status
            limit: Maximum number to return
            
        Returns:
            List of execution metadata dictionaries
        """
        return await storage.list_executions_async(status=status, limit=limit)
    
    @staticmethod
    async def load_by_id_async(
        execution_id: str,
        storage: DurableExecutionStorage
    ) -> Optional["DurableExecution"]:
        """
        Load a DurableExecution instance by execution ID asynchronously.
        
        Args:
            execution_id: The execution ID to load
            storage: Storage backend to load from
            
        Returns:
            DurableExecution instance if found, None otherwise
        """
        state = await storage.load_state_async(execution_id)
        if state is None:
            return None
        
        durable = DurableExecution(
            storage=storage,
            execution_id=execution_id,
            auto_cleanup=False  # Don't auto-cleanup loaded executions
        )
        return durable
    
    @staticmethod
    def load_by_id(
        execution_id: str,
        storage: DurableExecutionStorage
    ) -> Optional["DurableExecution"]:
        """
        Load a DurableExecution instance by execution ID synchronously.
        
        Synchronous wrapper for load_by_id_async.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call synchronous method from async context")
            return loop.run_until_complete(
                DurableExecution.load_by_id_async(execution_id, storage)
            )
        except RuntimeError:
            return asyncio.run(
                DurableExecution.load_by_id_async(execution_id, storage)
            )
    
    def __repr__(self) -> str:
        """String representation of DurableExecution."""
        return f"DurableExecution(execution_id='{self.execution_id}', storage={type(self.storage).__name__})"

