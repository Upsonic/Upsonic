try:
    import cloudpickle
except ImportError:
    from upsonic.utils.printing import import_error
    import_error("cloudpickle", "cloudpickle is not installed, please install it using `pip install cloudpickle`")

from typing import Any, Dict, Optional
from datetime import datetime, timezone


def _prepare_context_for_serialization(context: Any) -> Dict[str, Any]:
    """
    Prepare StepContext for serialization by extracting serializable data.
    
    This removes references to agent, model, and other unpicklable objects
    while preserving the essential state (messages, response, final_output, etc.).
        
        Args:
        context: StepContext object
            
        Returns:
        Dictionary with serializable context data
        """
    # Extract only the serializable fields from context
    # Don't serialize agent/model - they'll be provided on resume
    context_data = {
        "is_streaming": getattr(context, 'is_streaming', False),
        "messages": getattr(context, 'messages', []),
        "response": getattr(context, 'response', None),
        "final_output": getattr(context, 'final_output', None),
        # Don't serialize these - they're runtime-only or will be reconstructed:
        # - agent (has locks, will be provided on resume)
        # - model (has locks, will be provided on resume)  
        # - task (serialized separately)
        # - state (graph state, complex)
        # - stream_result (runtime object)
        # - streaming_events (ephemeral)
        # - continuation fields (handled separately)
        # - _memory_handler (runtime context manager)
    }
    
    return context_data


def serialize_state(
    task: Any,
    context: Any,
    step_index: int,
    step_name: str,
    status: str = "running",
    error: Optional[str] = None,
    agent_state: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Serialize complete execution state using cloudpickle.
    
    This function serializes the entire execution state including the task,
    step context, and any additional agent state. Cloudpickle handles
    complex objects including:
    - Pydantic models (with full class information)
    - Lambdas and closures
    - Custom classes and functions
    - Nested data structures
        
        Args:
        task: Task object to serialize
        context: StepContext object to serialize
        step_index: Index of current/last step
        step_name: Name of current/last step
        status: Execution status ('running', 'paused', 'completed', 'failed')
        error: Error message if any
        agent_state: Additional agent state to preserve
        
    Returns:
        Bytes containing pickled state
        
    Example:
        ```python
        state_bytes = serialize_state(
            task=task,
            context=context,
            step_index=5,
            step_name="model_execution",
            status="running"
        )
        ```
    """
    # Prepare context for serialization (remove unpicklable objects)
    context_data = _prepare_context_for_serialization(context)
    
    state_dict = {
        "version": "2.0",  # Version 2.0 indicates cloudpickle format
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "context_data": context_data,  # Store as dict, not full context
        "step_index": step_index,
        "step_name": step_name,
        "status": status,
        "error": error,
        "agent_state": agent_state or {},
    }
    
    return cloudpickle.dumps(state_dict)


def deserialize_state(data: bytes) -> Dict[str, Any]:
    """
    Deserialize execution state from cloudpickle bytes.
    
    This function deserializes the complete execution state, reconstructing
    all objects exactly as they were when serialized. Unlike JSON serialization,
    cloudpickle preserves full type information and can reconstruct complex
    objects including class instances and functions.
    
    Args:
        data: Bytes containing pickled state
        
    Returns:
        Dictionary containing deserialized state:
            - version: str (serialization version)
            - timestamp: str (ISO format timestamp)
            - task: Task object (fully reconstructed)
            - context_data: Dict (context data, needs reconstruction with current agent/model)
            - step_index: int
            - step_name: str
            - status: str
            - error: Optional[str]
            - agent_state: Dict[str, Any]
            
    Raises:
        Exception: If deserialization fails
        
    Example:
        ```python
        state = deserialize_state(state_bytes)
        task = state['task']
        context_data = state['context_data']
        step_index = state['step_index']
        ```
    """
    state_dict = cloudpickle.loads(data)
    
    # For backward compatibility, rename 'context' to 'context_data' if present
    if 'context' in state_dict and 'context_data' not in state_dict:
        state_dict['context_data'] = state_dict.pop('context')
    
    return state_dict


def reconstruct_context(context_data: Dict[str, Any], task: Any, agent: Any, model: Any) -> Any:
    """
    Reconstruct StepContext from serialized data with current agent and model references.
    
    When resuming execution, we need to create a new StepContext with the current
    agent and model instances, then restore the serialized state (messages, response, etc.).
    
    Args:
    context_data: Dictionary containing serialized context data
    task: Current task object
        agent: Current agent instance
        model: Current model instance
        
    Returns:
    Reconstructed StepContext object with current agent/model references
    
    Example:
        ```python
        checkpoint = deserialize_state(state_bytes)
        context = reconstruct_context(
            checkpoint['context_data'],
            task=checkpoint['task'],
            agent=current_agent,
            model=current_model
        )
        ```
    """
    from upsonic.agent.pipeline import StepContext
    
    # Create new context with current agent and model
    context = StepContext(
        task=task,
        agent=agent,
        model=model,
        is_streaming=context_data.get('is_streaming', False),
    )
    
    # Restore saved state
    context.messages = context_data.get('messages', [])
    context.response = context_data.get('response')
    context.final_output = context_data.get('final_output')
    
    return context
    

def serialize_to_base64(
        task: Any,
        context: Any,
        step_index: int,
        step_name: str,
        status: str = "running",
        error: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    Serialize execution state to base64 string.
    
    Convenience function that serializes state and encodes as base64.
    Useful for storage systems that work better with text than binary data.
    
    Args:
    task: Task object to serialize
    context: StepContext object to serialize
        step_index: Index of current/last step
        step_name: Name of current/last step
        status: Execution status
        error: Error message if any
        agent_state: Additional agent state to preserve
        
    Returns:
    Base64-encoded string containing pickled state
    
    Example:
        ```python
        state_str = serialize_to_base64(task, context, 5, "model_execution")
        ```
    """
    import base64
    
    state_bytes = serialize_state(
        task=task,
        context=context,
        step_index=step_index,
        step_name=step_name,
        status=status,
        error=error,
        agent_state=agent_state
    )
    
    return base64.b64encode(state_bytes).decode('utf-8')


def deserialize_from_base64(data: str) -> Dict[str, Any]:
    """
    Deserialize execution state from base64 string.
    
    Convenience function that decodes base64 and deserializes state.
    Counterpart to serialize_to_base64().
        
        Args:
        data: Base64-encoded string containing pickled state
            
        Returns:
        Dictionary containing deserialized state
        
    Raises:
        Exception: If deserialization fails
        binascii.Error: If base64 decoding fails
        
    Example:
        ```python
        state = deserialize_from_base64(state_str)
        ```
    """
    import base64
    
    state_bytes = base64.b64decode(data.encode('utf-8'))
    return deserialize_state(state_bytes)


# Legacy compatibility functions for migration
def to_json_compatible(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert cloudpickle state to JSON-compatible format for inspection.
    
    This function is provided for debugging and inspection purposes.
    It creates a simplified JSON-compatible view of the state that can
    be printed or logged, but cannot be used for deserialization.
        
        Args:
        state_dict: State dictionary from deserialize_state()
            
        Returns:
        JSON-compatible dictionary (simplified, for inspection only)
        
    Note:
        This is for debugging only. The returned dict cannot be used
        to reconstruct the original state.
    """
    return {
        "version": state_dict.get("version"),
        "timestamp": state_dict.get("timestamp"),
        "step_index": state_dict.get("step_index"),
        "step_name": state_dict.get("step_name"),
        "status": state_dict.get("status"),
        "error": state_dict.get("error"),
        "agent_state": state_dict.get("agent_state", {}),
        "task_type": type(state_dict.get("task")).__name__ if state_dict.get("task") else None,
        "context_type": type(state_dict.get("context")).__name__ if state_dict.get("context") else None,
    }
