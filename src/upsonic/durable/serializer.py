import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from pydantic_core import to_jsonable_python


class DurableStateSerializer:
    """
    Serializes and deserializes execution state for durable storage.
    
    Uses JSON-based serialization to avoid pickle's issues:
    - No unpicklable objects (locks, connections, etc.)
    - Safe for untrusted data
    - Version-independent
    - Human-readable for debugging
    """
    
    @staticmethod
    def serialize_task(task: Any) -> Dict[str, Any]:
        """
        Serialize a Task object to a JSON-safe dictionary.
        
        Extracts only the essential state needed to reconstruct the task.
        
        Args:
            task: Task object to serialize
            
        Returns:
            Dictionary containing serialized task data
        """
        # Serialize response_format properly
        response_format = getattr(task, 'response_format', str)
        if response_format == str or response_format is str:
            response_format_serialized = "str"
        elif hasattr(response_format, '__name__'):
            # It's a Pydantic model class - store class info
            response_format_serialized = {
                "_is_pydantic_class": True,
                "class_name": response_format.__name__,
                "module": getattr(response_format, '__module__', None),
                # Note: We can't reconstruct the class, tools will need to be re-registered
            }
        else:
            response_format_serialized = "str"
        
        # Extract task data - only serialize what's needed
        start_time = getattr(task, 'start_time', None)
        end_time = getattr(task, 'end_time', None)
        
        # Convert floats to ints for time fields (Pydantic expects int)
        if isinstance(start_time, float):
            start_time = int(start_time)
        if isinstance(end_time, float):
            end_time = int(end_time)
        
        serialized = {
            "type": "task",
            "description": getattr(task, 'description', ''),
            "attachments": getattr(task, 'attachments', None),
            "tools": [],  # Tools will be re-registered on resume
            "response_format": response_format_serialized,
            "response_lang": getattr(task, 'response_lang', 'en'),
            "_response": DurableStateSerializer._serialize_response(getattr(task, '_response', None)),
            "context": DurableStateSerializer._serialize_context_data(getattr(task, 'context', None)),
            "_context_formatted": getattr(task, '_context_formatted', None),
            "price_id_": getattr(task, 'price_id_', None),
            "task_id_": getattr(task, 'task_id_', None),
            "not_main_task": getattr(task, 'not_main_task', False),
            "start_time": start_time,
            "end_time": end_time,
            "enable_thinking_tool": getattr(task, 'enable_thinking_tool', None),
            "enable_reasoning_tool": getattr(task, 'enable_reasoning_tool', None),
            "_tool_calls": getattr(task, '_tool_calls', []),
            "is_paused": getattr(task, 'is_paused', False),
            "enable_cache": getattr(task, 'enable_cache', False),
            "cache_method": getattr(task, 'cache_method', 'vector_search'),
            "cache_threshold": getattr(task, 'cache_threshold', 0.7),
            "cache_duration_minutes": getattr(task, 'cache_duration_minutes', 60),
        }
        
        return serialized
    
    @staticmethod
    def deserialize_task(data: Dict[str, Any]) -> Any:
        """
        Deserialize a Task object from a dictionary.
        
        Args:
            data: Dictionary containing serialized task data
            
        Returns:
            Reconstructed Task object
        """
        from upsonic.tasks.tasks import Task
        
        # Convert start_time and end_time to int if they're floats
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        if isinstance(start_time, float):
            start_time = int(start_time)
        if isinstance(end_time, float):
            end_time = int(end_time)
        
        # Reconstruct the task with essential data
        task = Task(
            description=data.get('description', ''),
            attachments=data.get('attachments'),
            response=DurableStateSerializer._deserialize_response(data.get('_response')),
            context=DurableStateSerializer._deserialize_context_data(data.get('context')),
            _context_formatted=data.get('_context_formatted'),
            price_id_=data.get('price_id_'),
            task_id_=data.get('task_id_'),
            not_main_task=data.get('not_main_task', False),
            start_time=start_time,
            end_time=end_time,
            response_lang=data.get('response_lang'),
            enable_thinking_tool=data.get('enable_thinking_tool'),
            enable_reasoning_tool=data.get('enable_reasoning_tool'),
            is_paused=data.get('is_paused', False),
            enable_cache=data.get('enable_cache', False),
            cache_method=data.get('cache_method', 'vector_search'),
            cache_threshold=data.get('cache_threshold', 0.7),
            cache_duration_minutes=data.get('cache_duration_minutes', 60),
        )
        
        # Restore tool calls
        if data.get('_tool_calls'):
            task._tool_calls = data['_tool_calls']
        
        return task
    
    @staticmethod
    def _serialize_response(response: Any) -> Optional[Any]:
        """
        Serialize task response.
        
        Can handle:
        - None
        - str, int, float, bool (primitive types)
        - Pydantic model instances
        - Other objects (converted to string)
        """
        if response is None:
            return None
        
        # If it's already a simple type, return as-is
        if isinstance(response, (str, int, float, bool)):
            return response
        
        # If it's a Pydantic model instance, use model_dump(mode="json")
        if hasattr(response, 'model_dump'):
            return {
                '_pydantic_model': True,
                'class_name': type(response).__name__,
                'module': type(response).__module__,
                'data': response.model_dump(mode="json")
            }
        
        # For lists and dicts, try to use to_jsonable_python
        if isinstance(response, (list, dict)):
            try:
                return to_jsonable_python(response)
            except Exception:
                return response
        
        # Otherwise convert to string
        return str(response)
    
    @staticmethod
    def _deserialize_response(data: Any) -> Any:
        """
        Deserialize task response.
        
        Returns the data in its serialized form (dict for Pydantic models).
        The actual Pydantic model reconstruction happens when tools are registered.
        """
        if data is None:
            return None
        
        # If it's a Pydantic model dump, return the data dict
        # (We can't reconstruct the class without importing it)
        if isinstance(data, dict) and data.get('_pydantic_model'):
            return data.get('data')
        
        return data
    
    @staticmethod
    def _serialize_context_data(context: Any) -> Any:
        """Serialize task context."""
        if context is None:
            return None
        
        if isinstance(context, (str, int, float, bool)):
            return context
        
        if isinstance(context, list):
            return [DurableStateSerializer._serialize_context_data(item) for item in context]
        
        if isinstance(context, dict):
            return {k: DurableStateSerializer._serialize_context_data(v) for k, v in context.items()}
        
        # For complex objects, try to extract essential data
        if hasattr(context, 'model_dump'):
            return context.model_dump()
        
        return str(context)
    
    @staticmethod
    def _deserialize_context_data(data: Any) -> Any:
        """Deserialize task context."""
        return data
    
    @staticmethod
    def serialize_context(context: Any) -> Dict[str, Any]:
        """
        Serialize a StepContext object to a JSON-safe dictionary.
        
        Extracts only the essential state, avoiding unpicklable objects.
        
        Args:
            context: StepContext object to serialize
            
        Returns:
            Dictionary containing serialized context data
        """
        return {
            "type": "step_context",
            "is_streaming": getattr(context, 'is_streaming', False),
            "model_name": getattr(getattr(context, 'model', None), 'model_name', None),
            "messages": DurableStateSerializer.serialize_messages(getattr(context, 'messages', [])),
            "response": DurableStateSerializer._serialize_model_response(getattr(context, 'response', None)),
            "final_output": DurableStateSerializer._serialize_response(getattr(context, 'final_output', None)),
            "streaming_events": [],  # Don't serialize streaming events - they're ephemeral
        }
    
    @staticmethod
    def deserialize_context(data: Dict[str, Any], task: Any, agent: Any, model: Any) -> Any:
        """
        Deserialize a StepContext object from a dictionary.
        
        Args:
            data: Dictionary containing serialized context data
            task: Reconstructed task object
            agent: Current agent instance
            model: Current model instance
            
        Returns:
            Reconstructed StepContext object
        """
        from upsonic.agent.pipeline import StepContext
        
        # Create new context with current agent and model
        context = StepContext(
            task=task,
            agent=agent,
            model=model,
            is_streaming=data.get('is_streaming', False),
        )
        
        # Restore messages
        context.messages = DurableStateSerializer.deserialize_messages(data.get('messages', {"messages": []}))
        
        # Restore response if available
        if data.get('response'):
            context.response = DurableStateSerializer._deserialize_model_response(data['response'])
        
        # Restore final output
        context.final_output = DurableStateSerializer._deserialize_response(data.get('final_output'))
        
        return context
    
    @staticmethod
    def serialize_messages(messages: List[Any]) -> Dict[str, Any]:
        """
        Serialize message list to a JSON-safe dictionary.
        
        Uses the same approach as Memory class:
        - pydantic_core.to_jsonable_python() for serialization
        
        Args:
            messages: List of ModelMessage objects (ModelRequest/ModelResponse)
            
        Returns:
            Dictionary containing serialized messages
        """
        if not messages:
            return {
                "type": "messages",
                "messages": [],
                "count": 0
            }
        
        serialized_messages = to_jsonable_python(messages)
        
        return {
            "type": "messages",
            "messages": serialized_messages,
            "count": len(messages)
        }
    
    @staticmethod
    def deserialize_messages(data: Dict[str, Any]) -> List[Any]:
        """
        Deserialize message list from a dictionary.
        
        Uses the same approach as Memory class:
        - ModelMessagesTypeAdapter.validate_python() for deserialization
        
        Args:
            data: Dictionary containing serialized messages
            
        Returns:
            Reconstructed message list
        """
        raw_messages = data.get('messages', [])
        
        if not raw_messages:
            return []
        
        try:
            from upsonic.messages import ModelMessagesTypeAdapter
            validated_messages = ModelMessagesTypeAdapter.validate_python(raw_messages)
            return validated_messages
        except Exception as e:
            # If validation fails, return empty list
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not validate messages during deserialization: {e}", "DurableSerializer")
            return []
    
    @staticmethod
    def _serialize_model_response(response: Any) -> Optional[Dict[str, Any]]:
        """
        Serialize a ModelResponse object.
        
        Uses model_dump(mode="json") and to_jsonable_python for proper serialization.
        """
        if response is None:
            return None
        
        if hasattr(response, 'model_dump'):
            try:
                # Use to_jsonable_python for consistent serialization
                return {
                    'type': 'ModelResponse',
                    'data': to_jsonable_python(response)
                }
            except Exception:
                # Fallback to model_dump
                return {
                    'type': 'ModelResponse',
                    'data': response.model_dump(mode="json")
                }
        
        return None
    
    @staticmethod
    def _deserialize_model_response(data: Optional[Dict[str, Any]]) -> Any:
        """
        Deserialize a ModelResponse object.
        
        Uses model_validate for proper reconstruction.
        """
        if data is None:
            return None
        
        try:
            from upsonic.messages import ModelResponse
            return ModelResponse.model_validate(data.get('data'))
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not deserialize ModelResponse: {e}", "DurableSerializer")
            return None
    
    @staticmethod
    def serialize_state(
        task: Any,
        context: Any,
        step_index: int,
        step_name: str,
        status: str = "running",
        error: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Serialize complete execution state.
        
        Args:
            task: Task object
            context: StepContext object
            step_index: Index of current/last step
            step_name: Name of current/last step
            status: Execution status
            error: Error message if any
            agent_state: Additional agent state to preserve
            
        Returns:
            Complete serialized state dictionary
        """
        return {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": DurableStateSerializer.serialize_task(task),
            "context": DurableStateSerializer.serialize_context(context),
            "step_index": step_index,
            "step_name": step_name,
            "status": status,
            "error": error,
            "agent_state": agent_state or {},
        }
    
    @staticmethod
    def deserialize_state(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize execution state (task only, context data kept serialized).
        
        This method deserializes the task and keeps context data in serialized form.
        The context will be reconstructed later when agent/model are available.
        
        Args:
            data: Serialized state dictionary
            
        Returns:
            Dictionary with task and serialized context data:
                - task: Task object (deserialized)
                - context_data: Dict (serialized context, not reconstructed)
                - step_index: int
                - step_name: str
                - status: str
                - error: Optional[str]
                - agent_state: Dict
        """
        return {
            "version": data.get("version", "1.0"),
            "timestamp": data.get("timestamp"),
            "task": DurableStateSerializer.deserialize_task(data["task"]),
            "context_data": data["context"],  # Keep serialized for later reconstruction
            "step_index": data["step_index"],
            "step_name": data["step_name"],
            "status": data["status"],
            "error": data.get("error"),
            "agent_state": data.get("agent_state", {}),
        }
    
    @staticmethod
    def reconstruct_context(context_data: Dict[str, Any], task: Any, agent: Any, model: Any) -> Any:
        """
        Reconstruct StepContext from serialized data with current agent/model.
        
        This is called during resume with the current agent and model instances.
        
        Args:
            context_data: Serialized context data
            task: Reconstructed task object
            agent: Current agent instance
            model: Current model instance
            
        Returns:
            Reconstructed StepContext object
        """
        return DurableStateSerializer.deserialize_context(context_data, task, agent, model)
    
    @staticmethod
    def to_json(data: Dict[str, Any]) -> str:
        """
        Convert serialized state to JSON string.
        
        Args:
            data: Serialized state dictionary
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> Dict[str, Any]:
        """
        Parse JSON string to serialized state dictionary.
        
        Args:
            json_str: JSON string
            
        Returns:
            Serialized state dictionary
        """
        return json.loads(json_str)

