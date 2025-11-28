"""Tool wrapper implementations."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from upsonic.tools.base import Tool, ToolMetadata
from upsonic.tools.config import ToolConfig
from upsonic.tools.schema import FunctionSchema, function_schema, GenerateToolJsonSchema

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task


class FunctionTool(Tool):
    """Wrapper for function-based tools."""
    
    def __init__(
        self,
        function: Callable,
        schema: FunctionSchema,
        config: Optional[ToolConfig] = None
    ):
        self.function = function
        self.config = config or ToolConfig()
        
        # Create metadata with universal fields
        metadata = ToolMetadata(
            name=function.__name__,
            description=schema.description,
            kind='function',
            is_async=schema.is_async,
            strict=config.strict if config and config.strict is not None else False
        )
        
        super().__init__(
            name=function.__name__,
            description=schema.description,
            schema=schema,
            metadata=metadata
        )
        
        self.is_async = schema.is_async
    
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool function."""
        # Convert dict arguments to Pydantic models if needed
        converted_kwargs = self._convert_dicts_to_pydantic(kwargs)
        
        if self.is_async:
            return await self.function(*args, **converted_kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.function(*args, **converted_kwargs)
            )
    
    def _convert_dicts_to_pydantic(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary arguments to Pydantic models based on function type hints."""
        import inspect
        from typing import get_type_hints, get_origin, get_args
        from pydantic import BaseModel
        
        # Get function signature and type hints
        sig = inspect.signature(self.function)
        
        try:
            type_hints = get_type_hints(self.function)
        except Exception:
            # Fallback if type hints can't be retrieved
            type_hints = {}
        
        converted_kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                value = kwargs[param_name]
                type_hint = type_hints.get(param_name)
                
                # Skip if no type hint or if value is already the correct type
                if not type_hint:
                    converted_kwargs[param_name] = value
                    continue
                
                # Check if value is already a Pydantic model instance
                if isinstance(value, BaseModel):
                    converted_kwargs[param_name] = value
                    continue
                
                # Handle Pydantic model conversion for dicts and lists
                if isinstance(value, (dict, list)):
                    try:
                        converted_value = self._convert_value_to_pydantic(value, type_hint, param_name)
                        converted_kwargs[param_name] = converted_value
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(
                            f"Failed to convert parameter '{param_name}' to Pydantic model: {e}. "
                            f"Type hint: {type_hint}, Value type: {type(value)}",
                            "PydanticConverter"
                        )
                        raise ValueError(
                            f"Cannot convert parameter '{param_name}' to expected type {type_hint}. "
                            f"Received {type(value).__name__}. Error: {e}"
                        )
                else:
                    converted_kwargs[param_name] = value
        
        return converted_kwargs
    
    def _is_pydantic_model_type(self, type_hint: Any) -> bool:
        """Check if a type hint represents a Pydantic BaseModel."""
        from pydantic import BaseModel
        try:
            return isinstance(type_hint, type) and issubclass(type_hint, BaseModel)
        except (TypeError, AttributeError):
            return False
    
    def _convert_value_to_pydantic(self, value: Any, type_hint: Any, param_name: str = "") -> Any:
        """Convert a value to a Pydantic model if the type hint indicates it should be one."""
        from pydantic import BaseModel
        from typing import get_origin, get_args, Union
        
        # Handle direct Pydantic model types
        if self._is_pydantic_model_type(type_hint):
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected dict for Pydantic model {type_hint.__name__}, got {type(value).__name__}"
                )
            
            try:
                converted = self._convert_dict_to_pydantic_recursive(value, type_hint)
                # Validate the conversion succeeded
                if not isinstance(converted, type_hint):
                    raise ValueError(
                        f"Conversion to {type_hint.__name__} failed: result is {type(converted).__name__}"
                    )
                return converted
            except Exception as e:
                raise ValueError(
                    f"Failed to convert dict to {type_hint.__name__}: {e}"
                )
        
        # Handle Optional[PydanticModel] and Union types
        origin = get_origin(type_hint)
        
        if origin is Union:
            args = get_args(type_hint)
            # Filter out NoneType for Optional
            pydantic_args = [arg for arg in args if self._is_pydantic_model_type(arg)]
            
            if pydantic_args:
                # Handle None values for Optional
                if value is None:
                    return None
                
                # Try to convert using Pydantic models in the Union
                if isinstance(value, dict):
                    last_error = None
                    for pydantic_type in pydantic_args:
                        try:
                            converted = self._convert_dict_to_pydantic_recursive(value, pydantic_type)
                            # Validate conversion succeeded
                            if isinstance(converted, pydantic_type):
                                return converted
                        except Exception as e:
                            last_error = e
                            continue
                    # If all conversions failed, raise the last error with context
                    raise ValueError(
                        f"Failed to convert dict to any Union type {[t.__name__ for t in pydantic_args]}. "
                        f"Last error: {last_error}"
                    )
                elif isinstance(value, BaseModel):
                    # Already a Pydantic model, check if it's one of the Union types
                    for pydantic_type in pydantic_args:
                        if isinstance(value, pydantic_type):
                            return value
            # If not a dict/model or no Pydantic types in Union, return as-is
            return value
        
        # Handle List[PydanticModel] - recursively convert each item
        if origin is list:
            if not isinstance(value, list):
                raise TypeError(f"Expected list, got {type(value).__name__}")
            
            args = get_args(type_hint)
            if args and self._is_pydantic_model_type(args[0]):
                pydantic_type = args[0]
                converted_list = []
                
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        try:
                            converted_item = self._convert_dict_to_pydantic_recursive(item, pydantic_type)
                            # Validate the conversion
                            if not isinstance(converted_item, pydantic_type):
                                raise ValueError(
                                    f"List item {i} conversion failed: expected {pydantic_type.__name__}, "
                                    f"got {type(converted_item).__name__}"
                                )
                            converted_list.append(converted_item)
                        except Exception as e:
                            raise ValueError(
                                f"Failed to convert list item {i} to {pydantic_type.__name__}: {e}"
                            )
                    elif isinstance(item, pydantic_type):
                        # Already the correct type
                        converted_list.append(item)
                    else:
                        raise TypeError(
                            f"List item {i} must be dict or {pydantic_type.__name__}, "
                            f"got {type(item).__name__}"
                        )
                
                return converted_list
            else:
                # List of non-Pydantic types, return as-is
                return value
        
        # Return original value if no conversion needed
        return value
    
    def _convert_dict_to_pydantic_recursive(self, data: dict, model_class: type) -> Any:
        """Recursively convert a dictionary to a Pydantic model, handling nested models."""
        from pydantic import BaseModel, ValidationError
        from typing import get_origin, get_args, Union
        
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for {model_class.__name__}, got {type(data).__name__}")
        
        # Get the model's field annotations
        try:
            from typing import get_type_hints
            field_annotations = get_type_hints(model_class)
        except Exception:
            # Fallback to model fields or annotations
            if hasattr(model_class, 'model_fields'):
                # Pydantic v2
                field_annotations = {name: field.annotation for name, field in model_class.model_fields.items()}
            else:
                # Pydantic v1 fallback
                field_annotations = getattr(model_class, '__annotations__', {})
        
        converted_data = {}
        
        for field_name, field_value in data.items():
            if field_name in field_annotations:
                field_type = field_annotations[field_name]
                try:
                    converted_data[field_name] = self._convert_field_value(
                        field_name, field_value, field_type, model_class.__name__
                    )
                except Exception as e:
                    # If field conversion fails, keep the original value
                    # Pydantic will handle validation
                    converted_data[field_name] = field_value
            else:
                # Field not in annotations - might be extra field or Pydantic will handle it
                converted_data[field_name] = field_value
        
        # Try to create the Pydantic model
        try:
            result = model_class(**converted_data)
            return result
        except ValidationError as e:
            # Pydantic validation failed - try to add defaults and retry
            if hasattr(model_class, 'model_fields'):
                # Pydantic v2 - try adding defaults for missing fields
                for field_name, field_info in model_class.model_fields.items():
                    if field_name not in converted_data:
                        # Check if field has a default value
                        if hasattr(field_info, 'default') and field_info.default is not None:
                            from pydantic_core import PydanticUndefined
                            if field_info.default != PydanticUndefined:
                                converted_data[field_name] = field_info.default
                        elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                            converted_data[field_name] = field_info.default_factory()
                
                # Try again with defaults
                try:
                    result = model_class(**converted_data)
                    return result
                except Exception as e2:
                    # Still failed - raise original validation error
                    raise ValueError(
                        f"Validation failed for {model_class.__name__}: {str(e)}"
                    )
            
            # No defaults available or Pydantic v1 - raise original error
            raise ValueError(
                f"Validation failed for {model_class.__name__}: {str(e)}"
            )
        except Exception as e:
            # Other errors (not validation)
            raise ValueError(
                f"Failed to create {model_class.__name__}: {str(e)}"
            )
    
    def _convert_field_value(self, field_name: str, field_value: Any, field_type: Any, parent_class: str) -> Any:
        """Convert a single field value to its expected type."""
        from pydantic import BaseModel
        from typing import get_origin, get_args, Union
        
        # Handle None values
        if field_value is None:
            return None
        
        # Handle nested Pydantic models
        if self._is_pydantic_model_type(field_type):
            if isinstance(field_value, BaseModel):
                return field_value
            if isinstance(field_value, dict):
                return self._convert_dict_to_pydantic_recursive(field_value, field_type)
            raise TypeError(
                f"Field '{field_name}' in {parent_class}: expected dict or {field_type.__name__}, "
                f"got {type(field_value).__name__}"
            )
        
        # Handle List[PydanticModel]
        origin = get_origin(field_type)
        if origin is list:
            if not isinstance(field_value, list):
                raise TypeError(
                    f"Field '{field_name}' in {parent_class}: expected list, got {type(field_value).__name__}"
                )
            
            args = get_args(field_type)
            if args and self._is_pydantic_model_type(args[0]):
                pydantic_type = args[0]
                converted_list = []
                for i, item in enumerate(field_value):
                    if isinstance(item, pydantic_type):
                        converted_list.append(item)
                    elif isinstance(item, dict):
                        converted_item = self._convert_dict_to_pydantic_recursive(item, pydantic_type)
                        converted_list.append(converted_item)
                    else:
                        raise TypeError(
                            f"Field '{field_name}' in {parent_class}, list item {i}: "
                            f"expected dict or {pydantic_type.__name__}, got {type(item).__name__}"
                        )
                return converted_list
            else:
                # List of non-Pydantic types
                return field_value
        
        # Handle Optional[PydanticModel] and Union types
        if origin is Union:
            args = get_args(field_type)
            pydantic_args = [arg for arg in args if self._is_pydantic_model_type(arg)]
            
            if pydantic_args:
                # Try to convert to one of the Pydantic types
                if isinstance(field_value, dict):
                    for pydantic_type in pydantic_args:
                        try:
                            return self._convert_dict_to_pydantic_recursive(field_value, pydantic_type)
                        except Exception:
                            continue
                    raise ValueError(
                        f"Field '{field_name}' in {parent_class}: failed to convert to any Union type"
                    )
                elif isinstance(field_value, BaseModel):
                    return field_value
        
        # Return as-is for other types
        return field_value


class AgentTool(Tool):
    """Wrapper for agent-based tools."""
    
    def __init__(self, agent: Any):
        self.agent = agent
        
        # Generate tool name and description
        agent_name = getattr(agent, 'name', None) or f"Agent_{id(agent)}"
        agent_role = getattr(agent, 'role', None)
        agent_goal = getattr(agent, 'goal', None)
        agent_name = getattr(agent, 'name', None)
        system_prompt = getattr(agent, 'system_prompt', None)
        
        # Create method name
        method_name = f"ask_{self._sanitize_name(agent_name)}"
        
        # Create description
        description_parts = [f"Delegate tasks to {agent_name}"]
        if agent_role:
            description_parts.append(f"Role: {agent_role}")
        if agent_goal:
            description_parts.append(f"Goal: {agent_goal}")
        if agent_name:
            description_parts.append(f"Name: {agent_name}")
        if system_prompt:
            description_parts.append(f"Specialty: {system_prompt}...")
        
        description = ". ".join(description_parts)
        
        # Create a FunctionSchema for this agent tool
        # Create schema with agent parameters
        agent_func = self._create_agent_function()
        agent_func.__name__ = method_name
        agent_func.__doc__ = description
        
        agent_schema = function_schema(
            function=agent_func,
            schema_generator=GenerateToolJsonSchema,
            require_parameter_descriptions=False
        )
        
        # Create metadata with universal fields
        metadata = ToolMetadata(
            name=method_name,
            description=description,
            kind='agent',
            is_async=True,
            strict=False
        )
        
        super().__init__(
            name=method_name,
            description=description,
            schema=agent_schema,
            metadata=metadata
        )
    
    def _create_agent_function(self) -> Callable:
        """Create a dummy function for the schema."""
        async def agent_function(request: str) -> Any:
            """Delegate request to agent."""
            pass
        return agent_function
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as method name."""
        import re
        # Remove non-alphanumeric characters and convert to snake_case
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.lower().strip('_')
    
    async def execute(self, request: str, **kwargs: Any) -> Any:
        """Execute the agent with the given request."""
        # Import here to avoid circular imports
        from upsonic.tasks.tasks import Task
        
        # Create task for the agent
        task = Task(description=request)
        
        # Execute based on agent capabilities
        if hasattr(self.agent, 'do_async'):
            result = await self.agent.do_async(task)
        elif hasattr(self.agent, 'do'):
            # Run sync method in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.agent.do(task)
            )
        else:
            raise AttributeError(f"Agent {self.agent} has no do or do_async method")
        
        # Convert result to string if needed
        return str(result) if result is not None else "No response from agent"
