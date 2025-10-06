from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args, get_origin
from inspect import Parameter, signature

from pydantic import BaseModel, create_model
from pydantic.json_schema import JsonSchemaValue, model_json_schema

from upsonic.tools.base import ToolSchema
from upsonic._griffe import doc_descriptions


class SchemaGenerationError(Exception):
    """Error raised during schema generation."""
    pass


@dataclass
class FunctionSchema:
    """Schema information extracted from a function."""
    
    function: Callable[..., Any]
    name: str
    description: Optional[str]
    parameters_schema: Dict[str, Any]
    return_schema: Optional[Dict[str, Any]]
    is_async: bool
    takes_ctx: bool = False
    positional_params: List[str] = None
    var_positional_param: Optional[str] = None
    
    def __post_init__(self):
        if self.positional_params is None:
            self.positional_params = []


def is_async_callable(obj: Any) -> bool:
    """Check if an object is an async callable."""
    return inspect.iscoroutinefunction(obj) or (
        hasattr(obj, '__call__') and inspect.iscoroutinefunction(obj.__call__)
    )


def extract_type_schema(type_hint: Any) -> Dict[str, Any]:
    """Extract JSON schema from a Python type hint."""
    if type_hint is inspect.Parameter.empty or type_hint is None:
        return {"type": "null"}
    
    # Handle basic types
    if type_hint is str:
        return {"type": "string"}
    elif type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif type_hint is list or get_origin(type_hint) is list:
        items_type = get_args(type_hint)[0] if get_args(type_hint) else Any
        return {
            "type": "array",
            "items": extract_type_schema(items_type)
        }
    elif type_hint is dict or get_origin(type_hint) is dict:
        return {"type": "object"}
    elif isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        # For Pydantic models, use their schema
        return type_hint.model_json_schema()
    else:
        # Default to any type
        return {}


def generate_function_schema(
    function: Callable[..., Any],
    docstring_format: str = 'auto',
    require_parameter_descriptions: bool = False,
    takes_ctx: bool = False,
) -> FunctionSchema:
    """
    Generate schema information from a function.
    
    Args:
        function: The function to analyze
        docstring_format: The docstring format ('google', 'numpy', 'sphinx', 'auto')
        require_parameter_descriptions: Whether to require parameter descriptions
        takes_ctx: Whether the function takes a RunContext as first parameter
        
    Returns:
        FunctionSchema object containing all schema information
        
    Raises:
        SchemaGenerationError: If schema generation fails
    """
    try:
        sig = signature(function)
    except ValueError as e:
        raise SchemaGenerationError(f"Cannot get signature for {function.__name__}: {e}")
    
    # Get function name and async status
    func_name = function.__name__
    is_async = is_async_callable(function)
    
    # Extract description and parameter descriptions from docstring
    description, param_descriptions = doc_descriptions(
        function, sig, docstring_format=docstring_format
    )
    
    # Build parameter schema
    properties = {}
    required = []
    positional_params = []
    var_positional_param = None
    
    # Get type hints
    try:
        type_hints = inspect.get_annotations(function)
    except Exception:
        type_hints = {}
    
    # Process parameters
    param_list = list(sig.parameters.items())
    for idx, (param_name, param) in enumerate(param_list):
        # Skip RunContext parameter if present
        if takes_ctx and idx == 0:
            continue
            
        # Skip self/cls parameters
        if param_name in ('self', 'cls'):
            continue
            
        # Get type hint
        type_hint = type_hints.get(param_name, param.annotation)
        if type_hint is inspect.Parameter.empty:
            type_hint = Any
            
        # Handle different parameter kinds
        if param.kind == Parameter.VAR_KEYWORD:
            # **kwargs - allow additional properties
            continue
        elif param.kind == Parameter.VAR_POSITIONAL:
            # *args
            var_positional_param = param_name
            properties[param_name] = {
                "type": "array",
                "items": extract_type_schema(type_hint),
                "description": param_descriptions.get(param_name)
            }
        else:
            # Regular parameter
            param_schema = extract_type_schema(type_hint)
            if param_descriptions.get(param_name):
                param_schema["description"] = param_descriptions[param_name]
                
            properties[param_name] = param_schema
            
            # Check if required
            if param.default is Parameter.empty:
                required.append(param_name)
                
            # Track positional parameters
            if param.kind == Parameter.POSITIONAL_ONLY:
                positional_params.append(param_name)
    
    # Check for missing parameter descriptions if required
    if require_parameter_descriptions:
        missing = set(required) - set(param_descriptions.keys())
        if missing:
            raise SchemaGenerationError(
                f"Missing parameter descriptions for {func_name}: {', '.join(missing)}"
            )
    
    # Build final parameters schema
    parameters_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }
    
    # Extract return type schema
    return_type = type_hints.get('return', sig.return_annotation)
    return_schema = None
    if return_type is not sig.empty and return_type is not None:
        return_schema = extract_type_schema(return_type)
    
    return FunctionSchema(
        function=function,
        name=func_name,
        description=description,
        parameters_schema=parameters_schema,
        return_schema=return_schema,
        is_async=is_async,
        takes_ctx=takes_ctx,
        positional_params=positional_params,
        var_positional_param=var_positional_param
    )


def validate_tool_function(func: Callable) -> List[str]:
    """
    Validate a function to ensure it meets tool requirements.
    
    Args:
        func: The function to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        sig = signature(func)
    except ValueError:
        errors.append(f"Cannot get signature for function")
        return errors
    
    # Check for type hints
    type_hints = inspect.get_annotations(func)
    
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue
            
        # Check parameter type hints
        if param_name not in type_hints and param.annotation is inspect.Parameter.empty:
            errors.append(f"Parameter '{param_name}' is missing type hint")
    
    # Check return type hint
    if 'return' not in type_hints and sig.return_annotation is inspect.Signature.empty:
        errors.append("Function is missing return type hint")
    
    # Check docstring
    if not inspect.getdoc(func):
        errors.append("Function is missing docstring")
    
    return errors
