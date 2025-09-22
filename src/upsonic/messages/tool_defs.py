# my_agent_framework/messages/tool_defs.py

from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")

class ToolReturn(BaseModel, Generic[T]):
    """
    A structured return value for tools that separates the raw output from the
    content that should be sent back to the model.
    
    This is a powerful optimization pattern:
    - 'return_value' can be a complex object (e.g., a DataFrame, a custom class)
      for use in your application's internal logic.
    - 'content' is a concise string summary of the result, optimized for the
      LLM to conserve tokens and reduce noise.
    """
    return_value: T = Field(..., description="The raw, structured return value of the tool.")
    content: str = Field(..., description="The string content to be passed back to the model.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata associated with the tool's execution.")