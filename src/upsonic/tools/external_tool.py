from pydantic import BaseModel


from typing import Any, Dict, Optional


class ExternalToolCall(BaseModel):
    """Represents a tool call that must be executed externally."""
    tool_name: str
    tool_args: Dict[str, Any]
    result: Optional[Any] = None