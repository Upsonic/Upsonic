from typing import Literal, Dict, List
from pydantic import BaseModel, Field, ConfigDict


class Todo(BaseModel):
    """
    Represents a single todo item for task tracking.
    
    Attributes:
        content: Description of the task
        status: Current status (pending, in_progress, completed)
    """
    content: str = Field(description="Description of the task to complete")
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending",
        description="Current status of the task"
    )


class DeepAgentState(BaseModel):
    """
    Extended state for Deep Agents with todo tracking and virtual filesystem.
    
    This state is maintained across the agent's execution and provides:
    - Todo list management for complex task tracking
    - Virtual filesystem for file operations (isolated per agent)
    
    Attributes:
        todos: List of todo items for task planning and tracking
        files: Virtual filesystem mapping file paths to content
    """
    todos: List[Todo] = Field(
        default_factory=list,
        description="List of todos for task planning and tracking"
    )
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Virtual filesystem mapping file paths to content"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

