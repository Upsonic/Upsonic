"""
Planning ToolKit - Task decomposition and planning for DeepAgent.

Provides the write_todos tool that forces deliberate task decomposition.
This is a "cognitive forcing function" - the act of calling the tool and
structuring the plan induces more careful reasoning in the model.
"""

from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from upsonic.tools import tool, ToolKit
from upsonic.agent.deepagent.constants import WRITE_TODOS_TOOL_DESCRIPTION


class Todo(BaseModel):
    """
    A single todo item for task decomposition.
    
    Attributes:
        content: Description of the todo item
        status: Current status (pending, in_progress, completed, cancelled)
        id: Unique identifier for the todo
    """
    content: str = Field(
        description="Description of the todo item",
    )
    status: Literal["pending", "in_progress", "completed", "cancelled"] = Field(
        default="pending",
        description="Current status of the todo"
    )
    id: str = Field(
        description="Unique identifier for the todo"
    )
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty and within length limit."""
        if not v or not v.strip():
            raise ValueError("Todo content cannot be empty")
        return v.strip()
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Todo ID cannot be empty")
        return v.strip()


class TodoList(BaseModel):
    """
    A list of todos for task decomposition.
    - Minimum 2 todos required
    - All todos must have valid content, status, and id
    """
    todos: List[Todo] = Field(
        min_length=2,
        description="List of todo items (minimum 2 required)"
    )
    
    @field_validator('todos')
    @classmethod
    def validate_unique_ids(cls, v: List[Todo]) -> List[Todo]:
        """Ensure all todo IDs are unique."""
        ids = [todo.id for todo in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Todo IDs must be unique")
        return v


class PlanningToolKit(ToolKit):
    """
    Planning and task decomposition toolkit for DeepAgent.
    
    Provides the write_todos tool which is a "no-op" cognitive forcing function.
    The tool performs minimal computation - its value comes from forcing the LLM
    to explicitly decompose complex tasks into discrete steps.
    
    Features:
    - Task-specific todo storage
    - Merge-by-ID update strategy
    - Automatic system prompt injection via pipeline step
    - Contextual feedback messages
    
    Usage:
        ```python
        from upsonic import Agent, Task
        from upsonic.agent.deepagent.tools import PlanningToolKit
        
        # Create toolkit (needs reference to current task)
        toolkit = PlanningToolKit()
        
        # The toolkit gets the task reference dynamically during execution
        agent = Agent(model="openai/gpt-4o", tools=[toolkit])
        
        task = Task(description="Complex multi-step task")
        result = agent.do(task)
        ```
    """
    
    def __init__(self):
        """
        Initialize the planning toolkit.
        
        Note: The toolkit will receive task reference dynamically
        during tool execution via the tool call context.
        """
        # The current task will be set dynamically during execution
        self._current_task = None
    
    def set_current_task(self, task: Any) -> None:
        """
        Set the current task for todo storage.
        
        This is called by the agent before tool execution to ensure
        todos are stored in the correct task instance.
        
        Args:
            task: The task being executed
        """
        self._current_task = task
    
    @tool
    async def write_todos(self, todos: Optional[List[Dict[str, Any]]] = None) -> str:
        """Placeholder docstring - will be replaced."""
        # Validate we have a current task
        if self._current_task is None:
            return (
                "❌ Error: No active task found.\n\n"
                "The write_todos tool requires an active task context.\n"
                "This is an internal error - please report this issue."
            )
        
        # Validate todos parameter is provided
        if todos is None or not todos:
            return (
                "❌ Error: Missing 'todos' parameter.\n\n"
                "The write_todos tool requires a 'todos' parameter with a list of todo items.\n"
                "Each todo must have: 'content', 'status', and 'id' fields.\n"
                "Minimum 2 todos required.\n\n"
                "Example:\n"
                "write_todos([\n"
                '  {"content": "Research topic", "status": "pending", "id": "1"},\n'
                '  {"content": "Write report", "status": "pending", "id": "2"}\n'
                "])"
            )
        
        try:
            # Get current todos from task (if any)
            current_todos = getattr(self._current_task, '_task_todos', None)
            
            is_first_call = current_todos is None or len(current_todos) == 0
            
            # For first call, enforce minimum 2 todos
            # For updates, allow any number since we're merging with existing plan
            if is_first_call:
                # Validate with TodoList (enforces min 2 todos)
                todo_list = TodoList(todos=[Todo(**t) for t in todos])
                from upsonic.utils.printing import planning_todo_list
                planning_todo_list(todo_list, debug=True)
                new_todos = todo_list.todos
            else:
                # For updates, just validate individual todos (no minimum requirement)
                new_todos = [Todo(**t) for t in todos]
                
                # Validate unique IDs among the new todos
                new_ids = [todo.id for todo in new_todos]
                if len(new_ids) != len(set(new_ids)):
                    raise ValueError("Todo IDs must be unique")
            
            if is_first_call:
                # First call: Store all todos
                self._current_task._task_todos = new_todos
                
                # Format response for first call
                todo_summary = []
                for i, todo in enumerate(new_todos, 1):
                    todo_summary.append(f"{i}. [{todo.status}] {todo.content}")
                
                response = f"✅ Plan created with {len(new_todos)} tasks:\n\n"
                response += "\n".join(todo_summary)
                response += "\n\nYou can now proceed with your work. Update the plan as you progress."
                
                return response
            else:
                # Update call: Merge by ID
                # Create a dict of current todos by ID for fast lookup
                current_by_id = {todo.id: todo for todo in current_todos}
                
                # Merge new todos
                updated_count = 0
                added_count = 0
                
                for new_todo in new_todos:
                    if new_todo.id in current_by_id:
                        # Update existing todo
                        old_todo = current_by_id[new_todo.id]
                        
                        # Check what changed
                        status_changed = old_todo.status != new_todo.status
                        content_changed = old_todo.content != new_todo.content
                        
                        if status_changed or content_changed:
                            # Update the todo
                            current_by_id[new_todo.id] = new_todo
                            updated_count += 1
                    else:
                        # Add new todo
                        current_by_id[new_todo.id] = new_todo
                        added_count += 1
                
                # Rebuild the list (maintain order by ID)
                self._current_task._task_todos = list(current_by_id.values())
                
                # Format response for update
                response = "✅ Plan updated:\n"
                
                if updated_count > 0:
                    response += f"   • Updated: {updated_count} task(s)\n"
                if added_count > 0:
                    response += f"   • Added: {added_count} new task(s)\n"
                
                if updated_count == 0 and added_count == 0:
                    response += "   • No changes (all todos already exist with same values)\n"
                
                response += f"\nCurrent plan: {len(self._current_task._task_todos)} total task(s)"
                
                # Show current status breakdown
                status_counts = {}
                for todo in self._current_task._task_todos:
                    status_counts[todo.status] = status_counts.get(todo.status, 0) + 1
                
                status_parts = []
                if status_counts.get("completed", 0) > 0:
                    status_parts.append(f"{status_counts['completed']} completed")
                if status_counts.get("in_progress", 0) > 0:
                    status_parts.append(f"{status_counts['in_progress']} in progress")
                if status_counts.get("pending", 0) > 0:
                    status_parts.append(f"{status_counts['pending']} pending")
                if status_counts.get("cancelled", 0) > 0:
                    status_parts.append(f"{status_counts['cancelled']} cancelled")
                
                if status_parts:
                    response += f" ({', '.join(status_parts)})"
                # Print update visualization
                from upsonic.utils.printing import planning_todo_update
                planning_todo_update(
                    self._current_task._task_todos,
                    updated_count,
                    added_count,
                    status_counts,
                    debug=True
                )
                
                return response
                
        except ValueError as e:
            # Pydantic validation error
            error_msg = str(e)
            
            # Provide helpful guidance based on error type
            if "too_short" in error_msg.lower() or "at least 2" in error_msg.lower() or "min_length" in error_msg.lower() or "minimum" in error_msg.lower():
                return (
                    "❌ Error: Minimum 2 todos required\n\n"
                    "The write_todos tool requires at least 2 tasks for proper planning.\n"
                    "If you only have 1 task, consider breaking it down into subtasks,\n"
                    "or just proceed without using the planning tool."
                )
            elif "unique" in error_msg.lower():
                return (
                    "❌ Error: Todo IDs must be unique\n\n"
                    "Each todo must have a unique ID.\n"
                    "Please ensure all IDs are different (e.g., '1', '2', '3')."
                )
            elif "empty" in error_msg.lower():
                return (
                    "❌ Error: Todo content or ID cannot be empty\n\n"
                    "Each todo must have:\n"
                    "- content: Non-empty description\n"
                    "- id: Non-empty identifier\n"
                    "- status: One of: pending, in_progress, completed, cancelled"
                )
            else:
                return f"❌ Error: Invalid todo format\n\n{error_msg}"
        
        except Exception as e:
            return f"❌ Error creating plan: {str(e)}"
    
    def get_current_todos(self) -> List[Dict[str, Any]]:
        """
        Get the current todo list from the active task.
        
        Returns:
            List of todo dictionaries with content, status, and id keys
        """
        if self._current_task is None:
            return []
        
        current_todos = getattr(self._current_task, '_task_todos', None)
        
        if not current_todos:
            return []
        
        return [
            {
                "id": todo.id,
                "content": todo.content,
                "status": todo.status
            }
            for todo in current_todos
        ]


# Set tool descriptions from constants
PlanningToolKit.write_todos.__doc__ = WRITE_TODOS_TOOL_DESCRIPTION
