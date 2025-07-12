import json
from pydantic import BaseModel
from typing import Any, Optional, Union

from ..tasks.tasks import Task
from .base_strategy import ContextProcessingStrategy


class TaskContextStrategy(ContextProcessingStrategy):
    """Strategy for processing Task context items."""

    def can_process(self, context_item: Any) -> bool:
        """Check if this strategy can process the given context item."""
        return isinstance(context_item, Task)

    def process(self, context_item: Any) -> str:
        """Process the task context item."""
        if not self.can_process(context_item):
            raise ValueError(f"TaskContextStrategy cannot process {type(context_item)}")

        task = context_item
        return f"Task ID ({task.get_task_id()}): {turn_task_to_string(task)}\n"

    def get_section_name(self) -> str:
        """Get the XML section name for tasks."""
        return "Tasks"

    def validate(self, context_item: Any) -> Optional[str]:
        """Validate the task context item."""
        if not isinstance(context_item, Task):
            return f"Expected Task, got {type(context_item)}"

        if not hasattr(context_item, "get_task_id"):
            return "Task missing get_task_id method"

        return None


def turn_task_to_string(task: Task):
    """Convert task to JSON string representation."""
    the_dict = {}
    the_dict["id"] = task.task_id
    the_dict["description"] = task.description
    the_dict["images"] = task.images
    the_dict["response"] = str(task.response)

    string_of_dict = json.dumps(the_dict)
    return string_of_dict
