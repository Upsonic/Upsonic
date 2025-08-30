import json
from pydantic import BaseModel
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task

def turn_task_to_string(task: "Task"):
    the_dict = {}
    the_dict["id"] = task.task_id
    the_dict["description"] = task.description
    the_dict["attachments"] = task.attachments
    the_dict["response"] = str(task.response)

    string_of_dict = json.dumps(the_dict)
    return string_of_dict