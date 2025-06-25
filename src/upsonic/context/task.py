# upsonic/context/task.py
# DEPRECATED – kept only for backward compatibility
from strategies import TaskStrategy


def turn_task_to_string(task):
    return TaskStrategy().format(task)
