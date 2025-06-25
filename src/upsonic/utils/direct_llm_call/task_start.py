import time
from ...context.builder import build_context


def task_start(task, agent):
    task.start_time = time.time()

    if agent.canvas:
        task.add_canvas(agent.canvas)

    # Always treat task.context as a list (can be None)
    ctx_objects = task.context or []

    # Append the consolidated context to the task description
    task.description += build_context(ctx_objects)
