import time



def task_start(task, agent):
    task.start_time = time.time()
    if agent.canvas:
        task.add_canvas(agent.canvas)

    from ...context.context import ContextBuilder
    task.description += ContextBuilder(task.context).build()
