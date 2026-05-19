from upsonic import Agent, Task

# Create agent and execute task
agent = Agent(model="openai/gpt-4o")
task = Task(description="Explain quantum computing in simple terms")
result = agent.do(task)

# Access task metadata
print(f"Task ID: {task.task_id}")
print(f"Task Usage ID: {task.task_usage_id}")
print(f"Duration: {task.usage.duration}")
print(f"Start time: {task.start_time}")
print(f"End time: {task.end_time}")