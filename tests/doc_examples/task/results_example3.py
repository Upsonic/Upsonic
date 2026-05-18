from upsonic import Agent, Task

# Create agent and execute task
agent = Agent(model="openai/gpt-4o")
task = Task(description="Write a short poem about technology")
result = agent.do(task)

# Access cost information from the task's TaskUsage object.
usage = task.usage
print(f"Total cost: ${usage.cost}")
print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
