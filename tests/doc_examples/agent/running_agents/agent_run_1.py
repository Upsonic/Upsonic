from upsonic import Agent, Task

# Create agent
agent = Agent("openai/gpt-4o")

# Execute with Task object
task = Task("What is the capital of France?")
result = agent.do(task)
print(result)  # Output: Paris

# Or execute directly with a string
result = agent.do("What is the capital of France?")
print(result)  # Output: Paris