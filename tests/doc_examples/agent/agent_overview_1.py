from upsonic import Agent, Task

# Create an agent
agent = Agent("openai/gpt-4o")

# Execute with Task object
task = Task("What is 2 + 2?")
result = agent.do(task)
print(result)  # Output: 4

# Or execute directly with a string (auto-converted to Task)
result = agent.do("What is 2 + 2?")
print(result)  # Output: 4