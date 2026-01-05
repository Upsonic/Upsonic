from upsonic import Agent, Task, Team

# Create specialized agents
researcher = Agent(
    "openai/gpt-4o",
    name="Researcher",
    role="Research specialist"
)

writer = Agent(
    "openai/gpt-4o",
    name="Writer",
    role="Content writer"
)

# Create team
team = Team(
    agents=[researcher, writer],
    mode="sequential"
)

# Create tasks
tasks = [
    Task("Research latest AI trends"),
    Task("Write a blog post about the research")
]

# Execute team
result = team.do(tasks)
print(result)