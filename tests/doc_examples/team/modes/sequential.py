from upsonic import Agent, Task, Team

# Create specialized agents
researcher = Agent(
    model="openai/gpt-4o",
    name="Researcher",
    role="Research Specialist",
    goal="Find accurate information and data"
)

writer = Agent(
    model="openai/gpt-4o",
    name="Writer",
    role="Content Writer",
    goal="Create clear and engaging content"
)

# Create sequential team
team = Team(
    agents=[researcher, writer],
    mode="sequential"
)

# Define tasks
tasks = [
    Task(description="Research the latest developments in quantum computing"),
    Task(description="Write a blog post about quantum computing for general audience")
]

# Execute team workflow
result = team.do(tasks)
print(result)