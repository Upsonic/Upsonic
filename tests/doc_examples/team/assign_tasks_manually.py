from upsonic import Agent, Task, Team

# Create specialized agents
writer = Agent(
    model="openai/gpt-4o",
    name="Writer",
    role="Content Writer",
    goal="Create engaging content"
)

editor = Agent(
    model="openai/gpt-4o",
    name="Editor",
    role="Content Editor",
    goal="Polish and improve content"
)

# Create team
team = Team(agents=[writer, editor], mode="sequential")

# Explicitly assign agents to tasks
task1 = Task(description="Write a product description")
task1.agent = writer  # Force this task to use writer

task2 = Task(description="Edit and polish the description")
task2.agent = editor  # Force this task to use editor

# Execute with manual assignments
result = team.do([task1, task2])
print(result)