from upsonic import Agent, Task, Team

# Create specialized agents
data_analyst = Agent(
    model="openai/gpt-4o",
    name="Data Analyst",
    role="Data Analysis Expert",
    goal="Analyze data and extract insights"
)

report_writer = Agent(
    model="openai/gpt-4o",
    name="Report Writer",
    role="Business Report Specialist",
    goal="Create professional business reports"
)

# Create team with coordination
team = Team(
    agents=[data_analyst, report_writer],
    mode="coordinate",
    model="openai/gpt-4o"  # Required for leader agent
)

# Define tasks
tasks = [
    Task(description="Analyze Q4 sales data and identify trends"),
    Task(description="Create executive summary of findings")
]

# Leader agent coordinates the team
result = team.do(tasks)
print(result)