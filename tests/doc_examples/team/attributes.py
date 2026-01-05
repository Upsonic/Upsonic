from upsonic import Agent, Task, Team
from upsonic import Memory
from upsonic.storage.providers.sqlite import SqliteStorage

# Create SQLite storage and memory
storage = SqliteStorage(
    db_file="memory.db",
    sessions_table_name="sessions",
    profiles_table_name="profiles"
)

memory = Memory(storage=storage, session_id="team_001")

# Create specialized agents
data_analyst = Agent(
    "openai/gpt-4o",
    name="DataAnalyst",
    role="Data analysis expert"
)

report_writer = Agent(
    "openai/gpt-4o",
    name="ReportWriter",
    role="Technical writer"
)

# Create team with configuration
team = Team(
    agents=[data_analyst, report_writer],
    mode="coordinate",
    model="openai/gpt-4o",
    memory=memory,
    ask_other_team_members=True
)

# Define tasks
tasks = [
    Task("Analyze quarterly sales data"),
    Task("Write a comprehensive report based on the analysis")
]

# Execute team
result = team.do(tasks)
print(result)