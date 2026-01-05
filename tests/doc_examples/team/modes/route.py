from upsonic import Agent, Task, Team

# Create domain specialists
legal_expert = Agent(
    model="openai/gpt-4o",
    name="Legal Expert",
    role="Legal Advisor",
    goal="Provide legal guidance and compliance information",
    system_prompt="You are an expert in corporate law and regulations"
)

tech_expert = Agent(
    model="openai/gpt-4o",
    name="Tech Expert",
    role="Technology Specialist",
    goal="Provide technical solutions and architecture advice",
    system_prompt="You are an expert in software architecture and cloud systems"
)

# Create routing team
team = Team(
    agents=[legal_expert, tech_expert],
    mode="route",
    model="openai/gpt-4o"  # Required for router agent
)

# Router selects best expert
task = Task(description="What are the best practices for implementing OAuth 2.0?")
result = team.do(task)  # Automatically routed to tech_expert
print(result)