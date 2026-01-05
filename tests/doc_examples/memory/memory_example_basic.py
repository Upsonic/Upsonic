from upsonic import Agent, Task
from upsonic.storage import Memory, SqliteStorage

# Create persistent storage
storage = SqliteStorage(
    agent_sessions_table_name="support_sessions",
    db_file="./support_memory.db"
)

# Create memory for customer support
memory = Memory(
    storage=storage,
    session_id="support_session_001",
    user_id="customer_123",
    full_session_memory=True,      # Remember conversation history
    user_analysis_memory=True,     # Learn about the customer
    num_last_messages=10,          # Keep last 10 turns for context
    model="openai/gpt-4o"
)

# Support agent
agent = Agent(
    model="openai/gpt-4o",
    name="Support Agent",
    memory=memory
)

# Conversation 1
task1 = Task(description="I'm having trouble logging in to my account")
result1 = agent.do(task1)
print("Response 1:", result1)

# Conversation 2 - agent remembers context and user
task2 = Task(description="I tried resetting my password but didn't receive the email")
result2 = agent.do(task2)
print("Response 2:", result2)

# Next session - agent remembers customer history
memory_new_session = Memory(
    storage=storage,
    session_id="support_session_002",  # New session
    user_id="customer_123",            # Same customer
    full_session_memory=True,
    user_analysis_memory=True,         # Loads existing profile
    model="openai/gpt-4o"
)

agent_new = Agent(model="openai/gpt-4o", memory=memory_new_session)
task3 = Task(description="Hello, I'm back. What was my problem?")
result3 = agent_new.do(task3)  # Agent knows customer's previous issues
print("Response 3:", result3)