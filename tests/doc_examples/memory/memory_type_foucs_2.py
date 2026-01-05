from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage
from pydantic import BaseModel, Field
from typing import Optional

# Custom user profile schema
class UserProfile(BaseModel):
    name: Optional[str] = Field(None, description="User's name")
    expertise: Optional[str] = Field(None, description="Technical expertise level")
    interests: Optional[list] = Field(None, description="Areas of interest")

storage = InMemoryStorage()
memory = Memory(
    storage=storage,
    session_id="session_003",
    user_id="user_001",
    user_analysis_memory=True,
    user_profile_schema=UserProfile,
    model="openai/gpt-4o"
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("I'm Alice, an expert in NLP, interested in chatbots and language models")
result1 = agent.do(task1)

task2 = Task("What are my areas of interest?")
result2 = agent.do(task2)
print(result2)