
from .utils.db import agents


from .llm import get_model


class Agent():
    def __init__(self, role, goal, backstory):
        from crewai import Agent as crewai_Agent
        global agents
        agents.append(crewai_Agent(role=role, goal=goal, backstory=backstory, llm=get_model()))



