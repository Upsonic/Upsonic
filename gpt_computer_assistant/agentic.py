
from .utils.db import agents

class Agent():
    def __init__(self, role, goal, backstory):
        from crewai import Agent as crewai_Agent
        global agents
        agents.append(crewai_Agent(role=role, goal=goal, backstory=backstory))



