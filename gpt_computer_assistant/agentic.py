
from .utils.db import agents

class Agent():
    def __init__(self, *args, **kwargs):
        from crewai import Agent as crewai_Agent
        global agents
        agents.append(crewai_Agent(*args, **kwargs))



