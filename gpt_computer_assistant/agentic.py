from crewai import Agent as crewai_Agent

from .utils.db import agents

class Agent(crewai_Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global agents
        agents.append(self)



