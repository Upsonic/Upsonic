from .utils.db import agents


class Agent:
    def __init__(self, role, goal, backstory):

        global agents
        agents.append({"role": role, "goal": goal, "backstory": backstory})
