from .utils.db import agents

class Agent:
    """
    Represents an agent within the system.

    This class defines an agent with a specific role, goal, and backstory. Upon initialization,
    the agent is added to the global list of agents.

    Attributes:
    - role (str): The role of the agent.
    - goal (str): The goal or objective of the agent.
    - backstory (str): The backstory or history of the agent.

    Methods:
    - __init__(role, goal, backstory): Initializes the Agent object and adds it to the global list of agents.

    Global Variables:
    - agents (list): A global list containing information about all agents in the system.
    """
    def __init__(self, role, goal, backstory):
        """
        Initializes a new Agent object and adds it to the global list of agents.

        Parameters:
        - role (str): The role of the agent.
        - goal (str): The goal or objective of the agent.
        - backstory (str): The backstory or history of the agent.

        Returns:
        None
        """
        global agents
        agents.append({"role": role, "goal": goal, "backstory": backstory})
