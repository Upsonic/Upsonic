from abc import ABC

class BaseAgent(ABC):
    """
    An abstract base class for all agent implementations.

    This class serves as a common, non-functional parent to break circular
    dependencies between complex components like Graph and specific agent
    implementations. Any class inheriting from this can be used as an agent
    in a Graph.
    """
    pass