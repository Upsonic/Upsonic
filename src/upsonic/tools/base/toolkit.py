from abc import ABC, abstractmethod
from typing import List, Any, Optional
from upsonic.tools.decorators.tool_decorator import get_tools_from_instance


class Toolkit(ABC):
    """
    Abstract base class for all tool implementations.

    Provides a common interface for tools and handles automatic discovery
    of methods decorated with @tool.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the toolkit.

        Args:
            name: Optional name for the toolkit. If not provided,
                  will use the class name.
        """
        self._name = name or self.__class__.__name__

    def __control__(self) -> bool:
        """
        Validate that the toolkit is ready to use.

        This method can be overridden by subclasses to perform
        specific validation checks (e.g., API keys, connections).

        Returns:
            True if the toolkit is ready to use, False otherwise
        """
        return True

    @property
    def name(self) -> str:
        """Get the toolkit name."""
        return self._name

    def functions(self) -> List[Any]:
        """
        Return the list of tool functions to be used by the agent.

        This method automatically discovers all methods decorated with @tool
        and returns them as a list.

        Returns:
            List of bound methods for each decorated method
        """
        return get_tools_from_instance(self)

    @abstractmethod
    def get_description(self) -> str:
        """
        Return a description of what this toolkit does.

        This method must be implemented by subclasses to provide
        a clear description of the toolkit's purpose.

        Returns:
            A string description of the toolkit
        """
        pass
