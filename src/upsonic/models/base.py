from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict, List

from pydantic import BaseModel

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings


class BaseModelProvider(BaseModel, ABC):
    """
    An abstract base class that defines the contract for all model providers.

    This class serves as a blueprint for creating specific, self-contained
    provider classes (e.g., OpenAI, Anthropic). It ensures that every provider
    adheres to a consistent interface for configuration, metadata access, and
    provisioning a ready-to-use pydantic-ai model instance.
    """

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    @abstractmethod
    async def _provision(self) -> Tuple[Model, Optional[ModelSettings]]:
        """
        The core abstract method that every concrete provider must implement.

        This asynchronous method is responsible for taking the configuration
        stored within the instance's attributes, performing any necessary setup
        (like initializing an API client), and returning the final, ready-to-use
        pydantic-ai model and its corresponding settings object.

        Returns:
            A tuple containing:
            - The instantiated pydantic-ai model object (e.g., OpenAIResponsesModel).
            - An optional pydantic-ai model settings object if required by the model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pricing(self) -> Dict[str, float]:
        """
        Returns the pricing information for the specific model configured
        in this provider instance.
        
        Returns:
            A dictionary containing 'input' and 'output' cost per million tokens.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """
        Returns a list of special capabilities for the specific model
        configured in this provider instance (e.g., 'vision', 'audio').
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def required_environment_variables(self) -> List[str]:
        """
        Returns a list of environment variable names that are required
        for this provider to function correctly.
        """
        raise NotImplementedError