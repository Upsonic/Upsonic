from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic, Type

from upsonic.uel.runnable import Runnable
from upsonic.messages import ModelResponse, TextPart

T = TypeVar('T')


class BaseOutputParser(Runnable[ModelResponse, Any], ABC, Generic[T]):
    """Base class for all output parsers.
    
    Output parsers extract structured data from ModelResponse objects.
    They inherit from Runnable, making them compatible with UEL chains.
    
    Example:
        ```python
        from upsonic.uel import ChatPromptTemplate, StrOutputParser
        from upsonic import infer_model
        
        prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
        model = infer_model("openai/gpt-4o")
        parser = StrOutputParser()
        
        # Chain components together
        chain = prompt | model | parser
        result = await chain.ainvoke({"topic": "AI"})
        # result is a string
        ```
    """
    
    @abstractmethod
    def parse(self, response: ModelResponse) -> T:
        """Parse a ModelResponse and extract the output.
        
        This is the core parsing logic that subclasses must implement.
        
        Args:
            response: The ModelResponse to parse
            
        Returns:
            The parsed output (type depends on subclass)
        """
        raise NotImplementedError()
    
    def invoke(
        self, 
        input: ModelResponse, 
        config: Optional[dict[str, Any]] = None
    ) -> T:
        """Execute the parser synchronously.
        
        Args:
            input: The ModelResponse to parse
            config: Optional runtime configuration (unused for parsers)
            
        Returns:
            The parsed output
        """
        return self.parse(input)
    
    async def ainvoke(
        self, 
        input: ModelResponse, 
        config: Optional[dict[str, Any]] = None
    ) -> T:
        """Execute the parser asynchronously.
        
        Args:
            input: The ModelResponse to parse
            config: Optional runtime configuration (unused for parsers)
            
        Returns:
            The parsed output
        """
        # Parsing is typically synchronous, but we support async for consistency
        return self.parse(input)
    
    def __repr__(self) -> str:
        """Return a string representation of the parser."""
        return f"{self.__class__.__name__}()"


class StrOutputParser(BaseOutputParser[str]):
    """Parser that extracts text content from ModelResponse as a string.
    
    This parser gets the last TextPart from the response and returns its content.
    This is the most common output format.
    
    Example:
        ```python
        from upsonic.uel import StrOutputParser
        from upsonic.messages import ModelResponse, TextPart
        
        parser = StrOutputParser()
        
        # Create a mock response
        response = ModelResponse(parts=[TextPart(content="Hello, world!")])
        
        # Parse it
        result = parser.parse(response)
        # result == "Hello, world!"
        ```
    """
    
    def parse(self, response: ModelResponse) -> str:
        """Extract text content from ModelResponse.
        
        Gets the last element from response.parts and returns its content if it's a TextPart.
        
        Args:
            response: The ModelResponse to parse
            
        Returns:
            The content of the last TextPart if it exists, otherwise empty string
        """
        # Get the last element of parts
        if not response.parts:
            return ""
        
        last_part = response.parts[-1]
        
        # Check if it's a TextPart
        if isinstance(last_part, TextPart):
            return last_part.content
        
        # If last part is not a TextPart, return empty string
        return ""
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return "StrOutputParser()"


class PydanticOutputParser(BaseOutputParser[T], Generic[T]):
    """Parser that extracts structured data from ModelResponse as a Pydantic model.
    
    This parser extracts text content from the response, parses it as JSON,
    and validates it against a Pydantic model schema.
    
    Example:
        ```python
        from pydantic import BaseModel
        from upsonic.uel import PydanticOutputParser
        from upsonic.messages import ModelResponse, TextPart
        
        class Person(BaseModel):
            name: str
            age: int
        
        parser = PydanticOutputParser(Person)
        
        # Create a mock response with JSON
        response = ModelResponse(
            parts=[TextPart(content='{"name": "Alice", "age": 30}')]
        )
        
        # Parse it
        result = parser.parse(response)
        # result is a Person instance with name="Alice", age=30
        ```
    """
    
    def __init__(self, model_class: Type[T]):
        """Initialize the Pydantic output parser.
        
        Args:
            model_class: The Pydantic BaseModel class to parse into
            
        Raises:
            ValueError: If model_class is not a Pydantic BaseModel
        """
        from pydantic import BaseModel
        
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise ValueError(
                f"model_class must be a Pydantic BaseModel class, got {type(model_class)}"
            )
        
        self.model_class = model_class
    
    def parse(self, response: ModelResponse) -> T:
        """Extract and parse structured data from ModelResponse.
        
        Gets the last element from response.parts and parses it as JSON if it's a TextPart.
        
        Args:
            response: The ModelResponse to parse
            
        Returns:
            An instance of the Pydantic model class
            
        Raises:
            ValueError: If response.parts is empty, last part is not a TextPart, or if the response cannot be parsed as JSON or doesn't match the schema
        """
        # Get the last element of parts
        if not response.parts:
            raise ValueError(
                f"ModelResponse has no parts to parse as {self.model_class.__name__}"
            )
        
        last_part = response.parts[-1]
        
        # Check if it's a TextPart
        if not isinstance(last_part, TextPart):
            raise ValueError(
                f"Last part in ModelResponse is not a TextPart (got {type(last_part).__name__}), "
                f"cannot parse as {self.model_class.__name__}"
            )
        
        text_content = last_part.content
        
        if not text_content:
            raise ValueError(
                f"TextPart content is empty, cannot parse as {self.model_class.__name__}"
            )
        
        # Parse JSON
        try:
            import json
            parsed = json.loads(text_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from model response: {e}. "
                f"Content: {text_content[:200]}"
            ) from e
        
        # Validate against Pydantic model
        try:
            if hasattr(self.model_class, 'model_validate'):
                return self.model_class.model_validate(parsed)
            else:
                # Fallback for older Pydantic versions
                return self.model_class(**parsed)
        except Exception as e:
            raise ValueError(
                f"Failed to validate response against {self.model_class.__name__}: {e}. "
                f"Parsed data: {parsed}"
            ) from e
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return f"PydanticOutputParser({self.model_class.__name__})"

