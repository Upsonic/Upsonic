"""Unit tests for tool schema generation."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from upsonic.tools.schema import (
    generate_function_schema,
    validate_tool_function,
    SchemaGenerationError,
    FunctionSchema,
)
from pydantic import BaseModel


class TestToolSchema:
    """Test suite for tool schema generation."""

    def test_tool_schema_generation(self):
        """Test schema generation."""

        def test_function(query: str, limit: int = 10) -> str:
            """Test function.

            Args:
                query: The search query.
                limit: Maximum results.

            Returns:
                The result string.
            """
            return f"Result: {query}"

        schema = generate_function_schema(test_function)

        assert isinstance(schema, FunctionSchema)
        assert schema.name == "test_function"
        assert schema.description is not None
        assert "query" in schema.parameters_schema["properties"]
        assert "limit" in schema.parameters_schema["properties"]
        assert "query" in schema.parameters_schema["required"]
        assert "limit" not in schema.parameters_schema["required"]

    def test_tool_schema_validation(self):
        """Test schema validation."""

        def valid_function(query: str) -> str:
            """Valid function."""
            return f"Result: {query}"

        def invalid_function(query):
            """Invalid function without type hints."""
            return f"Result: {query}"

        errors = validate_tool_function(valid_function)
        assert len(errors) == 0

        errors = validate_tool_function(invalid_function)
        assert len(errors) > 0

    def test_tool_schema_from_function(self):
        """Test schema from function."""

        def test_function(
            name: str, age: int, active: bool = True, tags: list[str] = None
        ) -> dict:
            """Test function with various types.

            Args:
                name: Person name.
                age: Person age.
                active: Whether active.
                tags: List of tags.

            Returns:
                Result dictionary.
            """
            return {"name": name, "age": age}

        schema = generate_function_schema(test_function)

        assert schema.name == "test_function"
        assert schema.parameters_schema["properties"]["name"]["type"] == "string"
        assert schema.parameters_schema["properties"]["age"]["type"] == "integer"
        assert schema.parameters_schema["properties"]["active"]["type"] == "boolean"
        assert "name" in schema.parameters_schema["required"]
        assert "age" in schema.parameters_schema["required"]
        assert "active" not in schema.parameters_schema["required"]

    def test_tool_schema_with_pydantic_model(self):
        """Test schema with Pydantic models."""

        class UserModel(BaseModel):
            name: str
            age: int

        def test_function(user: UserModel) -> str:
            """Test function with Pydantic model.

            Args:
                user: User model.

            Returns:
                Result string.
            """
            return f"User: {user.name}"

        schema = generate_function_schema(test_function)

        assert schema.name == "test_function"
        assert "user" in schema.parameters_schema["properties"]

    def test_tool_schema_async_function(self):
        """Test schema for async function."""

        async def async_function(query: str) -> str:
            """Async function.

            Args:
                query: The query.

            Returns:
                Result string.
            """
            return f"Result: {query}"

        schema = generate_function_schema(async_function)

        assert schema.is_async is True
        assert schema.name == "async_function"

    def test_tool_schema_missing_docstring(self):
        """Test schema with missing docstring."""

        def no_docstring(query: str) -> str:
            return f"Result: {query}"

        schema = generate_function_schema(no_docstring)

        assert schema.name == "no_docstring"
        # Should still generate schema even without docstring

    def test_tool_schema_validation_errors(self):
        """Test validation error reporting."""

        def function_without_type_hints(param):
            """Function without type hints."""
            return param

        errors = validate_tool_function(function_without_type_hints)
        assert len(errors) > 0
        assert any("type hint" in error.lower() for error in errors)

    def test_tool_schema_optional_parameters(self):
        """Test schema with optional parameters."""

        def test_function(required: str, optional: Optional[str] = None) -> str:
            """Test function.

            Args:
                required: Required parameter.
                optional: Optional parameter.

            Returns:
                Result string.
            """
            return f"{required}: {optional}"

        schema = generate_function_schema(test_function)

        assert "required" in schema.parameters_schema["required"]
        assert "optional" not in schema.parameters_schema["required"]
