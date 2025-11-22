from upsonic.output import (
    OutputObjectDefinition,
    ToolOutput,
    NativeOutput,
    PromptedOutput,
    TextOutput,
    StructuredDict,
)


def test_output_object_definition():
    """Test OutputObjectDefinition."""
    from upsonic.tools import ObjectJsonSchema

    schema: ObjectJsonSchema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    output_def = OutputObjectDefinition(
        json_schema=schema, name="Person", description="A person object", strict=True
    )

    assert output_def.json_schema == schema
    assert output_def.name == "Person"
    assert output_def.description == "A person object"
    assert output_def.strict is True


def test_output_serialization():
    """Test output serialization."""

    # Test ToolOutput
    def output_func(text: str) -> str:
        return text.upper()

    tool_output = ToolOutput(
        type_=output_func,
        name="uppercase",
        description="Convert to uppercase",
        max_retries=3,
        strict=True,
    )

    assert tool_output.output == output_func
    assert tool_output.name == "uppercase"
    assert tool_output.description == "Convert to uppercase"
    assert tool_output.max_retries == 3
    assert tool_output.strict is True

    # Test NativeOutput
    native_output = NativeOutput(
        outputs=[str, int],
        name="StringOrInt",
        description="Either string or integer",
        strict=False,
    )

    assert len(native_output.outputs) == 2
    assert native_output.name == "StringOrInt"
    assert native_output.strict is False

    # Test PromptedOutput
    prompted_output = PromptedOutput(
        outputs=[str],
        name="TextOutput",
        description="Text output",
        template="Output: {schema}",
    )

    assert len(prompted_output.outputs) == 1
    assert prompted_output.template == "Output: {schema}"

    # Test TextOutput
    text_output = TextOutput(output_function=output_func)
    assert text_output.output_function == output_func

    # Test StructuredDict
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    structured_dict = StructuredDict(
        json_schema=schema, name="ValueDict", description="Dictionary with value"
    )

    assert structured_dict.__name__ == "_StructuredDict"
    assert hasattr(structured_dict, "__get_pydantic_json_schema__")
