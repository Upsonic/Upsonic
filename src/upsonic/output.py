from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal
from abc import ABC

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import TypeAliasType, TypeVar

from upsonic import _utils
from upsonic._json_schema import InlineDefsJsonSchemaTransformer
from upsonic.tools import ObjectJsonSchema
from upsonic.utils.package.exception import UserError


__all__ = (
    # classes
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    'OutputObjectDefinition',
    # types
    'OutputDataT',
    'OutputMode',
    'StructuredOutputMode',
    'OutputSpec',
    'OutputTypeOrFunction',
    'TextOutputFunc',
)

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

DEFAULT_OUTPUT_TOOL_NAME = 'final_result'

OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the output data type of a run."""

OutputMode = Literal['text', 'tool', 'native', 'prompted', 'tool_or_text', 'image', 'auto']
"""All output modes.

- `tool_or_text` is deprecated and no longer in use.
- `auto` means the model will automatically choose a structured output mode based on the model's `ModelProfile.default_structured_output_mode`.
"""
StructuredOutputMode = Literal['tool', 'native', 'prompted']
"""Output modes that can be used for structured output. Used by ModelProfile.default_structured_output_mode"""


OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction', type[T_co] | Callable[..., Awaitable[T_co] | T_co], type_params=(T_co,)
)
"""Definition of an output type or function.

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""


TextOutputFunc = TypeAliasType(
    'TextOutputFunc',
    Callable[[str], Awaitable[T_co] | T_co],
    type_params=(T_co,),
)
"""Definition of a function that will be called to process the model's plain text output. The function must take a single string argument.

You should not need to import or use this type directly.

See [text output docs](../output.md#text-output) for more information.
"""


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use a tool for output and optionally customize the tool."""

    output: OutputTypeOrFunction[OutputDataT]
    """An output type or function."""
    name: str | None
    """The name of the tool that will be passed to the model. If not specified and only one output is provided, `final_result` will be used. If multiple outputs are provided, the name of the output type or function will be added to the tool name."""
    description: str | None
    """The description of the tool that will be passed to the model. If not specified, the docstring of the output type or function will be used."""
    max_retries: int | None
    """The maximum number of retries for the tool."""
    strict: bool | None
    """Whether to use strict mode for the tool."""

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict


@dataclass(init=False)
class NativeOutput(Generic[OutputDataT]):
    """Marker class to use the model's native structured outputs functionality for outputs and optionally customize the name and description. """

    outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]]
    """The output types or functions."""
    name: str | None
    """The name of the structured output that will be passed to the model. If not specified and only one output is provided, the name of the output type or function will be used."""
    description: str | None
    """The description of the structured output that will be passed to the model. If not specified and only one output is provided, the docstring of the output type or function will be used."""
    strict: bool | None
    """Whether to use strict mode for the output, if the model supports it."""

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self.outputs = outputs
        self.name = name
        self.description = description
        self.strict = strict


@dataclass(init=False)
class PromptedOutput(Generic[OutputDataT]):
    """Marker class to use a prompt to tell the model what to output and optionally customize the prompt."""

    outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]]
    """The output types or functions."""
    name: str | None
    """The name of the structured output that will be passed to the model. If not specified and only one output is provided, the name of the output type or function will be used."""
    description: str | None
    """The description that will be passed to the model. If not specified and only one output is provided, the docstring of the output type or function will be used."""
    template: str | None
    """Template for the prompt passed to the model.
    The '{schema}' placeholder will be replaced with the output JSON schema.
    If not specified, the default template specified on the model's profile will be used.
    """

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        template: str | None = None,
    ):
        self.outputs = outputs
        self.name = name
        self.description = description
        self.template = template


@dataclass
class OutputObjectDefinition:
    """Definition of an output object used for structured output generation."""

    json_schema: ObjectJsonSchema
    name: str | None = None
    description: str | None = None
    strict: bool | None = None


@dataclass
class TextOutput(Generic[OutputDataT]):
    """Marker class to use text output for an output function taking a string argument."""

    output_function: TextOutputFunc[OutputDataT]
    """The function that will be called to process the model's plain text output. The function must take a single string argument."""


def StructuredDict(
    json_schema: JsonSchemaValue, name: str | None = None, description: str | None = None
) -> type[JsonSchemaValue]:
    """Returns a `dict[str, Any]` subclass with a JSON schema attached that will be used for structured output."""
    json_schema = _utils.check_object_json_schema(json_schema)

    # Pydantic `TypeAdapter` fails when `object.__get_pydantic_json_schema__` has `$defs`, so we inline them
    # See https://github.com/pydantic/pydantic/issues/12145
    if '$defs' in json_schema:
        json_schema = InlineDefsJsonSchemaTransformer(json_schema).walk()
        if '$defs' in json_schema:
            raise UserError(
                '`StructuredDict` does not currently support recursive `$ref`s and `$defs`. See https://github.com/pydantic/pydantic/issues/12145 for more information.'
            )

    if name:
        json_schema['title'] = name

    if description:
        json_schema['description'] = description

    class _StructuredDict(JsonSchemaValue):
        __is_model_like__ = True

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.dict_schema(
                keys_schema=core_schema.str_schema(),
                values_schema=core_schema.any_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            return json_schema

    return _StructuredDict


_OutputSpecItem = TypeAliasType(
    '_OutputSpecItem',
    OutputTypeOrFunction[T_co] | ToolOutput[T_co] | NativeOutput[T_co] | PromptedOutput[T_co] | TextOutput[T_co],
    type_params=(T_co,),
)

OutputSpec = TypeAliasType(
    'OutputSpec',
    _OutputSpecItem[T_co] | Sequence['OutputSpec[T_co]'],
    type_params=(T_co,),
)