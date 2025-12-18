"""Used to build pydantic validators and JSON schemas from functions.

This module uses numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, cast, get_origin

from pydantic import ConfigDict
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema

from upsonic._griffe import doc_descriptions
from upsonic._utils import is_async_callable, is_model_like

if TYPE_CHECKING:
    from upsonic.tools.base import DocstringFormat, ObjectJsonSchema


__all__ = ('FunctionSchema', 'function_schema', 'SchemaGenerationError', 'GenerateToolJsonSchema')


class GenerateToolJsonSchema(GenerateJsonSchema):
    """Custom JSON schema generator for tools with improved TypedDict and field handling."""
    
    def typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> Any:
        """Generate JSON schema for TypedDict with proper additionalProperties handling."""
        json_schema = super().typed_dict_schema(schema)
        # Workaround for https://github.com/pydantic/pydantic/issues/12123
        if 'additionalProperties' not in json_schema:  # pragma: no branch
            extra = schema.get('extra_behavior') or schema.get('config', {}).get('extra_fields_behavior')
            if extra == 'allow':
                extras_schema = schema.get('extras_schema', None)
                if extras_schema is not None:
                    json_schema['additionalProperties'] = self.generate_inner(extras_schema) or True
                else:
                    json_schema['additionalProperties'] = True  # pragma: no cover
            elif extra == 'forbid':
                json_schema['additionalProperties'] = False
        return json_schema

    def _named_required_fields_schema(self, named_required_fields: Any) -> Any:
        """Generate schema for named required fields, removing unnecessary titles."""
        # Remove largely-useless property titles
        s = super()._named_required_fields_schema(named_required_fields)
        for p in s.get('properties', {}):
            s['properties'][p].pop('title', None)
        return s


class SchemaGenerationError(Exception):
    """Error raised during schema generation."""
    pass


@dataclass(kw_only=True)
class FunctionSchema:
    """Internal information about a function schema."""

    function: Callable[..., Any]
    description: str | None
    validator: SchemaValidator
    json_schema: dict[str, Any]
    is_async: bool
    single_arg_name: str | None = None
    positional_fields: list[str] = field(default_factory=list)
    var_positional_field: str | None = None

    async def call(self, args_dict: dict[str, Any]) -> Any:
        """Execute the function with the given arguments.
        
        Args:
            args_dict: Dictionary of arguments to pass to the function
            
        Returns:
            The function result
        """
        args, kwargs = self._call_args(args_dict)
        if self.is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args, **kwargs)
        else:
            # Run sync function in executor to avoid blocking
            import asyncio
            loop = asyncio.get_running_loop()
            function = cast(Callable[[Any], str], self.function)
            return await loop.run_in_executor(None, function, *args, **kwargs)

    def _call_args(
        self,
        args_dict: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Prepare arguments for function call.
        
        Args:
            args_dict: Dictionary of arguments
            
        Returns:
            Tuple of (positional_args, keyword_args)
        """
        if self.single_arg_name:
            args_dict = {self.single_arg_name: args_dict}

        args = []
        for positional_field in self.positional_fields:
            args.append(args_dict.pop(positional_field))  # pragma: no cover
        if self.var_positional_field:
            args.extend(args_dict.pop(self.var_positional_field))

        return args, args_dict


def function_schema(  # noqa: C901
    function: Callable[..., Any],
    schema_generator: type[GenerateJsonSchema],
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
) -> FunctionSchema:
    """Build a Pydantic validator and JSON schema from a tool function.

    Args:
        function: The function to build a validator and JSON schema for.
        schema_generator: The JSON schema generator class to use.
        docstring_format: The docstring format to use.
        require_parameter_descriptions: Whether to require descriptions for all tool function parameters.

    Returns:
        A `FunctionSchema` instance.
        
    Raises:
        SchemaGenerationError: If schema generation fails
    """
    config = ConfigDict(title=function.__name__, use_attribute_docstrings=True)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)
    errors: list[str] = []

    # === VALIDATION 1: Check function signature ===
    try:
        sig = signature(function)
    except ValueError as e:
        errors.append(str(e))
        sig = signature(lambda: None)

    # === VALIDATION 2: Check docstring exists ===
    if not function.__doc__ or not function.__doc__.strip():
        errors.append(f"Tool function '{function.__name__}' must have a docstring")

    # === VALIDATION 3: Check return type annotation exists ===
    type_hints = _typing_extra.get_function_type_hints(function)
    if 'return' not in type_hints:
        errors.append(f"Tool function '{function.__name__}' must have a return type annotation")

    # === VALIDATION 4: Check input parameters have type annotations ===
    if sig.parameters:
        missing_annotations = []
        for param_name, param in sig.parameters.items():
            if param.annotation is sig.empty and param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                missing_annotations.append(param_name)
        
        if missing_annotations:
            errors.append(
                f"Tool function '{function.__name__}' has parameters without type annotations: "
                f'{", ".join(missing_annotations)}'
            )

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    positional_fields: list[str] = []
    var_positional_field: str | None = None
    decorators = _decorators.DecoratorInfos()

    description, field_descriptions = doc_descriptions(function, sig, docstring_format=docstring_format)

    # === VALIDATION 5: Check parameter descriptions (if required) ===
    if require_parameter_descriptions:
        missing_params = set(sig.parameters) - set(field_descriptions)
        if missing_params:
            errors.append(f'Missing parameter descriptions for {", ".join(missing_params)}')

    for index, (name, p) in enumerate(sig.parameters.items()):
        if p.annotation is sig.empty:
            # TODO warn?
            annotation = Any
        else:
            annotation = type_hints[name]

        field_name = p.name
        if p.kind == Parameter.VAR_KEYWORD:
            var_kwargs_schema = gen_schema.generate_schema(annotation)
        else:
            if p.kind == Parameter.VAR_POSITIONAL:
                annotation = list[annotation]

            required = p.default is Parameter.empty
            # FieldInfo.from_annotated_attribute expects a type, `annotation` is Any
            annotation = cast(type[Any], annotation)
            if required:
                field_info = FieldInfo.from_annotation(annotation)
            else:
                field_info = FieldInfo.from_annotated_attribute(annotation, p.default)
            if field_info.description is None:
                field_info.description = field_descriptions.get(field_name)

            fields[field_name] = td_schema = gen_schema._generate_td_field_schema(  # pyright: ignore[reportPrivateUsage]
                field_name,
                field_info,
                decorators,
                required=required,
            )
            # noinspection PyTypeChecker
            td_schema.setdefault('metadata', {})['is_model_like'] = is_model_like(annotation)

            if p.kind == Parameter.POSITIONAL_ONLY:
                positional_fields.append(field_name)
            elif p.kind == Parameter.VAR_POSITIONAL:
                var_positional_field = field_name

    if errors:
        error_details = '\n  '.join(errors)
        raise SchemaGenerationError(f'Error generating schema for {function.__qualname__}:\n  {error_details}')

    core_config = config_wrapper.core_config(None)
    # noinspection PyTypedDict
    core_config['extra_fields_behavior'] = 'allow' if var_kwargs_schema else 'forbid'

    schema, single_arg_name = _build_schema(fields, var_kwargs_schema, gen_schema, core_config)
    schema = gen_schema.clean_schema(schema)
    # noinspection PyUnresolvedReferences
    schema_validator = create_schema_validator(
        schema,
        function,
        function.__module__,
        function.__qualname__,
        'validate_call',
        core_config,
        config_wrapper.plugin_settings,
    )
    # PluggableSchemaValidator is api compatible with SchemaValidator
    schema_validator = cast(SchemaValidator, schema_validator)
    json_schema = schema_generator().generate(schema)

    # workaround for https://github.com/pydantic/pydantic/issues/10785
    # if we build a custom TypedDict schema (matches when `single_arg_name is None`), we manually set
    # `additionalProperties` in the JSON Schema
    if single_arg_name is not None and not description:
        # if the tool description is not set, and we have a single parameter, take the description from that
        # and set it on the tool
        description = json_schema.pop('description', None)

    return FunctionSchema(
        description=description,
        validator=schema_validator,
        json_schema=json_schema,
        single_arg_name=single_arg_name,
        positional_fields=positional_fields,
        var_positional_field=var_positional_field,
        is_async=is_async_callable(function),
        function=function,
    )


def _build_schema(
    fields: dict[str, core_schema.TypedDictField],
    var_kwargs_schema: core_schema.CoreSchema | None,
    gen_schema: _generate_schema.GenerateSchema,
    core_config: core_schema.CoreConfig,
) -> tuple[core_schema.CoreSchema, str | None]:
    """Generate a typed dict schema for function parameters.

    Args:
        fields: The fields to generate a typed dict schema for.
        var_kwargs_schema: The variable keyword arguments schema.
        gen_schema: The `GenerateSchema` instance.
        core_config: The core configuration.

    Returns:
        tuple of (generated core schema, single arg name).
    """
    if len(fields) == 1 and var_kwargs_schema is None:
        name = next(iter(fields))
        td_field = fields[name]
        if td_field['metadata']['is_model_like']:  # type: ignore
            return td_field['schema'], name

    td_schema = core_schema.typed_dict_schema(
        fields,
        config=core_config,
        extras_schema=gen_schema.generate_schema(var_kwargs_schema) if var_kwargs_schema else None,
    )
    return td_schema, None
