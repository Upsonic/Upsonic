from __future__ import annotations as _annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from upsonic import _utils
from upsonic.output import (
    OutputDataT,
    OutputMode,
    OutputObjectDefinition,
)


@dataclass(kw_only=True)
class OutputSchema(ABC, Generic[OutputDataT]):
    """Base class for output schemas."""
    text_processor: BaseOutputProcessor[OutputDataT] | None = None
    toolset: None = None
    object_def: OutputObjectDefinition | None = None
    allows_deferred_tools: bool = False
    allows_image: bool = False

    @property
    def mode(self) -> OutputMode:
        raise NotImplementedError()

    @property
    def allows_text(self) -> bool:
        return self.text_processor is not None


@dataclass(init=False)
class StructuredTextOutputSchema(OutputSchema[OutputDataT], ABC):
    """Base class for structured text output schemas."""
    processor: BaseObjectOutputProcessor[OutputDataT]

    def __init__(
        self, *, processor: BaseObjectOutputProcessor[OutputDataT], allows_deferred_tools: bool, allows_image: bool
    ):
        super().__init__(
            text_processor=processor,
            object_def=processor.object_def,
            allows_deferred_tools=allows_deferred_tools,
            allows_image=allows_image,
        )
        self.processor = processor


class BaseOutputProcessor(ABC, Generic[OutputDataT]):
    """Base class for output processors."""
    @abstractmethod
    async def process(
        self,
        data: str,
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function."""
        raise NotImplementedError()


@dataclass(kw_only=True)
class BaseObjectOutputProcessor(BaseOutputProcessor[OutputDataT]):
    """Base class for object output processors."""
    object_def: OutputObjectDefinition


@dataclass(init=False)
class PromptedOutputProcessor(BaseObjectOutputProcessor[OutputDataT]):
    """Processor for prompted output that strips markdown fences."""
    wrapped: BaseObjectOutputProcessor[OutputDataT]

    def __init__(self, wrapped: BaseObjectOutputProcessor[OutputDataT]):
        self.wrapped = wrapped
        super().__init__(object_def=wrapped.object_def)

    async def process(
        self,
        data: str,
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        text = _utils.strip_markdown_fences(data)

        return await self.wrapped.process(
            text, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class PromptedOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    """Schema for prompted output that uses a template to instruct the model."""
    template: str | None

    def __init__(
        self,
        *,
        template: str | None = None,
        processor: BaseObjectOutputProcessor[OutputDataT],
        allows_deferred_tools: bool,
        allows_image: bool,
    ):
        super().__init__(
            processor=PromptedOutputProcessor(processor),
            allows_deferred_tools=allows_deferred_tools,
            allows_image=allows_image,
        )
        self.template = template

    @property
    def mode(self) -> OutputMode:
        return 'prompted'

    @classmethod
    def build_instructions(cls, template: str, object_def: OutputObjectDefinition) -> str:
        """Build instructions from a template and an object definition."""
        schema = object_def.json_schema.copy()
        if object_def.name:
            schema['title'] = object_def.name
        if object_def.description:
            schema['description'] = object_def.description

        if '{schema}' not in template:
            template = '\n\n'.join([template, '{schema}'])

        return template.format(schema=json.dumps(schema))

    def instructions(self, default_template: str) -> str:  # pragma: no cover
        """Get instructions to tell model to output JSON matching the schema."""
        template = self.template or default_template
        object_def = self.object_def
        assert object_def is not None
        return self.build_instructions(template, object_def)
