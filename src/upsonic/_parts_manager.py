"""Manage and update parts of a model's streamed response.

Tracks parts (text and tool calls) with vendor-specific identifiers and produces
Upsonic-format events for streaming APIs.
"""

from __future__ import annotations as _annotations

from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from typing import Any

from upsonic.utils.package.exception import UnexpectedModelBehavior
from upsonic.messages.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

from upsonic._utils import generate_tool_call_id as _generate_tool_call_id

VendorId = Hashable
"""Type alias for vendor identifier (string, UUID, etc.)"""

ManagedPart = ModelResponsePart | ToolCallPartDelta
"""Union of managed part types including deltas for streaming APIs."""


@dataclass
class ModelResponsePartsManager:
    """Manages parts of a model's streamed response.

    Parts are added/updated via deltas tracked by vendor-specific IDs.
    """

    _parts: list[ManagedPart] = field(default_factory=list, init=False)
    """List of parts making up the current response state."""
    _vendor_id_to_part_index: dict[VendorId, int] = field(default_factory=dict, init=False)
    """Maps vendor part ID to index in _parts."""

    def get_parts(self) -> list[ModelResponsePart]:
        """Return complete model response parts (excluding deltas)."""
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def handle_text_delta(
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None = None,
        thinking_tags: tuple[str, str] | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> ModelResponseStreamEvent | None:
        """Handle incoming text content, creating or updating TextPart.

        Args:
            vendor_part_id: Vendor ID for this text piece. If None, updates latest TextPart or creates new.
            content: Text content to append.
            id: Optional ID for text part.
            thinking_tags: Handle content between tags as thinking parts.
            ignore_leading_whitespace: Ignore leading whitespace if True.

        Returns:
            PartStartEvent, PartDeltaEvent, or None.

        Raises:
            UnexpectedModelBehavior: If applying text to non-TextPart.
        """
        existing_text_part_and_index: tuple[TextPart, int] | None = None

        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    existing_text_part_and_index = latest_part, part_index
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]

                if thinking_tags and isinstance(existing_part, ThinkingPart):
                    if content == thinking_tags[1]:
                        self._vendor_id_to_part_index.pop(vendor_part_id)
                        return None
                    else:
                        return self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=content)
                elif isinstance(existing_part, TextPart):
                    existing_text_part_and_index = existing_part, part_index
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')

        if thinking_tags and content == thinking_tags[0]:
            self._vendor_id_to_part_index.pop(vendor_part_id, None)
            return self.handle_thinking_delta(vendor_part_id=vendor_part_id, content='')

        if existing_text_part_and_index is None:
            if ignore_leading_whitespace and (len(content) == 0 or content.isspace()):
                return None

            new_part_index = len(self._parts)
            part = TextPart(content=content, id=id)
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            return PartStartEvent(index=new_part_index, part=part)
        else:
            existing_text_part, part_index = existing_text_part_and_index
            part_delta = TextPartDelta(content_delta=content)
            self._parts[part_index] = part_delta.apply(existing_text_part)
            return PartDeltaEvent(index=part_index, delta=part_delta)

    def handle_thinking_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        content: str | None = None,
        id: str | None = None,
        signature: str | None = None,
        provider_name: str | None = None,
    ) -> ModelResponseStreamEvent:
        """Handle incoming thinking content, creating or updating ThinkingPart.

        Args:
            vendor_part_id: Vendor ID for thinking piece. If None, updates latest ThinkingPart or creates new.
            content: Thinking content to append.
            id: Optional ID for thinking part.
            signature: Optional signature for thinking content.
            provider_name: Optional provider name.

        Returns:
            PartStartEvent or PartDeltaEvent.

        Raises:
            UnexpectedModelBehavior: If applying thinking delta to non-ThinkingPart.
        """
        existing_thinking_part_and_index: tuple[ThinkingPart, int] | None = None

        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):
                    existing_thinking_part_and_index = latest_part, part_index
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is not None or signature is not None:
                new_part_index = len(self._parts)
                part = ThinkingPart(content=content or '', id=id, signature=signature, provider_name=provider_name)
                if vendor_part_id is not None:
                    self._vendor_id_to_part_index[vendor_part_id] = new_part_index
                self._parts.append(part)
                return PartStartEvent(index=new_part_index, part=part)
            else:
                raise UnexpectedModelBehavior('Cannot create a ThinkingPart with no content or signature')
        else:
            if content is not None or signature is not None:
                existing_thinking_part, part_index = existing_thinking_part_and_index
                part_delta = ThinkingPartDelta(
                    content_delta=content, signature_delta=signature, provider_name=provider_name
                )
                self._parts[part_index] = part_delta.apply(existing_thinking_part)
                return PartDeltaEvent(index=part_index, delta=part_delta)
            else:
                raise UnexpectedModelBehavior('Cannot update a ThinkingPart with no content or signature')

    def handle_tool_call_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str | None = None,
        args: str | dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent | None:
        """Handle or update tool call, creating/updating ToolCallPart, BuiltinToolCallPart, or ToolCallPartDelta.

        Args:
            vendor_part_id: Vendor ID for tool call. If None, updates latest matching tool call.
            tool_name: Tool name. If None, no name match enforced when vendor_part_id is None.
            args: Tool call arguments as string, dict, or None.
            tool_call_id: Optional identifier for tool call.

        Returns:
            PartStartEvent, PartDeltaEvent, or None.

        Raises:
            UnexpectedModelBehavior: If applying tool call delta to incompatible part.
        """
        existing_matching_part_and_index: tuple[ToolCallPartDelta | ToolCallPart | BuiltinToolCallPart, int] | None = (
            None
        )

        if vendor_part_id is None:
            if tool_name is None and self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta):
                    existing_matching_part_and_index = latest_part, part_index
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ToolCallPartDelta | ToolCallPart | BuiltinToolCallPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a tool call delta to {existing_part=}')
                existing_matching_part_and_index = existing_part, part_index

        if existing_matching_part_and_index is None:
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            part = delta.as_part() or delta
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = len(self._parts)
            new_part_index = len(self._parts)
            self._parts.append(part)
            if isinstance(part, ToolCallPart | BuiltinToolCallPart):
                return PartStartEvent(index=new_part_index, part=part)
        else:
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            updated_part = delta.apply(existing_part)
            self._parts[part_index] = updated_part
            if isinstance(updated_part, ToolCallPart | BuiltinToolCallPart):
                if isinstance(existing_part, ToolCallPartDelta):
                    return PartStartEvent(index=part_index, part=updated_part)
                else:
                    if updated_part.tool_call_id and not delta.tool_call_id:
                        delta = replace(delta, tool_call_id=updated_part.tool_call_id)
                    return PartDeltaEvent(index=part_index, delta=delta)

    def handle_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str,
        args: str | dict[str, Any] | None,
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent:
        """Create or overwrite ToolCallPart with given information.

        Args:
            vendor_part_id: Vendor ID for tool call part. If not None and existing part found, overwrites it.
            tool_name: Name of tool being invoked.
            args: Tool call arguments as string, dict, or None.
            tool_call_id: Optional string identifier for tool call.

        Returns:
            PartStartEvent indicating new tool call part added or replaced.
        """
        new_part = ToolCallPart(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id or _generate_tool_call_id(),
        )
        if vendor_part_id is None:
            new_part_index = len(self._parts)
            self._parts.append(new_part)
        else:
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], ToolCallPart):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = len(self._parts)
                self._parts.append(new_part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=new_part)

    def handle_builtin_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        part: BuiltinToolCallPart,
    ) -> ModelResponseStreamEvent:
        """Create or overwrite BuiltinToolCallPart.

        Args:
            vendor_part_id: Vendor ID for tool call part. If not None and existing part found, overwrites it.
            part: The BuiltinToolCallPart.

        Returns:
            PartStartEvent indicating new tool call part added or replaced.
        """
        if vendor_part_id is None:
            new_part_index = len(self._parts)
            self._parts.append(part)
        else:
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], BuiltinToolCallPart):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = part
            else:
                new_part_index = len(self._parts)
                self._parts.append(part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=part)

    def handle_builtin_tool_return_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        part: BuiltinToolReturnPart,
    ) -> ModelResponseStreamEvent:
        """Create or overwrite BuiltinToolReturnPart.

        Args:
            vendor_part_id: Vendor ID for tool call part. If not None and existing part found, overwrites it.
            part: The BuiltinToolReturnPart.

        Returns:
            PartStartEvent indicating new tool call part added or replaced.
        """
        if vendor_part_id is None:
            new_part_index = len(self._parts)
            self._parts.append(part)
        else:
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], BuiltinToolReturnPart):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = part
            else:
                new_part_index = len(self._parts)
                self._parts.append(part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=part)