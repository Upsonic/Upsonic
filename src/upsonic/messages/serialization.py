"""Serialization utilities for message classes using Pydantic TypeAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    from upsonic.messages.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
    )


def _get_model_messages_type_adapter() -> pydantic.TypeAdapter:
    """Lazy import and return ModelMessagesTypeAdapter."""
    from upsonic.messages.messages import ModelMessagesTypeAdapter
    return ModelMessagesTypeAdapter


def _get_model_request_type_adapter() -> pydantic.TypeAdapter:
    """Lazy create and return ModelRequest TypeAdapter."""
    from upsonic.messages.messages import ModelRequest
    return pydantic.TypeAdapter(
        ModelRequest,
        config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
    )


def _get_model_response_type_adapter() -> pydantic.TypeAdapter:
    """Lazy create and return ModelResponse TypeAdapter."""
    from upsonic.messages.messages import ModelResponse
    return pydantic.TypeAdapter(
        ModelResponse,
        config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
    )


def serialize_model_request(request: "ModelRequest") -> bytes:
    """Serialize a ModelRequest to bytes."""
    ta = _get_model_request_type_adapter()
    return ta.dump_json(request)


def deserialize_model_request(data: bytes) -> "ModelRequest":
    """Deserialize bytes to a ModelRequest."""
    ta = _get_model_request_type_adapter()
    return ta.validate_json(data)


def serialize_model_response(response: "ModelResponse") -> bytes:
    """Serialize a ModelResponse to bytes."""
    ta = _get_model_response_type_adapter()
    return ta.dump_json(response)


def deserialize_model_response(data: bytes) -> "ModelResponse":
    """Deserialize bytes to a ModelResponse."""
    ta = _get_model_response_type_adapter()
    return ta.validate_json(data)


def serialize_messages(messages: list["ModelMessage"]) -> bytes:
    """Serialize a list of ModelMessages to bytes."""
    ta = _get_model_messages_type_adapter()
    return ta.dump_json(messages)


def deserialize_messages(data: bytes) -> list["ModelMessage"]:
    """Deserialize bytes to a list of ModelMessages."""
    ta = _get_model_messages_type_adapter()
    return ta.validate_json(data)

