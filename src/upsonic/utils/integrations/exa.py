"""Exa integration utility helpers."""

from __future__ import annotations

import json
from typing import Any


def serialize_exa_response(result: Any) -> str:
    """Serialize an Exa SDK response to a JSON string.

    exa-py returns Pydantic-like result objects. This helper calls
    model_dump() when available so json.dumps receives a plain dict/list.

    Args:
        result: The raw Exa SDK response object.

    Returns:
        JSON-encoded string of the response.
    """
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), default=str)
    if hasattr(result, "__dict__"):
        data = {}
        for key, value in result.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list):
                data[key] = [
                    v.model_dump() if hasattr(v, "model_dump")
                    else v.__dict__ if hasattr(v, "__dict__") and not isinstance(v, (str, int, float, bool))
                    else v
                    for v in value
                ]
            elif hasattr(value, "model_dump"):
                data[key] = value.model_dump()
            elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool)):
                data[key] = value.__dict__
            else:
                data[key] = value
        return json.dumps(data, default=str)
    return json.dumps(result, default=str)
