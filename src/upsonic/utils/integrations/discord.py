"""Discord integration utility helpers."""

from __future__ import annotations

import re


def sanitize_text_for_discord(text: str) -> str:
    """Sanitize text to be safe for Discord's message API.

    Strips control characters that may cause issues.

    Args:
        text: Raw text that may contain problematic characters.

    Returns:
        Sanitized plain text safe for Discord.
    """
    sanitized: str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    sanitized = sanitized.strip()
    return sanitized if sanitized else "(empty response)"
