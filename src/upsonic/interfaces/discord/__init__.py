"""
Discord Interface Module for Upsonic Framework.

This module provides the DiscordInterface class for integrating AI agents
with the Discord Bot API.
"""

from .discord import DiscordInterface
from .schemas import (
    DiscordGatewayPayload,
    DiscordMessage,
    DiscordUser,
    DiscordChannel,
    DiscordGuild,
    DiscordAttachment,
    DiscordEmbed,
    DiscordInteraction,
)

__all__ = [
    "DiscordInterface",
    "DiscordGatewayPayload",
    "DiscordMessage",
    "DiscordUser",
    "DiscordChannel",
    "DiscordGuild",
    "DiscordAttachment",
    "DiscordEmbed",
    "DiscordInteraction",
]
