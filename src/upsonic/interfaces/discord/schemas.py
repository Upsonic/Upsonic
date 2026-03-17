"""
Schemas for Discord Bot API Integration.

This module contains Pydantic models for Discord Bot API requests and responses.
Based on the official Discord API: https://discord.com/developers/docs
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DiscordUser(BaseModel):
    """Represents a Discord user."""

    id: str = Field(..., description="The user's snowflake ID")
    username: str = Field(..., description="The user's username")
    discriminator: Optional[str] = Field(None, description="The user's discriminator (legacy)")
    global_name: Optional[str] = Field(None, description="The user's display name")
    avatar: Optional[str] = Field(None, description="The user's avatar hash")
    bot: Optional[bool] = Field(None, description="Whether the user is a bot")
    system: Optional[bool] = Field(None, description="Whether the user is a system user")


class DiscordGuild(BaseModel):
    """Represents a Discord guild (server)."""

    id: str = Field(..., description="Guild snowflake ID")
    name: str = Field(..., description="Guild name")
    icon: Optional[str] = Field(None, description="Icon hash")
    owner_id: Optional[str] = Field(None, description="ID of the guild owner")
    member_count: Optional[int] = Field(None, description="Approximate number of members")


class DiscordChannel(BaseModel):
    """Represents a Discord channel."""

    id: str = Field(..., description="Channel snowflake ID")
    type: int = Field(..., description="Channel type (0=text, 1=DM, 2=voice, etc.)")
    guild_id: Optional[str] = Field(None, description="Guild ID this channel belongs to")
    name: Optional[str] = Field(None, description="Channel name")
    topic: Optional[str] = Field(None, description="Channel topic")
    position: Optional[int] = Field(None, description="Sorting position of the channel")
    parent_id: Optional[str] = Field(None, description="ID of the parent category")
    last_message_id: Optional[str] = Field(None, description="ID of the last message sent")


class DiscordAttachment(BaseModel):
    """Represents a file attachment in a message."""

    id: str = Field(..., description="Attachment snowflake ID")
    filename: str = Field(..., description="Name of the file")
    size: int = Field(..., description="Size of the file in bytes")
    url: str = Field(..., description="Source URL of the file")
    proxy_url: Optional[str] = Field(None, description="Proxied URL of the file")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    width: Optional[int] = Field(None, description="Width of the image (if image)")
    height: Optional[int] = Field(None, description="Height of the image (if image)")


class DiscordEmbedField(BaseModel):
    """Represents a field in an embed."""

    name: str = Field(..., description="Name of the field")
    value: str = Field(..., description="Value of the field")
    inline: Optional[bool] = Field(None, description="Whether the field is inline")


class DiscordEmbedFooter(BaseModel):
    """Represents the footer of an embed."""

    text: str = Field(..., description="Footer text")
    icon_url: Optional[str] = Field(None, description="URL of footer icon")


class DiscordEmbedImage(BaseModel):
    """Represents an image in an embed."""

    url: str = Field(..., description="Source URL of the image")
    width: Optional[int] = Field(None, description="Width of the image")
    height: Optional[int] = Field(None, description="Height of the image")


class DiscordEmbedAuthor(BaseModel):
    """Represents the author of an embed."""

    name: str = Field(..., description="Name of the author")
    url: Optional[str] = Field(None, description="URL of the author")
    icon_url: Optional[str] = Field(None, description="URL of the author icon")


class DiscordEmbed(BaseModel):
    """Represents a rich embed in a message."""

    title: Optional[str] = Field(None, description="Title of the embed")
    description: Optional[str] = Field(None, description="Description of the embed")
    url: Optional[str] = Field(None, description="URL of the embed")
    color: Optional[int] = Field(None, description="Color code of the embed")
    fields: Optional[List[DiscordEmbedField]] = Field(None, description="Fields in the embed")
    footer: Optional[DiscordEmbedFooter] = Field(None, description="Footer information")
    image: Optional[DiscordEmbedImage] = Field(None, description="Image information")
    thumbnail: Optional[DiscordEmbedImage] = Field(None, description="Thumbnail information")
    author: Optional[DiscordEmbedAuthor] = Field(None, description="Author information")
    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp")


class DiscordEmoji(BaseModel):
    """Represents an emoji."""

    id: Optional[str] = Field(None, description="Emoji snowflake ID (None for standard emoji)")
    name: Optional[str] = Field(None, description="Emoji name")
    animated: Optional[bool] = Field(None, description="Whether the emoji is animated")


class DiscordReaction(BaseModel):
    """Represents a reaction to a message."""

    count: int = Field(..., description="Number of times this emoji was used")
    me: bool = Field(False, description="Whether the current user reacted")
    emoji: DiscordEmoji = Field(..., description="Emoji information")


class DiscordComponent(BaseModel):
    """Represents a message component (button, select menu, etc.)."""

    type: int = Field(..., description="Component type (1=action row, 2=button, 3=select menu)")
    custom_id: Optional[str] = Field(None, description="Developer-defined identifier")
    label: Optional[str] = Field(None, description="Text on the button")
    style: Optional[int] = Field(None, description="Button style (1=primary, 2=secondary, 3=success, 4=danger, 5=link)")
    url: Optional[str] = Field(None, description="URL for link buttons")
    disabled: Optional[bool] = Field(None, description="Whether the component is disabled")
    components: Optional[List["DiscordComponent"]] = Field(None, description="Child components (for action rows)")


class DiscordMember(BaseModel):
    """Represents a guild member."""

    user: Optional[DiscordUser] = Field(None, description="The user this guild member represents")
    nick: Optional[str] = Field(None, description="User's guild nickname")
    roles: List[str] = Field(default_factory=list, description="Array of role snowflake IDs")
    joined_at: Optional[str] = Field(None, description="When the user joined the guild")


class DiscordMessageReference(BaseModel):
    """Represents a reference to another message (reply)."""

    message_id: Optional[str] = Field(None, description="ID of the originating message")
    channel_id: Optional[str] = Field(None, description="ID of the originating channel")
    guild_id: Optional[str] = Field(None, description="ID of the originating guild")


class DiscordMessage(BaseModel):
    """Represents a Discord message."""

    id: str = Field(..., description="Message snowflake ID")
    channel_id: str = Field(..., description="ID of the channel the message was sent in")
    guild_id: Optional[str] = Field(None, description="ID of the guild the message was sent in")
    author: DiscordUser = Field(..., description="The author of the message")
    member: Optional[DiscordMember] = Field(None, description="Member properties for the author")
    content: str = Field("", description="Content of the message")
    timestamp: str = Field(..., description="When the message was sent")
    edited_timestamp: Optional[str] = Field(None, description="When the message was edited")
    tts: bool = Field(False, description="Whether this was a TTS message")
    mention_everyone: bool = Field(False, description="Whether this message mentions everyone")
    mentions: List[DiscordUser] = Field(default_factory=list, description="Users mentioned in the message")
    attachments: List[DiscordAttachment] = Field(default_factory=list, description="Attached files")
    embeds: List[DiscordEmbed] = Field(default_factory=list, description="Embedded content")
    reactions: Optional[List[DiscordReaction]] = Field(None, description="Reactions to the message")
    components: Optional[List[DiscordComponent]] = Field(None, description="Components in the message")
    referenced_message: Optional["DiscordMessage"] = Field(None, description="Referenced message (reply)")
    message_reference: Optional[DiscordMessageReference] = Field(None, description="Message reference data")
    type: int = Field(0, description="Type of message")


class DiscordInteractionData(BaseModel):
    """Represents the data payload of an interaction."""

    id: Optional[str] = Field(None, description="ID of the invoked command")
    name: Optional[str] = Field(None, description="Name of the invoked command")
    type: Optional[int] = Field(None, description="Type of the invoked command")
    custom_id: Optional[str] = Field(None, description="Custom ID for components")
    component_type: Optional[int] = Field(None, description="Type of the component")
    values: Optional[List[str]] = Field(None, description="Selected values (for select menus)")


class DiscordInteraction(BaseModel):
    """Represents a Discord interaction (slash command, button click, etc.)."""

    id: str = Field(..., description="Interaction snowflake ID")
    application_id: str = Field(..., description="ID of the application")
    type: int = Field(..., description="Type of interaction (1=ping, 2=command, 3=component, 4=autocomplete)")
    data: Optional[DiscordInteractionData] = Field(None, description="Interaction data payload")
    guild_id: Optional[str] = Field(None, description="Guild the interaction was sent from")
    channel_id: Optional[str] = Field(None, description="Channel the interaction was sent from")
    member: Optional[DiscordMember] = Field(None, description="Guild member who invoked (guild context)")
    user: Optional[DiscordUser] = Field(None, description="User who invoked (DM context)")
    token: str = Field(..., description="Continuation token for responding")
    message: Optional[DiscordMessage] = Field(None, description="Message the component was attached to")
    version: int = Field(1, description="Always 1")


class DiscordGatewayPayload(BaseModel):
    """Generic Discord Gateway event payload."""

    op: int = Field(..., description="Gateway opcode")
    d: Optional[Any] = Field(None, description="Event data")
    s: Optional[int] = Field(None, description="Sequence number (for op 0)")
    t: Optional[str] = Field(None, description="Event name (for op 0)")


# Rebuild models for forward references
DiscordComponent.model_rebuild()
DiscordMessage.model_rebuild()
