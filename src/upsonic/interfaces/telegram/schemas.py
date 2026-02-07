"""
Schemas for Telegram Bot API Integration.

This module contains Pydantic models for Telegram Bot API requests and responses.
Based on the official Telegram Bot API: https://core.telegram.org/bots/api
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TelegramUser(BaseModel):
    """Represents a Telegram user or bot."""
    
    id: int = Field(..., description="Unique identifier for this user or bot")
    is_bot: bool = Field(False, description="True if this user is a bot")
    first_name: str = Field(..., description="User's or bot's first name")
    last_name: Optional[str] = Field(None, description="User's or bot's last name")
    username: Optional[str] = Field(None, description="User's or bot's username")
    language_code: Optional[str] = Field(None, description="IETF language tag of the user's language")
    is_premium: Optional[bool] = Field(None, description="True if this user is a Telegram Premium user")


class TelegramChat(BaseModel):
    """Represents a Telegram chat."""
    
    id: int = Field(..., description="Unique identifier for this chat")
    type: Literal["private", "group", "supergroup", "channel"] = Field(
        ..., description="Type of chat"
    )
    title: Optional[str] = Field(None, description="Title for groups/channels")
    username: Optional[str] = Field(None, description="Username for private chats/channels")
    first_name: Optional[str] = Field(None, description="First name of the other party in a private chat")
    last_name: Optional[str] = Field(None, description="Last name of the other party in a private chat")


class TelegramPhotoSize(BaseModel):
    """Represents one size of a photo or file/sticker thumbnail."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    width: int = Field(..., description="Photo width")
    height: int = Field(..., description="Photo height")
    file_size: Optional[int] = Field(None, description="File size in bytes")


class TelegramAudio(BaseModel):
    """Represents an audio file."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    duration: int = Field(..., description="Duration of the audio in seconds")
    performer: Optional[str] = Field(None, description="Performer of the audio")
    title: Optional[str] = Field(None, description="Title of the audio")
    file_name: Optional[str] = Field(None, description="Original filename")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    file_size: Optional[int] = Field(None, description="File size in bytes")


class TelegramDocument(BaseModel):
    """Represents a general file."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    file_name: Optional[str] = Field(None, description="Original filename")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    thumbnail: Optional[TelegramPhotoSize] = Field(None, description="Document thumbnail")


class TelegramVideo(BaseModel):
    """Represents a video file."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")
    duration: int = Field(..., description="Duration of the video in seconds")
    file_name: Optional[str] = Field(None, description="Original filename")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    thumbnail: Optional[TelegramPhotoSize] = Field(None, description="Video thumbnail")


class TelegramVoice(BaseModel):
    """Represents a voice note."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    duration: int = Field(..., description="Duration of the audio in seconds")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    file_size: Optional[int] = Field(None, description="File size in bytes")


class TelegramVideoNote(BaseModel):
    """Represents a video message (round video)."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    length: int = Field(..., description="Video width and height")
    duration: int = Field(..., description="Duration of the video in seconds")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    thumbnail: Optional[TelegramPhotoSize] = Field(None, description="Video thumbnail")


class TelegramSticker(BaseModel):
    """Represents a sticker."""
    
    file_id: str = Field(..., description="Identifier for this file")
    file_unique_id: str = Field(..., description="Unique identifier for this file")
    type: Literal["regular", "mask", "custom_emoji"] = Field(..., description="Type of the sticker")
    width: int = Field(..., description="Sticker width")
    height: int = Field(..., description="Sticker height")
    is_animated: bool = Field(False, description="True if the sticker is animated")
    is_video: bool = Field(False, description="True if the sticker is a video sticker")
    emoji: Optional[str] = Field(None, description="Emoji associated with the sticker")
    file_size: Optional[int] = Field(None, description="File size in bytes")


class TelegramContact(BaseModel):
    """Represents a phone contact."""
    
    phone_number: str = Field(..., description="Contact's phone number")
    first_name: str = Field(..., description="Contact's first name")
    last_name: Optional[str] = Field(None, description="Contact's last name")
    user_id: Optional[int] = Field(None, description="Contact's user identifier in Telegram")
    vcard: Optional[str] = Field(None, description="Additional data about the contact in vCard")


class TelegramLocation(BaseModel):
    """Represents a point on the map."""
    
    longitude: float = Field(..., description="Longitude as defined by sender")
    latitude: float = Field(..., description="Latitude as defined by sender")
    horizontal_accuracy: Optional[float] = Field(None, description="The radius of uncertainty")
    live_period: Optional[int] = Field(None, description="Time relative to the message sending date")
    heading: Optional[int] = Field(None, description="Direction in which the user is moving")
    proximity_alert_radius: Optional[int] = Field(None, description="Maximum distance for proximity alerts")


class TelegramVenue(BaseModel):
    """Represents a venue."""
    
    location: TelegramLocation = Field(..., description="Venue location")
    title: str = Field(..., description="Name of the venue")
    address: str = Field(..., description="Address of the venue")
    foursquare_id: Optional[str] = Field(None, description="Foursquare identifier")
    foursquare_type: Optional[str] = Field(None, description="Foursquare type")
    google_place_id: Optional[str] = Field(None, description="Google Places identifier")
    google_place_type: Optional[str] = Field(None, description="Google Places type")


class TelegramPollOption(BaseModel):
    """Represents one answer option in a poll."""
    
    text: str = Field(..., description="Option text (1-100 characters)")
    voter_count: int = Field(0, description="Number of users that voted for this option")


class TelegramPoll(BaseModel):
    """Represents a poll."""
    
    id: str = Field(..., description="Unique poll identifier")
    question: str = Field(..., description="Poll question")
    options: List["TelegramPollOption"] = Field(default_factory=list, description="List of poll options")
    total_voter_count: int = Field(..., description="Total number of users that voted")
    is_closed: bool = Field(False, description="True if the poll is closed")
    is_anonymous: bool = Field(True, description="True if the poll is anonymous")
    type: Literal["regular", "quiz"] = Field(..., description="Poll type")
    allows_multiple_answers: bool = Field(False, description="True if the poll allows multiple answers")
    correct_option_id: Optional[int] = Field(None, description="0-based index of the correct option (quiz only)")
    explanation: Optional[str] = Field(None, description="Text shown when user selects wrong answer")


class TelegramMessageEntity(BaseModel):
    """Represents a special entity in a text message."""
    
    type: str = Field(..., description="Type of the entity")
    offset: int = Field(..., description="Offset in UTF-16 code units to the start")
    length: int = Field(..., description="Length of the entity in UTF-16 code units")
    url: Optional[str] = Field(None, description="URL that will be opened")
    user: Optional[TelegramUser] = Field(None, description="The mentioned user")
    language: Optional[str] = Field(None, description="The programming language of the entity text")
    custom_emoji_id: Optional[str] = Field(None, description="Unique identifier of the custom emoji")


class InlineKeyboardButton(BaseModel):
    """Represents one button of an inline keyboard."""
    
    text: str = Field(..., description="Label text on the button")
    url: Optional[str] = Field(None, description="HTTP or tg:// URL to be opened")
    callback_data: Optional[str] = Field(None, description="Data to be sent in a callback query")
    switch_inline_query: Optional[str] = Field(None, description="Switch inline query")
    switch_inline_query_current_chat: Optional[str] = Field(None, description="Switch inline query current chat")


class InlineKeyboardMarkup(BaseModel):
    """Represents an inline keyboard."""
    
    inline_keyboard: List[List[InlineKeyboardButton]] = Field(
        ..., description="Array of button rows"
    )


class KeyboardButton(BaseModel):
    """Represents one button of a reply keyboard."""
    
    text: str = Field(..., description="Text of the button")
    request_contact: Optional[bool] = Field(None, description="Request user's phone number")
    request_location: Optional[bool] = Field(None, description="Request user's location")


class ReplyKeyboardMarkup(BaseModel):
    """Represents a custom reply keyboard."""
    
    keyboard: List[List[KeyboardButton]] = Field(..., description="Array of button rows")
    resize_keyboard: Optional[bool] = Field(None, description="Resize keyboard vertically")
    one_time_keyboard: Optional[bool] = Field(None, description="Hide keyboard after use")
    input_field_placeholder: Optional[str] = Field(None, description="Placeholder text")
    selective: Optional[bool] = Field(None, description="Show keyboard to specific users only")


class ReplyKeyboardRemove(BaseModel):
    """Remove the current custom keyboard."""
    
    remove_keyboard: Literal[True] = Field(True, description="Requests clients to remove the keyboard")
    selective: Optional[bool] = Field(None, description="Remove keyboard for specific users only")


class TelegramMessage(BaseModel):
    """Represents a Telegram message."""
    
    message_id: int = Field(..., description="Unique message identifier")
    message_thread_id: Optional[int] = Field(None, description="Unique identifier of a message thread")
    from_user: Optional[TelegramUser] = Field(None, alias="from", description="Sender of the message")
    sender_chat: Optional[TelegramChat] = Field(None, description="Sender of the message if sent on behalf of a chat")
    date: int = Field(..., description="Date the message was sent (Unix time)")
    chat: TelegramChat = Field(..., description="Conversation the message belongs to")
    forward_origin: Optional[Dict[str, Any]] = Field(None, description="Information about the original message")
    is_topic_message: Optional[bool] = Field(None, description="True if the message is sent to a forum topic")
    is_automatic_forward: Optional[bool] = Field(None, description="True if the message is automatically forwarded")
    reply_to_message: Optional["TelegramMessage"] = Field(None, description="Original message for replies")
    edit_date: Optional[int] = Field(None, description="Date the message was last edited")
    has_protected_content: Optional[bool] = Field(None, description="True if protected from forwarding and saving")
    media_group_id: Optional[str] = Field(None, description="Unique identifier of a media message group")
    author_signature: Optional[str] = Field(None, description="Signature of the post author")
    text: Optional[str] = Field(None, description="UTF-8 text of the message")
    entities: Optional[List[TelegramMessageEntity]] = Field(None, description="Special entities in text")
    caption: Optional[str] = Field(None, description="Caption for media messages")
    caption_entities: Optional[List[TelegramMessageEntity]] = Field(None, description="Special entities in caption")
    audio: Optional[TelegramAudio] = Field(None, description="Audio file information")
    document: Optional[TelegramDocument] = Field(None, description="General file information")
    photo: Optional[List[TelegramPhotoSize]] = Field(None, description="Available sizes of the photo")
    sticker: Optional[TelegramSticker] = Field(None, description="Sticker information")
    video: Optional[TelegramVideo] = Field(None, description="Video information")
    video_note: Optional[TelegramVideoNote] = Field(None, description="Video message information")
    voice: Optional[TelegramVoice] = Field(None, description="Voice message information")
    contact: Optional[TelegramContact] = Field(None, description="Shared contact information")
    location: Optional[TelegramLocation] = Field(None, description="Shared location information")
    venue: Optional[TelegramVenue] = Field(None, description="Venue information")
    poll: Optional[TelegramPoll] = Field(None, description="Poll information")
    new_chat_members: Optional[List[TelegramUser]] = Field(None, description="New members added to the group")
    left_chat_member: Optional[TelegramUser] = Field(None, description="Member removed from the group")
    new_chat_title: Optional[str] = Field(None, description="New chat title")
    new_chat_photo: Optional[List[TelegramPhotoSize]] = Field(None, description="New chat photo")
    delete_chat_photo: Optional[bool] = Field(None, description="Chat photo was deleted")
    group_chat_created: Optional[bool] = Field(None, description="Group has been created")
    supergroup_chat_created: Optional[bool] = Field(None, description="Supergroup has been created")
    channel_chat_created: Optional[bool] = Field(None, description="Channel has been created")
    reply_markup: Optional[InlineKeyboardMarkup] = Field(None, description="Inline keyboard attached to the message")
    
    class Config:
        populate_by_name = True


class TelegramCallbackQuery(BaseModel):
    """Represents an incoming callback query from a callback button."""
    
    id: str = Field(..., description="Unique identifier for this query")
    from_user: TelegramUser = Field(..., alias="from", description="Sender")
    message: Optional[TelegramMessage] = Field(None, description="Message with the callback button")
    inline_message_id: Optional[str] = Field(None, description="Identifier of the message sent via the bot")
    chat_instance: str = Field(..., description="Global identifier for the chat")
    data: Optional[str] = Field(None, description="Data associated with the callback button")
    game_short_name: Optional[str] = Field(None, description="Short name of a Game")
    
    class Config:
        populate_by_name = True


class TelegramWebhookPayload(BaseModel):
    """Webhook payload from Telegram Bot API."""
    
    update_id: int = Field(..., description="The update's unique identifier")
    message: Optional[TelegramMessage] = Field(None, description="New incoming message")
    edited_message: Optional[TelegramMessage] = Field(None, description="Edited message")
    channel_post: Optional[TelegramMessage] = Field(None, description="New incoming channel post")
    edited_channel_post: Optional[TelegramMessage] = Field(None, description="Edited channel post")
    callback_query: Optional[TelegramCallbackQuery] = Field(None, description="New incoming callback query")


# Re-export for convenience
TelegramMessage.model_rebuild()
