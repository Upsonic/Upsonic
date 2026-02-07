"""
Telegram Interface Module for Upsonic Framework.

This module provides the TelegramInterface class for integrating AI agents
with the Telegram Bot API.
"""

from .telegram import TelegramInterface
from .schemas import (
    TelegramWebhookPayload,
    TelegramMessage,
    TelegramUser,
    TelegramChat,
    TelegramCallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)

__all__ = [
    "TelegramInterface",
    "TelegramWebhookPayload",
    "TelegramMessage",
    "TelegramUser",
    "TelegramChat",
    "TelegramCallbackQuery",
    "InlineKeyboardButton",
    "InlineKeyboardMarkup",
    "ReplyKeyboardMarkup",
    "KeyboardButton",
]
