"""Telegram integration module (MTProto and Bot API)."""

from .bot_service import GymTelegramBot
from .mtproto_reader import TelegramWorkoutReader

__all__ = ["TelegramWorkoutReader", "GymTelegramBot"]
