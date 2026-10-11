"""Módulo de integración con Telegram (MTProto y Bot API)."""

from .bot_service import GymTelegramBot
from .mtproto_reader import TelegramWorkoutReader

__all__ = ["TelegramWorkoutReader", "GymTelegramBot"]
