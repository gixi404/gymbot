"""Pruebas unitarias de autorización por whitelist para el bot secundario."""

import unittest
from unittest.mock import MagicMock

from config.settings import Settings
from telegram_service.bot_service import GymTelegramBot


class TestBotWhitelistAuthorization(unittest.TestCase):
    def test_settings_allowed_user_ids_parsing(self):
        # Caso 1: string separado por comas
        s1 = Settings(
            TELEGRAM_API_ID=123,
            TELEGRAM_API_HASH="abc",
            TELEGRAM_WORKOUT_CHAT_ID=-1001,
            TELEGRAM_BOT_TOKEN="token",
            ALLOWED_TELEGRAM_USER_IDS="11111, 22222, 33333",
            GEMINI_API_KEY="key",
        )
        self.assertEqual(s1.ALLOWED_TELEGRAM_USER_IDS, [11111, 22222, 33333])
        self.assertTrue(s1.is_user_allowed(22222))
        self.assertFalse(s1.is_user_allowed(99999))

        # Caso 2: entero individual
        s2 = Settings(
            TELEGRAM_API_ID=123,
            TELEGRAM_API_HASH="abc",
            TELEGRAM_WORKOUT_CHAT_ID=-1001,
            TELEGRAM_BOT_TOKEN="token",
            ALLOWED_TELEGRAM_USER_IDS=55555,
            GEMINI_API_KEY="key",
        )
        self.assertEqual(s2.ALLOWED_TELEGRAM_USER_IDS, [55555])

    def test_bot_is_authorized_check(self):
        settings = Settings(
            TELEGRAM_API_ID=123,
            TELEGRAM_API_HASH="abc",
            TELEGRAM_WORKOUT_CHAT_ID=-1001,
            TELEGRAM_BOT_TOKEN="token",
            ALLOWED_TELEGRAM_USER_IDS="100, 200",
            GEMINI_API_KEY="key",
        )

        bot = GymTelegramBot(
            settings=settings,
            database=MagicMock(),
            reader=MagicMock(),
            analyzer=MagicMock(),
        )

        # Usuario permitido
        update_allowed = MagicMock()
        update_allowed.effective_user.id = 100
        self.assertTrue(bot._is_authorized(update_allowed))

        # Usuario no permitido (atacante o tercero)
        update_denied = MagicMock()
        update_denied.effective_user.id = 999
        self.assertFalse(bot._is_authorized(update_denied))

        # Update sin usuario
        update_none = MagicMock()
        update_none.effective_user = None
        self.assertFalse(bot._is_authorized(update_none))

    def test_safe_reply_fallback_on_markdown_error(self):
        import asyncio
        from unittest.mock import AsyncMock

        msg_mock = AsyncMock()
        # Primer llamado falla con error de markdown, segundo tiene éxito
        msg_mock.reply_text.side_effect = [Exception("Bad Request: can't parse entities"), None]

        asyncio.run(GymTelegramBot._safe_reply(msg_mock, "texto *mal_formateado"))
        self.assertEqual(msg_mock.reply_text.call_count, 2)
        # El segundo llamado debe ser sin parse_mode (texto plano)
        call_args = msg_mock.reply_text.call_args_list[1]
        self.assertEqual(call_args[0][0], "texto *mal_formateado")
        self.assertNotIn("parse_mode", call_args[1])

    def test_safe_edit_fallback_on_markdown_error(self):
        import asyncio
        from unittest.mock import AsyncMock

        msg_mock = AsyncMock()
        msg_mock.edit_text.side_effect = [Exception("Bad Request: can't parse entities"), None]

        asyncio.run(GymTelegramBot._safe_edit(msg_mock, "texto *mal_formateado"))
        self.assertEqual(msg_mock.edit_text.call_count, 2)

    def test_photo_and_document_whitelist_protection(self):
        import asyncio
        from unittest.mock import AsyncMock

        settings = Settings(
            TELEGRAM_API_ID=123,
            TELEGRAM_API_HASH="abc",
            TELEGRAM_WORKOUT_CHAT_ID=-1001,
            TELEGRAM_BOT_TOKEN="token",
            ALLOWED_TELEGRAM_USER_IDS="100",
            GEMINI_API_KEY="key",
        )
        bot = GymTelegramBot(
            settings=settings,
            database=MagicMock(),
            reader=MagicMock(),
            analyzer=MagicMock(),
        )

        # Update con usuario no autorizado
        unauth_update = MagicMock()
        unauth_update.effective_user.id = 999
        unauth_update.message.reply_text = AsyncMock()

        # Ambos manejadores deben salir de inmediato sin interactuar
        asyncio.run(bot._handle_photo(unauth_update, MagicMock()))
        unauth_update.message.reply_text.assert_not_called()

        asyncio.run(bot._handle_document(unauth_update, MagicMock()))
        unauth_update.message.reply_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
