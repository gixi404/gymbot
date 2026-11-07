"""Unit tests for CSV export, keyboards and analysis helpers."""

import csv
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ai.analyzer import GymAnalyzer
from storage.database import GymDatabase
from storage.models import RawMessage
from telegram_service.bot_service import GymTelegramBot


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_features.db"
        self.db = GymDatabase(self.db_path)

        now = datetime.now()
        msgs = [
            RawMessage(id=1, date=now, chat_id="-1", chat_title="GYM", text="Press militar 50kg 3x8"),
            RawMessage(id=2, date=now, chat_id="-2", chat_title="Comidas", text="Almuerzo: 200g carne, 150g arroz"),
            RawMessage(id=3, date=now, chat_id="-3", chat_title="PesoCorporal", text="70.5 kg"),
        ]
        self.db.upsert_messages(msgs)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_to_csv(self):
        buf = self.db.export_to_csv()
        self.assertIsInstance(buf, io.BytesIO)

        # Verify UTF-8-sig decoding
        content = buf.getvalue().decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        # 1 header row + 3 data rows
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], ["ID", "Fecha_UTC", "Origen", "Texto", "Tipo_Media", "Fecha_Edicion"])
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[1][2], "GYM")
        self.assertIn("Press militar", rows[1][3])
        self.assertEqual(rows[2][2], "Comidas")
        self.assertEqual(rows[3][2], "PesoCorporal")

    def test_inline_keyboards(self):
        main_kb = GymTelegramBot._get_main_keyboard()
        self.assertIsNotNone(main_kb.inline_keyboard)
        callbacks = [btn.callback_data for row in main_kb.inline_keyboard for btn in row]
        self.assertIn("btn_chart_menu", callbacks)
        self.assertIn("btn_weekly", callbacks)
        self.assertIn("btn_macros", callbacks)
        self.assertIn("btn_progression_menu", callbacks)
        self.assertIn("btn_export", callbacks)

        chart_kb = GymTelegramBot._get_chart_keyboard()
        chart_callbacks = [btn.callback_data for row in chart_kb.inline_keyboard for btn in row]
        self.assertIn("chart_weight", chart_callbacks)
        self.assertIn("chart_banca", chart_callbacks)
        self.assertIn("chart_prensa", chart_callbacks)
        self.assertNotIn("chart_sentadilla", chart_callbacks)

        prog_kb = GymTelegramBot._get_progression_keyboard()
        prog_callbacks = [btn.callback_data for row in prog_kb.inline_keyboard for btn in row]
        self.assertIn("prog_banca", prog_callbacks)
        self.assertIn("prog_prensa", prog_callbacks)
        self.assertNotIn("prog_sentadilla", prog_callbacks)

    def test_analyzer_helper_methods(self):
        mock_gemini = MagicMock()
        mock_gemini.generate_response.return_value = "Respuesta simulada exitosa"

        analyzer = GymAnalyzer(database=self.db, gemini_client=mock_gemini)

        # Test today's nutrition analysis
        res_nutri = analyzer.analyze_nutrition_today()
        self.assertEqual(res_nutri, "Respuesta simulada exitosa")
        mock_gemini.generate_response.assert_called()

        # Test progression recommendation
        mock_gemini.reset_mock()
        res_prog = analyzer.recommend_next_progression("militar")
        self.assertEqual(res_prog, "Respuesta simulada exitosa")
        mock_gemini.generate_response.assert_called()

        # Test weekly summary
        mock_gemini.reset_mock()
        res_week = analyzer.generate_weekly_summary()
        self.assertEqual(res_week, "Respuesta simulada exitosa")
        mock_gemini.generate_response.assert_called()


class TestTelegramDownloadFile(unittest.IsolatedAsyncioTestCase):
    async def test_download_success_primary(self):
        from telegram_service.bot_service import _download_telegram_file

        mock_bot = MagicMock()
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio-data"))
        mock_bot.get_file = AsyncMock(return_value=mock_file)

        result = await _download_telegram_file(mock_bot, "file-123")
        self.assertEqual(result, b"audio-data")
        mock_bot.get_file.assert_called_once()
        mock_file.download_as_bytearray.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_download_fallback_to_httpx_on_error(self, mock_client_cls):
        from telegram_service.bot_service import _download_telegram_file

        mock_bot = MagicMock()
        mock_file = MagicMock()
        mock_file.file_path = "https://api.telegram.org/file/botTOKEN/voice.oga"
        mock_file.download_as_bytearray = AsyncMock(side_effect=TimeoutError("Timed out"))
        mock_bot.get_file = AsyncMock(return_value=mock_file)

        mock_resp = MagicMock()
        mock_resp.content = b"httpx-downloaded-audio"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await _download_telegram_file(mock_bot, "file-123", max_retries=1)
        self.assertEqual(result, b"httpx-downloaded-audio")
        mock_client.get.assert_called_once_with(mock_file.file_path)

    async def test_download_all_retries_fail(self):
        from telegram_service.bot_service import _download_telegram_file

        mock_bot = MagicMock()
        mock_bot.get_file = AsyncMock(side_effect=TimeoutError("Get file timed out"))

        with self.assertRaises(TimeoutError):
            await _download_telegram_file(mock_bot, "file-123", max_retries=2)
        self.assertEqual(mock_bot.get_file.call_count, 2)


if __name__ == "__main__":
    unittest.main()
