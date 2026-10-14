"""Pruebas unitarias para exportación CSV, teclados y asistentes de análisis."""

import csv
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

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

        # Verificar decodificación UTF-8-sig
        content = buf.getvalue().decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        # 1 fila encabezado + 3 filas de datos
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

        # Probar análisis de nutrición de hoy
        res_nutri = analyzer.analyze_nutrition_today()
        self.assertEqual(res_nutri, "Respuesta simulada exitosa")
        mock_gemini.generate_response.assert_called()

        # Probar recomendación de progresión
        mock_gemini.reset_mock()
        res_prog = analyzer.recommend_next_progression("militar")
        self.assertEqual(res_prog, "Respuesta simulada exitosa")
        mock_gemini.generate_response.assert_called()

        # Probar resumen semanal
        mock_gemini.reset_mock()
        res_week = analyzer.generate_weekly_summary()
        self.assertEqual(res_week, "Respuesta simulada exitosa")
        mock_gemini.generate_response.assert_called()


if __name__ == "__main__":
    unittest.main()
