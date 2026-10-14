"""Pruebas unitarias para el motor de análisis de IA y construcción de contexto."""

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from ai.analyzer import GymAnalyzer
from config.settings import Settings
from storage.database import GymDatabase
from storage.models import RawMessage


class TestGymAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_analyzer.db"
        self.db = GymDatabase(self.db_path)

        self.settings = Settings(
            TELEGRAM_API_ID=123,
            TELEGRAM_API_HASH="abc",
            TELEGRAM_WORKOUT_CHAT_ID=-1001,
            TELEGRAM_BOT_TOKEN="token",
            ALLOWED_TELEGRAM_USER_IDS="100",
            GEMINI_API_KEY="test_key",
        )

        self.mock_gemini = MagicMock()
        self.analyzer = GymAnalyzer(
            settings=self.settings,
            database=self.db,
            gemini_client=self.mock_gemini,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_context_empty_db(self):
        context = self.analyzer.build_workout_context()
        self.assertIn("NO HAY REGISTROS DISPONIBLES", context)

    def test_build_context_with_messages(self):
        msgs = [
            RawMessage(
                id=1,
                date=datetime(2026, 4, 1, 9, 30),
                text="Press Banca: 80kg x 8, 85kg x 6",
            ),
            RawMessage(
                id=2,
                date=datetime(2026, 4, 3, 10, 0),
                text="Sentadillas: 100kg 3x5 @8",
            ),
        ]
        self.db.upsert_messages(msgs)

        context = self.analyzer.build_workout_context()
        self.assertIn("Press Banca: 80kg x 8", context)
        self.assertIn("Sentadillas: 100kg 3x5", context)
        self.assertIn("[enviado: 01/04/2026]", context)
        self.assertIn("[enviado: 03/04/2026]", context)

    def test_analyze_query_empty_database_warning(self):
        # Con BD vacía, no debe invocar la API de Gemini
        response = self.analyzer.analyze_query("¿Cuál fue mi mejor press de banca?")
        self.assertIn("La base de datos local está vacía", response)
        self.mock_gemini.generate_response.assert_not_called()

    def test_analyze_query_calls_gemini_with_context(self):
        # Insertamos un mensaje
        self.db.upsert_messages([RawMessage(id=1, date=datetime.now(), text="Press Militar: 50kg 4x6")])
        self.mock_gemini.generate_response.return_value = "Tu mejor press militar fue de 50kg para 6 repeticiones."

        response = self.analyzer.analyze_query("¿Cuál fue mi mejor press militar?")
        self.assertIn("50kg para 6 repeticiones", response)
        self.mock_gemini.generate_response.assert_called_once()

        call_args = self.mock_gemini.generate_response.call_args
        prompt_passed = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
        self.assertIn("Press Militar: 50kg 4x6", prompt_passed)
        self.assertIn("¿Cuál fue mi mejor press militar?", prompt_passed)

    def test_get_system_instruction_includes_custom_directives(self):
        # Probamos que el método devuelva la instrucción base por defecto
        instruction = self.analyzer.get_system_instruction()
        self.assertIn("Eres el Entrenador de Fuerza", instruction)

    def test_transcribe_audio_delegates_to_gemini(self):
        self.mock_gemini.transcribe_audio.return_value = "¿Cuál fue mi última sesión?"
        result = self.analyzer.transcribe_audio(b"fake_audio_bytes", mime_type="audio/ogg")
        self.assertEqual(result, "¿Cuál fue mi última sesión?")
        self.mock_gemini.transcribe_audio.assert_called_once_with(b"fake_audio_bytes", mime_type="audio/ogg")

    def test_analyze_query_with_attachments_calls_gemini(self):
        # Con BD vacía pero con adjuntos, sí debe procesar la consulta
        self.mock_gemini.generate_response.return_value = "La imagen muestra una serie de 100kg en press banca."
        attachments = [(b"image_bytes_here", "image/png")]

        response = self.analyzer.analyze_query(
            user_question="¿Qué ejercicio es este?",
            attachments=attachments,
        )

        self.assertIn("100kg en press banca", response)
        self.mock_gemini.generate_response.assert_called_once()
        call_kwargs = self.mock_gemini.generate_response.call_args[1]
        self.assertEqual(call_kwargs.get("attachments"), attachments)

    def test_context_cache_in_memory(self):
        self.db.upsert_messages([RawMessage(id=1, date=datetime.now(), text="Dominadas: 4x8 peso corporal")])
        context1 = self.analyzer.build_workout_context()
        self.assertIn("Dominadas: 4x8", context1)
        self.assertEqual(self.analyzer._cached_context, context1)
        context2 = self.analyzer.build_workout_context()
        self.assertIs(context1, context2)

    def test_analyze_query_stream_async(self):
        import asyncio

        async def _test():
            self.db.upsert_messages([RawMessage(id=1, date=datetime.now(), text="Sentadilla: 120kg 3x5")])

            async def mock_stream(*args, **kwargs):
                yield "Tu sentadilla "
                yield "fue de 120kg."

            self.mock_gemini.generate_response_stream_async = mock_stream
            chunks = []
            async for chunk in self.analyzer.analyze_query_stream_async("¿Cuánto hice en sentadilla?"):
                chunks.append(chunk)
            self.assertEqual("".join(chunks), "Tu sentadilla fue de 120kg.")

        asyncio.run(_test())

    def test_extract_peso_series_parses_dated_pairs(self):
        msgs = [
            RawMessage(
                id=1,
                date=datetime(2026, 10, 11),
                text="07/10/26 - 70.05 kg\n30/09/26 - 69.95 kg\n23/09/26 - 71.50 kg",
            )
        ]
        series = self.analyzer._extract_peso_series(msgs)
        self.assertEqual(series[0], (date(2026, 9, 23), 71.50))
        self.assertEqual(series[-1], (date(2026, 10, 7), 70.05))

    def test_nutrition_date_priority_uses_text_date_over_send_date(self):
        # id1: enviada el 14/10 pero la comida es del 13/10 (fecha en el texto).
        # id2: enviada el 13/10 pero la comida es del 11/10 (fecha en el texto).
        self.db.upsert_messages(
            [
                RawMessage(
                    id=1,
                    date=datetime(2026, 10, 14, 12, 0),
                    chat_id="-1",
                    chat_title="Comidas",
                    text="13/10/26 | Almuerzo: 200g carne",
                ),
                RawMessage(
                    id=2,
                    date=datetime(2026, 10, 13, 12, 0),
                    chat_id="-1",
                    chat_title="Comidas",
                    text="11/10/26 | Cena: arroz",
                ),
            ]
        )
        self.mock_gemini.generate_response.return_value = "estimación"

        self.analyzer._analyze_nutrition_for_date(datetime(2026, 10, 13))

        self.mock_gemini.generate_response.assert_called_once()
        prompt = self.mock_gemini.generate_response.call_args[1]["prompt"]
        self.assertIn("200g carne", prompt)
        self.assertNotIn("Cena: arroz", prompt)

    def test_attachment_default_uses_short_context(self):
        # 50 sesiones históricas que NO deben adjuntarse a una foto sin caption.
        self.db.upsert_messages(
            [
                RawMessage(
                    id=i,
                    date=datetime(2026, 1, 1) + timedelta(days=i),
                    chat_id="-1",
                    chat_title="Sesiones",
                    text=f"- Prensa: 100kg 10 8 8 VOLUMEN_SECRETO_{i}",
                )
                for i in range(50)
            ]
        )
        self.mock_gemini.generate_response.return_value = "análisis de imagen"

        self.analyzer.analyze_query(
            user_question="Analiza esta imagen",
            attachments=[(b"img", "image/jpeg")],
            attachment_default=True,
        )

        route = self.mock_gemini.generate_response.call_args[1]["route"]
        self.assertEqual(route.groups, ["Comidas", "PesoCorporal"])
        self.assertEqual(route.days_lookback, 3)
        prompt = self.mock_gemini.generate_response.call_args[1]["prompt"]
        self.assertNotIn("VOLUMEN_SECRETO", prompt, "una foto sin caption no debe arrastrar el historial de Sesiones")

    def test_weekly_summary_uses_focused_context_and_complex_route(self):
        self.db.upsert_messages(
            [
                RawMessage(
                    id=1,
                    date=datetime.now(),
                    chat_id="-1",
                    chat_title="Sesiones",
                    text="- Prensa: 140kg 11 10 10",
                ),
                RawMessage(
                    id=2,
                    date=datetime.now(),
                    chat_id="-2",
                    chat_title="Comidas",
                    text="11/10/26 | Desayuno: avena",
                ),
            ]
        )
        self.mock_gemini.generate_response.return_value = "resumen"

        self.analyzer.generate_weekly_summary()

        call_kwargs = self.mock_gemini.generate_response.call_args[1]
        route = call_kwargs["route"]
        self.assertEqual(route.complexity, "complex")
        self.assertEqual(route.days_lookback, 7)
        self.assertEqual(set(route.groups), {"Sesiones", "Cardio", "Comidas", "PesoCorporal"})
        self.assertIn("REGISTROS DISPONIBLES POR GRUPO", call_kwargs["prompt"])
        self.assertIn("Prensa: 140kg 11 10 10", call_kwargs["prompt"])


if __name__ == "__main__":
    unittest.main()
