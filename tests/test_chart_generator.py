"""Pruebas unitarias para la generación de gráficos de rendimiento y peso."""

import io
import unittest
from datetime import datetime, timedelta

from storage.models import RawMessage
from visualization.chart_generator import ChartGenerator


class TestChartGenerator(unittest.TestCase):
    def setUp(self):
        self.base_date = datetime(2026, 5, 1, 10, 0, 0)
        self.weight_messages = [
            RawMessage(
                id=1,
                date=self.base_date,
                chat_title="PesoCorporal",
                text="01/05/26 - 68.50 kg\n08/05/26 - 68.80 kg\n15/05/26 - 69.10 kg\n22/05/26 - 69.40 kg\n29/05/26 - 69.80 kg",
            ),
            RawMessage(
                id=2,
                date=self.base_date + timedelta(days=35),
                chat_title="PesoCorporal",
                text="70.2 kg",
            ),
        ]

        self.exercise_messages = [
            RawMessage(
                id=10,
                date=self.base_date,
                chat_title="GYM",
                text="LUNES\n- Press banca plano:\n50 kg 12.11.10\n- Dominadas: 4x8",
            ),
            RawMessage(
                id=11,
                date=self.base_date + timedelta(days=7),
                chat_title="GYM",
                text="LUNES\n- Press banca plano:\n52.5 kg 12.12.11\n- Dominadas: 4x8",
            ),
            RawMessage(
                id=12,
                date=self.base_date + timedelta(days=14),
                chat_title="GYM",
                text="LUNES\n- Press banca plano:\n55 kg 10.9.8\n- Dominadas: 4x8",
            ),
        ]

    def test_parse_weight_entries(self):
        entries = ChartGenerator.parse_weight_entries(self.weight_messages)
        self.assertGreaterEqual(len(entries), 5)
        # Verificar orden cronológico
        dates = [e[0] for e in entries]
        self.assertEqual(dates, sorted(dates))
        # Verificar pesos numéricos válidos
        for _, w in entries:
            self.assertTrue(60.0 <= w <= 80.0)

    def test_generate_weight_chart_returns_valid_png(self):
        buf = ChartGenerator.generate_weight_chart(self.weight_messages)
        self.assertIsNotNone(buf)
        self.assertIsInstance(buf, io.BytesIO)
        content = buf.getvalue()
        # Verificar firma mágica de archivo PNG: \x89PNG\r\n\x1a\n
        self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_parse_exercise_data(self):
        canonical, data = ChartGenerator.parse_exercise_data("banca", self.exercise_messages)
        self.assertIn("Press banca", canonical)
        self.assertEqual(len(data), 3)

        # Cada punto es (fecha, carga, 1rm_estimado)
        for _, load, e1rm in data:
            self.assertGreater(load, 45.0)
            self.assertGreater(e1rm, load)  # El 1RM estimado debe ser mayor que la carga de repeticiones

    def test_parse_exercise_data_accepts_real_formats(self):
        """Formatos reales: series en la misma línea, sin '-', con '*' y reps sin 'kg'."""
        d0 = self.base_date
        msgs = [
            RawMessage(
                id=30,
                date=d0,
                chat_title="Sesiones",
                text="LUNES\n- Prensa:\n145 kg 11.10.9\n- Press banca: 60kg 10 8 8 8",
            ),
            RawMessage(
                id=31,
                date=d0 + timedelta(days=7),
                chat_title="Sesiones",
                text="VIERNES\nPrensa: 125 11 11 9 9\nPress banca: 62.5 12 10 8 8",
            ),
            RawMessage(
                id=32,
                date=d0 + timedelta(days=14),
                chat_title="Sesiones",
                text="LUNES\n* Prensa de piernas (140 kg 11.10.10)\n- Press banca: 65kg 3x8",
            ),
        ]
        _, prensa = ChartGenerator.parse_exercise_data("prensa", msgs)
        _, banca = ChartGenerator.parse_exercise_data("banca", msgs)
        self.assertEqual(len(prensa), 3, "no parseó los 3 formatos de prensa")
        self.assertEqual(len(banca), 3, "no parseó los 3 formatos de press banca")
        self.assertEqual([round(p[1]) for p in prensa], [145, 125, 140])
        self.assertEqual([round(b[1]) for b in banca], [60, 62, 65])

    def test_parse_exercise_data_does_not_contaminate(self):
        """'prensa' no debe mezclar 'Gemelos en prensa'; 'banca' no debe mezclar 'Press militar'."""
        d0 = self.base_date
        msgs = [
            RawMessage(
                id=40,
                date=d0,
                chat_title="Sesiones",
                text="LUNES\n- Prensa: 120kg 10 8 8\n- Gemelos en prensa: 100kg 12 12 12\n- Press banca: 60kg 8 8 8\n- Press militar: 30kg 8 8 8",
            ),
            RawMessage(
                id=41,
                date=d0 + timedelta(days=7),
                chat_title="Sesiones",
                text="LUNES\n- Gemelos prensa: 100kg 12 12 12\n- Press militar: 32.5kg 8 8 8",
            ),
        ]
        _, prensa = ChartGenerator.parse_exercise_data("prensa", msgs)
        _, banca = ChartGenerator.parse_exercise_data("banca", msgs)
        _, gemelos = ChartGenerator.parse_exercise_data("gemelos", msgs)
        self.assertEqual(len(prensa), 1, "prensa se contaminó con gemelos")
        self.assertEqual(len(banca), 1, "banca se contaminó con press militar")
        self.assertEqual(len(gemelos), 2, "gemelos debe seguir encontrándose al pedirlo")

    def test_generate_exercise_chart_returns_valid_png(self):
        result = ChartGenerator.generate_exercise_chart("banca", self.exercise_messages)
        self.assertIsNotNone(result)
        buf, canonical = result
        self.assertIn("Press banca", canonical)
        self.assertIsInstance(buf, io.BytesIO)
        content = buf.getvalue()
        self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_empty_or_insufficient_data_returns_none(self):
        empty_chart = ChartGenerator.generate_weight_chart([])
        self.assertIsNone(empty_chart)

        one_msg = [self.exercise_messages[0]]
        single_pt_chart = ChartGenerator.generate_exercise_chart("banca", one_msg)
        self.assertIsNone(single_pt_chart)


if __name__ == "__main__":
    unittest.main()
