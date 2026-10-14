"""Pruebas del clasificador (Nivel 1) y del enrutador de ejecución (Nivel 3)."""

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from ai.analyzer import GymAnalyzer
from ai.llm_client import LLMClient, ThinkFilter, strip_think_blocks
from ai.router import GROUPS, QueryRoute, classify_query, fallback_route, resolve_group_targets
from config.settings import Settings
from storage.database import GymDatabase
from storage.models import RawMessage


class TestQueryRoute(unittest.TestCase):
    def test_valid_json_route(self):
        route = QueryRoute.model_validate(
            {"complexity": "complex", "groups": ["sesiones", "PesoCorporal"], "days_lookback": "90"}
        )
        self.assertEqual(route.complexity, "complex")
        self.assertEqual(route.groups, ["Sesiones", "PesoCorporal"])
        self.assertEqual(route.days_lookback, 90)

    def test_defaults_and_sanitization(self):
        route = QueryRoute.model_validate({"groups": ["GrupoInexistente"], "days_lookback": -5})
        self.assertEqual(route.groups, list(GROUPS))
        self.assertEqual(route.complexity, "complex")
        self.assertEqual(route.days_lookback, 0)


class TestClassifyQuery(unittest.TestCase):
    def test_successful_classification(self):
        llm = MagicMock()
        llm.generate_json.return_value = '{"complexity": "simple", "groups": ["Comidas"], "days_lookback": 1}'
        route = classify_query("¿Cuántas comidas registré ayer?", llm)
        self.assertEqual(route.complexity, "simple")
        self.assertEqual(route.groups, ["Comidas"])
        self.assertEqual(route.days_lookback, 1)

    def test_fallback_on_broken_json(self):
        llm = MagicMock()
        llm.generate_json.return_value = "no soy json"
        route = classify_query("analiza todo", llm)
        self.assertEqual(route, fallback_route())

    def test_fallback_on_exception(self):
        llm = MagicMock()
        llm.generate_json.side_effect = TimeoutError("clasificador caído")
        route = classify_query("analiza todo", llm)
        self.assertEqual(route.groups, list(GROUPS))
        self.assertEqual(route.days_lookback, 0)

    def test_fallback_without_client(self):
        self.assertEqual(classify_query("hola", None), fallback_route())

    def test_conversation_history_is_included_in_prompt(self):
        llm = MagicMock()
        llm.generate_json.return_value = '{"complexity": "simple", "groups": ["Comidas"], "days_lookback": 7}'
        history = [
            {"role": "user", "text": "3 ejercicios de pecho"},
            {"role": "model", "text": "Press banca, fondos y aperturas."},
        ]
        classify_query("¿cuál de esos 3?", llm, conversation_history=history)
        prompt = llm.generate_json.call_args.kwargs["prompt"]
        self.assertIn("CONTEXTO PREVIO:", prompt)
        self.assertIn("3 ejercicios de pecho", prompt)
        self.assertIn("PREGUNTA ACTUAL: ¿cuál de esos 3?", prompt)

    def test_without_history_prompt_is_bare_question(self):
        llm = MagicMock()
        llm.generate_json.return_value = '{"complexity": "simple", "groups": ["Comidas"], "days_lookback": 1}'
        classify_query("¿Qué comí?", llm)
        self.assertEqual(llm.generate_json.call_args.kwargs["prompt"], "¿Qué comí?")


class TestThinkFilter(unittest.TestCase):
    # Los marcadores se construyen por concatenación: un editor puede cortar el literal.
    OPEN = "<" + "think" + ">"
    CLOSE = "<" + "/" + "think" + ">"

    def test_blocks_are_stripped(self):
        f = ThinkFilter()
        self.assertEqual(f.feed("Hola "), "Hola ")
        self.assertEqual(f.feed(self.OPEN + "razonamiento" + self.CLOSE), "")
        self.assertEqual(f.feed(" lista"), " lista")
        self.assertEqual(f.flush(), "")

    def test_marker_split_across_chunks(self):
        f = ThinkFilter()
        self.assertEqual(f.feed("texto <th"), "texto ")
        self.assertEqual(f.feed("ink>oculto</thi"), "")
        self.assertEqual(f.feed("nk>visible"), "visible")

    def test_no_think_blocks_untouched(self):
        f = ThinkFilter()
        self.assertEqual(f.feed("80kg x 8"), "80kg x 8")
        self.assertEqual(f.feed(" y 85kg x 6"), " y 85kg x 6")
        self.assertEqual(f.flush(), "")

    def test_unclosed_block_dropped(self):
        self.assertEqual(strip_think_blocks("Hola\n" + self.OPEN + "borrador"), "Hola\n")
        self.assertEqual(strip_think_blocks("a" + self.OPEN + "x" + self.CLOSE + "b"), "ab")


class TestGroupResolution(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            TELEGRAM_API_ID=1,
            TELEGRAM_API_HASH="h",
            TELEGRAM_WORKOUT_CHAT_ID=-1,
            GEMINI_API_KEY="k",
            GROUP_CHAT_MAP={"Sesiones": ["-5000217070", "GYM"], "PesoCorporal": ["-5397846915"]},
        )

    def test_title_and_id_targets(self):
        titles, chat_ids = resolve_group_targets(["Sesiones"], self.settings)
        self.assertIn("sesiones", titles)
        self.assertIn("gym", titles)
        self.assertIn("-5000217070", chat_ids)
        self.assertIn("GYM", chat_ids)  # las entradas del mapa actúan como título o ID

    def test_unknown_group_falls_back_to_all(self):
        titles, chat_ids = resolve_group_targets([], self.settings)
        self.assertEqual((titles, chat_ids), ([], []))

    def test_all_groups_expand(self):
        titles, chat_ids = resolve_group_targets(GROUPS, self.settings)
        self.assertEqual(
            [t for t in ("sesiones", "cardio", "comidas", "pesocorporal") if t in titles],
            ["sesiones", "cardio", "comidas", "pesocorporal"],
        )
        # El grupo principal también captura los mensajes sin chat_title.
        self.assertIn("", titles)
        self.assertIn("-5397846915", chat_ids)
        self.assertIn("GYM", chat_ids)


class TestEngineRouting(unittest.TestCase):
    def _client(self, **kwargs) -> LLMClient:
        base = {
            "TELEGRAM_API_ID": 1,
            "TELEGRAM_API_HASH": "h",
            "TELEGRAM_WORKOUT_CHAT_ID": -1,
            "GEMINI_API_KEY": "k",
            "GROQ_API_KEY": "groq-key",
            "DEEPSEEK_API_KEY": "ds-key",
        }
        return LLMClient(Settings(**{**base, **kwargs}))

    def _simple_small(self):
        return QueryRoute(complexity="simple", groups=["Comidas"], days_lookback=1)

    def _complex(self):
        return QueryRoute(complexity="complex", groups=["Sesiones", "PesoCorporal"], days_lookback=90)

    def test_simple_small_chain_is_groq_then_deepseek(self):
        client = self._client()
        order = client._engine_order("p" * 4000, "s" * 1000, self._simple_small())
        self.assertEqual(order, ["light", "deepseek"])

    def test_big_prompt_goes_to_deepseek_only(self):
        client = self._client()
        order = client._engine_order("p" * 9000, "s" * 9000, self._simple_small())
        self.assertEqual(order, ["deepseek"])  # sin reintento en Groq ligero

    def test_multigroup_long_window_goes_to_deepseek_only(self):
        client = self._client()
        route = QueryRoute(complexity="simple", groups=["Sesiones", "PesoCorporal"], days_lookback=0)
        order = client._engine_order("p" * 100, "s" * 100, route)
        self.assertEqual(order, ["deepseek"])

    def test_complex_never_retries_groq(self):
        client = self._client()
        order = client._engine_order("p" * 100, "s" * 100, self._complex())
        self.assertEqual(order, ["deepseek"])
        self.assertNotIn("light", order)

    def test_complex_without_deepseek_goes_to_gemini(self):
        client = self._client(DEEPSEEK_API_KEY="")
        self.assertEqual(client._engine_order("p" * 100, "s" * 100, self._complex()), [])

    def test_simple_without_deepseek_stays_on_groq(self):
        client = self._client(DEEPSEEK_API_KEY="")
        self.assertEqual(client._engine_order("p" * 100, "s" * 100, self._simple_small()), ["light"])

    def test_simple_without_groq_goes_to_deepseek(self):
        client = self._client(GROQ_API_KEY="")
        self.assertEqual(client._engine_order("p" * 100, "s" * 100, self._simple_small()), ["deepseek"])

    def test_no_route_small_uses_groq(self):
        client = self._client()
        self.assertEqual(client._engine_order("p" * 100, "s" * 100, None), ["light", "deepseek"])

    def test_light_models_only_configured_one(self):
        # GROQ_FALLBACK_MODELS vacío: un único intento en Groq.
        client = self._client(GROQ_FALLBACK_MODELS=[])
        self.assertEqual(client.light_models, ["qwen/qwen3.8-27b"])


class TestFilteredContext(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = GymDatabase(Path(self.temp_dir.name) / "t.db")
        now = datetime.now()
        self.db.upsert_messages(
            [
                RawMessage(id=1, date=now, text="Press banca 50kg 3x10", chat_id="-1001", chat_title="Sesiones"),
                RawMessage(id=2, date=now, text="Peso 70kg", chat_id="-1002", chat_title="PesoCorporal"),
                RawMessage(
                    id=3,
                    date=now,
                    text="Desayuno: huevos y avena",
                    chat_id="-1003",
                    chat_title="Comidas",
                ),
            ]
        )
        self.settings = Settings(
            TELEGRAM_API_ID=1,
            TELEGRAM_API_HASH="h",
            TELEGRAM_WORKOUT_CHAT_ID=-1,
            GEMINI_API_KEY="k",
            GROQ_API_KEY="groq-key",
            DEEPSEEK_API_KEY="ds-key",
        )
        self.analyzer = GymAnalyzer(settings=self.settings, database=self.db, llm_client=MagicMock())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_filter_by_group(self):
        ctx = self.analyzer.build_focused_context("¿Qué comí?", groups=["Comidas"], days_lookback=7)
        self.assertIn("huevos", ctx)
        self.assertNotIn("Press banca", ctx)
        self.assertNotIn("Peso 70kg", ctx)

    def test_existing_rows_without_title_are_matched_by_chat_id(self):
        ctx = self.analyzer.build_focused_context("¿Cuánto peso?", groups=["PesoCorporal"], days_lookback=7)
        self.assertIn("Peso 70kg", ctx)
        self.assertNotIn("Press banca", ctx)

    def test_legacy_rows_without_title_and_no_map_fall_back_to_unfiltered(self):
        # Sin GROUP_CHAT_MAP ni título, el filtro no debe devolver contexto vacío.
        self.db.upsert_messages(
            [RawMessage(id=10, date=datetime.now(), text="Sentadilla 100kg 3x5", chat_id="", chat_title="")]
        )
        ctx = self.analyzer.build_focused_context("¿Qué hice en sentadilla?", groups=["Sesiones"], days_lookback=7)
        self.assertIn("Sentadilla 100kg", ctx)

    def test_days_lookback_excludes_old_records(self):
        old = datetime(2020, 1, 1, 10, 0)
        self.db.upsert_messages(
            [RawMessage(id=20, date=old, text="Correr 5km lento", chat_id="-1004", chat_title="Cardio")]
        )
        ctx = self.analyzer.build_focused_context("¿Hice cardio?", groups=["Cardio"], days_lookback=7)
        self.assertNotIn("Correr 5km", ctx)

    def test_memory_block_is_capped(self):
        # 4 turnos de 5000 chars: cada mensaje se recorta a 800 y el bloque a ~2400.
        history = [{"role": "user" if i % 2 == 0 else "model", "text": "x" * 5000} for i in range(4)]
        prompt = self.analyzer._build_prompt("¿y eso?", fallback_route(), history)
        block = prompt.split("MEMORIA DE LA CONVERSACIÓN PREVIA")[1].split("CONSULTA DEL ATLETA")[0]
        self.assertLessEqual(block.count("x"), 800 * 3 + 100)
        self.assertNotIn("x" * 801, block)


class _FakeEngine:
    """Motor falso que evita tocar el SDK real de OpenAI."""

    def __init__(self, tag: str, failing: bool = False):
        self.tag = tag
        self.failing = failing

    def generate_response(self, prompt, system_instruction, temperature=0.2, models=None):
        if self.failing:
            raise RuntimeError("simulated 429")
        return f"respuesta-{self.tag}"

    async def generate_response_stream_async(self, prompt, system_instruction, temperature=0.2, models=None):
        if self.failing:
            raise RuntimeError("simulated 429")
        for chunk in (f"parte-{self.tag}1 ", f"parte-{self.tag}2"):
            yield chunk


async def _async_iter(items):
    for i in items:
        yield i


class TestEngineFallback(unittest.TestCase):
    def _client(self):
        settings = Settings(
            TELEGRAM_API_ID=1,
            TELEGRAM_API_HASH="h",
            TELEGRAM_WORKOUT_CHAT_ID=-1,
            GEMINI_API_KEY="k",
            GROQ_API_KEY="groq-key",
            DEEPSEEK_API_KEY="ds-key",
        )
        return LLMClient(settings)

    @staticmethod
    def _complex_route():
        return QueryRoute(complexity="complex", groups=["Sesiones", "PesoCorporal"], days_lookback=90)

    def test_groq_429_falls_back_to_deepseek(self):
        client = self._client()
        client.groq = _FakeEngine("groq", failing=True)
        client.deepseek = _FakeEngine("deepseek")
        route = QueryRoute(complexity="simple", groups=["Comidas"], days_lookback=1)
        self.assertEqual(client.generate_response("p" * 100, "sys", route=route), "respuesta-deepseek")

    def test_groq_ok_uses_light_engine(self):
        client = self._client()
        client.groq = _FakeEngine("groq")
        client.deepseek = _FakeEngine("deepseek")
        route = QueryRoute(complexity="simple", groups=["Comidas"], days_lookback=1)
        self.assertEqual(client.generate_response("p" * 100, "sys", route=route), "respuesta-groq")

    def test_complex_deepseek_failure_goes_straight_to_gemini(self):
        # Cadena de complejas: DeepSeek -> Gemini. Nunca se reintenta en Groq ligero.
        client = self._client()
        client.deepseek = _FakeEngine("deepseek", failing=True)
        client.groq = MagicMock()
        client.gemini = MagicMock()
        client.gemini.generate_response.return_value = "respuesta-gemini"
        route = self._complex_route()

        self.assertEqual(client.generate_response("p" * 100, "sys", route=route), "respuesta-gemini")
        client.groq.generate_response.assert_not_called()
        client.gemini.generate_response.assert_called_once()

    def test_streaming_complex_deepseek_failure_goes_to_gemini(self):
        client = self._client()
        client.deepseek = _FakeEngine("deepseek", failing=True)
        client.groq = MagicMock()
        client.gemini = MagicMock()
        client.gemini.generate_response_stream_async.return_value = _async_iter(["respuesta-gemini"])

        async def _run():
            return [
                c async for c in client.generate_response_stream_async("p" * 100, "sys", route=self._complex_route())
            ]

        self.assertEqual(asyncio.run(_run()), ["respuesta-gemini"])
        client.groq.generate_response_stream_async.assert_not_called()

    def test_streaming_fallback_does_not_break(self):
        client = self._client()
        client.groq = _FakeEngine("groq", failing=True)
        client.deepseek = _FakeEngine("deepseek")
        route = QueryRoute(complexity="simple", groups=["Comidas"], days_lookback=1)

        async def _run():
            chunks = [c async for c in client.generate_response_stream_async("p" * 100, "sys", route=route)]
            return chunks

        self.assertEqual("".join(asyncio.run(_run())), "parte-deepseek1 parte-deepseek2")


class TestChatTitleBackfill(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = GymDatabase(Path(self.temp_dir.name) / "t.db")
        self.settings = Settings(
            TELEGRAM_API_ID=1,
            TELEGRAM_API_HASH="h",
            TELEGRAM_WORKOUT_CHAT_ID=-1,
            GEMINI_API_KEY="k",
            GROUP_CHAT_MAP={
                "Sesiones": ["-5000217070", "GYM", "-1004344988533"],
                "PesoCorporal": ["-5397846915"],
            },
        )
        self.db.upsert_messages(
            [
                RawMessage(id=1, date=datetime.now(), text="Prensa 140kg", chat_id="-5000217070"),
                RawMessage(id=2, date=datetime.now(), text="70kg", chat_id="-5397846915"),
                RawMessage(id=3, date=datetime.now(), text="Hip thrust", chat_id=""),
                # Título ya correcto: el backfill no debe pisarlo.
                RawMessage(id=4, date=datetime.now(), text="Otro", chat_id="-5000217070", chat_title="Sesiones"),
            ]
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_backfill_and_idempotency(self):
        from backfill_chat_titles import backfill_chat_titles

        self.assertEqual(backfill_chat_titles(self.db, self.settings), 3)
        titles = {(m.chat_id, m.chat_title) for m in self.db.get_messages()}
        self.assertIn(("-5000217070", "Sesiones"), titles)
        self.assertIn(("-5397846915", "PesoCorporal"), titles)
        self.assertIn(("", "Sesiones"), titles)  # legacy sin chat_id -> grupo principal

        # Segunda ejecución: nada que actualizar.
        self.assertEqual(backfill_chat_titles(self.db, self.settings), 0)

    def test_group_filtering_after_backfill(self):
        from backfill_chat_titles import backfill_chat_titles

        backfill_chat_titles(self.db, self.settings)
        analyzer = GymAnalyzer(settings=self.settings, database=self.db, llm_client=MagicMock())
        ctx = analyzer.build_focused_context("¿Cuánto peso?", groups=["PesoCorporal"], days_lookback=90)
        self.assertIn("70kg", ctx)
        self.assertNotIn("Prensa", ctx)
        self.assertNotIn("Hip thrust", ctx)


if __name__ == "__main__":
    unittest.main()
