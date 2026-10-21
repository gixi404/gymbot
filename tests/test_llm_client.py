"""Unit tests for the LLM hybrid client, think filter, and fallback logic."""

import unittest
from unittest.mock import patch

from ai.analyzer import FITNESS_COACH_SYSTEM_INSTRUCTION
from ai.llm_client import LLMClient, ThinkFilter, strip_think_blocks
from ai.router import QueryRoute
from config.settings import Settings


class TestThinkFilterAndStripper(unittest.TestCase):
    def test_strip_think_blocks_closed(self):
        text = "<think>some internal thought</think>This is the real answer."
        self.assertEqual(strip_think_blocks(text), "This is the real answer.")

    def test_strip_think_blocks_unclosed(self):
        text = "<think>unclosed thought that cut off due to max tokens"
        self.assertEqual(strip_think_blocks(text), "")

    def test_think_filter_streaming(self):
        think_filter = ThinkFilter()
        chunks = ["<th", "ink>I am thi", "nking</th", "ink>Hello ", "world!"]
        visible = []
        for c in chunks:
            out = think_filter.feed(c)
            if out:
                visible.append(out)
        tail = think_filter.flush()
        if tail:
            visible.append(tail)
        self.assertEqual("".join(visible), "Hello world!")

    def test_think_filter_only_thinking(self):
        think_filter = ThinkFilter()
        chunks = ["<think>", "internal reasoning that exhausts tokens"]
        visible = [think_filter.feed(c) for c in chunks if think_filter.feed(c)]
        tail = think_filter.flush()
        if tail:
            visible.append(tail)
        self.assertEqual("".join(visible), "")


class TestLLMFallbackOnEmptyReasoning(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(
            TELEGRAM_API_ID=123,
            TELEGRAM_API_HASH="abc",
            TELEGRAM_WORKOUT_CHAT_ID=-1001,
            TELEGRAM_BOT_TOKEN="token",
            ALLOWED_TELEGRAM_USER_IDS="100",
            GEMINI_API_KEY="test_gemini_key",
            GROQ_API_KEY="test_groq_key",
            DEEPSEEK_API_KEY="test_deepseek_key",
        )

    @patch("ai.llm_client.GeminiClient")
    @patch("ai.llm_client.DeepSeekClient")
    @patch("ai.llm_client.GroqClient")
    async def test_streaming_fallback_when_deepseek_produces_only_think(
        self, mock_groq_cls, mock_deepseek_cls, mock_gemini_cls
    ):
        mock_gemini = mock_gemini_cls.return_value
        mock_deepseek = mock_deepseek_cls.return_value

        # DeepSeek raises RuntimeError when only <think> was returned
        async def mock_deepseek_stream(*args, **kwargs):
            raise RuntimeError("DeepSeek 'deepseek-flash' produced only reasoning blocks without visible text.")
            yield "never"

        mock_deepseek.generate_response_stream_async = mock_deepseek_stream

        # Gemini stream yields the actual response
        async def mock_gemini_stream(*args, **kwargs):
            yield "Evaluación: "
            yield "Tu superávit actual es correcto."

        mock_gemini.generate_response_stream_async = mock_gemini_stream

        client = LLMClient(settings=self.settings)
        route = QueryRoute(complexity="complex", groups=["PesoCorporal"], days_lookback=7)

        chunks = []
        async for chunk in client.generate_response_stream_async(
            prompt="¿Qué opinas de mi peso?",
            system_instruction="inst",
            route=route,
        ):
            chunks.append(chunk)

        result = "".join(chunks)
        self.assertEqual(result, "Evaluación: Tu superávit actual es correcto.")

    @patch("ai.llm_client.GeminiClient")
    @patch("ai.llm_client.DeepSeekClient")
    @patch("ai.llm_client.GroqClient")
    def test_sync_fallback_when_engine_produces_empty_visible_text(
        self, mock_groq_cls, mock_deepseek_cls, mock_gemini_cls
    ):
        mock_gemini = mock_gemini_cls.return_value
        mock_deepseek = mock_deepseek_cls.return_value

        mock_deepseek.generate_response.side_effect = RuntimeError(
            "DeepSeek 'deepseek-flash' produced only reasoning blocks without visible text."
        )
        mock_gemini.generate_response.return_value = "Respuesta desde Gemini."

        client = LLMClient(settings=self.settings)
        route = QueryRoute(complexity="complex", groups=["PesoCorporal"], days_lookback=7)

        resp = client.generate_response(
            prompt="¿Qué opinas de mi peso?",
            system_instruction="inst",
            route=route,
        )
        self.assertEqual(resp, "Respuesta desde Gemini.")


class TestDirectivesAndInstructions(unittest.TestCase):
    def test_system_instruction_forbids_generic_refusals(self):
        instruction = FITNESS_COACH_SYSTEM_INSTRUCTION
        self.assertIn("PROHIBIDO RECHAZAR", instruction)
        self.assertIn("No pude generar una respuesta concreta", instruction)
        self.assertIn("preguntas breves", instruction)
