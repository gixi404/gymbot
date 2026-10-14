"""Hybrid GymBot LLM client (Groq / DeepSeek / Gemini).

- **Level 3 — Execution routing:** simple, light queries go to
  `qwen/qwen3-32b` on Groq (fast, avoids Gemini congestion); complex, large
  or multi-group queries spanning months go to DeepSeek.
- **Automatic fallback:** if the primary engine fails (429, outage), it retries
  with the other and finally with Gemini, without breaking streaming.
- **Filter`; the streaming generator suppresses them in real time.
- **Gemini** is reserved for vision (photos/documents) and as the last resort.

The public interface (`generate_response`, `transcribe_audio`,
`generate_response_stream_async`) is identical to that of the old GeminiClient so that
`GymAnalyzer` does not depend on the provider.
"""

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI, RateLimitError

from config.settings import Settings, get_settings

from .gemini_client import GeminiClient
from .router import QueryRoute

logger = logging.getLogger(__name__)

GROQ_RATE_LIMIT_BACKOFF = 2.0  # Seconds to wait on a Groq 429

# Characters per token used to estimate prompt size (~3.8 for Spanish).
CHARS_PER_TOKEN = 3.8

_THINK_RE = re.compile("<think>.*?</think>", re.DOTALL)
_THINK_TAIL_RE = re.compile("<think>.*$", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove reasoning blocks (including an unclosed one) from a full text."""
    return _THINK_TAIL_RE.sub("", _THINK_RE.sub("", text))


class ThinkFilter:
    """Suppress reasoning blocks while streaming, even if the marker arrives split."""

    # Markers built by concatenation: they prevent external tools from cutting the literal.
    _OPEN = "<" + "think" + ">"
    _CLOSE = "<" + "/" + "think" + ">"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def _hold_tail(self, marker: str) -> int:
        """Trailing buffer characters that could be the start of `marker`."""
        for size in range(len(marker) - 1, 0, -1):
            if self._buffer.endswith(marker[:size]):
                return size
        return 0

    def feed(self, chunk: str) -> str:
        """Process a stream chunk and return only the visible text."""
        self._buffer += chunk
        out: list[str] = []

        while True:
            if self._in_think:
                idx = self._buffer.find(self._CLOSE)
                if idx == -1:
                    self._buffer = self._buffer[-self._hold_tail(self._CLOSE) :]
                    break
                self._buffer = self._buffer[idx + len(self._CLOSE) :]
                self._in_think = False
                continue

            idx = self._buffer.find(self._OPEN)
            if idx != -1:
                out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self._OPEN) :]
                self._in_think = True
                continue

            hold = self._hold_tail(self._OPEN)
            if len(self._buffer) > hold:
                out.append(self._buffer[: len(self._buffer) - hold])
                self._buffer = self._buffer[-hold:] if hold else ""
            break

        return "".join(out)

    def flush(self) -> str:
        """Flush the remaining buffer on stream close (nothing if it ended inside the block)."""
        rest, self._buffer = self._buffer, ""
        return "" if self._in_think else rest


class GroqClient:
    """Groq client on top of the OpenAI SDK (no vision; Whisper for audio)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_BASE_URL
        self.audio_model = settings.GROQ_AUDIO_MODEL
        self.classifier_model = settings.GROQ_CLASSIFIER_MODEL

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables (.env).")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _dedupe(models) -> list[str]:
        seen: set[str] = set()
        candidates: list[str] = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                candidates.append(m)
        return candidates

    def _candidate_models(self, models=None) -> list[str]:
        """Candidate models (primary + fallbacks) without duplicates."""
        return self._dedupe(models if models is not None else self.settings.GROQ_FALLBACK_MODELS)

    def _messages(self, prompt: str, system_instruction: str) -> list[dict]:
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

    def generate_json(
        self,
        prompt: str,
        system_instruction: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Ask the model for a strict JSON response (used by the classifier)."""
        messages = self._messages(prompt, system_instruction)
        last_error: Exception | None = None

        for candidate in self._candidate_models([model or self.classifier_model]):
            try:
                response = self.client.chat.completions.create(
                    model=candidate,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - propagated so the fallback can act
                last_error = e
                logger.warning("Groq '%s' failed generating JSON (%s)", candidate, e)

        raise RuntimeError(f"Failed to generate JSON with Groq ({self._candidate_models([model])}): {last_error}")

    def generate_response(self, prompt: str, system_instruction: str, temperature: float = 0.2, models=None) -> str:
        """Generate text with immediate fallback across the given models."""
        messages = self._messages(prompt, system_instruction)
        last_error: Exception | None = None

        for model in self._candidate_models(models):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.settings.GROQ_MAX_OUTPUT_TOKENS,
                )
                text = (response.choices[0].message.content or "").strip()
                return strip_think_blocks(text).strip() or "No se obtuvo respuesta de texto del modelo."
            except (RateLimitError, APIStatusError, APIConnectionError) as e:
                last_error = e
                if isinstance(e, RateLimitError):
                    time.sleep(GROQ_RATE_LIMIT_BACKOFF)
                logger.warning("Groq '%s' failed (%s). Trying the next model...", model, e)
            except Exception as e:  # noqa: BLE001 - safety net to delegate to Gemini
                last_error = e
                logger.warning("Unexpected exception in Groq '%s': %s. Trying the next one...", model, e)

        raise RuntimeError(f"Failed to invoke Groq with models {self._candidate_models(models)}: {last_error}")

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe audio with Whisper (Groq)."""
        clean_mime = (mime_type or "audio/ogg").split(";")[0].strip().lower()
        if clean_mime in ("audio/oga", "audio/opus"):
            clean_mime = "audio/ogg"
        extension = {"audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/wav": "wav"}.get(
            clean_mime, "ogg"
        )
        response = self.client.audio.transcriptions.create(
            model=self.audio_model,
            file=(f"audio.{extension}", audio_bytes, clean_mime),
            language="es",
        )
        return (response.text or "").strip()

    async def generate_response_stream_async(
        self, prompt: str, system_instruction: str, temperature: float = 0.2, models=None
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming with `` filter and immediate fallback across Groq models."""
        messages = self._messages(prompt, system_instruction)
        last_error: Exception | None = None
        yielded_any = False
        yielded_visible = False
        think_filter = ThinkFilter()

        for model in self._candidate_models(models):
            try:
                stream = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.settings.GROQ_MAX_OUTPUT_TOKENS,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yielded_any = True
                        visible = think_filter.feed(chunk.choices[0].delta.content)
                        if visible:
                            yielded_visible = True
                            yield visible

                tail = think_filter.flush()
                if tail:
                    yield tail
                    yielded_visible = True

                if not yielded_visible:
                    # The whole stream was empty internal reasoning.
                    logger.warning("Groq model '%s' returned only <think> blocks.", model)
                    yield "No pude generar una respuesta concreta. Intenta reformular la pregunta."
                return
            except (RateLimitError, APIStatusError, APIConnectionError) as e:
                last_error = e
                if yielded_any:
                    logger.error("Intermediate error in Groq stream '%s' after emitting tokens: %s", model, e)
                    return
                if isinstance(e, RateLimitError):
                    await asyncio.sleep(GROQ_RATE_LIMIT_BACKOFF)
                logger.warning("Groq stream '%s' failed (%s). Trying the next one...", model, e)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if yielded_any:
                    logger.error("Intermediate exception in Groq stream '%s' after emitting tokens: %s", model, e)
                    return
                logger.warning("Unexpected exception in Groq stream '%s': %s. Trying the next one...", model, e)

        raise RuntimeError(f"Failed to stream with Groq {self._candidate_models(models)}: {last_error}")


class DeepSeekClient:
    """DeepSeek client on top of the OpenAI SDK (complex analysis and Groq fallback)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model_name = settings.DEEPSEEK_MODEL
        self.max_output_tokens = settings.DEEPSEEK_MAX_OUTPUT_TOKENS

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set in the environment variables (.env).")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _messages(prompt: str, system_instruction: str) -> list[dict]:
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

    def generate_response(self, prompt: str, system_instruction: str, temperature: float = 0.2) -> str:
        """Generate text by removing the `` blocks of explicit reasoning."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._messages(prompt, system_instruction),
                temperature=temperature,
                max_tokens=self.max_output_tokens,
            )
            text = strip_think_blocks(response.choices[0].message.content or "").strip()
            return text or "No se obtuvo respuesta de texto del modelo."
        except (RateLimitError, APIStatusError, APIConnectionError) as e:
            raise RuntimeError(f"DeepSeek '{self.model_name}' failed: {e}") from e

    async def generate_response_stream_async(
        self, prompt: str, system_instruction: str, temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming with `` filter in real time."""
        stream = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(prompt, system_instruction),
            temperature=temperature,
            max_tokens=self.max_output_tokens,
            stream=True,
        )
        think_filter = ThinkFilter()
        yielded_visible = False

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                visible = think_filter.feed(chunk.choices[0].delta.content)
                if visible:
                    yielded_visible = True
                    yield visible

        tail = think_filter.flush()
        if tail:
            yield tail
            yielded_visible = True

        if not yielded_visible:
            logger.warning("DeepSeek '%s' returned only <think> blocks.", self.model_name)
            yield "No pude generar una respuesta concreta. Intenta reformular la pregunta."


class LLMClient:
    """Hybrid router: light Groq / DeepSeek by complexity, Gemini for vision."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.gemini = GeminiClient(self.settings)

        self.groq: GroqClient | None = None
        if self.settings.GROQ_API_KEY:
            try:
                self.groq = GroqClient(self.settings)
            except Exception as e:  # noqa: BLE001
                logger.warning("Groq unavailable (%s). DeepSeek/Gemini will be used.", e)
        else:
            logger.warning("GROQ_API_KEY is empty. No light Groq engine.")

        self.deepseek: DeepSeekClient | None = None
        if self.settings.DEEPSEEK_API_KEY:
            try:
                self.deepseek = DeepSeekClient(self.settings)
            except Exception as e:  # noqa: BLE001
                logger.warning("DeepSeek unavailable (%s). Groq/Gemini will be used.", e)
        else:
            logger.info("DEEPSEEK_API_KEY is empty. DeepSeek stays disabled.")

        # Light engine models (Groq primary + configured fallbacks).
        self.light_models = GroqClient._dedupe([self.settings.GROQ_LIGHT_MODEL, *self.settings.GROQ_FALLBACK_MODELS])

    def _use_light_engine(self, route: QueryRoute | None, estimated_tokens: float, threshold: int) -> bool:
        """Decision rule: simple and small → Groq; the rest → DeepSeek."""
        long_multi = (
            route is not None
            and len(route.groups) >= 2
            and (route.days_lookback == 0 or route.days_lookback > self.settings.ROUTER_MULTI_GROUP_DAYS)
        )
        if route is None:
            # Unclassified (weekly summary, progression, macros...): small → Groq.
            return estimated_tokens < threshold
        return route.complexity == "simple" and estimated_tokens < threshold and not long_multi

    def _engine_order(self, prompt: str, system_instruction: str, route: QueryRoute | None) -> list[str]:
        """Strict engine chain according to the decision rule.

        - Simple/light queries: light Groq -> DeepSeek (then Gemini, outside here).
        - Complex queries: DeepSeek only (never retried on light Groq).
        """
        estimated_tokens = (len(prompt) + len(system_instruction)) / CHARS_PER_TOKEN
        order: list[str] = []

        if self._use_light_engine(route, estimated_tokens, self.settings.ROUTER_TOKEN_THRESHOLD):
            if self.groq:
                order.append("light")
            if self.deepseek:
                order.append("deepseek")
        else:
            if self.deepseek:
                order.append("deepseek")

        return order

    def _sync_call(self, engine: str, prompt: str, system_instruction: str, temperature: float) -> str:
        assert engine in ("light", "deepseek")
        if engine == "light" and self.groq:
            return self.groq.generate_response(prompt, system_instruction, temperature, models=self.light_models)
        if engine == "deepseek" and self.deepseek:
            return self.deepseek.generate_response(prompt, system_instruction, temperature)
        raise RuntimeError(f"Engine '{engine}' not configured")

    def _stream_call(
        self, engine: str, prompt: str, system_instruction: str, temperature: float
    ) -> AsyncGenerator[str, None]:
        assert engine in ("light", "deepseek")
        if engine == "light" and self.groq:
            return self.groq.generate_response_stream_async(
                prompt, system_instruction, temperature, models=self.light_models
            )
        if engine == "deepseek" and self.deepseek:
            return self.deepseek.generate_response_stream_async(prompt, system_instruction, temperature)
        raise RuntimeError(f"Engine '{engine}' not configured")

    def generate_json(self, prompt: str, system_instruction: str, max_tokens: int = 256) -> str:
        """Strict JSON for the classifier (Groq); Gemini as fallback."""
        if self.groq:
            try:
                return self.groq.generate_json(prompt, system_instruction, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                logger.warning("Groq classification unavailable (%s). Falling back to Gemini.", e)
        return self.gemini.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.0,
        )

    def generate_response(
        self,
        prompt: str,
        system_instruction: str,
        attachments: list[tuple[bytes, str]] | None = None,
        temperature: float = 0.2,
        route: QueryRoute | None = None,
    ) -> str:
        """Text → light Groq/DeepSeek; attachments (vision) → Gemini as fallback."""
        if not attachments and (self.groq or self.deepseek):
            for engine in self._engine_order(prompt, system_instruction, route):
                try:
                    return self._sync_call(engine, prompt, system_instruction, temperature)
                except Exception as e:  # noqa: BLE001 - degrades to the next engine
                    logger.warning("Engine '%s' failed (%s). Trying the next engine...", engine, e)
        return self.gemini.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            attachments=attachments,
            temperature=temperature,
        )

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Audio → Whisper (Groq); failure → Gemini."""
        if self.groq:
            try:
                return self.groq.transcribe_audio(audio_bytes, mime_type)
            except Exception as e:  # noqa: BLE001
                logger.warning("Whisper (Groq) failed (%s). Falling back to Gemini.", e)
        return self.gemini.transcribe_audio(audio_bytes, mime_type=mime_type)

    async def generate_response_stream_async(
        self,
        prompt: str,
        system_instruction: str,
        attachments: list[tuple[bytes, str]] | None = None,
        temperature: float = 0.2,
        route: QueryRoute | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming: text → light Groq/DeepSeek; attachments or total failure → Gemini."""
        if not attachments and (self.groq or self.deepseek):
            yielded_any = False
            for engine in self._engine_order(prompt, system_instruction, route):
                try:
                    async for chunk in self._stream_call(engine, prompt, system_instruction, temperature):
                        yielded_any = True
                        yield chunk
                    return
                except Exception as e:  # noqa: BLE001 - degrades to the next engine
                    if yielded_any:
                        logger.error("Intermediate error in engine '%s' stream after emitting tokens: %s", engine, e)
                        return
                    logger.warning("Engine '%s' stream failed (%s). Trying the next engine...", engine, e)

        async for chunk in self.gemini.generate_response_stream_async(
            prompt=prompt,
            system_instruction=system_instruction,
            attachments=attachments,
            temperature=temperature,
        ):
            yield chunk
