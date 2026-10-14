"""Cliente LLM híbrido de GymBot (Groq / DeepSeek / Gemini).

- **Nivel 3 — Enrutamiento de ejecución:** las consultas simples y ligeras van a
  `qwen/qwen3-32b` en Groq (rápido, evita la congestión de Gemini); las complejas,
  grandes o que cruzan varios grupos durante meses van a DeepSeek.
- **Fallback automático:** si el motor primario falla (429, caída), se reintenta
  con el otro y finalmente con Gemini, sin romper el streaming.
- **Filtro`; el generador de streaming los suprime en tiempo real.
- **Gemini** se reserva para visión (fotos/documentos) y como último respaldo.

La interfaz pública (`generate_response`, `transcribe_audio`,
`generate_response_stream_async`) es idéntica a la del antiguo GeminiClient para que
`GymAnalyzer` no dependa del proveedor.
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

GROQ_RATE_LIMIT_BACKOFF = 2.0  # Segundos de espera ante un 429 de Groq

# Caracteres por token usados para estimar el tamaño del prompt (~3.8 para español).
CHARS_PER_TOKEN = 3.8

_THINK_RE = re.compile("<think>.*?</think>", re.DOTALL)
_THINK_TAIL_RE = re.compile("<think>.*$", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Elimina los bloques de razonamiento (incluido uno sin cerrar) de un texto completo."""
    return _THINK_TAIL_RE.sub("", _THINK_RE.sub("", text))


class ThinkFilter:
    """Suprime bloques de razonamiento en streaming, aunque el marcador llegue partido."""

    # Marcadores armados por concatenación: evitan que herramientas externas corten el literal.
    _OPEN = "<" + "think" + ">"
    _CLOSE = "<" + "/" + "think" + ">"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def _hold_tail(self, marker: str) -> int:
        """Caracteres finales del buffer que podrían ser el inicio de `marker`."""
        for size in range(len(marker) - 1, 0, -1):
            if self._buffer.endswith(marker[:size]):
                return size
        return 0

    def feed(self, chunk: str) -> str:
        """Procesa un fragmento de stream y devuelve solo el texto visible."""
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
        """Vacía el buffer restante al cerrar el stream (nada si quedo dentro del bloque)."""
        rest, self._buffer = self._buffer, ""
        return "" if self._in_think else rest


class GroqClient:
    """Cliente Groq sobre el SDK de OpenAI (sin visión; Whisper para audio)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_BASE_URL
        self.audio_model = settings.GROQ_AUDIO_MODEL
        self.classifier_model = settings.GROQ_CLASSIFIER_MODEL

        if not self.api_key:
            raise ValueError("GROQ_API_KEY no está configurada en las variables de entorno (.env).")

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
        """Modelos candidatos (principales + respaldos) sin duplicados."""
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
        """Pide al modelo una respuesta en JSON estricto (usado por el clasificador)."""
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
            except Exception as e:  # noqa: BLE001 - se propaga para que actúe el fallback
                last_error = e
                logger.warning("Groq '%s' falló generando JSON (%s)", candidate, e)

        raise RuntimeError(f"Fallo al generar JSON con Groq ({self._candidate_models([model])}): {last_error}")

    def generate_response(self, prompt: str, system_instruction: str, temperature: float = 0.2, models=None) -> str:
        """Genera texto con fallback inmediato entre los modelos indicados."""
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
                logger.warning("Groq '%s' falló (%s). Probando siguiente modelo...", model, e)
            except Exception as e:  # noqa: BLE001 - red de seguridad para delegar en Gemini
                last_error = e
                logger.warning("Excepción inesperada en Groq '%s': %s. Probando siguiente...", model, e)

        raise RuntimeError(f"Fallo al invocar Groq con modelos {self._candidate_models(models)}: {last_error}")

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe audio con Whisper (Groq)."""
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
        """Streaming asíncrono con filtro `` y fallback inmediato entre modelos de Groq."""
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
                    # Todo el stream fue razonamiento interno vacío.
                    logger.warning("Modelo Groq '%s' solo devolvió bloques <think>.", model)
                    yield "No pude generar una respuesta concreta. Intenta reformular la pregunta."
                return
            except (RateLimitError, APIStatusError, APIConnectionError) as e:
                last_error = e
                if yielded_any:
                    logger.error("Error intermedio en streaming Groq '%s' tras emitir tokens: %s", model, e)
                    return
                if isinstance(e, RateLimitError):
                    await asyncio.sleep(GROQ_RATE_LIMIT_BACKOFF)
                logger.warning("Streaming Groq '%s' falló (%s). Probando siguiente...", model, e)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if yielded_any:
                    logger.error("Excepción intermedia en streaming Groq '%s' tras emitir tokens: %s", model, e)
                    return
                logger.warning("Excepción inesperada en streaming Groq '%s': %s. Probando siguiente...", model, e)

        raise RuntimeError(f"Fallo al transmitir con Groq {self._candidate_models(models)}: {last_error}")


class DeepSeekClient:
    """Cliente DeepSeek sobre el SDK de OpenAI (análisis complejo y respaldo de Groq)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model_name = settings.DEEPSEEK_MODEL
        self.max_output_tokens = settings.DEEPSEEK_MAX_OUTPUT_TOKENS

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY no está configurada en las variables de entorno (.env).")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _messages(prompt: str, system_instruction: str) -> list[dict]:
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

    def generate_response(self, prompt: str, system_instruction: str, temperature: float = 0.2) -> str:
        """Genera texto eliminando los bloques `` del razonamiento explícito."""
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
            raise RuntimeError(f"DeepSeek '{self.model_name}' falló: {e}") from e

    async def generate_response_stream_async(
        self, prompt: str, system_instruction: str, temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        """Streaming asíncrono con filtro `` en tiempo real."""
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
            logger.warning("DeepSeek '%s' solo devolvió bloques <think>.", self.model_name)
            yield "No pude generar una respuesta concreta. Intenta reformular la pregunta."


class LLMClient:
    """Enrutador híbrido: Groq ligero / DeepSeek según complejidad, Gemini para visión."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.gemini = GeminiClient(self.settings)

        self.groq: GroqClient | None = None
        if self.settings.GROQ_API_KEY:
            try:
                self.groq = GroqClient(self.settings)
            except Exception as e:  # noqa: BLE001
                logger.warning("Groq no disponible (%s). Se usará DeepSeek/Gemini.", e)
        else:
            logger.warning("GROQ_API_KEY vacía. Sin motor ligero de Groq.")

        self.deepseek: DeepSeekClient | None = None
        if self.settings.DEEPSEEK_API_KEY:
            try:
                self.deepseek = DeepSeekClient(self.settings)
            except Exception as e:  # noqa: BLE001
                logger.warning("DeepSeek no disponible (%s). Se usará Groq/Gemini.", e)
        else:
            logger.info("DEEPSEEK_API_KEY vacía. DeepSeek queda desactivado.")

        # Modelos del motor ligero (principal de Groq + respaldos configurados).
        self.light_models = GroqClient._dedupe([self.settings.GROQ_LIGHT_MODEL, *self.settings.GROQ_FALLBACK_MODELS])

    def _use_light_engine(self, route: QueryRoute | None, estimated_tokens: float, threshold: int) -> bool:
        """Regla de decisión: simple y pequeño → Groq; el resto → DeepSeek."""
        long_multi = (
            route is not None
            and len(route.groups) >= 2
            and (route.days_lookback == 0 or route.days_lookback > self.settings.ROUTER_MULTI_GROUP_DAYS)
        )
        if route is None:
            # Sin clasificar (resumen semanal, progresión, macros...): pequeño → Groq.
            return estimated_tokens < threshold
        return route.complexity == "simple" and estimated_tokens < threshold and not long_multi

    def _engine_order(self, prompt: str, system_instruction: str, route: QueryRoute | None) -> list[str]:
        """Cadena estricta de motores según la regla de decisión.

        - Consultas simples/livianas: Groq ligero -> DeepSeek (luego Gemini, fuera de aquí).
        - Consultas complejas: solo DeepSeek (nunca se reintenta en Groq ligero).
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
        raise RuntimeError(f"Motor '{engine}' no configurado")

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
        raise RuntimeError(f"Motor '{engine}' no configurado")

    def generate_json(self, prompt: str, system_instruction: str, max_tokens: int = 256) -> str:
        """JSON estricto para el clasificador (Groq); Gemini como respaldo."""
        if self.groq:
            try:
                return self.groq.generate_json(prompt, system_instruction, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                logger.warning("Clasificación con Groq no disponible (%s). Respaldo con Gemini.", e)
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
        """Texto → Groq ligero/DeepSeek; adjuntos (visión) → Gemini con respaldo."""
        if not attachments and (self.groq or self.deepseek):
            for engine in self._engine_order(prompt, system_instruction, route):
                try:
                    return self._sync_call(engine, prompt, system_instruction, temperature)
                except Exception as e:  # noqa: BLE001 - se degrada al siguiente motor
                    logger.warning("Motor '%s' falló (%s). Probando siguiente motor...", engine, e)
        return self.gemini.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            attachments=attachments,
            temperature=temperature,
        )

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Audio → Whisper (Groq); fallo → Gemini."""
        if self.groq:
            try:
                return self.groq.transcribe_audio(audio_bytes, mime_type)
            except Exception as e:  # noqa: BLE001
                logger.warning("Whisper (Groq) falló (%s). Respaldo con Gemini.", e)
        return self.gemini.transcribe_audio(audio_bytes, mime_type=mime_type)

    async def generate_response_stream_async(
        self,
        prompt: str,
        system_instruction: str,
        attachments: list[tuple[bytes, str]] | None = None,
        temperature: float = 0.2,
        route: QueryRoute | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming: texto → Groq ligero/DeepSeek; adjuntos o fallo total → Gemini."""
        if not attachments and (self.groq or self.deepseek):
            yielded_any = False
            for engine in self._engine_order(prompt, system_instruction, route):
                try:
                    async for chunk in self._stream_call(engine, prompt, system_instruction, temperature):
                        yielded_any = True
                        yield chunk
                    return
                except Exception as e:  # noqa: BLE001 - se degrada al siguiente motor
                    if yielded_any:
                        logger.error("Error intermedio en streaming del motor '%s' tras emitir tokens: %s", engine, e)
                        return
                    logger.warning("Streaming del motor '%s' falló (%s). Probando siguiente motor...", engine, e)

        async for chunk in self.gemini.generate_response_stream_async(
            prompt=prompt,
            system_instruction=system_instruction,
            attachments=attachments,
            temperature=temperature,
        ):
            yield chunk
