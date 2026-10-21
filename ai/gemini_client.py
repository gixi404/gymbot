"""Google GenAI client using the official SDK (google-genai) for gemini-2.5-flash."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Candidate models in priority order (verified with the current API key).
# NON-lite models (gemini-3.5-flash, gemini-flash-latest) return 503 UNAVAILABLE
# consistently and were excluded after latency tests (October 2026).
_FALLBACK_MODELS = (
    "gemini-flash-lite-latest",  # Always points to the latest stable lite version
)
RATE_LIMIT_BACKOFF = 2.0  # Seconds to wait on a 429 error (Free Tier rate limit)
MAX_GLOBAL_PASSES = 2  # Maximum number of full passes over the model list


def _is_rate_limit(error: APIError) -> bool:
    """True if the Gemini error is 429 (Free Tier request limit)."""
    return str(getattr(error, "code", "")) == "429"


class GeminiClient:
    """Client to interact with Gemini optimized for the Free Tier."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        self.model_name = self.settings.GEMINI_MODEL

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables (.env).")

        # Official google-genai client initialization
        self.client = genai.Client(api_key=self.api_key)

    def _build_candidate_models(self) -> list[str]:
        """Build the list of candidate models without duplicates."""
        seen: set[str] = set()
        candidates: list[str] = []
        for m in (self.model_name, *_FALLBACK_MODELS):
            if m not in seen:
                seen.add(m)
                candidates.append(m)
        return candidates

    def generate_response(
        self,
        prompt: str,
        system_instruction: str,
        attachments: list[tuple[bytes, str]] | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Send a request to Gemini with immediate fallback between models and global retries.

        Strategy:
        - 1 attempt per model → if it fails, move to the next one immediately.
        - If all models fail in a pass, wait 2 s and repeat (max. MAX_GLOBAL_PASSES).
        - No wait within each model; time is lost only between global passes.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        if attachments:
            contents_payload: list = []
            for file_bytes, mime_type in attachments:
                clean_mime = (mime_type or "application/octet-stream").split(";")[0].strip().lower()
                contents_payload.append(types.Part.from_bytes(data=file_bytes, mime_type=clean_mime))
            contents_payload.append(prompt)
            request_contents = contents_payload
        else:
            request_contents = prompt

        candidate_models = self._build_candidate_models()
        last_error = None

        for pass_num in range(1, MAX_GLOBAL_PASSES + 1):
            for current_model in candidate_models:
                try:
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=request_contents,
                        config=config,
                    )
                    if response and response.text and response.text.strip():
                        return response.text.strip()
                    logger.warning("Gemini model '%s' returned empty text. Trying the next model...", current_model)

                except APIError as e:
                    last_error = e
                    if _is_rate_limit(e):
                        logger.warning(
                            "Rate limit (429) on '%s'. Waiting %.1f s before continuing...",
                            current_model,
                            RATE_LIMIT_BACKOFF,
                        )
                        time.sleep(RATE_LIMIT_BACKOFF)
                    logger.warning(
                        "Model '%s' failed (pass %d/%d, code %s). Trying the next one...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        getattr(e, "code", "?"),
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Unexpected exception in '%s' (pass %d/%d): %s. Trying the next one...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        e,
                    )

            # All models failed in this pass
            if pass_num < MAX_GLOBAL_PASSES:
                logger.warning(
                    "All models failed in pass %d/%d. Retrying in 2 s...",
                    pass_num,
                    MAX_GLOBAL_PASSES,
                )
                time.sleep(2.0)

        raise RuntimeError(f"Failed to invoke Gemini with models {candidate_models}: {last_error}")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
    ) -> str:
        """Transcribe an audio file to text using Gemini with immediate fallback."""
        clean_mime = mime_type.split(";")[0].strip().lower()
        if clean_mime in ("audio/oga", "audio/opus"):
            clean_mime = "audio/ogg"

        part = types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime)
        prompt = (
            "Transcribe con máxima fidelidad el contenido de este audio a texto en español. "
            "Devuelve única y exclusivamente el texto transcrito de las palabras habladas. "
            "No incluyas saludos, ni comentarios, ni formato adicional ni comillas."
        )

        candidate_models = self._build_candidate_models()
        last_error = None

        for pass_num in range(1, MAX_GLOBAL_PASSES + 1):
            for current_model in candidate_models:
                try:
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=[part, prompt],
                    )
                    if response and response.text:
                        return response.text.strip()
                    return ""
                except APIError as e:
                    last_error = e
                    if _is_rate_limit(e):
                        logger.warning(
                            "Rate limit (429) while transcribing with '%s'. Waiting %.1f s...",
                            current_model,
                            RATE_LIMIT_BACKOFF,
                        )
                        time.sleep(RATE_LIMIT_BACKOFF)
                    logger.warning(
                        "Model '%s' failed while transcribing (pass %d/%d, code %s). Trying the next one...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        getattr(e, "code", "?"),
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Unexpected exception while transcribing with '%s' (pass %d/%d): %s. Trying the next one...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        e,
                    )

            if pass_num < MAX_GLOBAL_PASSES:
                logger.warning(
                    "All models failed while transcribing (pass %d/%d). Retrying in 2 s...",
                    pass_num,
                    MAX_GLOBAL_PASSES,
                )
                time.sleep(2.0)

        raise RuntimeError(f"Failed to transcribe audio with models {candidate_models}: {last_error}")

    async def generate_response_stream_async(
        self,
        prompt: str,
        system_instruction: str,
        attachments: list[tuple[bytes, str]] | None = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Generate a response in asynchronous streaming with immediate fallback between models.

        Strategy identical to generate_response: 1 attempt per model, no waits within
        the pass, MAX_GLOBAL_PASSES full passes before giving up.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        if attachments:
            contents_payload: list = []
            for file_bytes, mime_type in attachments:
                clean_mime = (mime_type or "application/octet-stream").split(";")[0].strip().lower()
                contents_payload.append(types.Part.from_bytes(data=file_bytes, mime_type=clean_mime))
            contents_payload.append(prompt)
            request_contents = contents_payload
        else:
            request_contents = prompt

        candidate_models = self._build_candidate_models()
        last_error = None
        yielded_any = False

        for pass_num in range(1, MAX_GLOBAL_PASSES + 1):
            for current_model in candidate_models:
                try:
                    response_stream = await self.client.aio.models.generate_content_stream(
                        model=current_model,
                        contents=request_contents,
                        config=config,
                    )
                    async for chunk in response_stream:
                        if chunk.text:
                            yielded_any = True
                            yield chunk.text
                    return  # Success: exit completely
                except APIError as e:
                    last_error = e
                    if yielded_any:
                        # We already emitted partial tokens, we cannot recover
                        logger.error("Intermediate API error after emitting tokens with '%s': %s", current_model, e)
                        return
                    if _is_rate_limit(e):
                        logger.warning(
                            "Rate limit (429) in streaming with '%s'. Waiting %.1f s...",
                            current_model,
                            RATE_LIMIT_BACKOFF,
                        )
                        await asyncio.sleep(RATE_LIMIT_BACKOFF)
                    logger.warning(
                        "Model '%s' failed in streaming (pass %d/%d, code %s). Trying the next one...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        getattr(e, "code", "?"),
                    )
                except Exception as e:
                    last_error = e
                    if yielded_any:
                        logger.error(
                            "Intermediate connection failure after emitting tokens with '%s': %s", current_model, e
                        )
                        return
                    logger.warning(
                        "Unexpected exception in streaming with '%s' (pass %d/%d): %s. Trying the next one...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        e,
                    )

            if pass_num < MAX_GLOBAL_PASSES and not yielded_any:
                logger.warning(
                    "All models failed in streaming (pass %d/%d). Retrying in 2 s...",
                    pass_num,
                    MAX_GLOBAL_PASSES,
                )
                await asyncio.sleep(2.0)

        # Synchronous fallback if streaming failed completely without emitting anything
        if not yielded_any:
            logger.warning("Streaming unavailable. Falling back to synchronous generation: %s", last_error)
            fallback_text = self.generate_response(prompt, system_instruction, attachments, temperature)
            yield fallback_text
