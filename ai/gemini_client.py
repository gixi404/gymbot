"""Cliente de Google GenAI usando el SDK oficial (google-genai) para gemini-2.5-flash."""

import logging
import time
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Modelos candidatos en orden de prioridad (todos verificados, octubre 2026).
# Si el modelo principal (GEMINI_MODEL) falla, se prueba el siguiente inmediatamente.
# Cuando todos fallen, se repite la rotación completa hasta MAX_GLOBAL_PASSES veces.
_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)
MAX_GLOBAL_PASSES = 2  # Número máximo de pasadas completas sobre la lista de modelos


class GeminiClient:
    """Cliente para interactuar con Gemini optimizado para el Free Tier."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        self.model_name = self.settings.GEMINI_MODEL

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno (.env).")

        # Inicialización del cliente oficial google-genai
        self.client = genai.Client(api_key=self.api_key)

    def _build_candidate_models(self) -> list[str]:
        """Construye la lista de modelos candidatos sin duplicados."""
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
        """Envía una solicitud a Gemini con fallback inmediato entre modelos y reintentos globales.

        Estrategia:
        - 1 intento por modelo → si falla, pasa al siguiente inmediatamente.
        - Si todos los modelos fallan en una pasada, espera 2 s y repite (máx. MAX_GLOBAL_PASSES).
        - Sin espera dentro de cada modelo; el tiempo se pierde sólo entre pasadas globales.
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
                    if response and response.text:
                        return response.text.strip()
                    return "No se obtuvo respuesta de texto del modelo."

                except APIError as e:
                    last_error = e
                    logger.warning(
                        "Modelo '%s' falló (pasada %d/%d, código %s). Probando siguiente...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        getattr(e, "code", "?"),
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Excepción inesperada en '%s' (pasada %d/%d): %s. Probando siguiente...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        e,
                    )

            # Todos los modelos fallaron en esta pasada
            if pass_num < MAX_GLOBAL_PASSES:
                logger.warning(
                    "Todos los modelos fallaron en la pasada %d/%d. Reintentando en 2 s...",
                    pass_num,
                    MAX_GLOBAL_PASSES,
                )
                time.sleep(2.0)

        raise RuntimeError(f"Fallo al invocar Gemini con modelos {candidate_models}: {last_error}")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
    ) -> str:
        """Transcribe un archivo de audio a texto usando Gemini con fallback inmediato."""
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
                    logger.warning(
                        "Modelo '%s' falló al transcribir (pasada %d/%d, código %s). Probando siguiente...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        getattr(e, "code", "?"),
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Excepción inesperada al transcribir con '%s' (pasada %d/%d): %s. Probando siguiente...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        e,
                    )

            if pass_num < MAX_GLOBAL_PASSES:
                logger.warning(
                    "Todos los modelos fallaron al transcribir (pasada %d/%d). Reintentando en 2 s...",
                    pass_num,
                    MAX_GLOBAL_PASSES,
                )
                time.sleep(2.0)

        raise RuntimeError(f"Fallo al transcribir audio con modelos {candidate_models}: {last_error}")

    async def generate_response_stream_async(
        self,
        prompt: str,
        system_instruction: str,
        attachments: list[tuple[bytes, str]] | None = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Genera respuesta en streaming asíncrono con fallback inmediato entre modelos.

        Estrategia idéntica a generate_response: 1 intento por modelo, sin esperas dentro
        de la pasada, MAX_GLOBAL_PASSES pasadas completas antes de rendirse.
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
                    return  # Éxito: salir completamente
                except APIError as e:
                    last_error = e
                    if yielded_any:
                        # Ya emitimos tokens parciales, no podemos recuperarnos
                        logger.error("Error de API intermedio tras emitir tokens con '%s': %s", current_model, e)
                        return
                    logger.warning(
                        "Modelo '%s' falló en streaming (pasada %d/%d, código %s). Probando siguiente...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        getattr(e, "code", "?"),
                    )
                except Exception as e:
                    last_error = e
                    if yielded_any:
                        logger.error("Fallo de conexión intermedio tras emitir tokens con '%s': %s", current_model, e)
                        return
                    logger.warning(
                        "Excepción inesperada en streaming con '%s' (pasada %d/%d): %s. Probando siguiente...",
                        current_model,
                        pass_num,
                        MAX_GLOBAL_PASSES,
                        e,
                    )

            if pass_num < MAX_GLOBAL_PASSES and not yielded_any:
                logger.warning(
                    "Todos los modelos fallaron en streaming (pasada %d/%d). Reintentando en 2 s...",
                    pass_num,
                    MAX_GLOBAL_PASSES,
                )
                import asyncio

                await asyncio.sleep(2.0)

        # Fallback síncrono si el streaming falló por completo sin emitir nada
        if not yielded_any:
            logger.warning("Streaming no disponible. Recurriendo a generación síncrona: %s", last_error)
            fallback_text = self.generate_response(prompt, system_instruction, attachments, temperature)
            yield fallback_text
