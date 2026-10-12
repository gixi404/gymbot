"""Cliente de Google GenAI usando el SDK oficial (google-genai) para gemini-2.5-flash."""

import logging
import time
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Cliente para interactuar con Gemini 2.5 Flash optimizado para el Free Tier."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        self.model_name = self.settings.GEMINI_MODEL

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY no está configurada en las variables de entorno (.env)."
            )

        # Inicialización del cliente oficial google-genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(
        self,
        prompt: str,
        system_instruction: str,
        attachments: Optional[List[Tuple[bytes, str]]] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
        base_backoff: float = 1.5,
    ) -> str:
        """Envía una solicitud a Gemini con reintentos, adjuntos multimodales y fallback automático."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        contents_payload = []
        if attachments:
            for file_bytes, mime_type in attachments:
                clean_mime = (mime_type or "application/octet-stream").split(";")[0].strip().lower()
                part = types.Part.from_bytes(data=file_bytes, mime_type=clean_mime)
                contents_payload.append(part)
            contents_payload.append(prompt)
            request_contents = contents_payload
        else:
            request_contents = prompt

        # Modelos candidatos en orden de prioridad para fallback si hay congestión global (503)
        candidate_models = [self.model_name]
        for fallback in ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for current_model in candidate_models:
            for attempt in range(1, max_retries + 1):
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
                    status_code = getattr(e, "code", None)
                    err_str = str(e)

                    # Si el modelo no existe o está deprecado (404), pasar al siguiente modelo
                    if status_code == 404 or "NOT_FOUND" in err_str:
                        logger.warning("Modelo '%s' no disponible (404). Intentando alternativa...", current_model)
                        break

                    # Código 429 = Rate Limit, Código 503 = Unavailable / High Demand
                    if status_code in (429, 503) or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str:
                        # Si es el último intento de este modelo y tenemos otros candidatos, pasar al siguiente
                        if attempt == max_retries:
                            logger.warning(
                                "Modelo '%s' saturado tras %d intentos. Intentando siguiente modelo candidato...",
                                current_model,
                                max_retries,
                            )
                            break
                        sleep_time = base_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Alta demanda/cuota en '%s' (código %s, intento %d/%d). Reintentando en %.1f s...",
                            current_model,
                            status_code,
                            attempt,
                            max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error("Error de API en Gemini (%s): %s", current_model, e)
                        raise
                except Exception as e:
                    last_error = e
                    logger.error("Excepción inesperada al invocar Gemini (%s): %s", current_model, e)
                    raise

        raise RuntimeError(
            f"Fallo al invocar Gemini con modelos {candidate_models}: {last_error}"
        )

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        max_retries: int = 3,
        base_backoff: float = 1.5,
    ) -> str:
        """Transcribe un archivo de audio a texto usando Gemini con fallback y reintentos."""
        clean_mime = mime_type.split(";")[0].strip().lower()
        if clean_mime in ("audio/oga", "audio/opus"):
            clean_mime = "audio/ogg"

        part = types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime)
        prompt = (
            "Transcribe con máxima fidelidad el contenido de este audio a texto en español. "
            "Devuelve única y exclusivamente el texto transcrito de las palabras habladas. "
            "No incluyas saludos, ni comentarios, ni formato adicional ni comillas."
        )

        candidate_models = [self.model_name]
        for fallback in ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for current_model in candidate_models:
            for attempt in range(1, max_retries + 1):
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
                    status_code = getattr(e, "code", None)
                    err_str = str(e)

                    if status_code == 404 or "NOT_FOUND" in err_str:
                        break

                    if status_code in (429, 503) or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str:
                        if attempt == max_retries:
                            break
                        sleep_time = base_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Alta demanda al transcribir en '%s' (intento %d/%d). Reintentando en %.1f s...",
                            current_model,
                            attempt,
                            max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error("Error de API en Gemini al transcribir (%s): %s", current_model, e)
                        raise
                except Exception as e:
                    last_error = e
                    logger.error("Excepción inesperada al transcribir audio (%s): %s", current_model, e)
                    raise

        raise RuntimeError(f"Fallo al transcribir audio con modelos {candidate_models}: {last_error}")
