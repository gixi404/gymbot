"""Motor de análisis de entrenamientos y generación de contexto para GymBot."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.settings import Settings, get_settings
from storage.database import GymDatabase
from storage.models import RawMessage

from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)

FITNESS_COACH_SYSTEM_INSTRUCTION = """
Eres el Entrenador de Fuerza y Especialista en Fisiología del Ejercicio de GymBot.
Tu función es responder las consultas del atleta y analizar sus registros de entrenamiento cuando lo solicite.

DIRECTIVAS GENERALES:
1. Pertinencia Estricta y Relevancia:
   - Responde ÚNICAMENTE a lo que el atleta pregunta, sin divagar ni agregar resúmenes, tablas o análisis que no hayan sido pedidos.
   - Si la consulta es conceptual, general, un saludo o sobre tus funciones (ej. "¿para qué sirves?", "hola", "¿qué es la doble progresión?"), responde de forma directa y puntual SIN generar tablas ni desglosar las sesiones del historial.
   - Solo desglosa, cita o tabula datos de sesiones cuando la consulta pregunte explícitamente sobre su historial, ejercicios, pesos, series o progresión.

2. Criterios de Entrenamiento y Métricas (cuando se solicite análisis):
   - Nomenclatura: Comprende '80kg 4x8', '100x5 @9', 'PMR 90 3x10', biseries, RPE/RIR, notas cualitativas de fatiga, descansos y molestias.
   - 1RM Estimado: Fórmulas estándar (Epley: peso * (1 + reps/30) o Brzycki).
   - Sobrecarga y Progresión: Compara cargas y repeticiones a igual peso o menor RPE.
   - Fidelidad de datos: Basa tus afirmaciones sobre sus marcas estrictamente en los registros provistos, sin inventar sesiones.

3. Prioridad de Directivas del Atleta:
   - Las directivas y preferencias del atleta (estilo, formato, objetivos, rango de repeticiones, tono, sin saludos ni despedidas) tienen PRIORIDAD ABSOLUTA sobre cualquier regla general.
"""


class GymAnalyzer:
    """Orquestador del análisis de entrenamientos utilizando Gemini y SQLite."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        database: Optional[GymDatabase] = None,
        gemini_client: Optional[GeminiClient] = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        self.gemini_client = gemini_client or GeminiClient(self.settings)

    def build_workout_context(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Extrae y formatea el historial de entrenamientos en orden cronológico."""
        messages: List[RawMessage] = self.database.get_messages(
            since=since,
            until=until,
            limit=limit,
            reverse=False,  # Cronológico (antiguo a reciente)
        )

        if not messages:
            return "NO HAY REGISTROS DISPONIBLES EN LA BASE DE DATOS LOCAL."

        context_lines = []
        for msg in messages:
            context_lines.append(msg.to_summary_line())

        return "\n".join(context_lines)

    def get_system_instruction(self) -> str:
        """Construye la instrucción del sistema combinando la base deportiva con directivas personalizadas."""
        base_instruction = FITNESS_COACH_SYSTEM_INSTRUCTION.strip()
        custom_file = Path(__file__).resolve().parent.parent / "custom_instructions.txt"
        if custom_file.exists():
            try:
                custom_text = custom_file.read_text(encoding="utf-8").strip()
                if custom_text:
                    return (
                        f"{base_instruction}\n\n"
                        "DIRECTIVAS Y PREFERENCIAS ESPECÍFICAS DEL ATLETA (PRIORIDAD ALTA, SIEMPRE CUMPLIR):\n"
                        f"{custom_text}\n"
                    )
            except Exception as e:
                logger.warning("No se pudo leer custom_instructions.txt: %s", e)
        return base_instruction

    def analyze_query(self, user_question: str) -> str:
        """Combina el historial local de entrenamientos con la pregunta y consulta a Gemini."""
        question_clean = user_question.strip()
        if not question_clean:
            return "Por favor ingresa una consulta válida sobre tus entrenamientos."

        # Verificamos si hay registros en la base de datos
        stats = self.database.get_stats()
        if stats.total_messages == 0:
            return (
                "⚠️ La base de datos local está vacía. "
                "Ejecuta primero una sincronización (`:sync` en la CLI o `/sync` en el bot de Telegram) "
                "para descargar tus entrenamientos desde el chat."
            )

        context_text = self.build_workout_context()

        stats_summary = (
            f"Total de registros analizados: {stats.total_messages}. "
            f"Período: {stats.first_message_date.strftime('%d/%m/%Y') if stats.first_message_date else 'N/A'} "
            f"al {stats.last_message_date.strftime('%d/%m/%Y') if stats.last_message_date else 'N/A'}."
        )

        prompt = f"""
METADATOS DEL ATLETA:
{stats_summary}

HISTORIAL CRONOLÓGICO DE SESIONES DE ENTRENAMIENTO:
=====================================================
{context_text}
=====================================================

CONSULTA DEL ATLETA:
{question_clean}

INSTRUCCIÓN: Responde directamente y con precisión quirúrgica a la consulta del atleta, cumpliendo rigurosamente todas sus directivas. No agregues tablas ni desgloses del historial a menos que la consulta lo pida explícitamente.
"""

        try:
            return self.gemini_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
            )
        except Exception as e:
            logger.error("Error al procesar consulta con Gemini: %s", e)
            return f"❌ Error al consultar el motor de IA (Gemini): {e}"

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe un archivo o nota de voz de audio a texto utilizando Gemini."""
        return self.gemini_client.transcribe_audio(audio_bytes, mime_type=mime_type)
