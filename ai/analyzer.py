"""Motor de análisis de entrenamientos y generación de contexto para GymBot."""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

from config.settings import Settings, get_settings
from storage.database import GymDatabase
from storage.models import RawMessage

from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)

FITNESS_COACH_SYSTEM_INSTRUCTION = """
Eres el Entrenador de Fuerza y Especialista en Rendimiento de GymBot.

DIRECTIVAS ESENCIALES:
1. Estilo y Longitud: Responde de forma concisa, técnica y fluida (1 a 3 oraciones completas). Aporta el dato clave junto a su contexto relevante (fecha, ejercicio, sesión o peso). No des palabras o números sueltos, ni párrafos extensos salvo solicitud explícita.
2. Tono: Español neutro, directo, sin saludos, sin despedidas y sin emojis.
3. Fidelidad a los Registros: Basa tus afirmaciones estrictamente en los registros provistos ([GYM], [Sesiones], [Cardio], [Comidas], [PesoCorporal]). No inventes ni asumas datos no registrados.
4. Desglose de Datos: Solo tabula o lista el historial si el atleta lo pide explícitamente. Ante preguntas conceptuales, responde con brevedad técnica sin desglosar registros.
"""


class GymAnalyzer:
    """Orquestador del análisis de entrenamientos utilizando Gemini y SQLite."""

    def __init__(
        self,
        settings: Settings | None = None,
        database: GymDatabase | None = None,
        gemini_client: GeminiClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        self.gemini_client = gemini_client or GeminiClient(self.settings)
        self._cached_context: str | None = None
        self._cached_total_messages: int = -1
        self._cached_last_sync: datetime | None = None

    def build_workout_context(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> str:
        """Extrae y formatea el historial de entrenamientos en orden cronológico (con caché en RAM)."""
        # Si es la consulta general completa, usamos caché en memoria RAM para máxima velocidad
        if since is None and until is None and limit is None:
            stats = self.database.get_stats()
            if (
                self._cached_context is not None
                and self._cached_total_messages == stats.total_messages
                and self._cached_last_sync == stats.last_sync_at
            ):
                return self._cached_context

            messages: list[RawMessage] = self.database.get_messages(reverse=False)
            if not messages:
                self._cached_context = "NO HAY REGISTROS DISPONIBLES EN LA BASE DE DATOS LOCAL."
            else:
                self._cached_context = "\n".join(msg.to_summary_line() for msg in messages)
            self._cached_total_messages = stats.total_messages
            self._cached_last_sync = stats.last_sync_at
            return self._cached_context

        messages: list[RawMessage] = self.database.get_messages(
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

    def analyze_query(
        self,
        user_question: str,
        conversation_history: list[dict] | None = None,
        attachments: list[tuple[bytes, str]] | None = None,
    ) -> str:
        """Combina el historial local de entrenamientos, la memoria de conversación y adjuntos con la pregunta."""
        question_clean = user_question.strip()
        if not question_clean:
            return "Por favor ingresa una consulta válida sobre tus entrenamientos."

        # Verificamos si hay registros en la base de datos
        stats = self.database.get_stats()
        if stats.total_messages == 0 and not attachments:
            return (
                "⚠️ La base de datos local está vacía. "
                "Ejecuta primero una sincronización (`:sync` en la CLI o `/sync` en el bot de Telegram) "
                "para descargar tus entrenamientos desde el chat."
            )

        if stats.total_messages > 0:
            context_text = self.build_workout_context()
            stats_summary = (
                f"Total de registros analizados: {stats.total_messages}. "
                f"Período: {stats.first_message_date.strftime('%d/%m/%Y') if stats.first_message_date else 'N/A'} "
                f"al {stats.last_message_date.strftime('%d/%m/%Y') if stats.last_message_date else 'N/A'}."
            )
        else:
            context_text = "Sin registros históricos previos en base de datos."
            stats_summary = "Base de datos local sin registros. Análisis centrado en el material adjunto."

        conversation_context = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role_label = "Atleta" if msg.get("role") == "user" else "Entrenador"
                history_lines.append(f"{role_label}: {msg.get('text', '')}")
            conversation_context = (
                "\nMEMORIA DE LA CONVERSACIÓN PREVIA (EN ESTA SESIÓN ACTIVA):\n"
                "-----------------------------------------------------\n"
                + "\n".join(history_lines)
                + "\n-----------------------------------------------------\n"
            )

        prompt = f"""
METADATOS DEL ATLETA:
{stats_summary}

HISTORIAL CRONOLÓGICO DE SESIONES DE ENTRENAMIENTO:
=====================================================
{context_text}
=====================================================
{conversation_context}
CONSULTA DEL ATLETA:
{question_clean}

INSTRUCCIÓN: Responde a la consulta del atleta aplicando tus reglas de estilo y perfil. Si hay adjuntos o memoria conversacional previa, intégralos con coherencia directa.
"""

        try:
            return self.gemini_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                attachments=attachments,
            )
        except Exception as e:
            logger.error("Error al procesar consulta con Gemini: %s", e)
            return f"❌ Error al consultar el motor de IA (Gemini): {e}"

    async def analyze_query_stream_async(
        self,
        user_question: str,
        conversation_history: list[dict] | None = None,
        attachments: list[tuple[bytes, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Versión en streaming asíncrono de alta velocidad para respuesta inmediata en Telegram."""
        question_clean = user_question.strip()
        if not question_clean:
            yield "Por favor ingresa una consulta válida sobre tus entrenamientos."
            return

        stats = self.database.get_stats()
        if stats.total_messages == 0 and not attachments:
            yield (
                "⚠️ La base de datos local está vacía. "
                "Ejecuta primero una sincronización (`:sync` en la CLI o `/sync` en el bot de Telegram) "
                "para descargar tus entrenamientos desde el chat."
            )
            return

        if stats.total_messages > 0:
            context_text = self.build_workout_context()
            stats_summary = (
                f"Total de registros analizados: {stats.total_messages}. "
                f"Período: {stats.first_message_date.strftime('%d/%m/%Y') if stats.first_message_date else 'N/A'} "
                f"al {stats.last_message_date.strftime('%d/%m/%Y') if stats.last_message_date else 'N/A'}."
            )
        else:
            context_text = "Sin registros históricos previos en base de datos."
            stats_summary = "Base de datos local sin registros. Análisis centrado en el material adjunto."

        conversation_context = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role_label = "Atleta" if msg.get("role") == "user" else "Entrenador"
                history_lines.append(f"{role_label}: {msg.get('text', '')}")
            conversation_context = (
                "\nMEMORIA DE LA CONVERSACIÓN PREVIA (EN ESTA SESIÓN ACTIVA):\n"
                "-----------------------------------------------------\n"
                + "\n".join(history_lines)
                + "\n-----------------------------------------------------\n"
            )

        prompt = f"""
METADATOS DEL ATLETA:
{stats_summary}

HISTORIAL CRONOLÓGICO DE SESIONES DE ENTRENAMIENTO:
=====================================================
{context_text}
=====================================================
{conversation_context}
CONSULTA DEL ATLETA:
{question_clean}

INSTRUCCIÓN: Responde a la consulta del atleta aplicando tus reglas de estilo y perfil. Si hay adjuntos o memoria conversacional previa, intégralos con coherencia directa.
"""

        async for chunk in self.gemini_client.generate_response_stream_async(
            prompt=prompt,
            system_instruction=self.get_system_instruction(),
            attachments=attachments,
        ):
            yield chunk

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe un archivo o nota de voz de audio a texto utilizando Gemini."""
        return self.gemini_client.transcribe_audio(audio_bytes, mime_type=mime_type)

    def analyze_nutrition_today(self) -> str:
        """Calcula el balance calórico y macronutrientes estimados de las comidas registradas hoy."""
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0)

        # Obtener mensajes de hoy
        today_messages = self.database.get_messages(since=start_of_day, reverse=False)
        food_messages = [
            m
            for m in today_messages
            if (m.chat_title and "comida" in m.chat_title.lower())
            or any(
                kw in m.text.lower()
                for kw in ("desayuno", "almuerzo", "cena", "merienda", "snack", "comida", "gr", "kcal", "proteina")
            )
        ]

        if not food_messages:
            # Fallback a los últimos 3 mensajes de comidas si hoy no hay nada aún
            recent_all = self.database.get_messages(limit=20, reverse=True)
            food_messages = [m for m in recent_all if m.chat_title and "comida" in m.chat_title.lower()][:3]
            if not food_messages:
                return "No hay registros recientes de comidas en el grupo Comidas para calcular macros."

        food_text = "\n".join(m.to_summary_line() for m in food_messages)

        prompt = f"""
REGISTROS DE COMIDAS A ANALIZAR:
================================
{food_text}
================================

INSTRUCCIÓN:
Como nutricionista deportivo, analiza las comidas registradas y devuelve un desglose claro:
1. Estimación calórica total aproximada (kcal).
2. Macronutrientes aproximados: Proteínas (g), Carbohidratos (g), Grasas (g).
3. Breve evaluación (1-2 oraciones) indicando si se alinea con la meta del atleta (hipertrofia, superávit leve o mantenimiento, sin déficit calórico).
Mantén el formato conciso, limpio y profesional.
"""
        try:
            return self.gemini_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
            )
        except Exception as e:
            logger.error("Error al analizar nutrición con Gemini: %s", e)
            return f"❌ Error al estimar nutrición: {e}"

    def recommend_next_progression(self, exercise_name: str) -> str:
        """Recomienda la carga y repeticiones para la próxima sesión aplicando doble progresión."""
        ex_query = exercise_name.strip()
        if not ex_query:
            return "Indica el ejercicio que deseas consultar (por ejemplo: /proxima banca o /proxima sentadilla)."

        # Buscar mensajes que contengan el ejercicio
        matched = self.database.search_messages(ex_query, limit=10)
        if not matched:
            return f"No se encontraron registros previos para '{ex_query}' en tu historial."

        history_lines = [m.to_summary_line() for m in matched[:5]]
        history_text = "\n".join(reversed(history_lines))

        prompt = f"""
EJERCICIO A CONSULTAR: {ex_query}

ÚLTIMAS SESIONES REGISTRADAS PARA ESTE EJERCICIO:
=================================================
{history_text}
=================================================

INSTRUCCIÓN:
Aplica la regla de DOBLE PROGRESIÓN del atleta (rango de hipertrofia/fuerza, técnica estricta sin trampas):
1. Identifica las cargas y repeticiones completadas en la última sesión para este ejercicio.
2. Si el atleta completó el tope de repeticiones objetivo en todas sus series efectivas (ej. 3x8 o 3x10), indica el incremento de carga sugerido (ej. +1 a 2.5 kg) y las repeticiones base para la próxima sesión (ej. 3x6).
3. Si no completó el tope de repeticiones, recomienda mantener el mismo peso y buscar añadir 1-2 repeticiones totales.
Responde de forma concisa y directa en 1 a 3 oraciones completas y motivantes.
"""
        try:
            return self.gemini_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
            )
        except Exception as e:
            logger.error("Error al sugerir progresión con Gemini: %s", e)
            return f"❌ Error al consultar la sugerencia de progresión: {e}"

    def generate_weekly_summary(self) -> str:
        """Genera un reporte ejecutivo semanal de los últimos 7 días."""
        now = datetime.now()
        # Obtener los últimos 7 días de registros
        since_date = datetime.fromtimestamp(now.timestamp() - 7 * 86400)
        messages = self.database.get_messages(since=since_date, reverse=False)

        if not messages:
            return "No se registraron entrenamientos ni pesajes en los últimos 7 días para generar el resumen semanal."

        context_lines = [m.to_summary_line() for m in messages]
        context_text = "\n".join(context_lines)

        prompt = f"""
HISTORIAL DE LOS ÚLTIMOS 7 DÍAS:
================================
{context_text}
================================

INSTRUCCIÓN:
Elabora el RESUMEN SEMANAL EJECUTIVO para el atleta:
1. Días y sesiones completadas vs el plan objetivo (Fullbody 3 días L-M-V, Cardio 2 días M-S en Zona 2).
2. Mejores levantamientos o récords de sobrecarga progresiva alcanzados esta semana.
3. Tendencia o promedio del peso corporal si hubo pesajes registrados.
4. Cumplimiento nutricional si hubo notas en comidas.
Estructura la respuesta de manera clara, concisa y profesional (3 a 5 párrafos cortos con viñetas limpias), sin saludos, sin despedidas y sin emojis en exceso.
"""
        try:
            return self.gemini_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
            )
        except Exception as e:
            logger.error("Error al generar resumen semanal con Gemini: %s", e)
            return f"❌ Error al generar resumen semanal: {e}"
