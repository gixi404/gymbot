"""Motor de análisis de entrenamientos y generación de contexto para GymBot."""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from config.settings import Settings, get_settings
from storage.database import GymDatabase
from storage.models import RawMessage, to_atleta_tz

from .llm_client import LLMClient
from .router import GROUPS, QueryRoute, classify_query, resolve_group_targets

logger = logging.getLogger(__name__)

# Los registros y tareas programadas se interpretan en horario del atleta.
ATLETA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# Fecha explícita escrita dentro del texto (dd/mm/yy, dd/mm/yyyy o dd-mm...).
_TEXT_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
# Par (fecha, peso kg) dentro de mensajes de PesoCorporal.
_PESO_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b[^\d]*?([0-9]+(?:[.,][0-9]+)?)\s*kg", re.IGNORECASE)

FITNESS_COACH_SYSTEM_INSTRUCTION = """
Eres el Entrenador de Fuerza y Nutricionista Deportivo de GymBot.

DIRECTIVAS ESENCIALES:
1. Profundidad proporcional a la consulta:
   - Ante preguntas simples o puntuales (ej. un dato de una sesión, qué entrené tal día, un pesaje puntual), responde de forma directa y concisa en 1 a 3 oraciones sin rodeos.
   - Ante preguntas de análisis, progreso, estancamientos, comparativas de meses, estimación de músculo vs. grasa, macros detallados o revisión de los 4 grupos, explayate lo necesario con un análisis técnico estructurado (usando viñetas limpias, métricas concretas, deltas de cargas/repeticiones y conclusiones claras). No recortes información valiosa en consultas complejas ni escribas párrafos de relleno en consultas simples.
2. Tono: Español directo, técnico y natural. Sin saludos, sin despedidas, sin emojis y sin frases hechas de cortesía.
3. Fidelidad a los Registros y Fechas: Basa tus afirmaciones estrictamente en los registros provistos ([Sesiones], [Cardio], [Comidas], [PesoCorporal]). Respeta rigurosamente la jerarquía de fechas: si el texto del mensaje contiene una fecha escrita adentro, esa es la fecha real del evento; si solo dice el día de la semana o no tiene fecha en el texto, usa la fecha de `[enviado: dd/mm/yyyy]`.
4. Estimación Nutricional Activa: En consultas sobre [Comidas], calcula y estima siempre las calorías y macronutrientes (proteínas, carbohidratos, grasas) usando valores nutricionales estándar (porciones argentinas/rioplatenses comunes) para los alimentos anotados. Evalúa siempre la ingesta contra las metas del atleta: 2800-3000 kcal, proteína mínima indispensable (112 g) y proteína óptima (126-155 g).
"""


class GymAnalyzer:
    """Orquestador del análisis de entrenamientos (contexto local + enrutador LLM + SQLite)."""

    def __init__(
        self,
        settings: Settings | None = None,
        database: GymDatabase | None = None,
        llm_client: object | None = None,
        gemini_client: object | None = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        # La interfaz de LLMClient coincide con la de GeminiClient (duck typing).
        self.llm_client = llm_client or gemini_client or LLMClient(self.settings)
        self._cached_context: str | None = None
        self._cached_total_messages: int = -1
        self._cached_last_sync: datetime | None = None

    @staticmethod
    def _now() -> datetime:
        """Hora actual en el huso horario del atleta."""
        return datetime.now(ATLETA_TZ)

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

    def _fetch_group_messages(self, group: str, since: datetime | None) -> list[RawMessage]:
        """Trae los mensajes de un grupo resolviendo sus títulos/IDs de forma aislada."""
        titles, chat_ids = resolve_group_targets([group], self.settings)
        # PesoCorporal suele agrupar varios pesajes por mensaje: se trae completo (son pocos).
        group_since = None if group == "PesoCorporal" else since
        return self.database.get_messages(
            since=group_since, titles=titles, chat_ids=chat_ids, reverse=False
        )

    def build_focused_context(
        self,
        question: str,
        groups: list[str] | None = None,
        days_lookback: int = 0,
    ) -> str:
        """Extrae solo los registros relevantes (grupos + ventana temporal) para la consulta.

        Cada grupo se resuelve por separado (para no mezclar conteos ni ventanas). Con
        `days_lookback > 0` la ventana es estricta. Con `days_lookback == 0` y sin
        coincidencias se recurre al historial completo (mensajes legados sin mapeo).
        """
        since = datetime.now(UTC) - timedelta(days=days_lookback) if days_lookback > 0 else None

        if not groups:
            messages = self.database.get_messages(since=since, reverse=False)
        else:
            seen: set[tuple[int, str | None]] = set()
            messages = []
            for group in groups:
                for msg in self._fetch_group_messages(group, since):
                    key = (msg.id, msg.chat_id)
                    if key not in seen:
                        seen.add(key)
                        messages.append(msg)
            messages.sort(key=lambda m: m.date)
            if not messages and days_lookback == 0:
                # Mensajes legados sin chat_title: mejor el historial completo que nada.
                messages = self.database.get_messages(since=since, reverse=False)

        if not messages:
            window = f" en los últimos {days_lookback} días" if days_lookback > 0 else ""
            return f"NO HAY REGISTROS DISPONIBLES{window.upper()} PARA LA CONSULTA."

        return "\n".join(msg.to_summary_line() for msg in messages)

    @staticmethod
    def _extract_peso_series(messages: list[RawMessage]) -> list[tuple[date, float]]:
        """Extrae pares (fecha_real_del_texto, peso_kg) de los mensajes de PesoCorporal."""
        series: list[tuple[date, float]] = []
        for msg in messages:
            for match in _PESO_RE.finditer(msg.text):
                day, month, year, weight = match.groups()
                y = int(year)
                y = 2000 + y if y < 100 else y
                try:
                    series.append((date(y, int(month), int(day)), float(weight.replace(",", "."))))
                except ValueError:
                    continue
        series.sort(key=lambda pair: pair[0])
        return series

    def _build_context_stats(self, groups: list[str] | None = None) -> str:
        """Resumen por grupo (conteo real) resolviendo cada grupo por separado.

        Para PesoCorporal, extrae todos los pares fecha↔peso escritos en el texto y
        reporta primer/último peso real con su delta en kg.
        """
        lines: list[str] = []
        for group in groups or list(GROUPS):
            messages = self._fetch_group_messages(group, since=None)
            if group == "PesoCorporal":
                series = self._extract_peso_series(messages)
                if series:
                    first_day, first_w = series[0]
                    last_day, last_w = series[-1]
                    lines.append(
                        f"{group}: {len(messages)} registros | primer peso {first_w:.2f} kg "
                        f"({first_day.strftime('%d/%m/%y')}) -> último {last_w:.2f} kg "
                        f"({last_day.strftime('%d/%m/%y')}) | delta {last_w - first_w:+.2f} kg"
                    )
                    continue
            lines.append(f"{group}: {len(messages)} registros")
        return "\n".join(lines)

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

    def _build_prompt(
        self,
        question: str,
        route: QueryRoute,
        conversation_history: list[dict] | None,
    ) -> str:
        """Ensambla el prompt: metadatos, contexto filtrado por ruta, memoria y consulta."""
        stats = self.database.get_stats()
        context_text = self.build_focused_context(question, route.groups, route.days_lookback)
        group_stats = self._build_context_stats(route.groups)
        stats_summary = (
            f"FECHA_ACTUAL: {self._now().strftime('%d/%m/%Y')} (America/Argentina/Buenos_Aires). "
            f"Total de registros en la base: {stats.total_messages}. "
            f"Período: {stats.first_message_date.strftime('%d/%m/%Y') if stats.first_message_date else 'N/A'} "
            f"al {stats.last_message_date.strftime('%d/%m/%Y') if stats.last_message_date else 'N/A'} "
            f"(grupos consultados: {', '.join(route.groups)}; ventana: "
            f"{'completa' if route.days_lookback == 0 else f'últimos {route.days_lookback} días'}).\n"
            f"REGISTROS DISPONIBLES POR GRUPO:\n{group_stats}"
        )

        conversation_context = ""
        if conversation_history:
            # ponytail: cap de memoria (~2400 chars / ~600 tokens) para no desbordar
            # el umbral Groq de 4200 tokens tras un análisis largo de DeepSeek.
            budget = 2400
            lines: list[str] = []
            for msg in reversed(conversation_history):
                role = "Atleta" if msg.get("role") == "user" else "Entrenador"
                line = f"{role}: {str(msg.get('text', ''))[:800]}"
                if lines and sum(len(x) + 1 for x in lines) + len(line) > budget:
                    break
                lines.append(line)
            lines.reverse()
            conversation_context = (
                "\nMEMORIA DE LA CONVERSACIÓN PREVIA (EN ESTA SESIÓN ACTIVA):\n"
                "-----------------------------------------------------\n"
                + "\n".join(lines)
                + "\n-----------------------------------------------------\n"
            )

        return f"""
METADATOS DEL ATLETA:
{stats_summary}

HISTORIAL CRONOLÓGICO RELEVANTE:
=====================================================
{context_text}
=====================================================
{conversation_context}
CONSULTA DEL ATLETA:
{question}

INSTRUCCIÓN: Responde a la consulta del atleta aplicando tus reglas de estilo y perfil. Si hay adjuntos o memoria conversacional previa, intégralos con coherencia directa.
"""

    def analyze_query(
        self,
        user_question: str,
        conversation_history: list[dict] | None = None,
        attachments: list[tuple[bytes, str]] | None = None,
        route: QueryRoute | None = None,
        attachment_default: bool = False,
    ) -> str:
        """Combina el historial local, la memoria de conversación y adjuntos con la pregunta."""
        question_clean = user_question.strip()
        if not question_clean:
            return "Por favor ingresa una consulta válida sobre tus entrenamientos."

        stats = self.database.get_stats()
        if stats.total_messages == 0 and not attachments:
            return (
                "⚠️ La base de datos local está vacía. "
                "Ejecuta primero una sincronización (`:sync` en la CLI o `/sync` en el bot de Telegram) "
                "para descargar tus entrenamientos desde el chat."
            )

        route = route or self._resolve_route(question_clean, conversation_history, attachments, attachment_default)
        prompt = self._build_prompt(question_clean, route, conversation_history)

        try:
            return self.llm_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                attachments=attachments,
                route=route,
            )
        except Exception as e:
            logger.error("Error al procesar consulta con el motor de IA: %s", e)
            return f"❌ Error al consultar el motor de IA: {e}"

    def _resolve_route(
        self,
        question_clean: str,
        conversation_history: list[dict] | None,
        attachments: list[tuple[bytes, str]] | None,
        attachment_default: bool,
    ) -> QueryRoute:
        """Elige la ruta de contexto. Adjunto sin caption: ventana corta en vez del historial completo."""
        if attachment_default and attachments:
            # ponytail: evita adjuntar las 93 Sesiones (27k chars) a cada foto/documento
            # genérico. Subir la ventana si Gemini necesita más contexto visual.
            return QueryRoute(complexity="complex", groups=["Comidas", "PesoCorporal"], days_lookback=3)
        return classify_query(question_clean, self.llm_client, conversation_history)

    async def analyze_query_stream_async(
        self,
        user_question: str,
        conversation_history: list[dict] | None = None,
        attachments: list[tuple[bytes, str]] | None = None,
        route: QueryRoute | None = None,
        attachment_default: bool = False,
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

        route = route or await asyncio.to_thread(
            self._resolve_route, question_clean, conversation_history, attachments, attachment_default
        )
        prompt = self._build_prompt(question_clean, route, conversation_history)

        async for chunk in self.llm_client.generate_response_stream_async(
            prompt=prompt,
            system_instruction=self.get_system_instruction(),
            attachments=attachments,
            route=route,
        ):
            yield chunk

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe un archivo o nota de voz de audio a texto utilizando Gemini."""
        return self.llm_client.transcribe_audio(audio_bytes, mime_type=mime_type)

    def analyze_nutrition_today(self) -> str:
        """Calcula el balance calórico y macronutrientes estimados de las comidas registradas hoy."""
        return self._analyze_nutrition_for_date(self._now())

    def analyze_nutrition_by_date(self, date_str: str) -> str:
        """Calcula el balance calórico y macronutrientes estimados de las comidas de una fecha dada.

        Args:
            date_str: Fecha en formatos DD/MM/YYYY, DD-MM-YYYY, DD/MM/YY, DD-MM o DD/MM.
                      También acepta 'hoy' o 'ayer'.
        """
        date_str = date_str.strip().lower()

        if date_str in ("hoy", "today"):
            target_date = datetime.now()
        elif date_str in ("ayer", "yesterday"):
            from datetime import timedelta

            target_date = datetime.now() - timedelta(days=1)
        else:
            # Intentar parsear varios formatos comunes
            parsed = None
            now = datetime.now()
            formats = [
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%d/%m/%y",
                "%d-%m-%y",
                "%d/%m",
                "%d-%m",
            ]
            for fmt in formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    # Si el formato no incluye año, asignamos el año actual
                    if fmt in ("%d/%m", "%d-%m"):
                        parsed = parsed.replace(year=now.year)
                    break
                except ValueError:
                    continue

            if parsed is None:
                return (
                    f"⚠️ No pude interpretar la fecha '{date_str}'. "
                    "Usa el formato DD/MM/YYYY, DD/MM o palabras como 'hoy' o 'ayer'."
                )
            target_date = parsed

        return self._analyze_nutrition_for_date(target_date)

    def _msg_event_date(self, msg: RawMessage) -> date:
        """Fecha real de un evento: prioriza la fecha escrita en el texto sobre la de envío."""
        match = _TEXT_DATE_RE.search(msg.text)
        if match:
            day, month = int(match.group(1)), int(match.group(2))
            year = match.group(3)
            if year:
                y = int(year)
                y = 2000 + y if y < 100 else y
            else:
                y = self._now().year
            try:
                return date(y, month, day)
            except ValueError:
                pass
        return to_atleta_tz(msg.date).date()

    def _analyze_nutrition_for_date(self, target_date: datetime) -> str:
        """Lógica común de análisis nutricional para una fecha concreta."""
        target_day = date(target_date.year, target_date.month, target_date.day)
        date_label = target_day.strftime("%d/%m/%Y")

        # Día completo en horario del atleta, convertido a UTC para comparar contra SQLite.
        start_utc = datetime(target_day.year, target_day.month, target_day.day, tzinfo=ATLETA_TZ).astimezone(UTC)
        end_utc = start_utc + timedelta(days=1)
        # Ventana ampliada (+2 días) para no perder registros enviados con retraso.
        window_end = end_utc + timedelta(days=2)

        day_messages = self.database.get_messages(since=start_utc, until=window_end, reverse=False)
        food_messages = [
            m
            for m in day_messages
            if (
                (m.chat_title and "comida" in m.chat_title.lower())
                or any(
                    kw in m.text.lower()
                    for kw in ("desayuno", "almuerzo", "cena", "merienda", "snack", "comida", "gr", "kcal", "proteina")
                )
            )
            and self._msg_event_date(m) == target_day
        ]

        if not food_messages:
            return f"No hay registros de comidas para el {date_label} en tu historial."

        food_text = "\n".join(m.to_summary_line() for m in food_messages)

        prompt = f"""
FECHA ANALIZADA: {date_label}

REGISTROS DE COMIDAS A ANALIZAR:
===============================
{food_text}
===============================

INSTRUCCIÓN:
Como nutricionista deportivo, considera ÚNICAMENTE las comidas que correspondan al {date_label}: si un mensaje tiene una fecha escrita en su texto, esa fecha manda; si no la tiene, vale su fecha de envío `[enviado: ...]`. Ignora comidas de otros días que hayan entrado en la ventana de seguridad.
Luego entrega:
1. Estimación calórica total aproximada (kcal).
2. Macronutrientes aproximados: Proteínas (g), Carbohidratos (g), Grasas (g).
3. Evaluación breve (2-3 oraciones) contrastando la ingesta con las metas del atleta: 2800-3000 kcal/día, proteína mínima 112 g y proteína óptima 126-155 g. Menciona puntos fuertes y áreas a mejorar si las hubiera.
Formato conciso, limpio y profesional. Usa las proporciones y pesos mencionados en los registros; si no están exactos, estima con valores estándar para porciones comunes (rioplatenses).
"""
        try:
            return self.llm_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
            )
        except Exception as e:
            logger.error("Error al analizar nutrición con el motor de IA: %s", e)
            return f"❌ Error al estimar nutrición: {e}"

    def recommend_next_progression(self, exercise_name: str) -> str:
        """Recomienda la carga y repeticiones para la próxima sesión aplicando doble progresión."""
        ex_query = exercise_name.strip()
        if not ex_query:
            return "Indica el ejercicio que deseas consultar (por ejemplo: /proxima banca o /proxima prensa)."

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
Aplica la regla de DOBLE PROGRESIÓN del atleta con técnica estricta y rango completo:
1. Identifica las cargas y repeticiones completadas en la última sesión para este ejercicio.
2. El esquema actual del atleta son 3 SERIES EFECTIVAS por ejercicio, con un rango general de 8 a 13 repeticiones según el ejercicio (en registros antiguos puede aparecer 4 series: usa el estándar actual de 3).
3. Si el atleta alcanzó 13 (o el tope del rango) repeticiones en las 3 series efectivas, indica el incremento de carga sugerido (ej. +1 a 2.5 kg) y las repeticiones base para la próxima sesión.
4. Si no alcanzó el tope del rango, recomienda mantener el mismo peso y buscar añadir 1-2 repeticiones totales hasta llegar a 13.
Responde de forma concisa y directa en 1 a 3 oraciones completas y motivantes.
"""
        try:
            return self.llm_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
            )
        except Exception as e:
            logger.error("Error al sugerir progresión con Gemini: %s", e)
            return f"❌ Error al consultar la sugerencia de progresión: {e}"

    def generate_weekly_summary(self) -> str:
        """Genera un reporte ejecutivo semanal de los últimos 7 días."""
        groups = list(GROUPS)
        context_text = self.build_focused_context("resumen semanal", groups, days_lookback=7)
        if context_text.startswith("NO HAY REGISTROS"):
            return "No se registraron entrenamientos ni pesajes en los últimos 7 días para generar el resumen semanal."

        group_stats = self._build_context_stats(groups)

        prompt = f"""
REGISTROS DISPONIBLES POR GRUPO:
{group_stats}

HISTORIAL DE LOS ÚLTIMOS 7 DÍAS (SEPARADO POR GRUPO):
=====================================================
{context_text}
=====================================================

INSTRUCCIÓN:
Elabora el RESUMEN SEMANAL EJECUTIVO para el atleta:
1. Días y sesiones completadas vs el plan objetivo (Fullbody 3 días L-M-V, Cardio 2 días M-S en Zona 2).
2. Mejores levantamientos o récords de sobrecarga progresiva alcanzados esta semana.
3. Tendencia o promedio del peso corporal usando la curva de PesoCorporal disponible.
4. Cumplimiento nutricional si hubo notas en comidas.
Estructura la respuesta de manera clara, concisa y profesional (3 a 5 párrafos cortos con viñetas limpias), sin saludos, sin despedidas y sin emojis en exceso.
"""
        try:
            return self.llm_client.generate_response(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                route=QueryRoute(complexity="complex", groups=groups, days_lookback=7),
            )
        except Exception as e:
            logger.error("Error al generar resumen semanal con Gemini: %s", e)
            return f"❌ Error al generar resumen semanal: {e}"
