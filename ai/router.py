"""Clasificador de consultas (Nivel 1) y resolución de grupos lógicos (Nivel 2).

El clasificador decide, con un modelo diminuto y barato (llama-3.1-8b-instant vía
Groq), qué hay que mirar ANTES de tocar SQLite: complejidad, grupos implicados y
ventana de días. Así la extracción de contexto solo recupera lo necesario.

Si el clasificador falla, da timeout o devuelve algo inválido, se aplica un
fallback seguro: análisis completo (todos los grupos, todo el historial).
"""

import json
import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # evita importación circular; solo se usa para tipado
    from config.settings import Settings

logger = logging.getLogger(__name__)

# Grupos/tópicos lógicos del diario de entrenamientos del atleta.
GROUPS: tuple[str, ...] = ("Sesiones", "Cardio", "Comidas", "PesoCorporal")

# Grupo al que se atribuyen los mensajes antiguos sin chat_title (ver resolve_group_targets).
DEFAULT_GROUP = "Sesiones"

CLASSIFIER_SYSTEM_PROMPT = """Clasifica consultas de un atleta sobre su registro de entrenamientos.

GRUPOS DISPONIBLES (elige SOLO los estrictamente necesarios):
- Sesiones: entrenamientos de fuerza, series, repeticiones, cargas, RMs.
- Cardio: actividad aeróbica, caminatas, Zona 2, bicicleta.
- Comidas: alimentación, desayunos, calorías, proteínas y macros.
- PesoCorporal: peso corporal, pesajes, tendencia de peso.

CAMPOS DEL JSON:
- "complexity": "simple" si se resuelve con registros puntuales (última sesión, día
  concreto, macros de una fecha, una sola métrica). "complex" si exige progreso
  histórico, detección de estancamientos, estimar ganancia de músculo vs grasa,
  cruzar varios grupos o analizar varios meses.
- "groups": lista con los grupos implicados (usa los nombres EXACTOS de arriba).
- "days_lookback": cuántos días hacia atrás debe consultarse el historial.
  Para consultas de "hoy" o "ayer" usa mínimo 3 (cubre desfases de envío y cercanía
  a la medianoche UTC), para esta semana 7, para 2 semanas 14, para este mes 30,
  para 3 meses 90, y 0 SOLO si se pide todo el historial o un análisis de larga duración.

Si se incluye CONTEXTO PREVIO, úsalo para resolver referencias implícitas en la
PREGUNTA ACTUAL (ej. "¿y la semana pasada?", "¿cuál de esos 3?", "¿cómo impacta eso
en mi peso?"). Clasifica según lo que requiere responder la PREGUNTA ACTUAL.

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional ni markdown.
Ejemplo: {"complexity": "complex", "groups": ["Sesiones", "PesoCorporal"], "days_lookback": 90}
"""


class QueryRoute(BaseModel):
    """Ruta decidida por el clasificador para una consulta del atleta."""

    model_config = ConfigDict(extra="forbid")

    complexity: Literal["simple", "complex"] = "complex"
    groups: list[str] = Field(default_factory=lambda: list(GROUPS))
    days_lookback: int = 0

    @field_validator("groups", mode="before")
    @classmethod
    def _normalize_groups(cls, v: object) -> list[str]:
        """Conserva solo nombres de grupo válidos (sin distinguir mayúsculas)."""
        if not isinstance(v, (list, tuple, set)):
            return []
        out: list[str] = []
        for item in v:
            name = str(item).strip().lower()
            match = next((g for g in GROUPS if g.lower() == name), None)
            if match and match not in out:
                out.append(match)
        return out

    @field_validator("days_lookback", mode="before")
    @classmethod
    def _normalize_days(cls, v: object) -> int:
        try:
            return max(0, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    @field_validator("groups", mode="after")
    @classmethod
    def _ensure_groups(cls, v: list[str]) -> list[str]:
        """Una lista vacía se interpreta como "todos los grupos" (fallback seguro)."""
        return v or list(GROUPS)


def fallback_route() -> QueryRoute:
    """Ruta segura cuando el clasificador no está disponible: análisis completo."""
    return QueryRoute(complexity="complex", groups=list(GROUPS), days_lookback=0)


def _classification_input(question: str, conversation_history: list[dict] | None) -> str:
    """Antepone los últimos turnos (resumidos) para resolver referencias implícitas.

    Sin historial devuelve la pregunta tal cual (comportamiento histórico).
    """
    if not conversation_history:
        return question
    lines: list[str] = []
    for msg in list(conversation_history)[-4:]:
        role = "Atleta" if msg.get("role") == "user" else "Entrenador"
        lines.append(f"{role}: {str(msg.get('text', ''))[:300]}")
    return "CONTEXTO PREVIO:\n" + "\n".join(lines) + f"\n\nPREGUNTA ACTUAL: {question}"


def classify_query(
    question: str,
    llm_client: object | None = None,
    conversation_history: list[dict] | None = None,
) -> QueryRoute:
    """Clasifica la consulta en complejidad/grupos/ventana temporal.

    `llm_client` solo necesita exponer `generate_json(prompt, system_instruction)`.
    `conversation_history` (opcional) aporta los últimos turnos para resolver
    referencias implícitas en la pregunta actual.
    Ante cualquier fallo (red, timeout, modelo caído o JSON inválido) devuelve
    `fallback_route()`.
    """
    if llm_client is None:
        return fallback_route()
    try:
        raw = llm_client.generate_json(  # type: ignore[attr-defined]
            prompt=_classification_input(question, conversation_history),
            system_instruction=CLASSIFIER_SYSTEM_PROMPT,
        )
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Respuesta del clasificador no es un objeto JSON: {raw[:200]}")
        route = QueryRoute.model_validate(data)
        logger.debug(
            "Ruta clasificada: complexity=%s groups=%s days_lookback=%s",
            route.complexity,
            route.groups,
            route.days_lookback,
        )
        return route
    except Exception as e:  # noqa: BLE001 - el fallback seguro es el comportamiento esperado
        logger.warning("Clasificador no disponible (%s). Usando análisis completo.", e)
        return fallback_route()


def resolve_group_targets(groups: list[str], settings: "Settings | None" = None) -> tuple[list[str], list[str]]:
    """Traduce grupos lógicos a (títulos de chat, IDs de chat) para filtrar en SQLite.

    Los mensajes antiguos de la BD carecen de `chat_title`, por lo que además del
    nombre del grupo se aceptan las entradas configuradas en `GROUP_CHAT_MAP`
    (IDs numéricos o títulos alternativos, ej. "GYM").
    """
    if not groups:
        return [], []

    wanted = [g for g in GROUPS if g in groups] or list(GROUPS)
    mapping: dict[str, list[str]] = dict(getattr(settings, "GROUP_CHAT_MAP", None) or {})

    titles: list[str] = []
    chat_ids: list[str] = []
    for group in wanted:
        names = {group.lower()}
        for entry in mapping.get(group, []):
            candidate = str(entry).strip()
            if not candidate:
                continue
            names.add(candidate.lower())  # título alternativo (comparación insensible a mayúsculas)
            if candidate not in chat_ids:
                chat_ids.append(candidate)  # por si es un ID de chat
        if group == DEFAULT_GROUP:
            # Los mensajes antiguos de la BD no tienen chat_title: se atribuyen al
            # grupo principal, que es donde vive el grueso del diario.
            names.add("")
        for name in names:
            if name not in titles:
                titles.append(name)

    logger.debug("Grupos %s resueltos a títulos=%s chat_ids=%s", wanted, titles, chat_ids)
    return titles, chat_ids
