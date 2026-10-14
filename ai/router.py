"""Query classifier (Level 1) and logical group resolution (Level 2).

The classifier decides, with a tiny and cheap model (llama-3.1-8b-instant via
Groq), what to look at BEFORE touching SQLite: complexity, groups involved and
day window. That way context extraction only retrieves what is needed.

If the classifier fails, times out or returns something invalid, a safe
fallback is applied: full analysis (all groups, all history).
"""

import json
import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # avoids circular import; only used for typing
    from config.settings import Settings

logger = logging.getLogger(__name__)

# Logical groups/topics of the athlete's training log.
GROUPS: tuple[str, ...] = ("Sesiones", "Cardio", "Comidas", "PesoCorporal")

# Group to which old messages without chat_title are attributed (see resolve_group_targets).
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
    """Route decided by the classifier for an athlete query."""

    model_config = ConfigDict(extra="forbid")

    complexity: Literal["simple", "complex"] = "complex"
    groups: list[str] = Field(default_factory=lambda: list(GROUPS))
    days_lookback: int = 0

    @field_validator("groups", mode="before")
    @classmethod
    def _normalize_groups(cls, v: object) -> list[str]:
        """Keep only valid group names (case-insensitive)."""
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
        """An empty list is interpreted as "all groups" (safe fallback)."""
        return v or list(GROUPS)


def fallback_route() -> QueryRoute:
    """Safe route when the classifier is unavailable: full analysis."""
    return QueryRoute(complexity="complex", groups=list(GROUPS), days_lookback=0)


def _classification_input(question: str, conversation_history: list[dict] | None) -> str:
    """Prepend the last turns (summarized) to resolve implicit references.

    Without history it returns the question as-is (legacy behavior).
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
    """Classify the query into complexity/groups/time window.

    `llm_client` only needs to expose `generate_json(prompt, system_instruction)`.
    `conversation_history` (optional) provides the last turns to resolve
    implicit references in the current question.
    On any failure (network, timeout, model down or invalid JSON) it returns
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
            raise ValueError(f"Classifier response is not a JSON object: {raw[:200]}")
        route = QueryRoute.model_validate(data)
        logger.debug(
            "Classified route: complexity=%s groups=%s days_lookback=%s",
            route.complexity,
            route.groups,
            route.days_lookback,
        )
        return route
    except Exception as e:  # noqa: BLE001 - the safe fallback is the expected behavior
        logger.warning("Classifier unavailable (%s). Using full analysis.", e)
        return fallback_route()


def resolve_group_targets(groups: list[str], settings: "Settings | None" = None) -> tuple[list[str], list[str]]:
    """Translate logical groups into (chat titles, chat IDs) to filter in SQLite.

    Old messages in the DB lack `chat_title`, so in addition to the group
    name, the entries configured in `GROUP_CHAT_MAP` are accepted
    (numeric IDs or alternative titles, e.g. "GYM").
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
            names.add(candidate.lower())  # alternative title (case-insensitive comparison)
            if candidate not in chat_ids:
                chat_ids.append(candidate)  # in case it is a chat ID
        if group == DEFAULT_GROUP:
            # Old messages in the DB have no chat_title: they are attributed to the
            # main group, which is where the bulk of the log lives.
            names.add("")
        for name in names:
            if name not in titles:
                titles.append(name)

    logger.debug("Groups %s resolved to titles=%s chat_ids=%s", wanted, titles, chat_ids)
    return titles, chat_ids
