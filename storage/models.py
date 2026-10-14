"""Modelos de datos para el almacenamiento local de GymBot."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

ATLETA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")


def to_atleta_tz(dt: datetime) -> datetime:
    """Normaliza una fecha a la zona horaria del atleta.

    Si es naive se asume UTC (así vienen de Telethon/SQLite); si es aware se convierte.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ATLETA_TZ)


@dataclass
class RawMessage:
    """Representación de un mensaje obtenido del chat de entrenamientos."""

    id: int
    date: datetime
    text: str
    edit_date: datetime | None = None
    media_type: str | None = None
    chat_id: str | None = None
    chat_title: str | None = None
    created_at: datetime | None = None

    def to_summary_line(self) -> str:
        """Formatea el mensaje para la ventana de contexto de IA (compacto, sin truncar).

        La fecha de envío se rotula `[enviado: ...]` en hora de Argentina para que el
        LLM no la confunda con una fecha escrita dentro del propio texto.
        """
        date_str = to_atleta_tz(self.date).strftime("%d/%m/%Y")
        origin = f" [{self.chat_title}]" if self.chat_title else ""
        # Colapsar saltos de línea múltiples por ' | ', mantener texto completo
        text = re.sub(r"\s*\n\s*", " | ", self.text.strip())
        return f"[enviado: {date_str}]{origin} {text}"


@dataclass
class SyncMetadata:
    """Estado y cursor de sincronización para un canal o chat específico."""

    channel_id: str
    last_message_id: int
    last_sync_at: datetime
    total_messages: int


@dataclass
class DatabaseStats:
    """Métricas y estadísticas de la base de datos local."""

    total_messages: int
    first_message_date: datetime | None = None
    last_message_date: datetime | None = None
    last_sync_at: datetime | None = None
