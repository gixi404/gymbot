"""Modelos de datos para el almacenamiento local de GymBot."""

from dataclasses import dataclass
from datetime import datetime


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
        """Formatea el mensaje para la ventana de contexto de IA con fecha y texto."""
        date_str = self.date.strftime("%Y-%m-%d %H:%M")
        origin = f" [{self.chat_title}]" if self.chat_title else ""
        return f"[{date_str}]{origin} (Msg #{self.id}):\n{self.text.strip()}\n"


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
