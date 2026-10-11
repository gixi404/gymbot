"""Modelos de datos para el almacenamiento local de GymBot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawMessage:
    """Representación de un mensaje obtenido del chat de entrenamientos."""

    id: int
    date: datetime
    text: str
    edit_date: Optional[datetime] = None
    media_type: Optional[str] = None
    chat_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_summary_line(self) -> str:
        """Formatea el mensaje para la ventana de contexto de IA con fecha y texto."""
        date_str = self.date.strftime("%Y-%m-%d %H:%M")
        return f"[{date_str}] (Msg #{self.id}):\n{self.text.strip()}\n"


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
    first_message_date: Optional[datetime] = None
    last_message_date: Optional[datetime] = None
    last_sync_at: Optional[datetime] = None
