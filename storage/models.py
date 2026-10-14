"""Data models for GymBot's local storage."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

ATLETA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")


def to_atleta_tz(dt: datetime) -> datetime:
    """Normalizes a date to the athlete's timezone.

    If naive it is assumed to be UTC (that's how they come from Telethon/SQLite); if aware it is converted.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ATLETA_TZ)


@dataclass
class RawMessage:
    """Representation of a message obtained from the workout chat."""

    id: int
    date: datetime
    text: str
    edit_date: datetime | None = None
    media_type: str | None = None
    chat_id: str | None = None
    chat_title: str | None = None
    created_at: datetime | None = None

    def to_summary_line(self) -> str:
        """Formats the message for the AI context window (compact, no truncation).

        The send date is labeled `[enviado: ...]` in Argentina time so the
        LLM does not confuse it with a date written within the text itself.
        """
        date_str = to_atleta_tz(self.date).strftime("%d/%m/%Y")
        origin = f" [{self.chat_title}]" if self.chat_title else ""
        # Collapse multiple line breaks into ' | ', keep the full text
        text = re.sub(r"\s*\n\s*", " | ", self.text.strip())
        return f"[enviado: {date_str}]{origin} {text}"


@dataclass
class SyncMetadata:
    """Sync state and cursor for a specific channel or chat."""

    channel_id: str
    last_message_id: int
    last_sync_at: datetime
    total_messages: int


@dataclass
class DatabaseStats:
    """Metrics and statistics for the local database."""

    total_messages: int
    first_message_date: datetime | None = None
    last_message_date: datetime | None = None
    last_sync_at: datetime | None = None
