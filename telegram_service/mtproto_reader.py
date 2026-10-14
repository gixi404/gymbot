"""Passive read-only reader via MTProto (Telethon) for the workouts chat."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    RPCError,
)
from telethon.sessions import StringSession
from telethon.tl.custom.message import Message

from config.settings import Settings, get_settings
from storage.database import GymDatabase
from storage.models import RawMessage


# Reverse mapping chat_id/title -> logical group, derived from GROUP_CHAT_MAP in settings.
def _build_chat_id_to_group(settings: Settings) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, entries in (settings.GROUP_CHAT_MAP or {}).items():
        for entry in entries:
            key = str(entry).strip()
            if key:
                mapping[key] = group
    return mapping


logger = logging.getLogger(__name__)


class TelegramWorkoutReader:
    """Passive READ-ONLY MTProto client.

    CRITICAL SECURITY REQUIREMENT:
    This client NEVER sends messages, does not write, react or alter
    the log channel or chat. It connects on demand in an ephemeral way
    to read the history and disconnects immediately to release the session file.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        database: GymDatabase | None = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        self.api_id = self.settings.TELEGRAM_API_ID
        self.api_hash = self.settings.TELEGRAM_API_HASH
        self.session_name = self.settings.TELEGRAM_SESSION_NAME
        self.chat_ids: list[int | str] = self.settings.workout_chat_ids
        self._chat_id_to_group = _build_chat_id_to_group(self.settings)
        self._sync_lock = asyncio.Lock()
        self._last_sync_timestamp: float = 0.0

    @property
    def _resolved_session_path(self) -> str:
        session_path = Path(self.session_name)
        if not session_path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent
            session_path = base_dir / self.session_name
        return str(session_path)

    @asynccontextmanager
    async def get_client(self, max_retries: int = 3, retry_delay: float = 1.5) -> AsyncGenerator[TelegramClient, None]:
        """Opens an ephemeral connection with Telethon and guarantees its disconnection at the end.

        Includes brief retries in case another local process is releasing the session file.
        """
        session_arg = (
            StringSession(self.settings.TELEGRAM_STRING_SESSION)
            if self.settings.TELEGRAM_STRING_SESSION
            else self._resolved_session_path
        )
        client = TelegramClient(session_arg, self.api_id, self.api_hash)

        connected = False
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                await client.connect()
                connected = True
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d of %d to connect MTProto client failed: %s",
                    attempt,
                    max_retries,
                    e,
                )
                await asyncio.sleep(retry_delay)

        if not connected:
            raise RuntimeError(f"Could not connect the MTProto client after {max_retries} attempts: {last_error}")

        try:
            if not await client.is_user_authorized():
                raise PermissionError(
                    f"The session '{self.session_name}.session' is not authorized. "
                    "Please run 'python auth_telethon.py' first to authenticate your account."
                )
            yield client
        finally:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Error closing MTProto client: %s", e)

    async def sync_channel(
        self,
        force_all: bool = False,
        min_interval_seconds: float = 15.0,
    ) -> tuple[int, int]:
        """Incrementally syncs the messages from all workout chats to SQLite.

        Arguments:
            force_all: If True, retrieves all available history without considering the last ID.
            min_interval_seconds: Minimum time in seconds between successive syncs (debounce).

        Returns:
            Tuple (new_synced_messages, total_stored_messages)
        """
        async with self._sync_lock:
            stats = self.database.get_stats()
            now = time.monotonic()
            if (
                not force_all
                and self._last_sync_timestamp > 0
                and (now - self._last_sync_timestamp < min_interval_seconds)
            ):
                logger.debug(
                    "Sync skipped due to debounce (last one %.1fs ago).",
                    now - self._last_sync_timestamp,
                )
                return 0, stats.total_messages

            total_synced_now = 0

            async with self.get_client() as client:
                for current_chat_id in self.chat_ids:
                    try:
                        entity = await client.get_entity(current_chat_id)
                    except ValueError:
                        logger.debug(
                            "Entity '%s' not found in session cache. Fetching dialogs...",
                            current_chat_id,
                        )
                        await client.get_dialogs(limit=50)
                        entity = await client.get_entity(current_chat_id)
                    except Exception as e:
                        logger.error("Could not resolve the chat entity '%s': %s", current_chat_id, e)
                        continue

                    # Logical group name (per GROUP_CHAT_MAP) or, if not mapped, the real Telegram title.
                    chat_title = self._chat_id_to_group.get(str(current_chat_id)) or (
                        getattr(entity, "title", None) or getattr(entity, "username", None) or ""
                    )
                    min_id = 0 if force_all else self.database.get_last_synced_id(current_chat_id)
                    logger.info(
                        "Starting sync of '%s' ('%s') (min_id=%d, force_all=%s)...",
                        current_chat_id,
                        chat_title,
                        min_id,
                        force_all,
                    )

                    raw_messages_to_save: list[RawMessage] = []
                    max_seen_id = min_id

                    try:
                        async for msg in client.iter_messages(
                            entity,
                            min_id=min_id,
                            reverse=True,
                        ):
                            if not isinstance(msg, Message):
                                continue

                            text_content = (msg.message or msg.raw_text or "").strip()
                            if not text_content:
                                continue

                            msg_date = msg.date
                            edit_date = msg.edit_date
                            media_type = msg.media.__class__.__name__ if msg.media else None

                            raw_msg = RawMessage(
                                id=msg.id,
                                chat_id=str(current_chat_id),
                                chat_title=chat_title,
                                date=msg_date,
                                text=text_content,
                                edit_date=edit_date,
                                media_type=media_type,
                                created_at=datetime.now(UTC),
                            )
                            raw_messages_to_save.append(raw_msg)
                            if msg.id > max_seen_id:
                                max_seen_id = msg.id

                    except FloodWaitError as e:
                        logger.error("Telegram rate-limit detected (FloodWait: %d seconds)", e.seconds)
                        raise RuntimeError(f"Telegram has temporarily paused queries. Wait {e.seconds} seconds.") from e
                    except RPCError as e:
                        logger.error("MTProto error while iterating messages of '%s': %s", current_chat_id, e)
                        raise

                    synced_count = self.database.upsert_messages(raw_messages_to_save)
                    self.database.update_sync_metadata(current_chat_id, max_seen_id)
                    total_synced_now += synced_count

        stats = self.database.get_stats()
        self._last_sync_timestamp = time.monotonic()
        logger.info(
            "Sync completed. New messages: %d. Cumulative total: %d.",
            total_synced_now,
            stats.total_messages,
        )
        return total_synced_now, stats.total_messages

    async def verify_connection(self) -> bool:
        """Verifies that the Telethon credentials and session are valid."""
        try:
            async with self.get_client() as client:
                me = await client.get_me()
                logger.info(
                    "MTProto connection successful with user: %s (ID: %s)",
                    getattr(me, "username", "N/A"),
                    me.id,
                )
                return True
        except Exception as e:
            logger.error("Error verifying MTProto connection: %s", e)
            return False
