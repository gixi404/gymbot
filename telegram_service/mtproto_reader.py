"""Lector pasivo de solo lectura mediante MTProto (Telethon) para el chat de entrenamientos."""

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


# Mapeo inverso chat_id/título -> grupo lógico, derivado de GROUP_CHAT_MAP en settings.
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
    """Cliente MTProto pasivo de SOLO LECTURA.

    REQUISITO CRÍTICO DE SEGURIDAD:
    Este cliente NUNCA envía mensajes, no escribe, no reacciona ni altera
    el canal o chat de registro. Se conecta bajo demanda de forma efímera
    para leer el historial y se desconecta de inmediato para liberar el archivo de sesión.
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
        """Abre una conexión efímera con Telethon y garantiza su desconexión al finalizar.

        Incluye reintentos breves por si otro proceso local está liberando el archivo de sesión.
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
                    "Intento %d de %d para conectar cliente MTProto falló: %s",
                    attempt,
                    max_retries,
                    e,
                )
                await asyncio.sleep(retry_delay)

        if not connected:
            raise RuntimeError(f"No se pudo conectar el cliente MTProto tras {max_retries} intentos: {last_error}")

        try:
            if not await client.is_user_authorized():
                raise PermissionError(
                    f"La sesión '{self.session_name}.session' no está autorizada. "
                    "Por favor ejecuta primero 'python auth_telethon.py' para autenticar tu cuenta."
                )
            yield client
        finally:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Error cerrando cliente MTProto: %s", e)

    async def sync_channel(
        self,
        force_all: bool = False,
        min_interval_seconds: float = 15.0,
    ) -> tuple[int, int]:
        """Sincroniza incrementalmente los mensajes de todos los chats de entrenamientos hacia SQLite.

        Argumentos:
            force_all: Si es True, recupera todo el historial disponible sin considerar el último ID.
            min_interval_seconds: Tiempo mínimo en segundos entre sincronizaciones sucesivas (debounce).

        Retorna:
            Tupla (mensajes_nuevos_sincronizados, total_mensajes_almacenados)
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
                    "Sincronización omitida por debounce (última hace %.1fs).",
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
                            "Entidad '%s' no encontrada en caché de sesión. Obteniendo diálogos...",
                            current_chat_id,
                        )
                        await client.get_dialogs(limit=50)
                        entity = await client.get_entity(current_chat_id)
                    except Exception as e:
                        logger.error("No se pudo resolver la entidad del chat '%s': %s", current_chat_id, e)
                        continue

                    # Nombre del grupo lógico (según GROUP_CHAT_MAP) o, si no está mapeado, el título real de Telegram.
                    chat_title = self._chat_id_to_group.get(str(current_chat_id)) or (
                        getattr(entity, "title", None) or getattr(entity, "username", None) or ""
                    )
                    min_id = 0 if force_all else self.database.get_last_synced_id(current_chat_id)
                    logger.info(
                        "Iniciando sincronización de '%s' ('%s') (min_id=%d, force_all=%s)...",
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
                        logger.error("Rate-limit de Telegram detectado (FloodWait: %d segundos)", e.seconds)
                        raise RuntimeError(
                            f"Telegram ha pausado temporalmente las consultas. Espera {e.seconds} segundos."
                        ) from e
                    except RPCError as e:
                        logger.error("Error MTProto al iterar mensajes de '%s': %s", current_chat_id, e)
                        raise

                    synced_count = self.database.upsert_messages(raw_messages_to_save)
                    self.database.update_sync_metadata(current_chat_id, max_seen_id)
                    total_synced_now += synced_count

        stats = self.database.get_stats()
        self._last_sync_timestamp = time.monotonic()
        logger.info(
            "Sincronización completada. Nuevos mensajes: %d. Total acumulado: %d.",
            total_synced_now,
            stats.total_messages,
        )
        return total_synced_now, stats.total_messages

    async def verify_connection(self) -> bool:
        """Verifica que las credenciales y la sesión de Telethon sean válidas."""
        try:
            async with self.get_client() as client:
                me = await client.get_me()
                logger.info(
                    "Conexión MTProto exitosa con usuario: %s (ID: %s)",
                    getattr(me, "username", "N/A"),
                    me.id,
                )
                return True
        except Exception as e:
            logger.error("Error verificando conexión MTProto: %s", e)
            return False
