"""Lector pasivo de solo lectura mediante MTProto (Telethon) para el chat de entrenamientos."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple, Union

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import (
    FloodWaitError,
    RPCError,
    SessionPasswordNeededError,
)
from telethon.tl.custom.message import Message

from config.settings import Settings, get_settings
from storage.database import GymDatabase
from storage.models import RawMessage

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
        settings: Optional[Settings] = None,
        database: Optional[GymDatabase] = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        self.api_id = self.settings.TELEGRAM_API_ID
        self.api_hash = self.settings.TELEGRAM_API_HASH
        self.session_name = self.settings.TELEGRAM_SESSION_NAME
        self.chat_ids: List[Union[int, str]] = self.settings.workout_chat_ids

    @property
    def _resolved_session_path(self) -> str:
        session_path = Path(self.session_name)
        if not session_path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent
            session_path = base_dir / self.session_name
        return str(session_path)

    @asynccontextmanager
    async def get_client(
        self, max_retries: int = 3, retry_delay: float = 1.5
    ) -> AsyncGenerator[TelegramClient, None]:
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
            raise RuntimeError(
                f"No se pudo conectar el cliente MTProto tras {max_retries} intentos: {last_error}"
            )

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

    async def sync_channel(self, force_all: bool = False) -> Tuple[int, int]:
        """Sincroniza incrementalmente los mensajes de todos los chats de entrenamientos hacia SQLite.

        Argumentos:
            force_all: Si es True, recupera todo el historial disponible sin considerar el último ID.

        Retorna:
            Tupla (mensajes_nuevos_sincronizados, total_mensajes_almacenados)
        """
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
                    await client.get_dialogs()
                    entity = await client.get_entity(current_chat_id)
                except Exception as e:
                    logger.error("No se pudo resolver la entidad del chat '%s': %s", current_chat_id, e)
                    continue

                min_id = 0 if force_all else self.database.get_last_synced_id(current_chat_id)
                logger.info(
                    "Iniciando sincronización de '%s' (min_id=%d, force_all=%s)...",
                    current_chat_id,
                    min_id,
                    force_all,
                )

                raw_messages_to_save: List[RawMessage] = []
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
                            date=msg_date,
                            text=text_content,
                            edit_date=edit_date,
                            media_type=media_type,
                            created_at=datetime.now(timezone.utc),
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
