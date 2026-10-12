"""Gestión de la base de datos local SQLite con modo WAL para GymBot."""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional, Union

from .models import DatabaseStats, RawMessage, SyncMetadata

logger = logging.getLogger(__name__)


class GymDatabase:
    """Manejador de base de datos SQLite optimizado para concurrencia."""

    def __init__(self, db_path: Union[str, Path] = "gymbot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Abre y configura una conexión SQLite con soporte concurrente WAL."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        try:
            # Optimizaciones de rendimiento y concurrencia por conexión
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Inicializa las tablas e índices si no existen."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_messages (
                    id INTEGER PRIMARY KEY,
                    chat_id TEXT DEFAULT '',
                    date TEXT NOT NULL,
                    text TEXT NOT NULL,
                    edit_date TEXT,
                    media_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Migración automática si la tabla no tenía la columna chat_id
            cursor = conn.execute("PRAGMA table_info(raw_messages);")
            col_names = [r["name"] for r in cursor.fetchall()]
            if "chat_id" not in col_names:
                conn.execute("ALTER TABLE raw_messages ADD COLUMN chat_id TEXT DEFAULT '';")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_messages_date
                ON raw_messages (date ASC);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    channel_id TEXT PRIMARY KEY,
                    last_message_id INTEGER NOT NULL DEFAULT 0,
                    last_sync_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_user_session
                ON conversation_messages (user_id, session_id, id ASC);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_session_state (
                    user_id TEXT PRIMARY KEY,
                    current_session_id INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        logger.debug("Base de datos SQLite inicializada en: %s", self.db_path)

    def upsert_messages(self, messages: List[RawMessage]) -> int:
        """Inserta o actualiza mensajes de forma idempotente.

        Retorna la cantidad de mensajes procesados.
        """
        if not messages:
            return 0

        query = """
        INSERT INTO raw_messages (id, chat_id, date, text, edit_date, media_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
        ON CONFLICT(id) DO UPDATE SET
            chat_id = excluded.chat_id,
            date = excluded.date,
            text = excluded.text,
            edit_date = excluded.edit_date,
            media_type = excluded.media_type;
        """

        data = [
            (
                m.id,
                m.chat_id or "",
                m.date.isoformat(),
                m.text,
                m.edit_date.isoformat() if m.edit_date else None,
                m.media_type,
                m.created_at.isoformat() if m.created_at else None,
            )
            for m in messages
        ]

        with self.get_connection() as conn:
            conn.executemany(query, data)

        return len(messages)

    def get_last_synced_id(self, channel_id: Union[str, int]) -> int:
        """Obtiene el ID del último mensaje sincronizado para el chat."""
        cid = str(channel_id)
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT last_message_id FROM sync_metadata WHERE channel_id = ?;",
                (cid,),
            )
            row = cursor.fetchone()
            if row:
                return int(row["last_message_id"])

            # Fallback: consultar el ID máximo en raw_messages para este chat si no hay metadata previa
            cursor = conn.execute(
                "SELECT MAX(id) as max_id FROM raw_messages WHERE chat_id = ?;",
                (cid,),
            )
            max_row = cursor.fetchone()
            return int(max_row["max_id"]) if max_row and max_row["max_id"] else 0

    def update_sync_metadata(
        self,
        channel_id: Union[str, int],
        last_message_id: int,
        total_messages: Optional[int] = None,
    ) -> None:
        """Actualiza el cursor de sincronización para el canal."""
        cid = str(channel_id)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as conn:
            if total_messages is None:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM raw_messages;")
                total_messages = int(cursor.fetchone()["cnt"])

            query = """
            INSERT INTO sync_metadata (channel_id, last_message_id, last_sync_at, total_messages)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                last_message_id = MAX(sync_metadata.last_message_id, excluded.last_message_id),
                last_sync_at = excluded.last_sync_at,
                total_messages = excluded.total_messages;
            """
            conn.execute(query, (cid, last_message_id, now_iso, total_messages))

    def get_messages(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
        reverse: bool = False,
    ) -> List[RawMessage]:
        """Obtiene los mensajes registrados ordenados por fecha.

        Por defecto, reverse=False devuelve los mensajes en orden cronológico (más antiguo a más reciente).
        """
        conditions = []
        params = []

        if since:
            conditions.append("date >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("date <= ?")
            params.append(until.isoformat())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_dir = "DESC" if reverse else "ASC"
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

        query = f"""
        SELECT id, date, text, edit_date, media_type, created_at
        FROM raw_messages
        {where_clause}
        ORDER BY date {order_dir}
        {limit_clause};
        """

        results: List[RawMessage] = []
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                results.append(
                    RawMessage(
                        id=row["id"],
                        date=datetime.fromisoformat(row["date"]),
                        text=row["text"],
                        edit_date=(
                            datetime.fromisoformat(row["edit_date"])
                            if row["edit_date"]
                            else None
                        ),
                        media_type=row["media_type"],
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row["created_at"]
                            else None
                        ),
                    )
                )
        return results

    def get_stats(self, channel_id: Optional[Union[str, int]] = None) -> DatabaseStats:
        """Recupera estadísticas de la base de datos local."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_count,
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM raw_messages;
                """
            )
            row = cursor.fetchone()
            total_count = int(row["total_count"]) if row else 0
            min_date = (
                datetime.fromisoformat(row["min_date"])
                if row and row["min_date"]
                else None
            )
            max_date = (
                datetime.fromisoformat(row["max_date"])
                if row and row["max_date"]
                else None
            )

            last_sync_at = None
            if isinstance(channel_id, (list, tuple, set)):
                c_list = [str(cid) for cid in channel_id if str(cid).strip()]
                if c_list:
                    placeholders = ",".join("?" * len(c_list))
                    c_cursor = conn.execute(
                        f"SELECT MAX(last_sync_at) as max_sync FROM sync_metadata WHERE channel_id IN ({placeholders});",
                        c_list,
                    )
                    c_row = c_cursor.fetchone()
                    if c_row and c_row["max_sync"]:
                        last_sync_at = datetime.fromisoformat(c_row["max_sync"])
            elif channel_id:
                c_cursor = conn.execute(
                    "SELECT last_sync_at FROM sync_metadata WHERE channel_id = ?;",
                    (str(channel_id),),
                )
                c_row = c_cursor.fetchone()
                if c_row and c_row["last_sync_at"]:
                    last_sync_at = datetime.fromisoformat(c_row["last_sync_at"])
            else:
                c_cursor = conn.execute(
                    "SELECT MAX(last_sync_at) as max_sync FROM sync_metadata;"
                )
                c_row = c_cursor.fetchone()
                if c_row and c_row["max_sync"]:
                    last_sync_at = datetime.fromisoformat(c_row["max_sync"])

            return DatabaseStats(
                total_messages=total_count,
                first_message_date=min_date,
                last_message_date=max_date,
                last_sync_at=last_sync_at,
            )

    def search_messages(self, query_term: str, limit: int = 50) -> List[RawMessage]:
        """Búsqueda simple por palabra clave dentro del texto de entrenamientos."""
        query = """
        SELECT id, date, text, edit_date, media_type, created_at
        FROM raw_messages
        WHERE text LIKE ?
        ORDER BY date DESC
        LIMIT ?;
        """
        pattern = f"%{query_term}%"
        results: List[RawMessage] = []
        with self.get_connection() as conn:
            cursor = conn.execute(query, (pattern, limit))
            for row in cursor.fetchall():
                results.append(
                    RawMessage(
                        id=row["id"],
                        date=datetime.fromisoformat(row["date"]),
                        text=row["text"],
                        edit_date=(
                            datetime.fromisoformat(row["edit_date"])
                            if row["edit_date"]
                            else None
                        ),
                        media_type=row["media_type"],
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row["created_at"]
                            else None
                        ),
                    )
                )
        return results

    def get_current_session_id(self, user_id: str) -> int:
        """Obtiene el identificador de sesión de memoria activa para el usuario."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT current_session_id FROM user_session_state WHERE user_id = ?;",
                (str(user_id),),
            )
            row = cursor.fetchone()
            if row:
                return int(row["current_session_id"])
            conn.execute(
                "INSERT INTO user_session_state (user_id, current_session_id) VALUES (?, 1);",
                (str(user_id),),
            )
            return 1

    def add_conversation_message(self, user_id: str, role: str, content: str) -> None:
        """Registra un mensaje en el historial persistente asociado a la sesión de memoria activa."""
        session_id = self.get_current_session_id(user_id)
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages (user_id, role, content, session_id, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (str(user_id), role, content, session_id),
            )

    def get_active_memory(self, user_id: str, limit: int = 10) -> List[dict]:
        """Obtiene los últimos mensajes de la sesión activa para proveer contexto de memoria al modelo."""
        session_id = self.get_current_session_id(user_id)
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (str(user_id), session_id, limit),
            )
            rows = cursor.fetchall()
            return [{"role": r["role"], "text": r["content"]} for r in reversed(rows)]

    def reset_active_memory(self, user_id: str) -> int:
        """Reinicia la memoria activa incrementando el session_id.

        IMPORTANTE: NO borra los mensajes anteriores de la base de datos;
        el historial completo de conversaciones permanece intacto para consulta.
        Retorna el nuevo session_id.
        """
        current_id = self.get_current_session_id(user_id)
        new_id = current_id + 1
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_session_state (user_id, current_session_id, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    current_session_id = excluded.current_session_id,
                    updated_at = excluded.updated_at;
                """,
                (str(user_id), new_id),
            )
        logger.info(
            "Memoria de sesión reiniciada para usuario '%s': sesión %d -> %d",
            user_id,
            current_id,
            new_id,
        )
        return new_id
