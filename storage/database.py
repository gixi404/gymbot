"""Gestión de la base de datos local SQLite con modo WAL para GymBot."""

import csv
import io
import logging
import sqlite3
import unicodedata
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import DatabaseStats, RawMessage

logger = logging.getLogger(__name__)


def _unaccent(value: str | None) -> str | None:
    """Quita diacríticos para búsquedas insensibles a tildes ('jalon' == 'Jalón')."""
    if value is None:
        return None
    decomposed = unicodedata.normalize("NFD", str(value))
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


class GymDatabase:
    """Manejador de base de datos SQLite optimizado para concurrencia."""

    def __init__(self, db_path: str | Path = "gymbot.db"):
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
        conn.create_function("unaccent", 1, _unaccent, deterministic=True)
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
                    id INTEGER NOT NULL,
                    chat_id TEXT NOT NULL DEFAULT '',
                    chat_title TEXT DEFAULT '',
                    date TEXT NOT NULL,
                    text TEXT NOT NULL,
                    edit_date TEXT,
                    media_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id, chat_id)
                );
                """
            )
            # Migración automática si la tabla fue creada con id como única clave primaria
            # o si le faltan las columnas chat_id o chat_title
            cursor = conn.execute("PRAGMA table_info(raw_messages);")
            table_info = cursor.fetchall()
            col_names = [r["name"] for r in table_info]
            pk_cols = [r["name"] for r in table_info if r["pk"] > 0]

            needs_pk_migration = "chat_id" not in pk_cols
            if needs_pk_migration:
                has_chat_id = "chat_id" in col_names
                has_chat_title = "chat_title" in col_names
                chat_id_expr = "COALESCE(chat_id, '')" if has_chat_id else "''"
                chat_title_expr = "COALESCE(chat_title, '')" if has_chat_title else "''"

                conn.execute(
                    """
                    CREATE TABLE raw_messages_migrated (
                        id INTEGER NOT NULL,
                        chat_id TEXT NOT NULL DEFAULT '',
                        chat_title TEXT DEFAULT '',
                        date TEXT NOT NULL,
                        text TEXT NOT NULL,
                        edit_date TEXT,
                        media_type TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id, chat_id)
                    );
                    """
                )
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO raw_messages_migrated (id, chat_id, chat_title, date, text, edit_date, media_type, created_at)
                    SELECT id, {chat_id_expr}, {chat_title_expr}, date, text, edit_date, media_type, created_at
                    FROM raw_messages;
                    """
                )
                conn.execute("DROP TABLE raw_messages;")
                conn.execute("ALTER TABLE raw_messages_migrated RENAME TO raw_messages;")
            else:
                if "chat_title" not in col_names:
                    conn.execute("ALTER TABLE raw_messages ADD COLUMN chat_title TEXT DEFAULT '';")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_messages_date
                ON raw_messages (date ASC);
                """
            )
            # Índices compuestos para filtros por grupo + rango temporal + orden
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_messages_chat_title_date
                ON raw_messages (chat_title, date ASC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_messages_chat_id_date
                ON raw_messages (chat_id, date ASC);
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

    def upsert_messages(self, messages: list[RawMessage]) -> int:
        """Inserta o actualiza mensajes de forma idempotente con soporte multi-chat.

        Retorna la cantidad de mensajes procesados.
        """
        if not messages:
            return 0

        query = """
        INSERT INTO raw_messages (id, chat_id, chat_title, date, text, edit_date, media_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
        ON CONFLICT(id, chat_id) DO UPDATE SET
            chat_title = CASE WHEN excluded.chat_title != '' THEN excluded.chat_title ELSE raw_messages.chat_title END,
            date = excluded.date,
            text = excluded.text,
            edit_date = excluded.edit_date,
            media_type = excluded.media_type;
        """

        data = [
            (
                m.id,
                m.chat_id or "",
                m.chat_title or "",
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

    def get_last_synced_id(self, channel_id: str | int) -> int:
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
        channel_id: str | int,
        last_message_id: int,
        total_messages: int | None = None,
    ) -> None:
        """Actualiza el cursor de sincronización para el canal."""
        cid = str(channel_id)
        now_iso = datetime.now(UTC).isoformat()

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

    @staticmethod
    def _group_filter_clause(titles: list[str] | None, chat_ids: list[str] | None) -> tuple[list[str], list[str]]:
        """Cláusulas SQL (OR entre título y chat_id) para filtrar por grupos lógicos."""
        clauses: list[str] = []
        params: list[str] = []
        if titles:
            placeholders = ",".join("?" * len(titles))
            clauses.append(f"LOWER(chat_title) IN ({placeholders})")
            params.extend(t.lower() for t in titles)
        if chat_ids:
            placeholders = ",".join("?" * len(chat_ids))
            clauses.append(f"chat_id IN ({placeholders})")
            params.extend(chat_ids)
        return clauses, params

    @staticmethod
    def _row_to_raw_message(row: sqlite3.Row) -> RawMessage:
        """Mapea una fila de base de datos a una instancia de RawMessage."""
        col_names = row.keys()
        return RawMessage(
            id=row["id"],
            date=datetime.fromisoformat(row["date"]),
            text=row["text"],
            edit_date=(datetime.fromisoformat(row["edit_date"]) if row["edit_date"] else None),
            media_type=row["media_type"],
            chat_id=row["chat_id"] if "chat_id" in col_names else None,
            chat_title=row["chat_title"] if "chat_title" in col_names else None,
            created_at=(datetime.fromisoformat(row["created_at"]) if row["created_at"] else None),
        )

    def get_messages(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        reverse: bool = False,
        titles: list[str] | None = None,
        chat_ids: list[str] | None = None,
    ) -> list[RawMessage]:
        """Obtiene los mensajes registrados ordenados por fecha.

        Por defecto, reverse=False devuelve los mensajes en orden cronológico (más antiguo a más reciente).
        `titles`/`chat_ids` restringen a grupos lógicos concretos (ver ai.router.resolve_group_targets).
        """
        conditions = []
        params = []

        if since:
            conditions.append("date >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("date <= ?")
            params.append(until.isoformat())

        group_clauses, group_params = self._group_filter_clause(titles, chat_ids)
        if group_clauses:
            conditions.append("(" + " OR ".join(group_clauses) + ")")
            params.extend(group_params)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_dir = "DESC" if reverse else "ASC"
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

        query = f"""
        SELECT id, chat_id, chat_title, date, text, edit_date, media_type, created_at
        FROM raw_messages
        {where_clause}
        ORDER BY date {order_dir}
        {limit_clause};
        """

        results: list[RawMessage] = []
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                results.append(self._row_to_raw_message(row))
        return results

    def get_stats(self, channel_id: str | int | None = None) -> DatabaseStats:
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
            min_date = datetime.fromisoformat(row["min_date"]) if row and row["min_date"] else None
            max_date = datetime.fromisoformat(row["max_date"]) if row and row["max_date"] else None

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
                c_cursor = conn.execute("SELECT MAX(last_sync_at) as max_sync FROM sync_metadata;")
                c_row = c_cursor.fetchone()
                if c_row and c_row["max_sync"]:
                    last_sync_at = datetime.fromisoformat(c_row["max_sync"])

            return DatabaseStats(
                total_messages=total_count,
                first_message_date=min_date,
                last_message_date=max_date,
                last_sync_at=last_sync_at,
            )

    def search_messages(self, query_term: str, limit: int = 50) -> list[RawMessage]:
        """Búsqueda simple por palabra clave dentro del texto de entrenamientos.

        Insensible a tildes y mayúsculas: 'jalon' encuentra 'Jalón'.
        """
        query = """
        SELECT id, chat_id, chat_title, date, text, edit_date, media_type, created_at
        FROM raw_messages
        WHERE unaccent(text) LIKE unaccent(?)
        ORDER BY date DESC
        LIMIT ?;
        """
        pattern = f"%{query_term}%"
        results: list[RawMessage] = []
        with self.get_connection() as conn:
            cursor = conn.execute(query, (pattern, limit))
            for row in cursor.fetchall():
                results.append(self._row_to_raw_message(row))
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

    def get_active_memory(self, user_id: str, limit: int = 10) -> list[dict]:
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
        """Reinicia la memoria activa: incrementa el session_id y borra los mensajes del usuario.

        Se eliminan los mensajes de `conversation_messages` para que la tabla no
        acumule basura indefinidamente con el paso de los meses.
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
            conn.execute(
                "DELETE FROM conversation_messages WHERE user_id = ?;",
                (str(user_id),),
            )
        logger.info(
            "Memoria de sesión reiniciada para usuario '%s': sesión %d -> %d (mensajes borrados)",
            user_id,
            current_id,
            new_id,
        )
        return new_id

    def export_to_csv(self) -> io.BytesIO:
        """Exporta todos los mensajes registrados a un buffer CSV con codificación UTF-8-SIG."""
        messages = self.get_messages(reverse=False)

        string_buffer = io.StringIO()
        writer = csv.writer(string_buffer, quoting=csv.QUOTE_MINIMAL)

        # Encabezados
        writer.writerow(["ID", "Fecha_UTC", "Origen", "Texto", "Tipo_Media", "Fecha_Edicion"])

        for msg in messages:
            writer.writerow(
                [
                    msg.id,
                    msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                    msg.chat_title or msg.chat_id or "",
                    msg.text,
                    msg.media_type or "",
                    msg.edit_date.strftime("%Y-%m-%d %H:%M:%S") if msg.edit_date else "",
                ]
            )

        bytes_buffer = io.BytesIO(string_buffer.getvalue().encode("utf-8-sig"))
        bytes_buffer.seek(0)
        return bytes_buffer
