#!/usr/bin/env python3
"""Backfill de `chat_title` en las filas antiguas de la base de datos.

Las primeras sincronizaciones guardaron los mensajes sin `chat_title` (la BD previa a
la sincronización multi-chat) y el chat principal de entrenamientos se llama "GYM" en
Telegram, no "Sesiones". El filtrado por grupos del router (Nivel 2) depende del título,
así que este script normaliza los títulos usando `GROUP_CHAT_MAP`.

Es idempotente: solo rellena filas con `chat_title` vacío, nunca pisa títulos reales.

Uso:
    python backfill_chat_titles.py [--dry-run]
"""

import argparse
import logging
import sys

from ai.router import DEFAULT_GROUP
from config.settings import Settings, get_settings
from storage.database import GymDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill")


def backfill_chat_titles(database: GymDatabase, settings: Settings, dry_run: bool = False) -> int:
    """Rellena `chat_title` a partir de `chat_id` y `GROUP_CHAT_MAP`.

    Retorna la cantidad de filas actualizadas.
    """
    mapping: dict[str, list[str]] = dict(getattr(settings, "GROUP_CHAT_MAP", None) or {})

    # chat_id -> grupo: solo se aceptan entradas con forma de ID (empiezan por '-' o son dígitos).
    id_to_group: dict[str, str] = {}
    for group, entries in mapping.items():
        for entry in entries:
            candidate = str(entry).strip()
            if candidate and (candidate.lstrip("-").isdigit()):
                id_to_group[candidate] = group

    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT chat_id FROM raw_messages WHERE chat_title IS NULL OR chat_title = '';"
        ).fetchall()
        unknown_ids = sorted({r["chat_id"] for r in rows} - set(id_to_group))
        logger.info("Chats sin título encontrados: %s", [r["chat_id"] for r in rows])

        updated = 0
        for chat_id, group in id_to_group.items():
            cursor = conn.execute(
                "UPDATE raw_messages SET chat_title = ? WHERE chat_id = ? AND (chat_title IS NULL OR chat_title = '');",
                (group, chat_id),
            )
            if cursor.rowcount:
                logger.info("  chat %s -> '%s' (%d filas)", chat_id, group, cursor.rowcount)
                updated += cursor.rowcount

        if unknown_ids:
            # Resto de filas legacy (sin chat_id identificable): diario principal.
            placeholders = ",".join("?" * len(unknown_ids))
            cursor = conn.execute(
                f"UPDATE raw_messages SET chat_title = ? WHERE chat_id IN ({placeholders}) "
                f"AND (chat_title IS NULL OR chat_title = '');",
                [DEFAULT_GROUP, *unknown_ids],
            )
            if cursor.rowcount:
                logger.info("  chats %s -> '%s' (%d filas)", unknown_ids, DEFAULT_GROUP, cursor.rowcount)
                updated += cursor.rowcount

        if dry_run:
            conn.rollback()
            logger.info("Dry-run: no se aplicaron cambios.")

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill de chat_title en la base de datos de GymBot.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra los cambios sin aplicarlos.")
    args = parser.parse_args()

    try:
        settings = get_settings()
        database = GymDatabase(settings.DATABASE_PATH)
    except Exception as e:
        logger.error("Error de inicialización: %s", e)
        sys.exit(1)

    updated = backfill_chat_titles(database, settings, dry_run=args.dry_run)
    logger.info("Filas actualizadas: %d", updated)


if __name__ == "__main__":
    main()
