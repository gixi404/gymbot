#!/usr/bin/env python3
"""Backfill of `chat_title` in old rows of the database.

The first synchronizations saved messages without `chat_title` (the DB prior to
multi-chat synchronization) and the main workout chat is called "GYM" on
Telegram, not "Sesiones". The router's group filtering (Level 2) depends on the title,
so this script normalizes the titles using `GROUP_CHAT_MAP`.

It is idempotent: it only fills rows with an empty `chat_title`, never overwrites real titles.

Usage:
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
    """Fills `chat_title` from `chat_id` and `GROUP_CHAT_MAP`.

    Returns the number of updated rows.
    """
    mapping: dict[str, list[str]] = dict(getattr(settings, "GROUP_CHAT_MAP", None) or {})

    # chat_id -> group: only entries shaped like an ID are accepted (starting with '-' or digits).
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
        logger.info("Chats without title found: %s", [r["chat_id"] for r in rows])

        updated = 0
        for chat_id, group in id_to_group.items():
            cursor = conn.execute(
                "UPDATE raw_messages SET chat_title = ? WHERE chat_id = ? AND (chat_title IS NULL OR chat_title = '');",
                (group, chat_id),
            )
            if cursor.rowcount:
                logger.info("  chat %s -> '%s' (%d rows)", chat_id, group, cursor.rowcount)
                updated += cursor.rowcount

        if unknown_ids:
            # Remaining legacy rows (without an identifiable chat_id): main diary.
            placeholders = ",".join("?" * len(unknown_ids))
            cursor = conn.execute(
                f"UPDATE raw_messages SET chat_title = ? WHERE chat_id IN ({placeholders}) "
                f"AND (chat_title IS NULL OR chat_title = '');",
                [DEFAULT_GROUP, *unknown_ids],
            )
            if cursor.rowcount:
                logger.info("  chats %s -> '%s' (%d rows)", unknown_ids, DEFAULT_GROUP, cursor.rowcount)
                updated += cursor.rowcount

        if dry_run:
            conn.rollback()
            logger.info("Dry-run: no changes were applied.")

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill de chat_title en la base de datos de GymBot.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra los cambios sin aplicarlos.")
    args = parser.parse_args()

    try:
        settings = get_settings()
        database = GymDatabase(settings.DATABASE_PATH)
    except Exception as e:
        logger.error("Initialization error: %s", e)
        sys.exit(1)

    updated = backfill_chat_titles(database, settings, dry_run=args.dry_run)
    logger.info("Rows updated: %d", updated)


if __name__ == "__main__":
    main()
