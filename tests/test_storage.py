"""Pruebas unitarias para la persistencia SQLite en GymDatabase."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from storage.database import GymDatabase
from storage.models import RawMessage


class TestGymDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_gymbot.db"
        self.db = GymDatabase(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_tables(self):
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('raw_messages', 'sync_metadata');"
            )
            tables = {row["name"] for row in cursor.fetchall()}
            self.assertIn("raw_messages", tables)
            self.assertIn("sync_metadata", tables)

    def test_upsert_messages_idempotent(self):
        now = datetime(2026, 1, 15, 10, 30, 0)
        msgs = [
            RawMessage(id=1, date=now, text="Banca: 80kg 3x8"),
            RawMessage(id=2, date=now + timedelta(hours=1), text="Sentadillas: 100kg 4x5"),
        ]

        inserted = self.db.upsert_messages(msgs)
        self.assertEqual(inserted, 2)

        # Verificar lectura
        stored = self.db.get_messages()
        self.assertEqual(len(stored), 2)
        self.assertEqual(stored[0].id, 1)
        self.assertEqual(stored[0].text, "Banca: 80kg 3x8")

        # Actualizar un mensaje editado
        edited_msgs = [
            RawMessage(
                id=1,
                date=now,
                text="Banca: 85kg 3x8 (corregido)",
                edit_date=now + timedelta(minutes=5),
            )
        ]
        self.db.upsert_messages(edited_msgs)

        stored_after = self.db.get_messages()
        self.assertEqual(len(stored_after), 2)
        self.assertEqual(stored_after[0].text, "Banca: 85kg 3x8 (corregido)")
        self.assertIsNotNone(stored_after[0].edit_date)

    def test_sync_metadata_tracking(self):
        channel_id = "-1001234567890"
        self.assertEqual(self.db.get_last_synced_id(channel_id), 0)

        self.db.update_sync_metadata(channel_id, last_message_id=55, total_messages=10)
        self.assertEqual(self.db.get_last_synced_id(channel_id), 55)

        # Actualización posterior
        self.db.update_sync_metadata(channel_id, last_message_id=60, total_messages=15)
        self.assertEqual(self.db.get_last_synced_id(channel_id), 60)

    def test_get_messages_date_filter_and_order(self):
        base_time = datetime(2026, 3, 1, 10, 0, 0)
        msgs = [
            RawMessage(id=1, date=base_time, text="Día 1"),
            RawMessage(id=2, date=base_time + timedelta(days=2), text="Día 2"),
            RawMessage(id=3, date=base_time + timedelta(days=5), text="Día 3"),
        ]
        self.db.upsert_messages(msgs)

        # Orden cronológico (por defecto)
        chronological = self.db.get_messages()
        self.assertEqual([m.id for m in chronological], [1, 2, 3])

        # Orden inverso
        reverse_order = self.db.get_messages(reverse=True)
        self.assertEqual([m.id for m in reverse_order], [3, 2, 1])

        # Filtro de fechas
        filtered = self.db.get_messages(
            since=base_time + timedelta(days=1),
            until=base_time + timedelta(days=3),
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].id, 2)

    def test_search_messages_by_keyword(self):
        msgs = [
            RawMessage(id=1, date=datetime.now(), text="Press Militar con mancuernas 24kg"),
            RawMessage(id=2, date=datetime.now(), text="Dominadas supinas 4x8"),
            RawMessage(id=3, date=datetime.now(), text="Press Banca inclinado 70kg"),
        ]
        self.db.upsert_messages(msgs)

        results = self.db.search_messages("Press")
        self.assertEqual(len(results), 2)
        ids = {r.id for r in results}
        self.assertEqual(ids, {1, 3})


if __name__ == "__main__":
    unittest.main()
