"""Pruebas unitarias para la persistencia SQLite en GymDatabase."""

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

    def test_conversation_memory_and_clear_preserves_history(self):
        user_id = "test_user_123"

        # Sesión inicial debe ser 1
        self.assertEqual(self.db.get_current_session_id(user_id), 1)

        # Agregar mensajes en sesión 1
        self.db.add_conversation_message(user_id, "user", "¿Cuáles fueron mis pesos en sentadilla?")
        self.db.add_conversation_message(user_id, "assistant", "Hiciste 100kg x 5 y 105kg x 3.")

        # Memoria activa debe tener 2 mensajes
        active = self.db.get_active_memory(user_id)
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["text"], "¿Cuáles fueron mis pesos en sentadilla?")
        self.assertEqual(active[1]["text"], "Hiciste 100kg x 5 y 105kg x 3.")

        # Ejecutar /clear (reset_active_memory)
        new_session = self.db.reset_active_memory(user_id)
        self.assertEqual(new_session, 2)
        self.assertEqual(self.db.get_current_session_id(user_id), 2)

        # La memoria activa ahora debe estar VACÍA para la IA
        active_after_clear = self.db.get_active_memory(user_id)
        self.assertEqual(len(active_after_clear), 0)

        # PERO en la base de datos TODOS los mensajes siguen existiendo (historial no borrado)
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM conversation_messages WHERE user_id = ?;",
                (user_id,),
            )
            total_historical_messages = cursor.fetchone()["count"]
            self.assertEqual(total_historical_messages, 2)

        # Enviar nuevo mensaje en la nueva sesión
        self.db.add_conversation_message(user_id, "user", "Hola, empezamos de nuevo.")
        active_session_2 = self.db.get_active_memory(user_id)
        self.assertEqual(len(active_session_2), 1)
        self.assertEqual(active_session_2[0]["text"], "Hola, empezamos de nuevo.")

        # Historial total en BD ahora es 3
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM conversation_messages WHERE user_id = ?;",
                (user_id,),
            )
            self.assertEqual(cursor.fetchone()["count"], 3)

    def test_get_stats_with_channel_id_list(self):
        # Registramos sincronización para dos canales distintos
        self.db.update_sync_metadata("-1001", last_message_id=10, total_messages=5)
        self.db.update_sync_metadata("-1002", last_message_id=20, total_messages=10)

        # Consultar pasando una lista de chats (como ocurre en la configuración)
        stats = self.db.get_stats(channel_id=["-1001", "-1002"])
        self.assertIsNotNone(stats.last_sync_at)

    def test_get_last_synced_id_fallback_chat_isolation(self):
        # Canal 1 tiene mensaje con ID 500
        msg_c1 = RawMessage(id=500, date=datetime.now(), text="C1", chat_id="-1001")
        self.db.upsert_messages([msg_c1])

        # Canal 2 es nuevo y no tiene metadata; su fallback debe ser 0, no 500
        last_id_c2 = self.db.get_last_synced_id("-1002")
        self.assertEqual(last_id_c2, 0)

    def test_composite_primary_key_multi_chat_coexistence(self):
        now = datetime.now()
        # Mismo id=100 en dos chats diferentes
        msg_gym = RawMessage(id=100, date=now, text="Press militar 60kg", chat_id="-5000217070", chat_title="GYM")
        msg_food = RawMessage(id=100, date=now, text="Avena 80g + Whey", chat_id="-5335679250", chat_title="Comidas")

        self.db.upsert_messages([msg_gym, msg_food])

        stored = self.db.get_messages()
        self.assertEqual(len(stored), 2)
        chats = {m.chat_id for m in stored}
        self.assertEqual(chats, {"-5000217070", "-5335679250"})

        # Comprobar formato to_summary_line
        summary_gym = msg_gym.to_summary_line()
        self.assertIn("[GYM]", summary_gym)
        self.assertIn("(Msg #100)", summary_gym)

        summary_food = msg_food.to_summary_line()
        self.assertIn("[Comidas]", summary_food)
        self.assertIn("Avena 80g", summary_food)


if __name__ == "__main__":
    unittest.main()
