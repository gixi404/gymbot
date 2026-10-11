"""Pruebas de seguridad que verifican el cumplimiento del requisito estricto de solo lectura."""

import inspect
import unittest

from telegram_service.mtproto_reader import TelegramWorkoutReader


class TestMTProtoReadOnlySafety(unittest.TestCase):
    def test_no_write_methods_in_reader(self):
        """Verifica que el lector MTProto no defina métodos de escritura o modificación."""
        forbidden_method_substrings = [
            "send",
            "post",
            "write",
            "edit",
            "delete",
            "reply",
            "react",
            "publish",
            "forward",
        ]

        methods = inspect.getmembers(TelegramWorkoutReader, predicate=inspect.isfunction)
        method_names = [m[0].lower() for m in methods]

        for method_name in method_names:
            for forbidden in forbidden_method_substrings:
                # Permitir métodos internos seguros si existieran, pero ningún método público de envío
                self.assertFalse(
                    forbidden in method_name and not method_name.startswith("_safe"),
                    f"Violación de seguridad de solo lectura: método sospechoso encontrado '{method_name}'",
                )

    def test_reader_source_code_inspection(self):
        """Inspecciona el código fuente de TelegramWorkoutReader para confirmar ausencia de llamadas de envío."""
        source = inspect.getsource(TelegramWorkoutReader)

        # Telethon methods for sending or modifying content
        forbidden_telethon_calls = [
            ".send_message(",
            ".send_file(",
            ".edit_message(",
            ".delete_messages(",
            ".send_read_acknowledge(",
            ".forward_messages(",
        ]

        for call in forbidden_telethon_calls:
            self.assertNotIn(
                call,
                source,
                f"Llamada prohibida detectada en el lector MTProto: '{call}'. El chat debe ser 100% solo lectura.",
            )


if __name__ == "__main__":
    unittest.main()
