"""Security tests that verify strict read-only compliance requirement."""

import inspect
import unittest

from telegram_service.mtproto_reader import TelegramWorkoutReader


class TestMTProtoReadOnlySafety(unittest.TestCase):
    def test_no_write_methods_in_reader(self):
        """Verifies that the MTProto reader does not define write or modification methods."""
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
                # Allow safe internal methods if they exist, but no public send methods
                self.assertFalse(
                    forbidden in method_name and not method_name.startswith("_safe"),
                    f"Read-only security violation: suspicious method found '{method_name}'",
                )

    def test_reader_source_code_inspection(self):
        """Inspects the source code of TelegramWorkoutReader to confirm no send calls."""
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
                f"Forbidden call detected in MTProto reader: '{call}'. The chat must be 100% read-only.",
            )


if __name__ == "__main__":
    unittest.main()
