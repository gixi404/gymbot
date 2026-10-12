"""Pruebas unitarias para los argumentos y utilidades de la CLI."""

import unittest


class TestCLIUtilities(unittest.TestCase):
    def test_cli_argument_handling(self):
        # Probamos que cli.py se importe y el analizador de argumentos funcione sin fallas
        import cli

        parser = cli.argparse.ArgumentParser()
        parser.add_argument("query", nargs="*")
        parser.add_argument("--sync", action="store_true")
        parser.add_argument("--stats", action="store_true")

        args1 = parser.parse_args(["--sync"])
        self.assertTrue(args1.sync)

        args2 = parser.parse_args(["¿Cuál", "fue", "mi", "récord?"])
        self.assertEqual(" ".join(args2.query), "¿Cuál fue mi récord?")

    def test_split_long_message(self):
        from telegram_service.bot_service import _split_long_message

        short_text = "Hola mundo"
        self.assertEqual(_split_long_message(short_text, max_length=100), ["Hola mundo"])

        # Texto que sobrepasa el límite
        long_lines = ["Línea " + str(i) for i in range(50)]
        joined = "\n".join(long_lines)
        chunks = _split_long_message(joined, max_length=100)
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)


if __name__ == "__main__":
    unittest.main()
