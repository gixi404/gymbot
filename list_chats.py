#!/usr/bin/env python3
"""Script de utilidad para listar los chats, grupos y canales de tu cuenta y obtener sus IDs."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient

# Carga de variables de entorno
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "gymbot_user")

if not API_ID or not API_HASH:
    print("❌ Error: No se encontraron TELEGRAM_API_ID o TELEGRAM_API_HASH en tu archivo .env.")
    print("Por favor agrégalos a tu archivo .env antes de ejecutar este script.")
    sys.exit(1)

try:
    API_ID = int(API_ID)
except ValueError:
    print("❌ Error: TELEGRAM_API_ID debe ser un número entero.")
    sys.exit(1)


async def main() -> None:
    session_path = Path(SESSION_NAME)
    if not session_path.is_absolute():
        session_path = Path(__file__).resolve().parent / SESSION_NAME

    client = TelegramClient(str(session_path), API_ID, API_HASH)

    print("Conectando con Telegram...")
    await client.start()

    print("\n" + "=" * 70)
    print("📋 LISTADO DE TUS CHATS Y GRUPOS EN TELEGRAM")
    print("=" * 70)
    print(f"{'TÍTULO / NOMBRE':<35} | {'TIPO':<12} | {'CHAT ID'}")
    print("-" * 70)

    found_matches = []

    async for dialog in client.iter_dialogs():
        title = dialog.name or "Sin nombre"
        chat_id = dialog.id

        chat_type = "Privado"
        if dialog.is_channel:
            chat_type = "Canal"
        elif dialog.is_group:
            chat_type = "Grupo"

        print(f"{title[:34]:<35} | {chat_type:<12} | {chat_id}")

        if "gimnasio" in title.lower() or "gym" in title.lower():
            found_matches.append((title, chat_id))

    print("=" * 70)

    if found_matches:
        print("\n🎯 ¡SE ENCONTRARON COINCIDENCIAS CON TU BÚSQUEDA!")
        for match_title, match_id in found_matches:
            print(f"👉 Grupo: '{match_title}' -> TELEGRAM_WORKOUT_CHAT_ID={match_id}")
        print("\nCopia ese ID en tu archivo .env en la variable TELEGRAM_WORKOUT_CHAT_ID.")
    else:
        print("\n💡 Si tu grupo tiene otro nombre, busca en la lista de arriba y copia su ID en el .env.")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
