#!/usr/bin/env python3
"""Guided interactive script to authenticate the Telethon MTProto session a single time."""

import asyncio
import sys
from pathlib import Path

from telethon import TelegramClient

from config.settings import get_settings


async def main() -> None:
    print("=" * 60)
    print("🏋️ GYMBOT - ASISTENTE DE AUTENTICACIÓN MTPROTO DE TELEGRAM")
    print("=" * 60)
    print("Este script generará el archivo de sesión para que GymBot pueda")
    print("leer tu chat de entrenamientos en modo 100% pasivo y de solo lectura.\n")

    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Error al cargar la configuración (.env): {e}")
        print("Por favor revisa que el archivo .env exista y contenga TELEGRAM_API_ID y TELEGRAM_API_HASH.")
        sys.exit(1)

    session_name = settings.TELEGRAM_SESSION_NAME
    session_path = Path(session_name)
    if not session_path.is_absolute():
        session_path = Path(__file__).resolve().parent / session_name
    session_file = Path(f"{session_path}.session")

    print(f"• API ID: {settings.TELEGRAM_API_ID}")
    print(f"• Archivo de sesión a generar: {session_file.resolve()}")
    print(f"• Chat de entrenamientos configurado: {settings.TELEGRAM_WORKOUT_CHAT_ID}\n")

    client = TelegramClient(str(session_path), settings.TELEGRAM_API_ID, settings.TELEGRAM_API_HASH)

    print("Iniciando cliente de Telegram...")
    print("Si es tu primera vez, Telethon te pedirá:")
    print("  1. Tu número de teléfono con código de país (ej. +54911xxxxxxxx)")
    print("  2. El código de verificación que Telegram te envía a tu app")
    print("  3. Tu contraseña de doble factor (2FA), si la tienes activada.\n")

    try:
        await client.start()
    except Exception as e:
        print(f"\n❌ Error durante el proceso de autenticación: {e}")
        sys.exit(1)

    if await client.is_user_authorized():
        me = await client.get_me()
        print("\n" + "=" * 60)
        print("✅ ¡AUTENTICACIÓN COMPLETADA EXITOSAMENTE!")
        print("=" * 60)
        print(f"• Sesión activa para: {me.first_name} {me.last_name or ''} (@{me.username or 'sin_username'})")
        print(f"• Telegram User ID: {me.id}")
        print(f"\n💡 [IMPORTANTE]: Copia tu User ID ({me.id}) en la variable ALLOWED_TELEGRAM_USER_IDS de tu .env")
        print("   para que tu bot secundario reconozca tu usuario móvil.")

        # Verification of the workout chat
        print(f"\nVerificando acceso al chat de entrenamientos '{settings.TELEGRAM_WORKOUT_CHAT_ID}'...")
        try:
            entity = await client.get_entity(settings.TELEGRAM_WORKOUT_CHAT_ID)
            title = getattr(entity, "title", getattr(entity, "first_name", "Chat Privado"))
            print(f"✅ ¡Chat encontrado! Título: '{title}' (ID: {entity.id})")
        except ValueError:
            try:
                await client.get_dialogs()
                entity = await client.get_entity(settings.TELEGRAM_WORKOUT_CHAT_ID)
                title = getattr(entity, "title", getattr(entity, "first_name", "Chat Privado"))
                print(f"✅ ¡Chat encontrado! Título: '{title}' (ID: {entity.id})")
            except Exception as e:
                print(
                    f"⚠️ Advertencia: No se pudo resolver automáticamente el chat '{settings.TELEGRAM_WORKOUT_CHAT_ID}'."
                )
                print(f"   Detalle: {e}")
                print(
                    "   Asegúrate de haber ingresado en ese chat desde esta cuenta o que el ID/username sea correcto."
                )
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo resolver automáticamente el chat '{settings.TELEGRAM_WORKOUT_CHAT_ID}'.")
            print(f"   Detalle: {e}")
            print("   Asegúrate de haber ingresado en ese chat desde esta cuenta o que el ID/username sea correcto.")

        print("\nEl archivo de sesión ha sido guardado de forma segura.")
        print("Ya puedes usar la CLI (`python cli.py`) o arrancar el servicio de fondo (`python service.py`).\n")

        try:
            from telethon.sessions import StringSession

            string_session = StringSession.save(client.session)
            print("=" * 60)
            print("🔑 SESIÓN EN TEXTO (para TELEGRAM_STRING_SESSION)")
            print("=" * 60)
            print("Úsala en despliegues headless (Docker/Railway) donde el archivo de")
            print("sesión no se incluye en la imagen. Cópiala en la variable")
            print("TELEGRAM_STRING_SESSION del proveedor de despliegue (es un secreto).\n")
            print(string_session)
            print()
        except Exception as e:
            print(f"⚠️ No se pudo exportar la sesión en texto: {e}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
