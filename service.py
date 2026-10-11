#!/usr/bin/env python3
"""Punto de entrada para el servicio de fondo (daemon) de GymBot.

Ejecuta el bot secundario de Telegram para consultas móviles y la sincronización periódica opcional.
"""

import asyncio
import shutil
import signal
import sys
from pathlib import Path
from typing import Optional

from ai.analyzer import GymAnalyzer
from config.settings import Settings, get_settings
from storage.database import GymDatabase
from telegram_service.bot_service import GymTelegramBot
from telegram_service.mtproto_reader import TelegramWorkoutReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gymbot.service")


def _seed_persistent_volume_if_needed(settings: Settings) -> None:
    """Si el volumen persistente (ej. en Railway o Docker) está vacío, inicializa con la sesión y BD base."""
    # 1. Base de datos
    db_target = Path(settings.DATABASE_PATH)
    if not db_target.exists():
        local_db = Path(__file__).resolve().parent / "gymbot.db"
        if local_db.exists() and local_db.resolve() != db_target.resolve():
            db_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_db, db_target)
            logger.info("Base de datos inicial copiada al volumen persistente: %s", db_target)

    # 2. Sesión MTProto
    session_file_name = f"{settings.TELEGRAM_SESSION_NAME}.session"
    session_target = Path(session_file_name)
    if not session_target.is_absolute():
        session_target = Path(__file__).resolve().parent / session_file_name
    if not session_target.exists():
        local_session = Path(__file__).resolve().parent / "gymbot_user.session"
        if local_session.exists() and local_session.resolve() != session_target.resolve():
            session_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_session, session_target)
            logger.info("Sesión MTProto copiada al volumen persistente: %s", session_target)


async def periodic_sync_task(reader: TelegramWorkoutReader, interval_minutes: int) -> None:
    """Tarea en segundo plano para sincronizar periódicamente de forma pasiva."""
    if interval_minutes <= 0:
        logger.info("Sincronización periódica automática desactivada (intervalo <= 0).")
        return

    interval_seconds = interval_minutes * 60
    logger.info(
        "Sincronización periódica automática activada (cada %d minutos).",
        interval_minutes,
    )

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logger.info("Ejecutando ciclo de sincronización automática...")
            new_msgs, total = await reader.sync_channel()
            logger.info(
                "Ciclo de sincronización finalizado: %d nuevos, %d total.",
                new_msgs,
                total,
            )
        except asyncio.CancelledError:
            logger.info("Tarea de sincronización periódica cancelada.")
            break
        except Exception as e:
            logger.warning("Advertencia en ciclo de sincronización automática: %s", e)


async def run_services() -> None:
    """Inicia concurrentemente el bot móvil y la tarea periódica."""
    settings = get_settings()
    _seed_persistent_volume_if_needed(settings)

    database = GymDatabase(settings.DATABASE_PATH)
    reader = TelegramWorkoutReader(settings, database)
    analyzer = GymAnalyzer(settings, database)

    # 1. Sincronización suave inicial al arrancar el servicio
    try:
        logger.info("Realizando sincronización inicial de arranque...")
        new_msgs, total_msgs = await reader.sync_channel()
        logger.info(
            "Sincronización inicial exitosa: %d nuevos mensajes, %d total.",
            new_msgs,
            total_msgs,
        )
    except Exception as e:
        logger.warning(
            "No se pudo completar la sincronización inicial al arrancar (se reintentará después): %s",
            e,
        )

    # 2. Construcción de la aplicación del Bot Móvil
    bot_service = GymTelegramBot(
        settings=settings,
        database=database,
        reader=reader,
        analyzer=analyzer,
    )
    application = bot_service.build_application()

    # 3. Inicialización del bot dentro del loop asyncio
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    logger.info("Bot secundario de Telegram iniciado y escuchando mensajes en polling...")

    # 4. Lanzamiento de sincronización periódica en background
    sync_task: Optional[asyncio.Task] = None
    if settings.AUTO_SYNC_INTERVAL_MINUTES > 0:
        sync_task = asyncio.create_task(
            periodic_sync_task(reader, settings.AUTO_SYNC_INTERVAL_MINUTES)
        )

    # 5. Espera hasta señal de terminación (SIGINT/SIGTERM)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # En caso de entornos que no soporten add_signal_handler
            pass

    try:
        await stop_event.wait()
    finally:
        logger.info("Deteniendo servicios de GymBot...")
        if sync_task:
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass

        if application.updater:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Servicios detenidos de forma limpia.")


def main() -> None:
    try:
        asyncio.run(run_services())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Servicio detenido por el usuario.")
    except Exception as e:
        logger.critical("Error fatal en el servicio de GymBot: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
