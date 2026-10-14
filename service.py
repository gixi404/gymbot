#!/usr/bin/env python3
"""Entry point for the GymBot background service (daemon).

Runs the secondary Telegram bot for mobile queries and the optional periodic synchronization.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from ai.analyzer import ATLETA_TZ, GymAnalyzer
from config.settings import get_settings
from storage.database import GymDatabase
from telegram_service.bot_service import GymTelegramBot
from telegram_service.mtproto_reader import TelegramWorkoutReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gymbot.service")

# httpx logs full request URLs at INFO; Telegram's bot token is embedded in the URL.
logging.getLogger("httpx").setLevel(logging.WARNING)


async def periodic_sync_task(reader: TelegramWorkoutReader, interval_minutes: int) -> None:
    """Background task to synchronize periodically in a passive way."""
    if interval_minutes <= 0:
        logger.info("Automatic periodic synchronization disabled (interval <= 0).")
        return

    interval_seconds = interval_minutes * 60
    logger.info(
        "Automatic periodic synchronization enabled (every %d minutes).",
        interval_minutes,
    )

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logger.info("Running automatic synchronization cycle...")
            new_msgs, total = await reader.sync_channel()
            logger.info(
                "Synchronization cycle finished: %d new, %d total.",
                new_msgs,
                total,
            )
        except asyncio.CancelledError:
            logger.info("Periodic synchronization task cancelled.")
            break
        except Exception as e:
            logger.warning("Warning in automatic synchronization cycle: %s", e)


async def weekly_summary_task(bot_service: GymTelegramBot, check_interval_seconds: int = 1800) -> None:
    """Monitors periodically and sends the proactive weekly summary on Sundays from 20:00 onward."""
    logger.info("Proactive weekly summary monitor enabled (Sundays 20:00).")
    last_sent_week: int | None = None

    while True:
        try:
            await asyncio.sleep(check_interval_seconds)
            now = datetime.now(ATLETA_TZ)
            iso_year, iso_week, iso_weekday = now.isocalendar()
            # iso_weekday 7 = Sunday
            if iso_weekday == 7 and now.hour >= 20 and last_sent_week != iso_week:
                logger.info("Generating proactive Sunday weekly summary...")
                summary = bot_service.analyzer.generate_weekly_summary()
                await bot_service.send_proactive_message(f"📢 *Resumen Semanal Proactivo*\n\n{summary}")
                last_sent_week = iso_week
        except asyncio.CancelledError:
            logger.info("Proactive weekly summary monitor cancelled.")
            break
        except Exception as e:
            logger.warning("Error in proactive weekly summary monitor: %s", e)


async def run_services() -> None:
    """Starts the mobile bot and the periodic tasks concurrently."""
    settings = get_settings()

    database = GymDatabase(settings.DATABASE_PATH)
    reader = TelegramWorkoutReader(settings, database)
    analyzer = GymAnalyzer(settings, database)

    # 1. Soft initial synchronization on service startup
    try:
        logger.info("Performing initial synchronization on startup...")
        new_msgs, total_msgs = await reader.sync_channel()
        logger.info(
            "Initial synchronization successful: %d new messages, %d total.",
            new_msgs,
            total_msgs,
        )
    except Exception as e:
        logger.warning(
            "The initial synchronization on startup could not be completed (it will be retried later): %s",
            e,
        )

    # 2. Building the Mobile Bot application
    bot_service = GymTelegramBot(
        settings=settings,
        database=database,
        reader=reader,
        analyzer=analyzer,
    )
    application = bot_service.build_application()

    # 3. Bot initialization inside the asyncio loop
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    logger.info("Secondary Telegram bot started and listening for messages via polling...")

    # 4. Launching periodic synchronization and proactive summary in the background
    sync_task: asyncio.Task | None = None
    if settings.AUTO_SYNC_INTERVAL_MINUTES > 0:
        sync_task = asyncio.create_task(periodic_sync_task(reader, settings.AUTO_SYNC_INTERVAL_MINUTES))

    weekly_task = asyncio.create_task(weekly_summary_task(bot_service))

    # 5. Wait until a termination signal (SIGINT/SIGTERM)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping GymBot services...")
        if sync_task:
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass

        if weekly_task:
            weekly_task.cancel()
            try:
                await weekly_task
            except asyncio.CancelledError:
                pass

        if application.updater:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Services stopped cleanly.")


def main() -> None:
    try:
        asyncio.run(run_services())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Service stopped by the user.")
    except Exception as e:
        logger.critical("Fatal error in the GymBot service: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
