"""Secondary Telegram bot for mobile queries with strict whitelist authentication."""

import asyncio
import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from ai.analyzer import GymAnalyzer
from config.settings import Settings, get_settings
from storage.database import GymDatabase
from telegram_service.mtproto_reader import TelegramWorkoutReader
from visualization.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)

# Maximum characters per Telegram message
TELEGRAM_MESSAGE_MAX_LENGTH = 4000

# Telegram Bot API download limit (bytes). Enforced before processing attachments.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def _send_typing_action(chat, interval: float = 4.0):
    """Keeps the Telegram 'typing...' indicator active while the request is processed."""

    async def _keep_typing():
        try:
            while True:
                await chat.send_action(action=ChatAction.TYPING)
                await asyncio.sleep(interval)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception:
            pass

    task = asyncio.create_task(_keep_typing())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, GeneratorExit):
            pass


def _split_long_message(text: str, max_length: int = TELEGRAM_MESSAGE_MAX_LENGTH) -> list[str]:
    """Splits long text into chunks on line boundaries so Telegram limits are not exceeded."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk: list[str] = []
    current_length = 0

    for line in lines:
        line_len = len(line) + 1
        if current_length + line_len > max_length:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            # If a single line exceeds the maximum length
            if len(line) > max_length:
                for i in range(0, len(line), max_length):
                    chunks.append(line[i : i + max_length])
            else:
                current_chunk.append(line)
                current_length += line_len
        else:
            current_chunk.append(line)
            current_length += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


class GymTelegramBot:
    """Telegram bot service that handles queries from mobile clients."""

    def __init__(
        self,
        settings: Settings | None = None,
        database: GymDatabase | None = None,
        reader: TelegramWorkoutReader | None = None,
        analyzer: GymAnalyzer | None = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        self.reader = reader or TelegramWorkoutReader(self.settings, self.database)
        self.analyzer = analyzer or GymAnalyzer(self.settings, self.database)
        self._app: Application | None = None

        if not self.settings.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env. Create a bot with @BotFather and add the token.")

        self.allowed_ids = set(self.settings.ALLOWED_TELEGRAM_USER_IDS)
        if not self.allowed_ids:
            logger.warning(
                "ALLOWED_TELEGRAM_USER_IDS is empty in .env. "
                "The bot will ignore all messages until you add your telegram_user_id!"
            )

    @staticmethod
    def _get_main_keyboard() -> InlineKeyboardMarkup:
        """Main interactive keyboard with quick actions."""
        buttons = [
            [
                InlineKeyboardButton("📈 Gráficos de Progreso", callback_data="btn_chart_menu"),
                InlineKeyboardButton("📅 Resumen Semanal", callback_data="btn_weekly"),
            ],
            [
                InlineKeyboardButton("🥗 Macros de Hoy", callback_data="btn_macros"),
                InlineKeyboardButton("🎯 Próxima Sesión", callback_data="btn_progression_menu"),
            ],
            [
                InlineKeyboardButton("📊 Estadísticas BD", callback_data="btn_stats"),
                InlineKeyboardButton("🔄 Sincronizar", callback_data="btn_sync"),
            ],
            [
                InlineKeyboardButton("💾 Exportar CSV", callback_data="btn_export"),
                InlineKeyboardButton("🧹 Reiniciar Memoria", callback_data="btn_clear"),
            ],
        ]
        return InlineKeyboardMarkup(buttons)

    @staticmethod
    def _get_chart_keyboard() -> InlineKeyboardMarkup:
        """Keyboard to select the chart to generate."""
        buttons = [
            [
                InlineKeyboardButton("⚖️ Peso Corporal", callback_data="chart_weight"),
                InlineKeyboardButton("🏋️ Press Banca", callback_data="chart_banca"),
            ],
            [
                InlineKeyboardButton("🦵 Prensa", callback_data="chart_prensa"),
                InlineKeyboardButton("🍑 Hip Thrust", callback_data="chart_hip_thrust"),
            ],
            [
                InlineKeyboardButton("🔙 Menú Principal", callback_data="btn_main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(buttons)

    @staticmethod
    def _get_progression_keyboard() -> InlineKeyboardMarkup:
        """Keyboard to query progression for the next workout."""
        buttons = [
            [
                InlineKeyboardButton("Press Banca", callback_data="prog_banca"),
                InlineKeyboardButton("Prensa", callback_data="prog_prensa"),
            ],
            [
                InlineKeyboardButton("Hip Thrust", callback_data="prog_hip_thrust"),
            ],
            [
                InlineKeyboardButton("🔙 Menú Principal", callback_data="btn_main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(buttons)

    def _is_authorized(self, update: Update) -> bool:
        """Checks whether the sender is authorized by the whitelist."""
        user = update.effective_user
        if not user:
            return False

        if user.id in self.allowed_ids:
            return True

        logger.warning(
            "⚠️ Blocked unauthorized access attempt: user '%s' (ID: %d)",
            user.username or "unknown",
            user.id,
        )
        return False

    @staticmethod
    async def _safe_reply(message, text: str, parse_mode: str = ParseMode.MARKDOWN, reply_markup=None) -> None:
        """Sends a message trying Markdown format, with automatic fallback to plain text."""
        try:
            await message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        except Exception:
            await message.reply_text(text, reply_markup=reply_markup)

    @staticmethod
    async def _safe_edit(message, text: str, parse_mode: str = ParseMode.MARKDOWN, reply_markup=None) -> None:
        """Edits a message trying Markdown format, with automatic fallback to plain text."""
        try:
            await message.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        except Exception:
            await message.edit_text(text, reply_markup=reply_markup)

    @staticmethod
    async def _safe_reply_photo(message, photo, caption: str = "", parse_mode: str = ParseMode.MARKDOWN) -> None:
        """Sends a photo trying Markdown format in the caption, with fallback to plain text."""
        try:
            await message.reply_photo(photo=photo, caption=caption, parse_mode=parse_mode)
        except Exception:
            await message.reply_photo(photo=photo, caption=caption)

    @staticmethod
    async def _safe_reply_document(
        message, document, filename: str, caption: str = "", parse_mode: str = ParseMode.MARKDOWN
    ) -> None:
        """Sends a document trying Markdown format in the caption, with fallback to plain text."""
        try:
            await message.reply_document(document=document, filename=filename, caption=caption, parse_mode=parse_mode)
        except Exception:
            await message.reply_document(document=document, filename=filename, caption=caption)

    def _remember(self, user_id: str, user_text: str, bot_reply: str) -> None:
        """Records a command/button exchange in the active conversational memory.

        Only used for analytical responses (macros, progression, weekly summary),
        not for system commands (stats, sync, export, charts).
        """
        if not bot_reply:
            return
        try:
            self.database.add_conversation_message(str(user_id), "user", user_text)
            self.database.add_conversation_message(str(user_id), "model", bot_reply)
        except Exception as e:
            logger.warning("Could not record conversational memory: %s", e)

    async def _background_sync_safe(self) -> None:
        """Syncs in the background without blocking or adding latency to queries."""
        try:
            await self.reader.sync_channel(min_interval_seconds=30.0)
        except Exception as e:
            logger.debug("Background sync skipped or deferred: %s", e)

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /inicio (alias /start)."""
        if not self._is_authorized(update):
            return

        raw_name = update.effective_user.first_name if update.effective_user else "Atleta"
        user_name = raw_name.replace("*", "").replace("_", "").replace("`", "")
        stats = self.database.get_stats()

        welcome_text = (
            f"💪 *¡Hola, {user_name}! Bienvenido a GymBot*\n\n"
            "Tu asistente inteligente de fuerza, nutrición y rendimiento físico.\n\n"
            "📊 *Estado Actual:*\n"
            f"• Registros sincronizados: `{stats.total_messages}`\n"
            f"• Última sincronización: `{stats.last_sync_at.strftime('%d/%m/%Y %H:%M') if stats.last_sync_at else 'Nunca'}`\n\n"
            "⚡ *Accesos Rápidos:* Usa los botones de abajo o escribe tus comandos:\n"
            "• `/grafico [ejercicio|peso]` - Gráfica de evolución en imagen.\n"
            "• `/macros [fecha]` - Macros y calorías de hoy o de una fecha (ej. `12/10/2026`).\n"
            "• `/proxima [ejercicio]` - Carga sugerida con doble progresión.\n"
            "• `/semana` - Resumen semanal ejecutivo.\n"
            "• `/exportar` - Descarga tu historial en archivo CSV.\n"
            "• `/sincronizar` | `/estadisticas` | `/limpiar` | `/ayuda`"
        )

        await self._safe_reply(update.message, welcome_text, reply_markup=self._get_main_keyboard())

    async def _handle_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /menu."""
        if not self._is_authorized(update):
            return
        await self._safe_reply(
            update.message,
            "📋 *Menú de Acciones Rápidas:*",
            reply_markup=self._get_main_keyboard(),
        )

    async def _handle_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /grafico [peso | exercise]."""
        if not self._is_authorized(update):
            return

        args = context.args or []
        if not args:
            await self._safe_reply(
                update.message,
                "📊 *Selecciona qué gráfico deseas generar:*",
                reply_markup=self._get_chart_keyboard(),
            )
            return

        query = " ".join(args).strip().lower()
        await self._generate_and_send_chart(update.message, query)

    async def _generate_and_send_chart(self, message, query: str) -> None:
        """Generates and sends the requested chart."""
        all_messages = self.database.get_messages(reverse=False)

        if "peso" in query:
            buf = ChartGenerator.generate_weight_chart(all_messages)
            if not buf:
                await self._safe_reply(
                    message,
                    "⚠️ No hay suficientes registros de peso para generar la gráfica (se requieren al menos 2 mediciones en PesoCorporal).",
                )
                return
            await self._safe_reply_photo(message, photo=buf, caption="📈 *Evolución de Peso Corporal*")
        else:
            res = ChartGenerator.generate_exercise_chart(query, all_messages)
            if not res:
                await self._safe_reply(
                    message,
                    f"⚠️ No se encontraron suficientes sesiones registradas para generar la gráfica de '{query}'.",
                )
                return
            buf, canonical = res
            await self._safe_reply_photo(message, photo=buf, caption=f"🏋️ *Progresión de Fuerza: {canonical}*")

    async def _handle_macros(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /macros [date] — with no argument analyzes today; with an argument analyzes the given date."""
        if not self._is_authorized(update):
            return

        args = context.args or []
        date_arg = " ".join(args).strip() if args else None

        async with _send_typing_action(update.message.chat):
            loop = asyncio.get_running_loop()
            if date_arg:
                result = await loop.run_in_executor(None, lambda: self.analyzer.analyze_nutrition_by_date(date_arg))
            else:
                result = await loop.run_in_executor(None, self.analyzer.analyze_nutrition_today)
            await self._safe_reply(update.message, result)
        self._remember(str(update.effective_user.id), f"/macros {date_arg or ''}".strip(), result)

    async def _handle_progression(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /proxima [exercise]."""
        if not self._is_authorized(update):
            return

        args = context.args or []
        if not args:
            await self._safe_reply(
                update.message,
                "🎯 *Selecciona el ejercicio para calcular tu próxima carga:*",
                reply_markup=self._get_progression_keyboard(),
            )
            return

        ex_name = " ".join(args).strip()
        async with _send_typing_action(update.message.chat):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.analyzer.recommend_next_progression(ex_name))
            await self._safe_reply(update.message, result)
        self._remember(str(update.effective_user.id), f"/proxima {ex_name}", result)

    async def _handle_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /semana or /resumen."""
        if not self._is_authorized(update):
            return

        async with _send_typing_action(update.message.chat):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.analyzer.generate_weekly_summary)
            await self._safe_reply(update.message, result)
        self._remember(str(update.effective_user.id), "/semana", result)

    async def _handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /exportar (alias /export, /backup)."""
        if not self._is_authorized(update):
            return

        status_msg = await update.message.reply_text("⏳ Generando archivo CSV con tu historial completo...")
        try:
            csv_buf = self.database.export_to_csv()
            now_str = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"gymbot_export_{now_str}.csv"
            await self._safe_reply_document(
                update.message,
                document=csv_buf,
                filename=filename,
                caption="💾 *Historial exportado con éxito* (Codificación UTF-8 compatible con Excel y Hojas de Cálculo).",
            )
            try:
                await status_msg.delete()
            except Exception:
                pass
        except Exception as e:
            logger.error("Error exporting data: %s", e)
            await status_msg.edit_text(f"❌ Error al exportar: {e}")

    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Centralized handler for interactive button presses."""
        query = update.callback_query
        if not query or not query.data:
            return

        user_id = query.from_user.id
        if user_id not in self.allowed_ids:
            await query.answer("Acceso no autorizado", show_alert=True)
            return

        await query.answer()
        data = query.data
        msg = query.message

        if data == "btn_main_menu":
            await self._safe_edit(msg, "📋 *Menú de Acciones Rápidas:*", reply_markup=self._get_main_keyboard())
        elif data == "btn_chart_menu":
            await self._safe_edit(
                msg, "📊 *Selecciona qué gráfico deseas generar:*", reply_markup=self._get_chart_keyboard()
            )
        elif data == "btn_progression_menu":
            await self._safe_edit(
                msg,
                "🎯 *Selecciona el ejercicio para calcular tu próxima carga:*",
                reply_markup=self._get_progression_keyboard(),
            )
        elif data == "chart_weight":
            await self._generate_and_send_chart(msg, "peso")
        elif data == "chart_banca":
            await self._generate_and_send_chart(msg, "banca")
        elif data == "chart_prensa":
            await self._generate_and_send_chart(msg, "prensa")
        elif data == "chart_hip_thrust":
            await self._generate_and_send_chart(msg, "hip thrust")
        elif data.startswith("prog_"):
            ex_name = data.replace("prog_", "").replace("_", " ")
            async with _send_typing_action(msg.chat):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self.analyzer.recommend_next_progression(ex_name))
                await self._safe_reply(msg, result)
            self._remember(str(user_id), f"/proxima {ex_name}", result)
        elif data == "btn_weekly":
            async with _send_typing_action(msg.chat):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.analyzer.generate_weekly_summary)
                await self._safe_reply(msg, result)
            self._remember(str(user_id), "/semana", result)
        elif data == "btn_macros":
            async with _send_typing_action(msg.chat):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.analyzer.analyze_nutrition_today)
                await self._safe_reply(msg, result)
            self._remember(str(user_id), "/macros", result)
        elif data == "btn_stats":
            stats = self.database.get_stats(self.settings.TELEGRAM_WORKOUT_CHAT_ID)
            stats_text = (
                "📈 *Estadísticas de la Base de Datos:*\n\n"
                f"• Total de mensajes registrados: `{stats.total_messages}`\n"
                f"• Primer registro: `{stats.first_message_date.strftime('%d/%m/%Y %H:%M') if stats.first_message_date else 'N/A'}`\n"
                f"• Último registro: `{stats.last_message_date.strftime('%d/%m/%Y %H:%M') if stats.last_message_date else 'N/A'}`\n"
                f"• Última sincronización: `{stats.last_sync_at.strftime('%d/%m/%Y %H:%M') if stats.last_sync_at else 'Pendiente'}`"
            )
            await self._safe_reply(msg, stats_text)
        elif data == "btn_sync":
            status_msg = await msg.reply_text("🔄 Sincronizando entrenamientos desde Telegram...")
            try:
                new_msgs, total_msgs = await self.reader.sync_channel()
                msg_text = (
                    f"✅ *Sincronización exitosa*\n\n"
                    f"• Nuevos mensajes descargados: `{new_msgs}`\n"
                    f"• Total acumulado en base de datos: `{total_msgs}`"
                )
                await self._safe_edit(status_msg, msg_text)
            except Exception as e:
                logger.error("Error during button-triggered sync: %s", e)
                await status_msg.edit_text(f"❌ Error al sincronizar: {str(e)}")
        elif data == "btn_export":
            try:
                csv_buf = self.database.export_to_csv()
                now_str = datetime.now().strftime("%Y%m%d_%H%M")
                await self._safe_reply_document(
                    msg,
                    document=csv_buf,
                    filename=f"gymbot_export_{now_str}.csv",
                    caption="💾 *Historial exportado con éxito*.",
                )
            except Exception as e:
                logger.error("Error exporting CSV: %s", e)
                await msg.reply_text(f"❌ Error al exportar: {e}")
        elif data == "btn_clear":
            self.database.reset_active_memory(str(user_id))
            await msg.reply_text("Memoria reiniciada")

    async def send_proactive_message(self, text: str) -> None:
        """Sends a proactive message to all whitelisted users."""
        if not self._app:
            return
        for uid in self.allowed_ids:
            try:
                chunks = _split_long_message(text)
                for chunk in chunks:
                    try:
                        await self._app.bot.send_message(
                            chat_id=uid,
                            text=chunk,
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    except Exception:
                        await self._app.bot.send_message(
                            chat_id=uid,
                            text=chunk,
                        )
            except Exception as e:
                logger.error("Error sending proactive message to user %d: %s", uid, e)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /ayuda (alias /help)."""
        if not self._is_authorized(update):
            return

        help_text = (
            "🏋️ *Ejemplos de consultas que puedes hacerme:*\n\n"
            "• _¿Cuál fue mi mejor press de banca este mes?_\n"
            "• _¿Qué pesos y series hice en prensa la última sesión?_\n"
            "• _Calcula mi 1RM estimado en peso muerto según mis últimos registros._\n"
            "• _¿En qué ejercicios tengo notas de molestias articulares o fatiga?_\n"
            "• _¿Cómo ha sido mi progresión en dominadas las últimas 4 semanas?_\n"
            "• 📸 _Adjuntar foto de libreta o captura de app con tu consulta._\n"
            "• 📄 _Adjuntar PDF de rutina o informe de composición corporal._\n\n"
            "⚙️ *Comandos útiles:*\n"
            "• `/menu` - Menú de acciones rápidas.\n"
            "• `/macros` - Macros y calorías de hoy.\n"
            "• `/macros 12/10/2026` - Macros de una fecha específica.\n"
            "• `/macros ayer` - Macros del día anterior.\n"
            "• `/proxima press banca` - Carga sugerida para el próximo entrenamiento.\n"
            "• `/grafico prensa` - Gráfica de evolución de un ejercicio (`/grafico peso` para peso corporal).\n"
            "• `/semana` - Resumen semanal ejecutivo.\n"
            "• `/exportar` - Descarga tu historial completo en CSV.\n"
            "• `/sincronizar` - Sincroniza inmediatamente nuevos entrenamientos anotados.\n"
            "• `/limpiar` - Reinicia la memoria conversacional activa para iniciar una nueva charla.\n"
            "• `/estadisticas` - Consulta el volumen de registros guardados y fechas.\n"
            "• `/ayuda` - Muestra esta ayuda."
        )
        await self._safe_reply(update.message, help_text)

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /estadisticas (alias /stats)."""
        if not self._is_authorized(update):
            return

        stats = self.database.get_stats(self.settings.TELEGRAM_WORKOUT_CHAT_ID)
        stats_text = (
            "📈 *Estadísticas de la Base de Datos:*\n\n"
            f"• Total de mensajes registrados: `{stats.total_messages}`\n"
            f"• Primer registro: `{stats.first_message_date.strftime('%d/%m/%Y %H:%M') if stats.first_message_date else 'N/A'}`\n"
            f"• Último registro: `{stats.last_message_date.strftime('%d/%m/%Y %H:%M') if stats.last_message_date else 'N/A'}`\n"
            f"• Última sincronización: `{stats.last_sync_at.strftime('%d/%m/%Y %H:%M') if stats.last_sync_at else 'Pendiente'}`"
        )
        await self._safe_reply(update.message, stats_text)

    async def _handle_sync(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /sincronizar (alias /sync) for on-demand synchronization."""
        if not self._is_authorized(update):
            return

        status_msg = await update.message.reply_text("🔄 Sincronizando entrenamientos desde Telegram...")
        try:
            # Runs the ephemeral MTProto sync
            new_msgs, total_msgs = await self.reader.sync_channel()
            msg_text = (
                f"✅ *Sincronización exitosa*\n\n"
                f"• Nuevos mensajes descargados: `{new_msgs}`\n"
                f"• Total acumulado en base de datos: `{total_msgs}`"
            )
            await self._safe_edit(status_msg, msg_text)
        except Exception as e:
            logger.error("Error during sync in the bot: %s", e)
            await status_msg.edit_text(f"❌ Error al sincronizar: {str(e)}")

    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Command /limpiar (aliases /clear, /reset): clears active memory without deleting the permanent history."""
        if not self._is_authorized(update):
            return

        user_id = str(update.effective_user.id)
        self.database.reset_active_memory(user_id)
        await self._safe_reply(update.message, "Memoria reiniciada")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Processes ordinary text questions by sending them to the model."""
        if not self._is_authorized(update):
            return

        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        async with _send_typing_action(update.message.chat):
            # Decoupled background sync (0 ms latency for the user)
            asyncio.create_task(self._background_sync_safe())

            user_id = str(update.effective_user.id)
            active_memory = self.database.get_active_memory(user_id, limit=8)

            placeholder = await update.message.reply_text("✍️ ...")
            full_text = ""
            last_edit_time = time.monotonic()
            edit_interval = 0.8  # Respect Telegram rate limits (~1 edit per second)

            try:
                async for chunk in self.analyzer.analyze_query_stream_async(
                    user_text,
                    conversation_history=active_memory,
                ):
                    full_text += chunk
                    now = time.monotonic()
                    if now - last_edit_time >= edit_interval and len(full_text) > 10:
                        try:
                            preview = full_text if len(full_text) < 3900 else (full_text[:3900] + "...")
                            await placeholder.edit_text(preview + " ▌")
                            last_edit_time = now
                        except Exception:
                            pass

                final_text = full_text.strip() or "No se obtuvo respuesta del modelo."

                # Store in the active session
                self.database.add_conversation_message(user_id, "user", user_text)
                self.database.add_conversation_message(user_id, "model", final_text)

                # If it exceeds the max message size, edit the first and send the rest
                chunks = _split_long_message(final_text)
                await self._safe_edit(placeholder, chunks[0])
                for extra_chunk in chunks[1:]:
                    await self._safe_reply(update.message, extra_chunk)

            except Exception as e:
                logger.error("Error streaming with the model: %s", e)
                # Safe fallback using synchronous execution
                loop = asyncio.get_running_loop()
                try:
                    fallback_resp = await loop.run_in_executor(
                        None,
                        lambda: self.analyzer.analyze_query(user_text, conversation_history=active_memory),
                    )
                except Exception as ex:
                    fallback_resp = f"❌ Ocurrió un error al procesar tu consulta: {ex}"
                self.database.add_conversation_message(user_id, "user", user_text)
                self.database.add_conversation_message(user_id, "model", fallback_resp)
                await self._safe_edit(placeholder, fallback_resp)

    async def _handle_voice_or_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Processes voice messages or audio files by transcribing and answering them."""
        if not self._is_authorized(update):
            return

        message = update.message
        voice_or_audio = message.voice or message.audio
        if not voice_or_audio:
            return

        if voice_or_audio.file_size and voice_or_audio.file_size > MAX_ATTACHMENT_BYTES:
            await message.reply_text("❌ El audio supera el límite de 20 MB permitido por Telegram.")
            return

        async with _send_typing_action(message.chat):
            try:
                file_obj = await context.bot.get_file(voice_or_audio.file_id)
                audio_bytearray = await file_obj.download_as_bytearray()
                audio_bytes = bytes(audio_bytearray)
            except Exception as e:
                logger.error("Error downloading audio file: %s", e)
                await message.reply_text("❌ No se pudo descargar el audio para su transcripción.")
                return

            if not audio_bytes:
                await message.reply_text("⚠️ El archivo de audio descargado está vacío.")
                return

            mime_type = getattr(voice_or_audio, "mime_type", None) or ("audio/ogg" if message.voice else "audio/mpeg")

            loop = asyncio.get_running_loop()
            try:
                transcription = await loop.run_in_executor(None, self.analyzer.transcribe_audio, audio_bytes, mime_type)
            except Exception as e:
                logger.error("Error transcribing audio with the model: %s", e)
                await message.reply_text(f"❌ Error al transcribir el audio: {str(e)}")
                return

            transcription = (transcription or "").strip()
            if not transcription:
                await message.reply_text("⚠️ No se detectó ninguna voz comprensible en el mensaje de audio.")
                return

            # Decoupled background sync (0 ms latency)
            asyncio.create_task(self._background_sync_safe())

            user_id = str(update.effective_user.id)
            active_memory = self.database.get_active_memory(user_id, limit=8)

            # Analyze the transcribed query with incremental streaming (same as text)
            placeholder = await message.reply_text("✍️ ...")
            full_text = ""
            last_edit_time = time.monotonic()
            edit_interval = 0.8  # Respect Telegram rate limits (~1 edit per second)

            try:
                async for chunk in self.analyzer.analyze_query_stream_async(
                    transcription,
                    conversation_history=active_memory,
                ):
                    full_text += chunk
                    now = time.monotonic()
                    if now - last_edit_time >= edit_interval and len(full_text) > 10:
                        try:
                            preview = full_text if len(full_text) < 3900 else (full_text[:3900] + "...")
                            await placeholder.edit_text(preview + " ▌")
                            last_edit_time = now
                        except Exception:
                            pass

                final_text = full_text.strip() or "No se obtuvo respuesta del modelo."

                # Store in the session's persistent history
                self.database.add_conversation_message(user_id, "user", transcription)
                self.database.add_conversation_message(user_id, "model", final_text)

                chunks = _split_long_message(final_text)
                await self._safe_edit(placeholder, chunks[0])
                for extra_chunk in chunks[1:]:
                    await self._safe_reply(message, extra_chunk)

            except Exception as e:
                logger.error("Error streaming voice: %s", e)
                # Safe fallback using synchronous execution
                try:
                    response_text = await loop.run_in_executor(
                        None,
                        lambda: self.analyzer.analyze_query(transcription, conversation_history=active_memory),
                    )
                except Exception as ex:
                    response_text = f"❌ Ocurrió un error al procesar tu consulta: {ex}"
                self.database.add_conversation_message(user_id, "user", transcription)
                self.database.add_conversation_message(user_id, "model", response_text)
                await self._safe_edit(placeholder, response_text)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Processes user-sent photos and analyzes them with the model respecting context."""
        if not self._is_authorized(update):
            return

        message = update.message
        if not message.photo:
            return

        async with _send_typing_action(message.chat):
            # Decoupled background sync (0 ms latency)
            asyncio.create_task(self._background_sync_safe())

            # Index -1 holds the highest resolution photo
            photo_obj = message.photo[-1]
            if photo_obj.file_size and photo_obj.file_size > MAX_ATTACHMENT_BYTES:
                await message.reply_text("❌ La imagen supera el límite de 20 MB permitido por Telegram.")
                return
            try:
                file_obj = await context.bot.get_file(photo_obj.file_id)
                img_bytearray = await file_obj.download_as_bytearray()
                img_bytes = bytes(img_bytearray)
            except Exception as e:
                logger.error("Error downloading photo from Telegram: %s", e)
                await message.reply_text("❌ No se pudo descargar la imagen para su análisis.")
                return

            user_query = (message.caption or "").strip()
            attachment_default = not user_query
            if attachment_default:
                user_query = "Analiza detalladamente esta imagen en el contexto de mis entrenamientos y directivas."

            user_id = str(update.effective_user.id)
            active_memory = self.database.get_active_memory(user_id, limit=8)
            attachments = [(img_bytes, "image/jpeg")]

            loop = asyncio.get_running_loop()
            try:
                response_text = await loop.run_in_executor(
                    None,
                    lambda: self.analyzer.analyze_query(
                        user_query,
                        conversation_history=active_memory,
                        attachments=attachments,
                        attachment_default=attachment_default,
                    ),
                )
            except Exception as e:
                logger.error("Error analyzing photo with the model: %s", e)
                response_text = f"❌ Ocurrió un error al procesar tu imagen: {str(e)}"

            self.database.add_conversation_message(user_id, "user", f"[Imagen adjunta] {user_query}")
            self.database.add_conversation_message(user_id, "model", response_text)

            chunks = _split_long_message(response_text)
            for chunk in chunks:
                try:
                    await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await message.reply_text(chunk)

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Processes documents (PDF, TXT, CSV, images sent as uncompressed files, etc.)."""
        if not self._is_authorized(update):
            return

        message = update.message
        doc = message.document
        if not doc:
            return

        if doc.file_size and doc.file_size > MAX_ATTACHMENT_BYTES:
            await message.reply_text("❌ El archivo supera el límite de 20 MB permitido por Telegram.")
            return

        async with _send_typing_action(message.chat):
            # Decoupled background sync (0 ms latency)
            asyncio.create_task(self._background_sync_safe())

            file_name = doc.file_name or "documento"
            mime_type = doc.mime_type
            if not mime_type or mime_type == "application/octet-stream":
                guessed, _ = mimetypes.guess_type(file_name)
                mime_type = guessed or "application/octet-stream"

            try:
                file_obj = await context.bot.get_file(doc.file_id)
                doc_bytearray = await file_obj.download_as_bytearray()
                doc_bytes = bytes(doc_bytearray)
            except Exception as e:
                logger.error("Error downloading document from Telegram: %s", e)
                await message.reply_text(f"❌ No se pudo descargar el archivo '{file_name}'.")
                return

            user_query = (message.caption or "").strip()
            attachment_default = not user_query
            if attachment_default:
                user_query = f"Analiza detalladamente este documento ({file_name}) en el contexto de mis entrenamientos y directivas."

            user_id = str(update.effective_user.id)
            active_memory = self.database.get_active_memory(user_id, limit=8)
            attachments = [(doc_bytes, mime_type)]

            loop = asyncio.get_running_loop()
            try:
                response_text = await loop.run_in_executor(
                    None,
                    lambda: self.analyzer.analyze_query(
                        user_query,
                        conversation_history=active_memory,
                        attachments=attachments,
                        attachment_default=attachment_default,
                    ),
                )
            except Exception as e:
                logger.error("Error analyzing document with the model: %s", e)
                response_text = f"❌ Ocurrió un error al procesar el archivo '{file_name}': {str(e)}"

            self.database.add_conversation_message(user_id, "user", f"[Archivo: {file_name}] {user_query}")
            self.database.add_conversation_message(user_id, "model", response_text)

            chunks = _split_long_message(response_text)
            for chunk in chunks:
                try:
                    await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await message.reply_text(chunk)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Captures and logs unhandled exceptions or transient network failures."""
        logger.error("Exception in Telegram bot: %s", context.error)

    async def _post_init(self, application: Application) -> None:
        """Registers the bot command menu shown in Telegram clients."""
        commands = [
            ("inicio", "Inicia el bot y muestra el menú principal"),
            ("menu", "Menú de acciones rápidas"),
            ("grafico", "Gráfica de progreso (peso corporal o ejercicio)"),
            ("macros", "Macros y calorías de hoy o de una fecha"),
            ("proxima", "Carga sugerida para el próximo entrenamiento"),
            ("semana", "Resumen semanal ejecutivo"),
            ("estadisticas", "Estadísticas de la base de datos"),
            ("sincronizar", "Sincroniza nuevos entrenamientos ahora"),
            ("exportar", "Exporta tu historial completo a CSV"),
            ("limpiar", "Reinicia la memoria conversacional activa"),
            ("ayuda", "Ayuda y ejemplos de consultas"),
        ]
        await application.bot.set_my_commands(commands)

    def build_application(self) -> Application:
        """Builds and configures the python-telegram-bot application."""
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=20.0,
            read_timeout=30.0,
            write_timeout=20.0,
        )
        app = (
            ApplicationBuilder()
            .token(self.settings.TELEGRAM_BOT_TOKEN)
            .request(request)
            .post_init(self._post_init)
            .build()
        )

        app.add_error_handler(self._error_handler)
        app.add_handler(CommandHandler("inicio", self._handle_start))
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("ayuda", self._handle_help))
        app.add_handler(CommandHandler("help", self._handle_help))
        app.add_handler(CommandHandler("menu", self._handle_menu))
        app.add_handler(CommandHandler("estadisticas", self._handle_stats))
        app.add_handler(CommandHandler("stats", self._handle_stats))
        app.add_handler(CommandHandler("sincronizar", self._handle_sync))
        app.add_handler(CommandHandler("sync", self._handle_sync))
        app.add_handler(CommandHandler("limpiar", self._handle_clear))
        app.add_handler(CommandHandler("clear", self._handle_clear))
        app.add_handler(CommandHandler("reset", self._handle_clear))
        app.add_handler(CommandHandler("grafico", self._handle_chart))
        app.add_handler(CommandHandler("plot", self._handle_chart))
        app.add_handler(CommandHandler("chart", self._handle_chart))
        app.add_handler(CommandHandler("macros", self._handle_macros))
        app.add_handler(CommandHandler("hoy", self._handle_macros))
        app.add_handler(CommandHandler("nutricion", self._handle_macros))
        app.add_handler(CommandHandler("proxima", self._handle_progression))
        app.add_handler(CommandHandler("siguiente", self._handle_progression))
        app.add_handler(CommandHandler("recomendar", self._handle_progression))
        app.add_handler(CommandHandler("semana", self._handle_weekly))
        app.add_handler(CommandHandler("resumen", self._handle_weekly))
        app.add_handler(CommandHandler("exportar", self._handle_export))
        app.add_handler(CommandHandler("export", self._handle_export))
        app.add_handler(CommandHandler("backup", self._handle_export))
        app.add_handler(CallbackQueryHandler(self._handle_callback_query))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice_or_audio))
        app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        self._app = app
        return app

    def run(self) -> None:
        """Starts the bot in continuous polling mode."""
        logger.info("Starting secondary Telegram bot service...")
        app = self.build_application()
        app.run_polling(drop_pending_updates=True)
