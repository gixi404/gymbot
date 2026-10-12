"""Bot secundario de Telegram para consultas móviles con autenticación estricta por whitelist."""

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

# Límite máximo de caracteres por mensaje en Telegram
TELEGRAM_MESSAGE_MAX_LENGTH = 4000


@asynccontextmanager
async def _send_typing_action(chat, interval: float = 4.0):
    """Mantiene activo el indicador 'escribiendo...' en Telegram mientras se procesa la solicitud."""

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
    """Divide un texto largo en bloques respetando saltos de línea para no exceder los límites de Telegram."""
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
            # Si una sola línea excede el límite máximo
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
    """Servicio del bot de Telegram para atención de consultas desde el móvil."""

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
            raise ValueError("TELEGRAM_BOT_TOKEN no configurado en .env. Crea un bot con @BotFather y añade el token.")

        self.allowed_ids = set(self.settings.ALLOWED_TELEGRAM_USER_IDS)
        if not self.allowed_ids:
            logger.warning(
                "ALLOWED_TELEGRAM_USER_IDS está vacío en .env. "
                "¡El bot ignorará todos los mensajes hasta que agregues tu telegram_user_id!"
            )

    @staticmethod
    def _get_main_keyboard() -> InlineKeyboardMarkup:
        """Teclado interactivo principal con accesos rápidos."""
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
        """Teclado para seleccionar el gráfico a generar."""
        buttons = [
            [
                InlineKeyboardButton("⚖️ Peso Corporal", callback_data="chart_weight"),
                InlineKeyboardButton("🏋️ Press Banca", callback_data="chart_banca"),
            ],
            [
                InlineKeyboardButton("🦵 Sentadilla", callback_data="chart_sentadilla"),
                InlineKeyboardButton("🍑 Hip Thrust", callback_data="chart_hip_thrust"),
            ],
            [
                InlineKeyboardButton("🔙 Menú Principal", callback_data="btn_main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(buttons)

    @staticmethod
    def _get_progression_keyboard() -> InlineKeyboardMarkup:
        """Teclado para consultar la progresión del próximo entrenamiento."""
        buttons = [
            [
                InlineKeyboardButton("Press Banca", callback_data="prog_banca"),
                InlineKeyboardButton("Sentadilla", callback_data="prog_sentadilla"),
            ],
            [
                InlineKeyboardButton("Prensa", callback_data="prog_prensa"),
                InlineKeyboardButton("Hip Thrust", callback_data="prog_hip_thrust"),
            ],
            [
                InlineKeyboardButton("🔙 Menú Principal", callback_data="btn_main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(buttons)

    def _is_authorized(self, update: Update) -> bool:
        """Verifica si el remitente está autorizado según la whitelist."""
        user = update.effective_user
        if not user:
            return False

        if user.id in self.allowed_ids:
            return True

        logger.warning(
            "⚠️ Intento de acceso no autorizado bloqueado: usuario '%s' (ID: %d)",
            user.username or "desconocido",
            user.id,
        )
        return False

    @staticmethod
    async def _safe_reply(message, text: str, parse_mode: str = ParseMode.MARKDOWN, reply_markup=None) -> None:
        """Envía un mensaje intentando formato Markdown, con fallback automático a texto plano."""
        try:
            await message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        except Exception:
            await message.reply_text(text, reply_markup=reply_markup)

    @staticmethod
    async def _safe_edit(message, text: str, parse_mode: str = ParseMode.MARKDOWN, reply_markup=None) -> None:
        """Edita un mensaje intentando formato Markdown, con fallback automático a texto plano."""
        try:
            await message.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        except Exception:
            await message.edit_text(text, reply_markup=reply_markup)

    @staticmethod
    async def _safe_reply_photo(message, photo, caption: str = "", parse_mode: str = ParseMode.MARKDOWN) -> None:
        """Envía una foto intentando formato Markdown en el caption, con fallback a texto plano."""
        try:
            await message.reply_photo(photo=photo, caption=caption, parse_mode=parse_mode)
        except Exception:
            await message.reply_photo(photo=photo, caption=caption)

    @staticmethod
    async def _safe_reply_document(
        message, document, filename: str, caption: str = "", parse_mode: str = ParseMode.MARKDOWN
    ) -> None:
        """Envía un documento intentando formato Markdown en el caption, con fallback a texto plano."""
        try:
            await message.reply_document(document=document, filename=filename, caption=caption, parse_mode=parse_mode)
        except Exception:
            await message.reply_document(document=document, filename=filename, caption=caption)

    async def _background_sync_safe(self) -> None:
        """Sincroniza en segundo plano sin bloquear ni añadir latencia a las consultas."""
        try:
            await self.reader.sync_channel(min_interval_seconds=30.0)
        except Exception as e:
            logger.debug("Sincronización en segundo plano omitida o diferida: %s", e)

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /start."""
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
            "• `/macros` - Estimación nutricional de hoy.\n"
            "• `/proxima [ejercicio]` - Carga sugerida con doble progresión.\n"
            "• `/semana` - Resumen semanal ejecutivo.\n"
            "• `/export` - Descarga tu historial en archivo CSV.\n"
            "• `/sync` | `/stats` | `/clear` | `/help`"
        )

        await self._safe_reply(update.message, welcome_text, reply_markup=self._get_main_keyboard())

    async def _handle_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /menu."""
        if not self._is_authorized(update):
            return
        await self._safe_reply(
            update.message,
            "📋 *Menú de Acciones Rápidas:*",
            reply_markup=self._get_main_keyboard(),
        )

    async def _handle_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /grafico [peso | ejercicio]."""
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
        """Genera y envía el gráfico solicitado."""
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
        """Comando /macros o /hoy."""
        if not self._is_authorized(update):
            return

        async with _send_typing_action(update.message.chat):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.analyzer.analyze_nutrition_today)
            await self._safe_reply(update.message, result)

    async def _handle_progression(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /proxima [ejercicio]."""
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

    async def _handle_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /semana o /resumen."""
        if not self._is_authorized(update):
            return

        async with _send_typing_action(update.message.chat):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.analyzer.generate_weekly_summary)
            await self._safe_reply(update.message, result)

    async def _handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /export o /backup."""
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
            logger.error("Error exportando datos: %s", e)
            await status_msg.edit_text(f"❌ Error al exportar: {e}")

    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manejador centralizado de pulsaciones de botones interactivos."""
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
        elif data == "chart_sentadilla":
            await self._generate_and_send_chart(msg, "sentadilla")
        elif data == "chart_hip_thrust":
            await self._generate_and_send_chart(msg, "hip thrust")
        elif data.startswith("prog_"):
            ex_name = data.replace("prog_", "").replace("_", " ")
            async with _send_typing_action(msg.chat):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self.analyzer.recommend_next_progression(ex_name))
                await self._safe_reply(msg, result)
        elif data == "btn_weekly":
            async with _send_typing_action(msg.chat):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.analyzer.generate_weekly_summary)
                await self._safe_reply(msg, result)
        elif data == "btn_macros":
            async with _send_typing_action(msg.chat):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.analyzer.analyze_nutrition_today)
                await self._safe_reply(msg, result)
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
                logger.error("Error durante sincronización por botón: %s", e)
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
                logger.error("Error exportando CSV: %s", e)
                await msg.reply_text(f"❌ Error al exportar: {e}")
        elif data == "btn_clear":
            self.database.reset_active_memory(str(user_id))
            await msg.reply_text("Memoria reiniciada")

    async def send_proactive_message(self, text: str) -> None:
        """Envía un mensaje proactivo a todos los usuarios de la whitelist."""
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
                logger.error("Error enviando mensaje proactivo a usuario %d: %s", uid, e)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /help."""
        if not self._is_authorized(update):
            return

        help_text = (
            "🏋️ *Ejemplos de consultas que puedes hacerme:*\n\n"
            "• _¿Cuál fue mi mejor press de banca este mes?_\n"
            "• _¿Qué pesos y series hice en sentadillas la última sesión?_\n"
            "• _Calcula mi 1RM estimado en peso muerto según mis últimos registros._\n"
            "• _¿En qué ejercicios tengo notas de molestias articulares o fatiga?_\n"
            "• _¿Cómo ha sido mi progresión en dominadas las últimas 4 semanas?_\n"
            "• 📸 _Adjuntar foto de libreta o captura de app con tu consulta._\n"
            "• 📄 _Adjuntar PDF de rutina o informe de composición corporal._\n\n"
            "⚙️ *Comandos útiles:*\n"
            "• `/sync` - Sincroniza inmediatamente nuevos entrenamientos anotados.\n"
            "• `/clear` - Reinicia la memoria conversacional activa para iniciar una nueva charla.\n"
            "• `/stats` - Consulta el volumen de registros guardados y fechas."
        )
        await self._safe_reply(update.message, help_text)

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /stats."""
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
        """Comando /sync para sincronización bajo demanda."""
        if not self._is_authorized(update):
            return

        status_msg = await update.message.reply_text("🔄 Sincronizando entrenamientos desde Telegram...")
        try:
            # Ejecuta la sincronización efímera MTProto
            new_msgs, total_msgs = await self.reader.sync_channel()
            msg_text = (
                f"✅ *Sincronización exitosa*\n\n"
                f"• Nuevos mensajes descargados: `{new_msgs}`\n"
                f"• Total acumulado en base de datos: `{total_msgs}`"
            )
            await self._safe_edit(status_msg, msg_text)
        except Exception as e:
            logger.error("Error durante sincronización en el bot: %s", e)
            await status_msg.edit_text(f"❌ Error al sincronizar: {str(e)}")

    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /clear o /reset: limpia la memoria activa sin borrar el historial permanente."""
        if not self._is_authorized(update):
            return

        user_id = str(update.effective_user.id)
        self.database.reset_active_memory(user_id)
        await self._safe_reply(update.message, "Memoria reiniciada")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Procesa preguntas ordinarias de texto enviándolas a Gemini."""
        if not self._is_authorized(update):
            return

        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        async with _send_typing_action(update.message.chat):
            # Sincronización desacoplada en segundo plano (0 ms de latencia para el usuario)
            asyncio.create_task(self._background_sync_safe())

            user_id = str(update.effective_user.id)
            active_memory = self.database.get_active_memory(user_id, limit=8)

            placeholder = await update.message.reply_text("✍️ ...")
            full_text = ""
            last_edit_time = time.monotonic()
            edit_interval = 0.8  # Respetar rate limits de Telegram (~1 edit por segundo)

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

                # Guardamos en la sesión activa
                self.database.add_conversation_message(user_id, "user", user_text)
                self.database.add_conversation_message(user_id, "model", final_text)

                # Si excede el tamaño máximo de mensaje, editamos el primero y enviamos los restantes
                chunks = _split_long_message(final_text)
                await self._safe_edit(placeholder, chunks[0])
                for extra_chunk in chunks[1:]:
                    await self._safe_reply(update.message, extra_chunk)

            except Exception as e:
                logger.error("Error en streaming con Gemini: %s", e)
                # Fallback seguro con ejecución síncrona
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
        """Procesa mensajes de voz o archivos de audio transcribiéndolos y respondiendo."""
        if not self._is_authorized(update):
            return

        message = update.message
        voice_or_audio = message.voice or message.audio
        if not voice_or_audio:
            return

        async with _send_typing_action(message.chat):
            try:
                file_obj = await context.bot.get_file(voice_or_audio.file_id)
                audio_bytearray = await file_obj.download_as_bytearray()
                audio_bytes = bytes(audio_bytearray)
            except Exception as e:
                logger.error("Error al descargar archivo de audio: %s", e)
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
                logger.error("Error transcribiendo audio con Gemini: %s", e)
                await message.reply_text(f"❌ Error al transcribir el audio: {str(e)}")
                return

            transcription = (transcription or "").strip()
            if not transcription:
                await message.reply_text("⚠️ No se detectó ninguna voz comprensible en el mensaje de audio.")
                return

            # Sincronización desacoplada en segundo plano (0 ms de latencia)
            asyncio.create_task(self._background_sync_safe())

            user_id = str(update.effective_user.id)
            active_memory = self.database.get_active_memory(user_id, limit=8)

            # Analizamos la consulta transcrita
            try:
                response_text = await loop.run_in_executor(
                    None,
                    lambda: self.analyzer.analyze_query(transcription, conversation_history=active_memory),
                )
            except Exception as e:
                logger.error("Error analizando consulta transcrita: %s", e)
                response_text = f"❌ Ocurrió un error al procesar tu consulta: {str(e)}"

            # Guardamos en el historial persistente de la sesión
            self.database.add_conversation_message(user_id, "user", transcription)
            self.database.add_conversation_message(user_id, "model", response_text)

            chunks = _split_long_message(response_text)
            for chunk in chunks:
                try:
                    await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await message.reply_text(chunk)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Procesa fotos enviadas por el usuario y las analiza con Gemini respetando el contexto."""
        if not self._is_authorized(update):
            return

        message = update.message
        if not message.photo:
            return

        async with _send_typing_action(message.chat):
            # Sincronización desacoplada en segundo plano (0 ms de latencia)
            asyncio.create_task(self._background_sync_safe())

            # La foto con el índice -1 contiene la mayor resolución
            photo_obj = message.photo[-1]
            try:
                file_obj = await context.bot.get_file(photo_obj.file_id)
                img_bytearray = await file_obj.download_as_bytearray()
                img_bytes = bytes(img_bytearray)
            except Exception as e:
                logger.error("Error al descargar foto en Telegram: %s", e)
                await message.reply_text("❌ No se pudo descargar la imagen para su análisis.")
                return

            user_query = (message.caption or "").strip()
            if not user_query:
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
                    ),
                )
            except Exception as e:
                logger.error("Error analizando foto con Gemini: %s", e)
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
        """Procesa documentos (PDF, TXT, CSV, imágenes enviadas como archivo sin comprimir, etc.)."""
        if not self._is_authorized(update):
            return

        message = update.message
        doc = message.document
        if not doc:
            return

        if doc.file_size and doc.file_size > 20 * 1024 * 1024:
            await message.reply_text("❌ El archivo supera el límite de 20 MB permitido por Telegram.")
            return

        async with _send_typing_action(message.chat):
            # Sincronización desacoplada en segundo plano (0 ms de latencia)
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
                logger.error("Error al descargar documento en Telegram: %s", e)
                await message.reply_text(f"❌ No se pudo descargar el archivo '{file_name}'.")
                return

            user_query = (message.caption or "").strip()
            if not user_query:
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
                    ),
                )
            except Exception as e:
                logger.error("Error analizando documento con Gemini: %s", e)
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
        """Captura y registra excepciones no controladas o fallos transitorios de red."""
        logger.error("Excepción en bot de Telegram: %s", context.error)

    def build_application(self) -> Application:
        """Construye y configura la aplicación de python-telegram-bot."""
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=20.0,
            read_timeout=30.0,
            write_timeout=20.0,
        )
        app = ApplicationBuilder().token(self.settings.TELEGRAM_BOT_TOKEN).request(request).build()

        app.add_error_handler(self._error_handler)
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("help", self._handle_help))
        app.add_handler(CommandHandler("menu", self._handle_menu))
        app.add_handler(CommandHandler("stats", self._handle_stats))
        app.add_handler(CommandHandler("sync", self._handle_sync))
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
        """Inicia el bot en modo polling continuo."""
        logger.info("Iniciando servicio de bot secundario de Telegram...")
        app = self.build_application()
        app.run_polling(drop_pending_updates=True)
