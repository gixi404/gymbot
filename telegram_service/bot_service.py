"""Bot secundario de Telegram para consultas móviles con autenticación estricta por whitelist."""

import asyncio
import html
import logging
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ai.analyzer import GymAnalyzer
from config.settings import Settings, get_settings
from storage.database import GymDatabase
from telegram_service.mtproto_reader import TelegramWorkoutReader

logger = logging.getLogger(__name__)

# Límite máximo de caracteres por mensaje en Telegram
TELEGRAM_MESSAGE_MAX_LENGTH = 4000


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
        settings: Optional[Settings] = None,
        database: Optional[GymDatabase] = None,
        reader: Optional[TelegramWorkoutReader] = None,
        analyzer: Optional[GymAnalyzer] = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or GymDatabase(self.settings.DATABASE_PATH)
        self.reader = reader or TelegramWorkoutReader(self.settings, self.database)
        self.analyzer = analyzer or GymAnalyzer(self.settings, self.database)

        if not self.settings.TELEGRAM_BOT_TOKEN:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN no configurado en .env. "
                "Crea un bot con @BotFather y añade el token."
            )

        self.allowed_ids = set(self.settings.ALLOWED_TELEGRAM_USER_IDS)
        if not self.allowed_ids:
            logger.warning(
                "ALLOWED_TELEGRAM_USER_IDS está vacío en .env. "
                "¡El bot ignorará todos los mensajes hasta que agregues tu telegram_user_id!"
            )

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
    async def _safe_reply(message, text: str, parse_mode: str = ParseMode.MARKDOWN) -> None:
        """Envía un mensaje intentando formato Markdown, con fallback automático a texto plano."""
        try:
            await message.reply_text(text, parse_mode=parse_mode)
        except Exception:
            await message.reply_text(text)

    @staticmethod
    async def _safe_edit(message, text: str, parse_mode: str = ParseMode.MARKDOWN) -> None:
        """Edita un mensaje intentando formato Markdown, con fallback automático a texto plano."""
        try:
            await message.edit_text(text, parse_mode=parse_mode)
        except Exception:
            await message.edit_text(text)

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /start."""
        if not self._is_authorized(update):
            return

        raw_name = update.effective_user.first_name if update.effective_user else "Atleta"
        user_name = raw_name.replace("*", "").replace("_", "").replace("`", "")
        stats = self.database.get_stats()

        welcome_text = (
            f"💪 *¡Hola, {user_name}! Bienvenido a GymBot*\n\n"
            "Soy tu asistente personal de entrenamientos impulsado por Gemini 2.5 Flash.\n\n"
            "📊 *Estado Actual:*\n"
            f"• Registros sincronizados: `{stats.total_messages}`\n"
            f"• Última sincronización: `{stats.last_sync_at.strftime('%d/%m/%Y %H:%M') if stats.last_sync_at else 'Nunca'}`\n\n"
            "🚀 *Comandos Disponibles:*\n"
            "• `/sync` - Sincroniza los últimos entrenamientos desde tu chat privado.\n"
            "• `/stats` - Muestra estadísticas de la base de datos local.\n"
            "• `/help` - Ayuda y ejemplos de consultas.\n\n"
            "💬 O simplemente *escríbeme cualquier pregunta* sobre tus cargas, series, 1RM o progresiones."
        )

        await self._safe_reply(update.message, welcome_text)

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
            "• _¿Cómo ha sido mi progresión en dominadas las últimas 4 semanas?_\n\n"
            "⚙️ *Sincronización:*\n"
            "Usa `/sync` cada vez que anotes un nuevo entrenamiento en tu chat privado para tener los datos al día."
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

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Procesa preguntas ordinarias de texto enviándolas a Gemini."""
        if not self._is_authorized(update):
            return

        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        # Notificamos acción de 'escribiendo' al usuario
        await update.message.chat.send_action(action=ChatAction.TYPING)

        # Sincronización incremental previa en tiempo real para capturar notas recién enviadas
        try:
            await self.reader.sync_channel()
        except Exception as e:
            logger.debug("Sincronización incremental previa omitida: %s", e)

        # Ejecutamos el análisis con Gemini en un hilo asíncrono seguro
        loop = asyncio.get_running_loop()
        try:
            response_text = await loop.run_in_executor(
                None, self.analyzer.analyze_query, user_text
            )
        except Exception as e:
            logger.error("Error analizando consulta: %s", e)
            response_text = f"❌ Ocurrió un error al procesar tu consulta: {str(e)}"

        # Dividimos el mensaje si supera el límite de Telegram
        chunks = _split_long_message(response_text)
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                # Fallback sin markdown si Telegram rechaza sintaxis de formato
                await update.message.reply_text(chunk)

    async def _handle_voice_or_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Procesa mensajes de voz o archivos de audio transcribiéndolos y respondiendo."""
        if not self._is_authorized(update):
            return

        message = update.message
        voice_or_audio = message.voice or message.audio
        if not voice_or_audio:
            return

        # Notificamos acción de 'escribiendo' al usuario
        await message.chat.send_action(action=ChatAction.TYPING)

        try:
            file_obj = await context.bot.get_file(voice_or_audio.file_id)
            audio_bytearray = await file_obj.download_as_bytearray()
            audio_bytes = bytes(audio_bytearray)
        except Exception as e:
            logger.error("Error al descargar archivo de audio: %s", e)
            await message.reply_text("❌ No se pudo descargar el audio para su transcripción.")
            return

        mime_type = getattr(voice_or_audio, "mime_type", None) or ("audio/ogg" if message.voice else "audio/mpeg")

        loop = asyncio.get_running_loop()
        try:
            transcription = await loop.run_in_executor(
                None, self.analyzer.transcribe_audio, audio_bytes, mime_type
            )
        except Exception as e:
            logger.error("Error transcribiendo audio con Gemini: %s", e)
            await message.reply_text(f"❌ Error al transcribir el audio: {str(e)}")
            return

        transcription = (transcription or "").strip()
        if not transcription:
            await message.reply_text("⚠️ No se detectó ninguna voz comprensible en el mensaje de audio.")
            return

        # Sincronización incremental previa en tiempo real
        try:
            await self.reader.sync_channel()
        except Exception as e:
            logger.debug("Sincronización incremental previa omitida: %s", e)

        # Analizamos la consulta transcrita
        await message.chat.send_action(action=ChatAction.TYPING)
        try:
            response_text = await loop.run_in_executor(
                None, self.analyzer.analyze_query, transcription
            )
        except Exception as e:
            logger.error("Error analizando consulta transcrita: %s", e)
            response_text = f"❌ Ocurrió un error al procesar tu consulta: {str(e)}"

        chunks = _split_long_message(response_text)
        for chunk in chunks:
            try:
                await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await message.reply_text(chunk)

    def build_application(self) -> Application:
        """Construye y configura la aplicación de python-telegram-bot."""
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
            .build()
        )

        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("help", self._handle_help))
        app.add_handler(CommandHandler("stats", self._handle_stats))
        app.add_handler(CommandHandler("sync", self._handle_sync))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice_or_audio))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        return app

    def run(self) -> None:
        """Inicia el bot en modo polling continuo."""
        logger.info("Iniciando servicio de bot secundario de Telegram...")
        app = self.build_application()
        app.run_polling(drop_pending_updates=True)
