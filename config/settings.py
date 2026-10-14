"""Configuración tipada y validación de variables de entorno para GymBot."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carga explícita del archivo .env si existe en el directorio de trabajo o en la raíz del proyecto
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


class Settings(BaseSettings):
    """Esquema de configuración de GymBot basado en variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Credenciales MTProto (Telegram Client - Lectura pasiva)
    TELEGRAM_API_ID: int
    TELEGRAM_API_HASH: str
    TELEGRAM_SESSION_NAME: str = "gymbot_user"
    TELEGRAM_STRING_SESSION: str = ""
    TELEGRAM_WORKOUT_CHAT_ID: int | str | list[int | str]

    # 2. Credenciales Bot Secundario (Telegram Bot API - Consultas móviles)
    TELEGRAM_BOT_TOKEN: str = ""
    ALLOWED_TELEGRAM_USER_IDS: list[int] = []

    # 3a. Inteligencia Artificial - Groq (proveedor principal de texto y audio)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    # Modelo diminuto y barato que solo decide complejidad/grupos/ventana temporal.
    # (Los IDs varían por cuenta/plan de Groq; verifica con `client.models.list()`.)
    GROQ_CLASSIFIER_MODEL: str = "qwen/qwen3.8-27b"
    # Modelo Groq para consultas simples (evita la congestión de Gemini).
    GROQ_LIGHT_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_AUDIO_MODEL: str = "whisper-large-v3"
    GROQ_MAX_OUTPUT_TOKENS: int = 1024
    # Vacío por defecto: si el modelo ligero falla, se salta directo a DeepSeek.
    # Añade IDs aquí solo si quieres reintentar con otros modelos de Groq primero.
    GROQ_FALLBACK_MODELS: list[str] = []

    # 3b. Inteligencia Artificial - DeepSeek (análisis complejo y respaldo de Groq)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    # "deepseek-chat" quedó deprecado en la API oficial; "deepseek-reasoner" añade razonamiento.
    DEEPSEEK_MODEL: str = "deepseek-flash"
    DEEPSEEK_MAX_OUTPUT_TOKENS: int = 2048

    # 3c. Enrutador de consultas (Nivel 1) y enrutamiento de ejecución (Nivel 3)
    # Umbral de tokens estimados por encima del cual una consulta "simple" sale de Groq
    # (protege el límite gratuito de ~6.000 TPM de Groq).
    ROUTER_TOKEN_THRESHOLD: int = 4200
    # Cruce de 2+ grupos en una ventana mayor a estos días siempre va a DeepSeek.
    ROUTER_MULTI_GROUP_DAYS: int = 14
    # Mapa de grupos lógicos -> chats reales (IDs o títulos). Necesario porque los
    # mensajes antiguos de la BD no tienen chat_title (ver ai/router.resolve_group_targets).
    GROUP_CHAT_MAP: dict[str, list[str]] = {}

    # 3c. Inteligencia Artificial - Google Gemini (visión y respaldo)
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # 4. Persistencia Local y Sincronización
    DATABASE_PATH: Path = Path("gymbot.db")
    AUTO_SYNC_INTERVAL_MINUTES: int = 60  # 0 desactiva la sincronización periódica

    @property
    def workout_chat_ids(self) -> list[int | str]:
        """Lista unificada de chats a sincronizar.

        Une TELEGRAM_WORKOUT_CHAT_ID con todos los IDs numéricos declarados en
        GROUP_CHAT_MAP, para no depender de mantener ambos campos en sincronía.
        """
        base = (
            self.TELEGRAM_WORKOUT_CHAT_ID
            if isinstance(self.TELEGRAM_WORKOUT_CHAT_ID, list)
            else [self.TELEGRAM_WORKOUT_CHAT_ID]
        )
        ids: list[int | str] = []
        seen: set[str] = set()

        def add(value: int | str) -> None:
            key = str(value)
            if key not in seen:
                seen.add(key)
                ids.append(value)

        for item in base:
            add(item)
        for entries in (self.GROUP_CHAT_MAP or {}).values():
            for entry in entries:
                text = str(entry).strip()
                if text.lstrip("-").isdigit():
                    add(int(text))
        return ids

    @field_validator("TELEGRAM_WORKOUT_CHAT_ID", mode="before")
    @classmethod
    def parse_workout_chat_id(cls, v: Any) -> int | str | list[int | str]:
        """Convierte identificadores numéricos a enteros y listas separadas por comas."""

        def parse_single(item: Any) -> int | str:
            if isinstance(item, (int, float)):
                return int(item)
            s = str(item).strip()
            if s.lstrip("-").isdigit():
                return int(s)
            return s

        if isinstance(v, (list, tuple, set)):
            return [parse_single(x) for x in v if str(x).strip()]
        if isinstance(v, str) and "," in v:
            return [parse_single(x) for x in v.split(",") if x.strip()]
        return parse_single(v)

    @field_validator("ALLOWED_TELEGRAM_USER_IDS", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, v: Any) -> list[int]:
        """Parsea listas de IDs separadas por comas o enteros individuales."""
        if isinstance(v, (list, tuple, set)):
            return [int(x) for x in v if str(x).strip()]
        if isinstance(v, (int, float)):
            return [int(v)]
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                return []
            ids = []
            for item in cleaned.split(","):
                item_clean = item.strip()
                if item_clean:
                    ids.append(int(item_clean))
            return ids
        return []

    @field_validator("DATABASE_PATH", mode="after")
    @classmethod
    def resolve_database_path(cls, v: Path) -> Path:
        """Garantiza que la ruta de la base de datos sea absoluta relativa a la raíz del proyecto."""
        if not v.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            return project_root / v
        return v

    def is_user_allowed(self, user_id: int) -> bool:
        """Verifica si un telegram_user_id se encuentra en la whitelist."""
        return user_id in self.ALLOWED_TELEGRAM_USER_IDS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve una instancia singleton en caché de las configuraciones."""
    return Settings()
