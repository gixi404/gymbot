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

    # 3. Inteligencia Artificial (Google GenAI)
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash"

    # 4. Persistencia Local y Sincronización
    DATABASE_PATH: Path = Path("gymbot.db")
    AUTO_SYNC_INTERVAL_MINUTES: int = 60  # 0 desactiva la sincronización periódica

    @property
    def workout_chat_ids(self) -> list[int | str]:
        """Devuelve una lista unificada de todos los IDs de chat configurados."""
        if isinstance(self.TELEGRAM_WORKOUT_CHAT_ID, list):
            return self.TELEGRAM_WORKOUT_CHAT_ID
        return [self.TELEGRAM_WORKOUT_CHAT_ID]

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
