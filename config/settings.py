"""Typed configuration and environment variable validation for GymBot."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Explicitly load the .env file if it exists in the working directory or the project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


class Settings(BaseSettings):
    """GymBot configuration schema based on environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. MTProto credentials (Telegram Client - Passive reading)
    TELEGRAM_API_ID: int
    TELEGRAM_API_HASH: str
    TELEGRAM_SESSION_NAME: str = "gymbot_user"
    TELEGRAM_STRING_SESSION: str = ""
    TELEGRAM_WORKOUT_CHAT_ID: int | str | list[int | str]

    # 2. Secondary Bot credentials (Telegram Bot API - Mobile queries)
    TELEGRAM_BOT_TOKEN: str = ""
    ALLOWED_TELEGRAM_USER_IDS: list[int] = []

    # 3a. Artificial Intelligence - Groq (main text and audio provider)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    # Tiny, cheap model that only decides complexity/groups/time window.
    # (IDs vary by Groq account/plan; verify with `client.models.list()`.)
    GROQ_CLASSIFIER_MODEL: str = "qwen/qwen3.8-27b"
    # Groq model for simple queries (avoids Gemini congestion).
    GROQ_LIGHT_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_AUDIO_MODEL: str = "whisper-large-v3"
    GROQ_MAX_OUTPUT_TOKENS: int = 1024
    # Empty by default: if the light model fails, it jumps straight to DeepSeek.
    # Add IDs here only if you want to retry with other Groq models first.
    GROQ_FALLBACK_MODELS: list[str] = []

    # 3b. Artificial Intelligence - DeepSeek (complex analysis and Groq fallback)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    # "deepseek-chat" was deprecated in the official API; "deepseek-reasoner" adds reasoning.
    DEEPSEEK_MODEL: str = "deepseek-flash"
    DEEPSEEK_MAX_OUTPUT_TOKENS: int = 2048

    # 3c. Query router (Level 1) and execution routing (Level 3)
    # Estimated token threshold above which a "simple" query leaves Groq
    # (protects Groq's free ~6,000 TPM limit).
    ROUTER_TOKEN_THRESHOLD: int = 4200
    # Crossing 2+ groups in a window larger than these days always goes to DeepSeek.
    ROUTER_MULTI_GROUP_DAYS: int = 14
    # Map of logical groups -> real chats (IDs or titles). Needed because old
    # messages in the DB have no chat_title (see ai/router.resolve_group_targets).
    GROUP_CHAT_MAP: dict[str, list[str]] = {}

    # 3c. Artificial Intelligence - Google Gemini (vision and fallback)
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # 4. Local Persistence and Synchronization
    DATABASE_PATH: Path = Path("gymbot.db")
    AUTO_SYNC_INTERVAL_MINUTES: int = 60  # 0 disables periodic synchronization

    @property
    def workout_chat_ids(self) -> list[int | str]:
        """Unified list of chats to synchronize.

        Merges TELEGRAM_WORKOUT_CHAT_ID with all numeric IDs declared in
        GROUP_CHAT_MAP, so it doesn't depend on keeping both fields in sync.
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
        """Converts numeric identifiers to integers and comma-separated lists."""

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
        """Parses comma-separated ID lists or single integers."""
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
        """Ensures the database path is absolute relative to the project root."""
        if not v.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            return project_root / v
        return v

    def is_user_allowed(self, user_id: int) -> bool:
        """Checks whether a telegram_user_id is in the whitelist."""
        return user_id in self.ALLOWED_TELEGRAM_USER_IDS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton instance of the settings."""
    return Settings()
