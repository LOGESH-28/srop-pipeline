"""
config.py — Application settings loaded from environment / .env file.
Uses pydantic-settings for validation and type coercion.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    app_name: str = "SROP"
    debug: bool = False

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./srop.db"

    # ── Vector store ──────────────────────────────────────────────────────────
    vector_store_path: str = "./vector_store"


# Singleton – import and use `settings` throughout the application.
settings = Settings()
