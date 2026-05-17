"""
Configuration centralisée via pydantic-settings.
Les valeurs peuvent être surchargées par variables d'environnement ou fichier .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"

    # ── Transcription ────────────────────────────────────────────────────────
    # Moteur par défaut : "google" (gratuit, sans clé) | "sphinx" (hors-ligne)
    DEFAULT_ENGINE: Literal["google", "sphinx"] = "google"
    DEFAULT_LANGUAGE: str = "fr-FR"

    # Taille max du fichier audio uploadé (en octets). Défaut : 25 MB
    MAX_AUDIO_SIZE_BYTES: int = 25 * 1024 * 1024

    # Durée max de silence avant stop automatique (en secondes)
    PAUSE_THRESHOLD: float = 0.8
    NON_SPEAKING_DURATION: float = 0.5

    # ── Clés API optionnelles (engines tiers) ────────────────────────────────
    GOOGLE_CLOUD_CREDENTIALS: str | None = None
    WHISPER_API_KEY: str | None = None          # OpenAI Whisper
    AZURE_SPEECH_KEY: str | None = None
    AZURE_SPEECH_REGION: str | None = None


settings = Settings()
