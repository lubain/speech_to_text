"""
Schémas Pydantic — modèles de requête et de réponse.
"""

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class SupportedEngine(str, Enum):
    google = "google"
    sphinx = "sphinx"


class TranscriptionResponse(BaseModel):
    """Réponse d'une transcription réussie."""

    transcript: str = Field(..., description="Texte transcrit")
    language: str = Field(..., description="Langue utilisée (ex: fr-FR)")
    engine: str = Field(..., description="Moteur de reconnaissance utilisé")
    duration_seconds: float = Field(..., description="Durée de traitement en secondes")
    confidence: float | None = Field(
        default=None,
        description="Score de confiance [0–1] si disponible selon le moteur",
    )
    audio_duration_seconds: float | None = Field(
        default=None, description="Durée de l'audio analysé"
    )


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    engines_available: list[str]


class EnginesResponse(BaseModel):
    engines: list[dict]
    default: str


class LanguagesResponse(BaseModel):
    languages: list[dict]
    default: str
