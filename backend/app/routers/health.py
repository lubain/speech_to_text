"""
Router — health check.
Expose aussi ffmpeg_available pour que le frontend adapte son mode d'enregistrement.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

from app.services.transcription import AVAILABLE_ENGINES, FFMPEG_OK

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    engines_available: list[str]
    ffmpeg_available: bool


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    """Vérifie que l'API est opérationnelle et expose les capacités système."""
    engines_ok: list[str] = []

    for eng in AVAILABLE_ENGINES:
        if eng["id"] == "sphinx":
            try:
                import pocketsphinx  # noqa: F401
                engines_ok.append("sphinx")
            except ImportError:
                pass
        else:
            engines_ok.append(eng["id"])

    return HealthResponse(
        status="ok",
        version="1.0.0",
        engines_available=engines_ok,
        ffmpeg_available=FFMPEG_OK,
    )
