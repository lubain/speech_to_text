"""
Router — health check.
"""

import speech_recognition as sr
from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services.transcription import AVAILABLE_ENGINES

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    """Vérifie que l'API et ses dépendances sont opérationnelles."""
    engines_ok: list[str] = []

    # Vérifie les engines disponibles
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
    )
