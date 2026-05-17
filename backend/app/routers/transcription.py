"""
Router — transcription.
Endpoints :
  POST /api/v1/transcribe          → upload fichier audio
  POST /api/v1/transcribe/base64   → audio encodé en base64
  GET  /api/v1/engines             → liste des moteurs disponibles
  GET  /api/v1/languages           → liste des langues supportées
"""

import base64
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config import settings
from app.schemas import (
    EnginesResponse,
    ErrorResponse,
    LanguagesResponse,
    SupportedEngine,
    TranscriptionResponse,
)
from app.services.transcription import (
    AVAILABLE_ENGINES,
    SUPPORTED_FORMATS,
    SUPPORTED_LANGUAGES,
    FileTooLargeError,
    ServiceUnavailableError,
    TranscriptionError,
    UnknownValueError,
    UnsupportedFormatError,
    transcribe,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Helper ───────────────────────────────────────────────────────────────────

def _raise_http(exc: TranscriptionError) -> None:
    """Convertit une TranscriptionError en HTTPException."""
    status_map = {
        "file_too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        "unsupported_format": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "no_speech_detected": status.HTTP_422_UNPROCESSABLE_CONTENT,
        "service_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
        "unsupported_engine": status.HTTP_400_BAD_REQUEST,
    }
    http_status = status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    raise HTTPException(
        status_code=http_status,
        detail={"code": exc.code, "message": exc.message},
    )


# ─── POST /transcribe  (multipart/form-data) ──────────────────────────────────

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
        415: {"model": ErrorResponse, "description": "Format audio non supporté"},
        422: {"model": ErrorResponse, "description": "Aucune parole détectée"},
        503: {"model": ErrorResponse, "description": "Service de reconnaissance indisponible"},
    },
    summary="Transcrire un fichier audio (upload)",
    description=(
        "Accepte un fichier audio en `multipart/form-data`. "
        "Formats supportés : WAV, WebM, OGG, MP3, MP4, FLAC. "
        "Taille max : 25 MB."
    ),
)
async def transcribe_file(
    file: Annotated[UploadFile, File(description="Fichier audio à transcrire")],
    language: Annotated[str, Form(description="Code BCP-47, ex: fr-FR")] = settings.DEFAULT_LANGUAGE,
    engine: Annotated[SupportedEngine, Form(description="Moteur de reconnaissance")] = SupportedEngine(settings.DEFAULT_ENGINE),
):
    content_type = file.content_type or "audio/wav"
    logger.info("POST /transcribe — filename=%s, content_type=%s, lang=%s, engine=%s",
                file.filename, content_type, language, engine)

    audio_bytes = await file.read()

    # Détection du format depuis le nom de fichier si le content-type est générique
    if content_type in ("application/octet-stream", "binary/octet-stream") and file.filename:
        ext_map = {v: k for k, v in {
            "wav": "audio/wav", "webm": "audio/webm", "ogg": "audio/ogg",
            "mp3": "audio/mpeg", "mp4": "audio/mp4", "flac": "audio/flac",
        }.items()}
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        content_type = {v: k for k, v in ext_map.items()}.get(ext, content_type)

    try:
        result = transcribe(audio_bytes, content_type, language, engine)
    except (UnsupportedFormatError, FileTooLargeError, UnknownValueError, ServiceUnavailableError, TranscriptionError) as exc:
        _raise_http(exc)

    return TranscriptionResponse(**result)


# ─── POST /transcribe/base64 ──────────────────────────────────────────────────

class Base64TranscribeRequest(BaseModel):
    audio_base64: str = Field(..., description="Audio encodé en base64")
    content_type: str = Field(default="audio/wav", description="MIME type du fichier")
    language: str = Field(default=settings.DEFAULT_LANGUAGE, description="Code BCP-47")
    engine: SupportedEngine = Field(default=SupportedEngine(settings.DEFAULT_ENGINE))


@router.post(
    "/transcribe/base64",
    response_model=TranscriptionResponse,
    summary="Transcrire un audio encodé en base64",
    description="Utile pour les clients web qui transmettent l'audio via JSON.",
)
async def transcribe_base64(body: Base64TranscribeRequest):
    logger.info("POST /transcribe/base64 — content_type=%s, lang=%s, engine=%s",
                body.content_type, body.language, body.engine)
    try:
        audio_bytes = base64.b64decode(body.audio_base64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_base64", "message": "La chaîne base64 est invalide."},
        )

    try:
        result = transcribe(audio_bytes, body.content_type, body.language, body.engine)
    except (UnsupportedFormatError, FileTooLargeError, UnknownValueError, ServiceUnavailableError, TranscriptionError) as exc:
        _raise_http(exc)

    return TranscriptionResponse(**result)


# ─── GET /engines ─────────────────────────────────────────────────────────────

@router.get(
    "/engines",
    response_model=EnginesResponse,
    summary="Lister les moteurs de reconnaissance disponibles",
)
def list_engines():
    return EnginesResponse(engines=AVAILABLE_ENGINES, default=settings.DEFAULT_ENGINE)


# ─── GET /languages ───────────────────────────────────────────────────────────

@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="Lister les langues supportées",
)
def list_languages():
    return LanguagesResponse(languages=SUPPORTED_LANGUAGES, default=settings.DEFAULT_LANGUAGE)
