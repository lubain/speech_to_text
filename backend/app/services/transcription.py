"""
Service de transcription — couche métier.
Encapsule speech_recognition et gère les différents moteurs / formats audio.
"""

import io
import time
import logging
import tempfile
import os
from pathlib import Path

import speech_recognition as sr
from pydub import AudioSegment

from app.config import settings
from app.schemas import SupportedEngine

logger = logging.getLogger(__name__)

# Formats audio acceptés → extension temporaire utilisée par pydub
SUPPORTED_FORMATS: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}

# Langues supportées (code BCP-47 → label)
SUPPORTED_LANGUAGES: list[dict] = [
    {"code": "fr-FR", "label": "Français (France)"},
    {"code": "fr-CA", "label": "Français (Canada)"},
    {"code": "en-US", "label": "English (US)"},
    {"code": "en-GB", "label": "English (UK)"},
    {"code": "es-ES", "label": "Español (España)"},
    {"code": "es-MX", "label": "Español (México)"},
    {"code": "de-DE", "label": "Deutsch"},
    {"code": "it-IT", "label": "Italiano"},
    {"code": "pt-BR", "label": "Português (Brasil)"},
    {"code": "ar-SA", "label": "العربية"},
    {"code": "zh-CN", "label": "中文 (简体)"},
    {"code": "ja-JP", "label": "日本語"},
]

# Moteurs disponibles
AVAILABLE_ENGINES: list[dict] = [
    {
        "id": "google",
        "label": "Google Web Speech API",
        "requires_key": False,
        "offline": False,
        "description": "Gratuit, sans configuration. Nécessite une connexion Internet.",
    },
    {
        "id": "sphinx",
        "label": "CMU Sphinx",
        "requires_key": False,
        "offline": True,
        "description": "Reconnaissance hors-ligne. Précision moindre. Nécessite pocketsphinx.",
    },
]


# ─── Exceptions métier ────────────────────────────────────────────────────────

class TranscriptionError(Exception):
    """Erreur générique de transcription."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class UnsupportedFormatError(TranscriptionError):
    pass


class FileTooLargeError(TranscriptionError):
    pass


class UnknownValueError(TranscriptionError):
    """Audio reçu mais aucune parole reconnue."""
    pass


class ServiceUnavailableError(TranscriptionError):
    """API tierce inaccessible."""
    pass


# ─── Conversion audio ─────────────────────────────────────────────────────────

def _to_wav_bytes(audio_bytes: bytes, content_type: str) -> tuple[bytes, float]:
    """
    Convertit n'importe quel format supporté en WAV PCM 16 kHz mono.
    Retourne (wav_bytes, duration_seconds).
    """
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    fmt = SUPPORTED_FORMATS.get(normalized_content_type)
    if fmt is None:
        raise UnsupportedFormatError(
            code="unsupported_format",
            message=f"Format '{content_type}' non supporté. Formats acceptés : {list(SUPPORTED_FORMATS.keys())}",
        )

    # Si c'est déjà du WAV natif, on passe via pydub quand même pour normaliser
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    try:
        segment = AudioSegment.from_file(tmp_in_path, format=fmt)
        duration = len(segment) / 1000.0  # ms → s

        # Normalisation : mono, 16 kHz, 16-bit
        segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        wav_buf = io.BytesIO()
        segment.export(wav_buf, format="wav")
        return wav_buf.getvalue(), duration
    finally:
        os.unlink(tmp_in_path)


# ─── Transcription ────────────────────────────────────────────────────────────

def transcribe(
    audio_bytes: bytes,
    content_type: str,
    language: str = settings.DEFAULT_LANGUAGE,
    engine: SupportedEngine = SupportedEngine(settings.DEFAULT_ENGINE),
) -> dict:
    """
    Transcrit un fichier audio et retourne un dict compatible TranscriptionResponse.

    Args:
        audio_bytes:  Contenu binaire du fichier audio.
        content_type: MIME type du fichier (ex: "audio/webm").
        language:     Code BCP-47 (ex: "fr-FR").
        engine:       Moteur de reconnaissance.

    Returns:
        dict avec les clés transcript, language, engine, duration_seconds,
        confidence, audio_duration_seconds.

    Raises:
        TranscriptionError et sous-classes en cas d'échec.
    """

    if len(audio_bytes) > settings.MAX_AUDIO_SIZE_BYTES:
        raise FileTooLargeError(
            code="file_too_large",
            message=f"Fichier trop volumineux ({len(audio_bytes) / 1e6:.1f} MB). Maximum : {settings.MAX_AUDIO_SIZE_BYTES / 1e6:.0f} MB.",
        )

    start = time.perf_counter()

    # 1. Conversion en WAV normalisé
    logger.info("Conversion audio : content_type=%s, size=%d bytes", content_type, len(audio_bytes))
    wav_bytes, audio_duration = _to_wav_bytes(audio_bytes, content_type)

    # 2. Chargement dans speech_recognition
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = settings.PAUSE_THRESHOLD
    recognizer.non_speaking_duration = settings.NON_SPEAKING_DURATION

    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)

    # 3. Reconnaissance
    transcript = ""
    confidence: float | None = None

    logger.info("Transcription : engine=%s, language=%s", engine.value, language)

    try:
        if engine == SupportedEngine.google:
            result = recognizer.recognize_google(
                audio_data,
                language=language,
                show_all=True,  # pour récupérer le score de confiance
            )
            if isinstance(result, dict) and result.get("alternative"):
                best = result["alternative"][0]
                transcript = best.get("transcript", "")
                confidence = best.get("confidence")
            else:
                # Parfois recognize_google retourne juste une string
                transcript = result if isinstance(result, str) else ""

        elif engine == SupportedEngine.sphinx:
            # Sphinx ne supporte que l'anglais nativement sans modèle custom
            transcript = recognizer.recognize_sphinx(audio_data)

        else:
            raise TranscriptionError(
                code="unsupported_engine",
                message=f"Moteur '{engine}' non disponible sur ce serveur.",
            )

    except sr.UnknownValueError:
        raise UnknownValueError(
            code="no_speech_detected",
            message="Aucune parole détectée dans l'audio. Vérifiez le signal ou réessayez.",
        )
    except sr.RequestError as exc:
        raise ServiceUnavailableError(
            code="service_unavailable",
            message=f"Le service de reconnaissance est inaccessible : {exc}",
        )

    elapsed = time.perf_counter() - start

    return {
        "transcript": transcript.strip(),
        "language": language,
        "engine": engine.value,
        "duration_seconds": round(elapsed, 3),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "audio_duration_seconds": round(audio_duration, 2),
    }
