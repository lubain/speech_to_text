"""
Service de transcription — couche métier.
Encapsule speech_recognition et gère les différents moteurs / formats audio.

Compatibilité Windows :
  - NamedTemporaryFile ouvert avec delete=False ET explicitement fermé AVANT
    que pydub/ffmpeg tente de l'ouvrir (Windows verrouille les fichiers ouverts,
    contrairement à POSIX).
  - Détection de ffmpeg au démarrage ; fallback pur-Python pour les WAV natifs
    (lecture via le module `wave` de la stdlib, sans ffmpeg).
"""

import io
import time
import logging
import tempfile
import os
import shutil
import wave as wave_mod

import speech_recognition as sr
from pydub import AudioSegment

from app.config import settings
from app.schemas import SupportedEngine

logger = logging.getLogger(__name__)

# ─── Détection ffmpeg ─────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None or shutil.which("avconv") is not None

FFMPEG_OK = _ffmpeg_available()
if not FFMPEG_OK:
    logger.warning(
        "ffmpeg introuvable sur ce système. Seuls les fichiers WAV natifs seront "
        "acceptés sans ffmpeg. Pour activer WebM/OGG/MP3 :\n"
        "  Windows : https://www.gyan.dev/ffmpeg/builds/  (ajoutez ffmpeg/bin au PATH)\n"
        "  macOS   : brew install ffmpeg\n"
        "  Linux   : sudo apt install ffmpeg"
    )

# ─── Formats audio acceptés ───────────────────────────────────────────────────

WAV_MIME_TYPES: set[str] = {"audio/wav", "audio/x-wav", "audio/wave"}

SUPPORTED_FORMATS: dict[str, str] = {
    # WAV (décodage pur Python, pas de ffmpeg requis)
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    # WebM — les navigateurs envoient souvent le MIME avec codec précisé
    "audio/webm": "webm",
    "audio/webm;codecs=opus": "webm",
    "audio/webm;codecs=vp8": "webm",
    "audio/webm;codecs=vp9": "webm",
    # OGG
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/ogg;codecs=vorbis": "ogg",
    # MP3 / MPEG
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    # MP4
    "audio/mp4": "mp4",
    "audio/mp4;codecs=mp4a.40.2": "mp4",
    # FLAC
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}

def _normalize_mime(mime: str) -> str:
    """Normalise un MIME type: retire espaces, met en minuscules.
    Ex: 'audio/webm; codecs=opus' → 'audio/webm;codecs=opus'"""
    return mime.lower().replace(" ", "")

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
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

class UnsupportedFormatError(TranscriptionError):
    pass

class FileTooLargeError(TranscriptionError):
    pass

class UnknownValueError(TranscriptionError):
    pass

class ServiceUnavailableError(TranscriptionError):
    pass


# ─── Fallback WAV pur Python (sans ffmpeg) ────────────────────────────────────

def _normalize_wav_python(audio_bytes: bytes) -> tuple[bytes, float]:
    """
    Normalise un WAV en mono 16 kHz 16-bit SANS ffmpeg.
    Utilise la stdlib `wave` + pydub en mode in-memory (pas d'appel subprocess).
    """
    buf_in = io.BytesIO(audio_bytes)
    with wave_mod.open(buf_in) as wf:
        n_channels  = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate  = wf.getframerate()
        n_frames    = wf.getnframes()
        raw_frames  = wf.readframes(n_frames)

    duration = n_frames / frame_rate

    # pydub peut construire un AudioSegment depuis du PCM brut sans ffmpeg
    segment = AudioSegment(
        data=raw_frames,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=n_channels,
    )
    segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    buf_out = io.BytesIO()
    segment.export(buf_out, format="wav")
    return buf_out.getvalue(), duration


# ─── Conversion avec ffmpeg ───────────────────────────────────────────────────

def _to_wav_bytes_ffmpeg(audio_bytes: bytes, fmt: str) -> tuple[bytes, float]:
    """
    Convertit n'importe quel format audio en WAV via pydub/ffmpeg.

    Fix Windows PermissionError [WinError 32] :
      On utilise mkstemp() qui retourne un fd entier.
      On ferme le fd via os.fdopen+close AVANT que pydub ouvre le fichier.
      Windows interdit à un 2e processus d'ouvrir un fichier déjà ouvert.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=f".{fmt}")
    try:
        # Écriture puis fermeture explicite du handle — fichier déverrouillé ensuite
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(audio_bytes)
        # À ce stade le handle est fermé, Windows peut ouvrir le fichier

        segment = AudioSegment.from_file(tmp_path, format=fmt)
        duration = len(segment) / 1000.0

        segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        buf_out = io.BytesIO()
        segment.export(buf_out, format="wav")
        return buf_out.getvalue(), duration

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Routeur de conversion ────────────────────────────────────────────────────

def _to_wav_bytes(audio_bytes: bytes, content_type: str) -> tuple[bytes, float]:
    """
    Choisit la stratégie optimale selon le format et la dispo de ffmpeg.
    WAV → pur Python (pas de ffmpeg requis).
    Autres → ffmpeg obligatoire (erreur claire si absent).

    Le content_type est normalisé (minuscules, sans espaces) avant lookup,
    pour gérer "audio/webm;codecs=opus", "audio/webm; codecs=opus", etc.
    """
    normalized = _normalize_mime(content_type)
    fmt = SUPPORTED_FORMATS.get(normalized)
    if fmt is None:
        raise UnsupportedFormatError(
            code="unsupported_format",
            message=(
                f"Format '{content_type}' non supporté. "
                f"Formats acceptés : {list(SUPPORTED_FORMATS.keys())}"
            ),
        )

    if normalized in WAV_MIME_TYPES:
        try:
            return _normalize_wav_python(audio_bytes)
        except Exception as exc:
            logger.warning("Décodage WAV Python échoué (%s), tentative ffmpeg", exc)
            if not FFMPEG_OK:
                raise UnsupportedFormatError(
                    code="wav_decode_failed",
                    message=f"Impossible de décoder le WAV : {exc}",
                ) from exc
            return _to_wav_bytes_ffmpeg(audio_bytes, fmt)

    # Format non-WAV : ffmpeg requis
    if not FFMPEG_OK:
        raise UnsupportedFormatError(
            code="ffmpeg_required",
            message=(
                f"Le format '{content_type}' nécessite ffmpeg, introuvable sur ce système.\n"
                "• Windows : https://www.gyan.dev/ffmpeg/builds/ → ajoutez ffmpeg\\bin au PATH\n"
                "• macOS   : brew install ffmpeg\n"
                "• Linux   : sudo apt install ffmpeg\n"
                "Sans ffmpeg, seul le format audio/wav est accepté."
            ),
        )

    return _to_wav_bytes_ffmpeg(audio_bytes, fmt)


# ─── Transcription ────────────────────────────────────────────────────────────

def transcribe(
    audio_bytes: bytes,
    content_type: str,
    language: str = settings.DEFAULT_LANGUAGE,
    engine: SupportedEngine = SupportedEngine(settings.DEFAULT_ENGINE),
) -> dict:
    """
    Transcrit un fichier audio et retourne un dict compatible TranscriptionResponse.
    """
    if len(audio_bytes) > settings.MAX_AUDIO_SIZE_BYTES:
        raise FileTooLargeError(
            code="file_too_large",
            message=(
                f"Fichier trop volumineux ({len(audio_bytes) / 1e6:.1f} MB). "
                f"Maximum : {settings.MAX_AUDIO_SIZE_BYTES / 1e6:.0f} MB."
            ),
        )

    start = time.perf_counter()

    logger.info(
        "Conversion audio : content_type=%s, size=%d bytes, ffmpeg=%s",
        content_type, len(audio_bytes), FFMPEG_OK,
    )
    wav_bytes, audio_duration = _to_wav_bytes(audio_bytes, content_type)

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = settings.PAUSE_THRESHOLD
    recognizer.non_speaking_duration = settings.NON_SPEAKING_DURATION

    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)

    transcript = ""
    confidence: float | None = None

    logger.info("Transcription : engine=%s, language=%s", engine.value, language)

    try:
        if engine == SupportedEngine.google:
            result = recognizer.recognize_google(
                audio_data,
                language=language,
                show_all=True,
            )
            if isinstance(result, dict) and result.get("alternative"):
                best = result["alternative"][0]
                transcript = best.get("transcript", "")
                confidence = best.get("confidence")
            else:
                transcript = result if isinstance(result, str) else ""

        elif engine == SupportedEngine.sphinx:
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
