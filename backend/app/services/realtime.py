"""
Service de transcription temps réel via WebSocket.
Architecture :
  - Le client envoie des chunks PCM bruts (16kHz, mono, int16) en binaire
  - Les chunks s'accumulent dans un buffer
  - Toutes les CHUNK_DURATION secondes, on tente une transcription
  - Les résultats (interim/final) sont renvoyés en JSON
"""

import io
import wave
import logging
import time
from collections import deque

import speech_recognition as sr

logger = logging.getLogger(__name__)

# Durée minimale d'audio à accumuler avant de tenter une transcription (secondes)
CHUNK_DURATION   = 1.5
# Durée max du buffer (évite la mémoire infinie en cas de long silence)
MAX_BUFFER_SECS  = 8.0
# Taux d'échantillonnage attendu du client
SAMPLE_RATE      = 16000
# Taille d'un sample int16 en octets
SAMPLE_WIDTH     = 2


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encapsule du PCM brut int16 mono dans un fichier WAV en mémoire."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class RealtimeTranscriber:
    """
    Gère le buffer audio et la transcription pour une session WebSocket.
    Une instance par connexion WebSocket.
    """

    def __init__(self, language: str = "fr-FR", engine: str = "google"):
        self.language   = language
        self.engine     = engine
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold        = 0.5
        self.recognizer.non_speaking_duration  = 0.3
        self.recognizer.energy_threshold       = 300

        # Buffer des chunks PCM reçus depuis le client
        self._buffer: bytearray = bytearray()
        # Timestamp du dernier envoi au recognizer
        self._last_transcribe   = time.monotonic()
        # Dernier texte final envoyé (pour dédup)
        self._last_final        = ""

    @property
    def buffer_duration(self) -> float:
        """Durée en secondes des données dans le buffer."""
        return len(self._buffer) / (SAMPLE_RATE * SAMPLE_WIDTH)

    def push_chunk(self, pcm_chunk: bytes) -> None:
        """Ajoute un chunk PCM au buffer. Tronque si trop long."""
        self._buffer.extend(pcm_chunk)
        # Garde au plus MAX_BUFFER_SECS de données
        max_bytes = int(MAX_BUFFER_SECS * SAMPLE_RATE * SAMPLE_WIDTH)
        if len(self._buffer) > max_bytes:
            # Supprime les données les plus anciennes
            self._buffer = self._buffer[-max_bytes:]

    def should_transcribe(self) -> bool:
        """Retourne True si on a assez d'audio ET assez de temps écoulé."""
        elapsed = time.monotonic() - self._last_transcribe
        return (
            self.buffer_duration >= CHUNK_DURATION
            and elapsed >= CHUNK_DURATION * 0.8
        )

    def transcribe(self) -> dict | None:
        """
        Tente une transcription du buffer actuel.
        Retourne un dict {type, transcript, language, engine} ou None.
        """
        if len(self._buffer) < SAMPLE_RATE * SAMPLE_WIDTH * 0.5:
            return None  # Moins de 0.5s — trop court

        self._last_transcribe = time.monotonic()

        wav_bytes = pcm_to_wav(bytes(self._buffer))

        try:
            with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                audio_data = self.recognizer.record(source)

            if self.engine == "google":
                result = self.recognizer.recognize_google(
                    audio_data,
                    language=self.language,
                    show_all=True,
                )
                if isinstance(result, dict) and result.get("alternative"):
                    transcript = result["alternative"][0].get("transcript", "").strip()
                    confidence = result["alternative"][0].get("confidence")
                elif isinstance(result, str):
                    transcript = result.strip()
                    confidence = None
                else:
                    return None
            else:
                transcript = self.recognizer.recognize_sphinx(audio_data)
                confidence = None

            if not transcript:
                return None

            # Si le texte est identique au dernier final → on ne renvoie pas
            if transcript == self._last_final:
                return None

            # On considère tout résultat comme "interim" (streaming)
            # Un résultat devient "final" quand un silence est détecté (buffer reset)
            return {
                "type":       "interim",
                "transcript": transcript,
                "language":   self.language,
                "engine":     self.engine,
                "confidence": round(confidence, 4) if confidence else None,
                "duration":   round(self.buffer_duration, 2),
            }

        except sr.UnknownValueError:
            # Pas de parole détectée — normal, on continue
            return None
        except sr.RequestError as e:
            logger.error("Recognizer request error: %s", e)
            return {
                "type":    "error",
                "code":    "service_unavailable",
                "message": str(e),
            }
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return None

    def finalize(self) -> dict | None:
        """
        Transcrit le buffer restant et le marque comme final.
        Appelé quand le client signale une pause ou la fin.
        """
        result = self.transcribe()
        if result and result.get("type") == "interim":
            result["type"] = "final"
            self._last_final = result["transcript"]
            # Vide le buffer après un final
            self._buffer.clear()
        return result

    def reset_buffer(self) -> None:
        """Vide le buffer (après une pause détectée côté client)."""
        self._buffer.clear()
