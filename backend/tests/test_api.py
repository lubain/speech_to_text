"""
Tests de l'API Voice Recognition.
Lancer avec : pytest tests/ -v
"""

import base64
import io
import wave
import struct
import math
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _generate_sine_wav(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: int = 16000,
) -> bytes:
    """Génère un fichier WAV PCM contenant une sinusoïde (audio silencieux mais valide)."""
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            val = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", val))
    return buf.getvalue()


def _silent_wav(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Génère un WAV silencieux — aucune parole → UnknownValueError attendue."""
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "engines_available" in data
        assert "version" in data


# ─── Engines & Languages ──────────────────────────────────────────────────────

class TestMeta:
    def test_list_engines(self):
        r = client.get("/api/v1/engines")
        assert r.status_code == 200
        data = r.json()
        assert "engines" in data
        assert "default" in data
        assert len(data["engines"]) >= 1

    def test_list_languages(self):
        r = client.get("/api/v1/languages")
        assert r.status_code == 200
        data = r.json()
        assert "languages" in data
        ids = [l["code"] for l in data["languages"]]
        assert "fr-FR" in ids
        assert "en-US" in ids


# ─── Transcription — upload ───────────────────────────────────────────────────

class TestTranscribeFile:
    def test_silent_audio_returns_422(self):
        """Un WAV silencieux doit renvoyer 422 (aucune parole détectée)."""
        wav = _silent_wav(duration=2.0)
        r = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.wav", wav, "audio/wav")},
            data={"language": "fr-FR", "engine": "google"},
        )
        # 422 si Google répond, 503 si pas de réseau, les deux sont valides en CI
        assert r.status_code in (422, 503)

    def test_unsupported_format_returns_415(self):
        r = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert r.status_code == 415
        assert r.json()["detail"]["code"] == "unsupported_format"

    def test_file_too_large_returns_413(self):
        big = b"\x00" * (26 * 1024 * 1024)  # 26 MB > 25 MB limit
        r = client.post(
            "/api/v1/transcribe",
            files={"file": ("big.wav", big, "audio/wav")},
        )
        assert r.status_code == 413
        assert r.json()["detail"]["code"] == "file_too_large"

    def test_valid_wav_response_shape(self):
        """Structure de la réponse pour un WAV valide (même si pas de parole)."""
        wav = _generate_sine_wav()
        r = client.post(
            "/api/v1/transcribe",
            files={"file": ("tone.wav", wav, "audio/wav")},
            data={"language": "en-US", "engine": "google"},
        )
        # On vérifie juste la structure si la requête aboutit
        if r.status_code == 200:
            data = r.json()
            assert "transcript" in data
            assert "language" in data
            assert "engine" in data
            assert "duration_seconds" in data
            assert data["language"] == "en-US"
            assert data["engine"] == "google"


# ─── Transcription — base64 ───────────────────────────────────────────────────

class TestTranscribeBase64:
    def test_invalid_base64_returns_400(self):
        r = client.post(
            "/api/v1/transcribe/base64",
            json={"audio_base64": "NOT_VALID_BASE64!!!!", "content_type": "audio/wav"},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "invalid_base64"

    def test_silent_base64_audio(self):
        wav = _silent_wav()
        encoded = base64.b64encode(wav).decode()
        r = client.post(
            "/api/v1/transcribe/base64",
            json={"audio_base64": encoded, "content_type": "audio/wav", "language": "fr-FR"},
        )
        assert r.status_code in (422, 503)

    def test_unsupported_engine_returns_400(self):
        wav = _silent_wav()
        encoded = base64.b64encode(wav).decode()
        r = client.post(
            "/api/v1/transcribe/base64",
            json={
                "audio_base64": encoded,
                "content_type": "audio/wav",
                "engine": "invalid_engine",
            },
        )
        assert r.status_code == 422  # Pydantic validation error
