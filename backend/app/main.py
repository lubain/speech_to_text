"""
Voice Recognition API — FastAPI backend
Utilise le package SpeechRecognition pour la transcription audio.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health
from app.routers import transcription

app = FastAPI(
    title="Voice Recognition API",
    description="API de reconnaissance vocale basée sur SpeechRecognition",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["health"])
app.include_router(transcription.router, prefix="/api/v1", tags=["transcription"])
