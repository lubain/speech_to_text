"""
Voice Recognition API — FastAPI backend
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import transcription, health

app = FastAPI(
    title="Voice Recognition API",
    description="API de reconnaissance vocale basée sur SpeechRecognition",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# IMPORTANT : le middleware CORS doit être ajouté EN PREMIER pour que les headers
# soient présents même sur les réponses d'erreur (4xx / 5xx).
# Sans ça, une exception 500 sort sans Access-Control-Allow-Origin
# et le navigateur la bloque avant même de voir le code d'erreur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← permissif en dev ; restreignez en prod
    allow_credentials=False,      # doit être False quand allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Handler global d'exceptions non catchées ─────────────────────────────────
# Garantit qu'une erreur interne inattendue retourne du JSON propre
# (et non une page HTML que le frontend ne sait pas parser).
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "internal_error",
                "message": str(exc),
            }
        },
    )

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["health"])
app.include_router(transcription.router, prefix="/api/v1", tags=["transcription"])
