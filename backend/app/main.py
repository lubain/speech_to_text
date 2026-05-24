"""
Voice Recognition API — FastAPI backend
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import transcription, health, websocket

app = FastAPI(
    title="Voice Recognition API",
    description="API de reconnaissance vocale — REST + WebSocket temps réel",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error", "message": str(exc)}},
    )

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router,        tags=["health"])
app.include_router(transcription.router, prefix="/api/v1", tags=["transcription"])
app.include_router(websocket.router,     prefix="/api/v1", tags=["realtime"])
