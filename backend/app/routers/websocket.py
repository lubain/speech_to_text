"""
Router WebSocket — transcription temps réel.

Endpoint : ws://host/api/v1/ws/transcribe

Protocole :
  Client → Serveur :
    1. Message JSON  : {"type":"config","language":"fr-FR","engine":"google"}
    2. Messages bin  : chunks PCM bruts (int16, mono, 16kHz)
    3. Message JSON  : {"type":"finalize"}   → demande un résultat final
    4. Message JSON  : {"type":"pause"}      → vide le buffer
    5. Message JSON  : {"type":"ping"}       → keepalive

  Serveur → Client :
    {"type":"ready"}                          → session prête
    {"type":"interim","transcript":"..."}     → résultat partiel
    {"type":"final","transcript":"..."}       → résultat final
    {"type":"error","code":"...","message"} → erreur
    {"type":"pong"}                           → réponse keepalive
"""

import json
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.realtime import RealtimeTranscriber

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected: %s", websocket.client)

    transcriber: RealtimeTranscriber | None = None
    transcribe_task: asyncio.Task | None = None

    async def send_json(data: dict):
        try:
            await websocket.send_json(data)
        except Exception:
            pass  # connexion déjà fermée

    async def transcription_loop():
        """Tourne en arrière-plan, transcrit périodiquement le buffer."""
        while True:
            await asyncio.sleep(0.5)
            if transcriber and transcriber.should_transcribe():
                # La transcription est bloquante → thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, transcriber.transcribe
                )
                if result:
                    await send_json(result)

    try:
        # ── Boucle de réception des messages ──────────────────────────────
        while True:
            message = await websocket.receive()

            # ── Message binaire : chunk PCM audio ─────────────────────────
            if "bytes" in message and message["bytes"]:
                if transcriber is None:
                    # Pas encore configuré → on ignore
                    continue
                transcriber.push_chunk(message["bytes"])

            # ── Message texte : commande JSON ──────────────────────────────
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_json({"type": "error", "code": "invalid_json", "message": "JSON invalide"})
                    continue

                msg_type = data.get("type", "")

                if msg_type == "config":
                    # Initialise ou reconfigure le transcriber
                    language = data.get("language", "fr-FR")
                    engine   = data.get("engine",   "google")
                    transcriber = RealtimeTranscriber(language=language, engine=engine)

                    # Lance la boucle de transcription en arrière-plan
                    if transcribe_task:
                        transcribe_task.cancel()
                    transcribe_task = asyncio.create_task(transcription_loop())

                    await send_json({
                        "type":     "ready",
                        "language": language,
                        "engine":   engine,
                    })
                    logger.info("Session configured: lang=%s engine=%s", language, engine)

                elif msg_type == "finalize":
                    # Transcrit et vide le buffer → résultat final
                    if transcriber:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, transcriber.finalize
                        )
                        if result:
                            await send_json(result)

                elif msg_type == "pause":
                    # Vide le buffer sans transcrire (silence long)
                    if transcriber:
                        transcriber.reset_buffer()

                elif msg_type == "ping":
                    await send_json({"type": "pong"})

                else:
                    await send_json({
                        "type":    "error",
                        "code":    "unknown_command",
                        "message": f"Commande inconnue : {msg_type}",
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", websocket.client)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await send_json({"type": "error", "code": "internal_error", "message": str(e)})
    finally:
        if transcribe_task:
            transcribe_task.cancel()
