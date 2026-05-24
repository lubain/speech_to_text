// Gère la connexion WebSocket + le streaming audio PCM vers le backend.
//
// Flux :
//  1. connect()     → ouvre WS + envoie config JSON
//  2. WS "ready"   → startStreaming() : getUserMedia + AudioContext → PCM chunks
//  3. Toutes les CHUNK_MS ms : encode PCM int16 → envoie via WS (binary)
//  4. Réception "interim" / "final" → callbacks
//  5. disconnect()  → envoie "finalize" + ferme WS + arrête micro

import { useRef, useCallback, useState } from "react";
import { API_BASE_URL } from "@/lib/api";

const WS_URL = API_BASE_URL.replace(/^http/, "ws");
const CHUNK_MS = 250; // envoi toutes les 250ms
const SAMPLE_RATE = 16000; // Hz
const FFT_SIZE = 512;

export type RTStatus =
  | "idle"
  | "connecting"
  | "ready"
  | "streaming"
  | "finalizing"
  | "error";

export type RTMessage = {
  type: "interim" | "final" | "error" | "ready" | "pong";
  transcript?: string;
  language?: string;
  engine?: string;
  confidence?: number | null;
  duration?: number;
  code?: string;
  message?: string;
};

type Options = {
  language?: string;
  engine?: string;
  onMessage: (msg: RTMessage) => void;
  onStatusChange?: (status: RTStatus) => void;
};

// ── Encode Float32Array PCM → Int16Array → ArrayBuffer ───────────────────────
function float32ToInt16(input: Float32Array): ArrayBuffer {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.max(-32768, Math.min(32767, Math.round(input[i] * 32767)));
  }
  return output.buffer;
}

export function useRealtimeTranscription({
  language = "fr-FR",
  engine = "google",
  onMessage,
  onStatusChange,
}: Options) {
  const [status, setStatus] = useState<RTStatus>("idle");

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pcmBufferRef = useRef<Float32Array[]>([]);
  const langRef = useRef(language);
  const engineRef = useRef(engine);

  langRef.current = language;
  engineRef.current = engine;

  const updateStatus = useCallback(
    (s: RTStatus) => {
      setStatus(s);
      onStatusChange?.(s);
    },
    [onStatusChange],
  );

  // ── Démarre le streaming audio ────────────────────────────────────────────
  const startStreaming = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const ac = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = ac;

      const source = ac.createMediaStreamSource(stream);

      // Analyser pour le visualiseur
      const analyser = ac.createAnalyser();
      analyser.fftSize = FFT_SIZE;
      analyserRef.current = analyser;
      source.connect(analyser);

      // ScriptProcessor pour capturer le PCM
      const processor = ac.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e) => {
        const data = new Float32Array(e.inputBuffer.getChannelData(0));
        pcmBufferRef.current.push(data);
      };
      source.connect(processor);
      processor.connect(ac.destination);

      // Envoie un chunk PCM toutes les CHUNK_MS ms
      intervalRef.current = setInterval(() => {
        if (
          !wsRef.current ||
          wsRef.current.readyState !== WebSocket.OPEN ||
          pcmBufferRef.current.length === 0
        )
          return;

        // Assemble les chunks en attente
        const chunks = pcmBufferRef.current.splice(0);
        const totalLen = chunks.reduce((s, c) => s + c.length, 0);
        const merged = new Float32Array(totalLen);
        let off = 0;
        for (const c of chunks) {
          merged.set(c, off);
          off += c.length;
        }

        // Encode en int16 et envoie en binaire
        wsRef.current.send(float32ToInt16(merged));
      }, CHUNK_MS);

      updateStatus("streaming");
    } catch (err) {
      console.error("getUserMedia error:", err);
      updateStatus("error");
      onMessage({
        type: "error",
        code: "mic_error",
        message: "Accès microphone refusé",
      });
    }
  }, [onMessage, updateStatus]);

  // ── Connecte le WebSocket ─────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current) return;
    updateStatus("connecting");

    const ws = new WebSocket(`${WS_URL}/api/v1/ws/transcribe`);
    wsRef.current = ws;
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      // Envoie la config initiale
      ws.send(
        JSON.stringify({
          type: "config",
          language: langRef.current,
          engine: engineRef.current,
        }),
      );
    };

    ws.onmessage = async (event) => {
      try {
        const msg: RTMessage = JSON.parse(event.data);
        if (msg.type === "ready") {
          updateStatus("ready");
          await startStreaming();
        }
        onMessage(msg);
      } catch {
        console.error("WS parse error:", event.data);
      }
    };

    ws.onerror = (e) => {
      console.error("WS error:", e);
      updateStatus("error");
      onMessage({
        type: "error",
        code: "ws_error",
        message: "Erreur WebSocket",
      });
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (status !== "error") updateStatus("idle");
    };
  }, [startStreaming, onMessage, updateStatus, status]);

  // ── Demande un résultat final au backend ──────────────────────────────────
  const finalize = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "finalize" }));
      updateStatus("finalizing");
    }
  }, [updateStatus]);

  // ── Déconnecte et nettoie tout ────────────────────────────────────────────
  const disconnect = useCallback(() => {
    // Arrête l'intervalle d'envoi
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Arrête le ScriptProcessor
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }

    // Ferme l'AudioContext et le stream
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    analyserRef.current = null;
    pcmBufferRef.current = [];

    // Ferme le WebSocket
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        /* ignore */
      }
      wsRef.current = null;
    }

    updateStatus("idle");
  }, [updateStatus]);

  // ── Pause (vide le buffer côté serveur) ──────────────────────────────────
  const pause = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "pause" }));
    }
    pcmBufferRef.current = [];
  }, []);

  return {
    status,
    connect,
    disconnect,
    finalize,
    pause,
    analyserRef, // exposé pour le visualiseur
  };
}
