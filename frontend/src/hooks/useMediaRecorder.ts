// Capture audio depuis les AudioNodes partagés créés par useAudioVisualizer.
//
// CORRECTIONS CLÉS :
//  1. Plus de getUserMedia() séparé → reçoit les nodes déjà ouverts
//  2. Mode WAV : ScriptProcessor branché sur le sourceNode existant
//     (même AudioContext que le visualiseur → un seul contexte audio)
//  3. Mode WebM : MediaRecorder sur le stream existant (inchangé)

import { useRef, useCallback } from "react";
import type { AudioNodes } from "./useAudioVisualizer";

type RecorderOptions = {
  onStop: (blob: Blob, mimeType: string) => void;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function getSupportedMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  for (const mime of candidates) {
    if (
      typeof MediaRecorder !== "undefined" &&
      MediaRecorder.isTypeSupported(mime)
    )
      return mime;
  }
  return "";
}

function encodeWav(samples: Int16Array, sampleRate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.byteLength);
  const v = new DataView(buf);
  const w = (off: number, str: string) => {
    for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + samples.byteLength, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true); // chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, samples.byteLength, true);
  new Int16Array(buf, 44).set(samples);
  return new Blob([buf], { type: "audio/wav" });
}

// ── Mode WAV : branchement sur le AudioContext/sourceNode partagé ─────────────
// Retourne une fonction stop() qui assemble et envoie le Blob WAV.

function startWavCapture(
  nodes: AudioNodes,
  onStop: (blob: Blob, mimeType: string) => void,
): () => void {
  const { audioCtx, sourceNode } = nodes;
  const sampleRate = audioCtx.sampleRate; // 16000 Hz
  const BUFFER_SIZE = 4096;

  // ScriptProcessor branché sur le source existant (pas de nouvel AudioContext)
  // eslint-disable-next-line @typescript-eslint/no-deprecated
  const processor = audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
  const chunks: Float32Array[] = [];

  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  sourceNode.connect(processor);
  // Connecter au destination pour que Chrome ne supprime pas le nœud inactif
  processor.connect(audioCtx.destination);

  return () => {
    // Déconnecter le processor uniquement (ne pas fermer l'audioCtx ici,
    // c'est stopViz() qui s'en charge)
    try {
      sourceNode.disconnect(processor);
    } catch {
      /* already disconnected */
    }
    try {
      processor.disconnect();
    } catch {
      /* already disconnected */
    }

    if (chunks.length === 0) {
      console.warn("WAV recorder: no audio chunks captured");
    }

    // Assemble le PCM Float32
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const pcm = new Float32Array(total);
    let off = 0;
    for (const c of chunks) {
      pcm.set(c, off);
      off += c.length;
    }

    // Float32 → Int16
    const int16 = new Int16Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32767)));
    }

    console.log(
      `WAV captured: ${(total / sampleRate).toFixed(2)}s, ${int16.byteLength} bytes`,
    );
    onStop(encodeWav(int16, sampleRate), "audio/wav");
  };
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useMediaRecorder({ onStop }: RecorderOptions) {
  const ffmpegRef = useRef<boolean>(true);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const mimeRef = useRef<string>("");
  const stopWavRef = useRef<(() => void) | null>(null);
  const wavModeRef = useRef(false);

  const setFfmpegAvailable = useCallback((v: boolean) => {
    ffmpegRef.current = v;
  }, []);

  // start() reçoit désormais les AudioNodes partagés, pas un simple MediaStream
  const start = useCallback(
    (nodes: AudioNodes) => {
      chunksRef.current = [];

      if (!ffmpegRef.current) {
        // ── Mode WAV pur JS ──────────────────────────────────────────────
        wavModeRef.current = true;
        stopWavRef.current = startWavCapture(nodes, onStop);
        return;
      }

      // ── Mode MediaRecorder (WebM/OGG, nécessite ffmpeg côté backend) ──
      wavModeRef.current = false;
      const mimeType = getSupportedMimeType();
      mimeRef.current = mimeType;

      const recorder = new MediaRecorder(
        nodes.stream,
        mimeType ? { mimeType } : undefined,
      );
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: mimeRef.current || "audio/webm",
        });
        onStop(blob, mimeRef.current || "audio/webm");
      };

      recorder.start(250);
    },
    [onStop],
  );

  const stop = useCallback(() => {
    if (wavModeRef.current && stopWavRef.current) {
      stopWavRef.current();
      stopWavRef.current = null;
    } else if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  return { start, stop, setFfmpegAvailable };
}
