// Gère le canvas animé + capture audio partagée.
//
// ARCHITECTURE CORRIGÉE :
//   Un seul getUserMedia() + un seul AudioContext partagé entre :
//     - le visualiseur (AnalyserNode → canvas)
//     - le recorder WAV (ScriptProcessorNode → PCM chunks)
//   Cela évite les conflits de stream sur Windows/Chrome.

import { useRef, useCallback, useEffect } from "react";

const NUM_BARS = 48;
const TEAL = ["#9FE1CB", "#5DCAA5", "#1D9E75", "#0F6E56", "#085041"];

export type AudioNodes = {
  stream: MediaStream;
  audioCtx: AudioContext;
  sourceNode: MediaStreamAudioSourceNode;
};

export function useAudioVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animRef = useRef<number>(0);
  const idleRef = useRef<number>(0);
  const idleT = useRef(0);
  // Référence vers les nodes audio partagés
  const audioNodesRef = useRef<AudioNodes | null>(null);

  // ── Dessin ────────────────────────────────────────────────────────────────

  const paint = useCallback(
    (getH: (i: number) => number, getC: (i: number) => string) => {
      const cv = canvasRef.current;
      if (!cv) return;
      const cx = cv.getContext("2d");
      if (!cx) return;
      const W = cv.width;
      const H = cv.height;
      const bw = Math.floor((W - (NUM_BARS - 1) * 3) / NUM_BARS);
      cx.clearRect(0, 0, W, H);
      for (let i = 0; i < NUM_BARS; i++) {
        const h = Math.max(4, getH(i));
        cx.fillStyle = getC(i);
        cx.beginPath();
        cx.roundRect(i * (bw + 3), H - h, bw, h, [3, 3, 0, 0]);
        cx.fill();
      }
    },
    [],
  );

  const drawIdle = useCallback(() => {
    idleT.current += 0.04;
    const t = idleT.current;
    const H = canvasRef.current?.height ?? 96;
    paint(
      (i) =>
        (Math.sin(t + (i / NUM_BARS) * Math.PI * 2) * 0.5 + 0.5) * 0.14 * H + 4,
      () => "rgba(136,135,128,0.25)",
    );
    idleRef.current = requestAnimationFrame(drawIdle);
  }, [paint]);

  const drawActive = useCallback(() => {
    if (!analyserRef.current) return;
    const d = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(d);
    const step = Math.floor(d.length / NUM_BARS);
    const H = canvasRef.current?.height ?? 96;
    paint(
      (i) => (d[i * step] / 255) * H,
      (i) => TEAL[Math.min(4, Math.floor((d[i * step] / 255) * 5))],
    );
    animRef.current = requestAnimationFrame(drawActive);
  }, [paint]);

  // ── Start : crée UN SEUL stream + AudioContext, retourne les nodes partagés

  const startViz = useCallback(async (): Promise<AudioNodes | null> => {
    try {
      // Demande le micro une seule fois pour tout (visuel + recording)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });

      // AudioContext à 16 kHz — bonne résolution pour la reconnaissance vocale
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const sourceNode = audioCtx.createMediaStreamSource(stream);

      // Analyser pour le visualiseur
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512; // plus de résolution fréquentielle
      analyser.smoothingTimeConstant = 0.75;
      sourceNode.connect(analyser);
      analyserRef.current = analyser;

      const nodes: AudioNodes = { stream, audioCtx, sourceNode };
      audioNodesRef.current = nodes;

      // Lance l'animation active
      cancelAnimationFrame(idleRef.current);
      drawActive();

      return nodes;
    } catch (err) {
      console.error("getUserMedia error:", err);
      return null;
    }
  }, [drawActive]);

  // ── Stop : ferme tout

  const stopViz = useCallback(() => {
    cancelAnimationFrame(animRef.current);

    if (audioNodesRef.current) {
      const { stream, audioCtx, sourceNode } = audioNodesRef.current;
      sourceNode.disconnect();
      audioCtx.close();
      stream.getTracks().forEach((t) => t.stop());
      audioNodesRef.current = null;
    }

    analyserRef.current = null;
    drawIdle();
  }, [drawIdle]);

  // ── Idle animation au montage

  useEffect(() => {
    idleRef.current = requestAnimationFrame(drawIdle);
    return () => {
      cancelAnimationFrame(idleRef.current);
      cancelAnimationFrame(animRef.current);
    };
  }, [drawIdle]);

  return { canvasRef, startViz, stopViz };
}
