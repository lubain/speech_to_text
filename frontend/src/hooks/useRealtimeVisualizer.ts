// Visualiseur canvas branché sur l'AnalyserNode du hook WebSocket.
// Partage le même AudioContext → un seul getUserMedia.

import { useRef, useCallback, useEffect } from "react";

const NUM_BARS = 48;
const TEAL = ["#9FE1CB", "#5DCAA5", "#1D9E75", "#0F6E56", "#085041"];

export function useRealtimeVisualizer(
  isStreaming: boolean,
  analyserRef: React.RefObject<AnalyserNode | null>,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const idleRef = useRef<number>(0);
  const idleT = useRef(0);

  const paint = useCallback(
    (getH: (i: number) => number, getC: (i: number) => string) => {
      const cv = canvasRef.current;
      if (!cv) return;
      const cx = cv.getContext("2d");
      if (!cx) return;
      const W = cv.width,
        H = cv.height;
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
        (Math.sin(t + (i / NUM_BARS) * Math.PI * 2) * 0.5 + 0.5) * 0.12 * H + 4,
      () => "rgba(136,135,128,0.25)",
    );
    idleRef.current = requestAnimationFrame(drawIdle);
  }, [paint]);

  const drawActive = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) {
      idleRef.current = requestAnimationFrame(drawIdle);
      return;
    }
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const step = Math.floor(data.length / NUM_BARS);
    const H = canvasRef.current?.height ?? 96;
    paint(
      (i) => (data[i * step] / 255) * H,
      (i) => TEAL[Math.min(4, Math.floor((data[i * step] / 255) * 5))],
    );
    animRef.current = requestAnimationFrame(drawActive);
  }, [paint, drawIdle, analyserRef]);

  useEffect(() => {
    if (isStreaming) {
      cancelAnimationFrame(idleRef.current);
      drawActive();
    } else {
      cancelAnimationFrame(animRef.current);
      drawIdle();
    }
    return () => {
      cancelAnimationFrame(animRef.current);
      cancelAnimationFrame(idleRef.current);
    };
  }, [isStreaming, drawActive, drawIdle]);

  useEffect(() => {
    idleRef.current = requestAnimationFrame(drawIdle);
    return () => {
      cancelAnimationFrame(idleRef.current);
      cancelAnimationFrame(animRef.current);
    };
  }, [drawIdle]);

  return { canvasRef };
}
