/**
 * VoiceRecognition.tsx
 * Composant React/TypeScript connecté au backend FastAPI.
 *
 * Flux :
 *  1. Au montage → fetch /api/v1/languages + /api/v1/engines
 *  2. Démarrer  → getUserMedia + MediaRecorder + visualiseur canvas
 *  3. Arrêter   → MediaRecorder.stop() → Blob → base64 → POST /api/v1/transcribe/base64
 *  4. Résultat  → affichage transcription + métadonnées (confiance, durée)
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  fetchEngines,
  fetchLanguages,
  transcribeAudio,
  checkHealth,
  type Engine,
  type Language,
  type TranscriptionResult,
  type ApiError,
} from "@/infrastructure/api";
import { useMediaRecorder } from "@/presentation/hooks/useMediaRecorder";

// ─── Constants ────────────────────────────────────────────────────────────────

const NUM_BARS = 48;
const FFT_SIZE = 256;
const TEAL_STOPS = ["#9FE1CB", "#5DCAA5", "#1D9E75", "#0F6E56", "#085041"];

type AppStatus =
  | "loading" // fetch initial langues/moteurs
  | "idle" // prêt
  | "recording" // enregistrement en cours
  | "transcribing" // envoi au backend
  | "done" // résultat affiché
  | "error"; // erreur

// ─── useAudioVisualizer ───────────────────────────────────────────────────────

function useAudioVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animRef = useRef<number>(0);
  const idleRef = useRef<number>(0);
  const idleT = useRef(0);

  const barColor = (v: number) => TEAL_STOPS[Math.min(4, Math.floor(v * 5))];

  const drawBars = useCallback(
    (getHeight: (i: number) => number, colorFn: (i: number) => string) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const W = canvas.width;
      const H = canvas.height;
      const bw = Math.floor((W - (NUM_BARS - 1) * 3) / NUM_BARS);
      ctx.clearRect(0, 0, W, H);
      for (let i = 0; i < NUM_BARS; i++) {
        const h = Math.max(4, getHeight(i));
        ctx.fillStyle = colorFn(i);
        ctx.beginPath();
        ctx.roundRect(i * (bw + 3), H - h, bw, h, [3, 3, 0, 0]);
        ctx.fill();
      }
    },
    []
  );

  const drawIdle = useCallback(() => {
    idleT.current += 0.04;
    const t = idleT.current;
    const H = canvasRef.current?.height ?? 96;
    drawBars(
      (i) =>
        (Math.sin(t + (i / NUM_BARS) * Math.PI * 2) * 0.5 + 0.5) * 0.12 * H + 4,
      () => "rgba(136,135,128,0.25)"
    );
    idleRef.current = requestAnimationFrame(drawIdle);
  }, [drawBars]);

  const drawActive = useCallback(() => {
    if (!analyserRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const step = Math.floor(data.length / NUM_BARS);
    const H = canvasRef.current?.height ?? 96;
    drawBars(
      (i) => (data[i * step] / 255) * H,
      (i) => barColor(data[i * step] / 255)
    );
    animRef.current = requestAnimationFrame(drawActive);
  }, [drawBars]);

  const startVisualizer = useCallback(async (): Promise<MediaStream | null> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ac = new AudioContext();
      audioCtxRef.current = ac;
      const an = ac.createAnalyser();
      an.fftSize = FFT_SIZE;
      analyserRef.current = an;
      ac.createMediaStreamSource(stream).connect(an);
      cancelAnimationFrame(idleRef.current);
      drawActive();
      return stream;
    } catch {
      return null;
    }
  }, [drawActive]);

  const stopVisualizer = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    analyserRef.current = null;
    drawIdle();
  }, [drawIdle]);

  useEffect(() => {
    idleRef.current = requestAnimationFrame(drawIdle);
    return () => {
      cancelAnimationFrame(idleRef.current);
      cancelAnimationFrame(animRef.current);
    };
  }, [drawIdle]);

  return { canvasRef, startVisualizer, stopVisualizer };
}

// ─── Icons ────────────────────────────────────────────────────────────────────

const MicIcon = () => (
  <svg
    width={16}
    height={16}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const StopIcon = () => (
  <svg width={16} height={16} viewBox="0 0 24 24" fill="currentColor">
    <rect x="4" y="4" width="16" height="16" rx="2" />
  </svg>
);

const SpinnerIcon = () => (
  <svg
    width={16}
    height={16}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    style={{ animation: "spin 0.9s linear infinite" }}
  >
    <circle cx="12" cy="12" r="10" strokeOpacity={0.25} />
    <path d="M12 2a10 10 0 0 1 10 10" />
  </svg>
);

// ─── MetaBadge ────────────────────────────────────────────────────────────────

function MetaBadge({ label, value }: { label: string; value: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md bg-stone-100 dark:bg-stone-800 text-stone-400 dark:text-stone-500"
      style={{ fontFamily: "'Space Mono', monospace" }}
    >
      <span className="text-stone-300 dark:text-stone-600">{label}</span>
      {value}
    </span>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function VoiceRecognition() {
  const [appStatus, setAppStatus] = useState<AppStatus>("loading");
  const [statusMsg, setStatusMsg] = useState("Connexion au backend…");

  const [languages, setLanguages] = useState<Language[]>([]);
  const [engines, setEngines] = useState<Engine[]>([]);
  const [lang, setLang] = useState("fr-FR");
  const [engine, setEngine] = useState("google");

  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const { canvasRef, startVisualizer, stopVisualizer } = useAudioVisualizer();

  // ─── MediaRecorder : appelé quand l'enregistrement s'arrête ───────────────
  const handleRecordingStop = useCallback(
    async (blob: Blob) => {
      setAppStatus("transcribing");
      setStatusMsg("Transcription en cours…");
      setApiError(null);
      try {
        const res = await transcribeAudio(blob, lang, engine);
        setTranscript(res.transcript);
        setResult(res);
        setAppStatus("done");
        setStatusMsg("Transcription terminée");
      } catch (err) {
        const e = err as ApiError;
        setApiError(e);
        setAppStatus("error");
        setStatusMsg(`Erreur : ${e.message}`);
      }
    },
    [lang, engine]
  );

  const { start: startRecorder, stop: stopRecorder } = useMediaRecorder({
    onStop: handleRecordingStop,
  });

  // ─── Initialisation : health + langues + moteurs ──────────────────────────
  useEffect(() => {
    (async () => {
      const online = await checkHealth();
      setBackendOnline(online);

      if (!online) {
        setAppStatus("error");
        setStatusMsg(
          "Backend inaccessible — démarrez uvicorn sur le port 8000"
        );
        return;
      }

      try {
        const [langs, engs] = await Promise.all([
          fetchLanguages(),
          fetchEngines(),
        ]);
        setLanguages(langs);
        setEngines(engs);
        if (langs.length) setLang(langs[0].code);
        if (engs.length) setEngine(engs[0].id);
        setAppStatus("idle");
        setStatusMsg("Prêt — cliquez sur Démarrer pour enregistrer");
      } catch {
        setAppStatus("error");
        setStatusMsg("Impossible de charger la configuration du backend");
      }
    })();
  }, []);

  // ─── Actions ──────────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    setApiError(null);
    setResult(null);
    setTranscript("");
    const stream = await startVisualizer();
    if (!stream) {
      setAppStatus("error");
      setStatusMsg("Accès microphone refusé");
      return;
    }
    streamRef.current = stream;
    startRecorder(stream);
    setAppStatus("recording");
    setStatusMsg("Enregistrement en cours — parlez maintenant");
  }, [startVisualizer, startRecorder]);

  const stopRecording = useCallback(() => {
    stopRecorder(); // déclenche handleRecordingStop
    stopVisualizer();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, [stopRecorder, stopVisualizer]);

  const handleToggle = () => {
    if (appStatus === "recording") stopRecording();
    else if (
      appStatus === "idle" ||
      appStatus === "done" ||
      appStatus === "error"
    )
      startRecording();
  };

  const handleClear = () => {
    setTranscript("");
    setResult(null);
    setApiError(null);
    if (appStatus === "done" || appStatus === "error") {
      setAppStatus("idle");
      setStatusMsg("Prêt — cliquez sur Démarrer pour enregistrer");
    }
  };

  // ─── Derived UI state ─────────────────────────────────────────────────────
  const isRecording = appStatus === "recording";
  const isTranscribing = appStatus === "transcribing";
  const isLoading = appStatus === "loading" || isTranscribing;
  const canToggle = ["idle", "recording", "done", "error"].includes(appStatus);
  const btnLabel = isRecording
    ? "Arrêter"
    : isTranscribing
    ? "Transcription…"
    : "Démarrer";

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950 flex items-center justify-center p-4">
      {/* Grid background */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(136,135,128,0.1) 1px,transparent 1px),linear-gradient(90deg,rgba(136,135,128,0.1) 1px,transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10 w-full max-w-xl">
        <div className="bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 rounded-2xl p-8 flex flex-col items-center gap-6 shadow-sm">
          {/* ── Header ────────────────────────────────────────────────────── */}
          <div className="text-center w-full">
            <h1
              className="text-2xl font-black tracking-tight text-stone-900 dark:text-stone-100"
              style={{ fontFamily: "'Syne','DM Sans',sans-serif" }}
            >
              Reconnaissance Vocale
            </h1>
            <div className="flex items-center justify-center gap-2 mt-1">
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{
                  background:
                    backendOnline === null
                      ? "#B4B2A9"
                      : backendOnline
                      ? "#1D9E75"
                      : "#E24B4A",
                }}
              />
              <p
                className="text-xs text-stone-400 dark:text-stone-500"
                style={{ fontFamily: "'Space Mono',monospace" }}
              >
                {backendOnline === null
                  ? "Vérification…"
                  : backendOnline
                  ? "Backend connecté"
                  : "Backend hors ligne"}
              </p>
            </div>
          </div>

          {/* ── Canvas visualizer ──────────────────────────────────────────── */}
          <div className="w-full h-28 bg-stone-50 dark:bg-stone-800/60 border border-stone-100 dark:border-stone-700 rounded-xl overflow-hidden flex items-end px-2 pb-2">
            <canvas
              ref={canvasRef}
              width={560}
              height={96}
              className="w-full h-full"
              style={{ display: "block" }}
            />
          </div>

          {/* ── Status row ────────────────────────────────────────────────── */}
          <div className="flex items-center gap-3 w-full">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0 transition-all duration-300"
              style={{
                background: isRecording
                  ? "#1D9E75"
                  : isTranscribing
                  ? "#EF9F27"
                  : "#B4B2A9",
                boxShadow: isRecording
                  ? "0 0 0 4px rgba(29,158,117,0.15)"
                  : isTranscribing
                  ? "0 0 0 4px rgba(239,159,39,0.15)"
                  : "none",
                animation:
                  isRecording || isTranscribing
                    ? "pulse-dot 1.2s infinite"
                    : "none",
              }}
            />
            <span
              className="text-xs text-stone-400 dark:text-stone-500 flex-1 truncate"
              style={{ fontFamily: "'Space Mono',monospace" }}
            >
              {statusMsg}
            </span>

            {/* Sélecteurs langue + moteur */}
            <div className="flex gap-1.5 flex-shrink-0">
              <select
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                disabled={isRecording || isLoading}
                className="text-xs border border-stone-200 dark:border-stone-600 rounded-md px-2 py-1 bg-transparent text-stone-500 dark:text-stone-400 focus:outline-none disabled:opacity-40 cursor-pointer"
                style={{ fontFamily: "'Space Mono',monospace" }}
              >
                {languages.length ? (
                  languages.map((l) => (
                    <option key={l.code} value={l.code}>
                      {l.label}
                    </option>
                  ))
                ) : (
                  <option value="fr-FR">Français</option>
                )}
              </select>

              <select
                value={engine}
                onChange={(e) => setEngine(e.target.value)}
                disabled={isRecording || isLoading}
                className="text-xs border border-stone-200 dark:border-stone-600 rounded-md px-2 py-1 bg-transparent text-stone-500 dark:text-stone-400 focus:outline-none disabled:opacity-40 cursor-pointer"
                style={{ fontFamily: "'Space Mono',monospace" }}
              >
                {engines.length ? (
                  engines.map((eng) => (
                    <option key={eng.id} value={eng.id}>
                      {eng.label.split(" ")[0]}
                    </option>
                  ))
                ) : (
                  <option value="google">Google</option>
                )}
              </select>
            </div>
          </div>

          {/* ── Transcript box ────────────────────────────────────────────── */}
          <div className="w-full min-h-24 bg-stone-50 dark:bg-stone-800/50 border border-stone-100 dark:border-stone-700 rounded-xl px-4 py-3 flex flex-col gap-2">
            {isTranscribing ? (
              <div className="flex items-center gap-3">
                <SpinnerIcon />
                <span
                  className="text-sm text-stone-400 dark:text-stone-500 italic"
                  style={{ fontFamily: "'Space Mono',monospace" }}
                >
                  Envoi au backend…
                </span>
              </div>
            ) : apiError ? (
              <div className="flex flex-col gap-1">
                <p className="text-sm text-red-600 dark:text-red-400 font-medium">
                  {apiError.message}
                </p>
                <p
                  className="text-xs text-stone-400"
                  style={{ fontFamily: "'Space Mono',monospace" }}
                >
                  code: {apiError.code}
                </p>
              </div>
            ) : transcript ? (
              <>
                <p className="text-[15px] leading-relaxed text-stone-800 dark:text-stone-200 break-words">
                  {transcript}
                  {isRecording && (
                    <span
                      className="inline-block w-0.5 h-[1em] bg-stone-400 ml-0.5 align-text-bottom"
                      style={{ animation: "blink 1s step-end infinite" }}
                    />
                  )}
                </p>
                {/* Métadonnées */}
                {result && (
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    <MetaBadge label="moteur" value={result.engine} />
                    <MetaBadge label="langue" value={result.language} />
                    {result.confidence !== null &&
                      result.confidence !== undefined && (
                        <MetaBadge
                          label="confiance"
                          value={`${Math.round(result.confidence * 100)}%`}
                        />
                      )}
                    {result.audio_duration_seconds !== null &&
                      result.audio_duration_seconds !== undefined && (
                        <MetaBadge
                          label="durée audio"
                          value={`${result.audio_duration_seconds.toFixed(1)}s`}
                        />
                      )}
                    <MetaBadge
                      label="traitement"
                      value={`${result.duration_seconds.toFixed(2)}s`}
                    />
                  </div>
                )}
              </>
            ) : (
              <p
                className="text-xs text-stone-300 dark:text-stone-600"
                style={{ fontFamily: "'Space Mono',monospace" }}
              >
                {isRecording
                  ? "Enregistrement… appuyez sur Arrêter pour transcrire"
                  : "Votre transcription apparaîtra ici après l'enregistrement"}
              </p>
            )}
          </div>

          {/* ── Boutons ───────────────────────────────────────────────────── */}
          <div className="flex gap-3 w-full justify-center">
            <button
              onClick={handleToggle}
              disabled={!canToggle || isLoading || backendOnline === false}
              className={[
                "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 min-w-[160px] justify-center",
                "border focus:outline-none active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed",
                isRecording
                  ? "bg-teal-50 dark:bg-teal-950 border-teal-300 dark:border-teal-700 text-teal-700 dark:text-teal-300"
                  : isTranscribing
                  ? "bg-amber-50 dark:bg-amber-950 border-amber-300 dark:border-amber-700 text-amber-700 dark:text-amber-300"
                  : "bg-white dark:bg-stone-800 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-200 hover:bg-stone-50 dark:hover:bg-stone-700",
              ].join(" ")}
              style={{ fontFamily: "'Syne',sans-serif" }}
            >
              {isTranscribing ? (
                <SpinnerIcon />
              ) : isRecording ? (
                <StopIcon />
              ) : (
                <MicIcon />
              )}
              {btnLabel}
            </button>

            <button
              onClick={handleClear}
              disabled={isRecording || isTranscribing}
              className="px-4 py-2.5 rounded-xl text-sm border border-stone-200 dark:border-stone-700 bg-transparent text-stone-400 dark:text-stone-500 hover:bg-stone-100 dark:hover:bg-stone-800 hover:text-stone-700 dark:hover:text-stone-200 transition-all focus:outline-none disabled:opacity-30 disabled:cursor-not-allowed"
              style={{ fontFamily: "'Syne',sans-serif" }}
            >
              Effacer
            </button>
          </div>

          {/* ── Backend hors ligne ────────────────────────────────────────── */}
          {backendOnline === false && (
            <div
              className="w-full text-xs text-center text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-xl px-4 py-3"
              style={{ fontFamily: "'Space Mono',monospace" }}
            >
              Backend inaccessible. Lancez :<br />
              <code className="font-bold">
                uvicorn app.main:app --reload --port 8000
              </code>
            </div>
          )}
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@400;600;800&display=swap');
        @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes spin { to{transform:rotate(360deg)} }
      `}</style>
    </div>
  );
}
