// ARCHITECTURE AUDIO CORRIGÉE :
//   startViz() → 1 seul getUserMedia + 1 AudioContext → retourne AudioNodes
//   startRecorder(nodes) → branche sur les mêmes nodes (WAV ou WebM)
//   stopRecording() → stopRecorder() puis stopViz() (qui ferme tout)

import { useState, useEffect, useRef, useCallback } from "react";
import {
  checkHealth,
  fetchEngines,
  fetchLanguages,
  transcribeAudio,
  type Engine,
  type Language,
  type TranscriptionResult,
  type ApiError,
  type HealthStatus,
} from "@/lib/api";
import { useMediaRecorder } from "@/hooks/useMediaRecorder";
import { useAudioVisualizer } from "@/hooks/useAudioVisualizer";
import { AudioVisualizer } from "./AudioVisualizer";
import { StatusBar } from "./StatusBar";
import { TranscriptBox } from "./TranscriptBox";
import { FfmpegBanner, OfflineBanner } from "./Banners";
import { MicIcon, StopIcon, SpinnerIcon } from "./Icons";

type AppStatus =
  | "loading"
  | "idle"
  | "recording"
  | "transcribing"
  | "done"
  | "error";

export default function VoiceRecognition() {
  const [status, setStatus] = useState<AppStatus>("loading");
  const [msg, setMsg] = useState("Connexion au backend…");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [engines, setEngines] = useState<Engine[]>([]);
  const [lang, setLang] = useState("fr-FR");
  const [engine, setEngine] = useState("google");
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);

  const { canvasRef, startViz, stopViz } = useAudioVisualizer();

  // ── Callback : blob audio prêt → envoi backend ──────────────────────────
  const onRecordingStop = useCallback(
    async (blob: Blob) => {
      setStatus("transcribing");
      setMsg("Transcription en cours…");
      setApiError(null);
      try {
        const res = await transcribeAudio(blob, lang, engine);
        setTranscript(res.transcript);
        setResult(res);
        setStatus("done");
        setMsg("Transcription terminée");
      } catch (err) {
        const e = err as ApiError;
        setApiError(e);
        setStatus("error");
        setMsg(`Erreur : ${e.message}`);
      }
    },
    [lang, engine],
  );

  const {
    start: startRecorder,
    stop: stopRecorder,
    setFfmpegAvailable,
  } = useMediaRecorder({ onStop: onRecordingStop });

  // ── Init ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      const h = await checkHealth();
      setHealth(h);
      setFfmpegAvailable(h.ffmpegAvailable);

      if (!h.online) {
        setStatus("error");
        setMsg(
          "Backend inaccessible — lancez : uvicorn app.main:app --reload --port 8000",
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
        setStatus("idle");
        setMsg(
          h.ffmpegAvailable
            ? "Prêt — cliquez sur Démarrer"
            : "Prêt — mode WAV (ffmpeg absent)",
        );
      } catch {
        setStatus("error");
        setMsg("Impossible de charger la config du backend");
      }
    })();
  }, [setFfmpegAvailable]);

  // ── Démarrer ─────────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    setApiError(null);
    setResult(null);
    setTranscript("");
    if (health) setFfmpegAvailable(health.ffmpegAvailable);

    // startViz() ouvre le micro + crée l'AudioContext → retourne les nodes partagés
    const nodes = await startViz();
    if (!nodes) {
      setStatus("error");
      setMsg(
        "Accès microphone refusé — vérifiez les permissions du navigateur",
      );
      return;
    }

    // Le recorder utilise les mêmes nodes (pas de second getUserMedia)
    startRecorder(nodes);

    setStatus("recording");
    setMsg(
      health?.ffmpegAvailable
        ? "Enregistrement en cours — parlez maintenant"
        : "Enregistrement WAV — parlez maintenant",
    );
  }, [startViz, startRecorder, setFfmpegAvailable, health]);

  // ── Arrêter ──────────────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    // 1. Le recorder assembles le blob (WAV ou WebM) et appelle onRecordingStop
    stopRecorder();
    // 2. Le visualiseur ferme stream + AudioContext
    stopViz();
  }, [stopRecorder, stopViz]);

  const handleToggle = () => {
    if (status === "recording") stopRecording();
    else if (["idle", "done", "error"].includes(status)) startRecording();
  };

  const handleClear = () => {
    setTranscript("");
    setResult(null);
    setApiError(null);
    if (["done", "error"].includes(status)) {
      setStatus("idle");
      setMsg("Prêt — cliquez sur Démarrer");
    }
  };

  // ── Derived ──────────────────────────────────────────────────────────────
  const isRec = status === "recording";
  const isTx = status === "transcribing";
  const busy = status === "loading" || isTx;
  const canAct = ["idle", "recording", "done", "error"].includes(status);
  const online = health?.online ?? null;

  const btnClass = [
    "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold",
    "transition-all duration-200 min-w-[160px] justify-center border",
    "focus:outline-none active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed",
    isRec
      ? "bg-teal-50 dark:bg-teal-950 border-teal-300 dark:border-teal-700 text-teal-700 dark:text-teal-300"
      : isTx
        ? "bg-amber-50 dark:bg-amber-950 border-amber-300 dark:border-amber-700 text-amber-700 dark:text-amber-300"
        : "bg-white dark:bg-stone-800 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-200 hover:bg-stone-50 dark:hover:bg-stone-700",
  ].join(" ");

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950 flex items-center justify-center p-4">
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(136,135,128,0.1) 1px,transparent 1px)," +
            "linear-gradient(90deg,rgba(136,135,128,0.1) 1px,transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10 w-full max-w-xl">
        <div
          className="bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700
          rounded-2xl p-8 flex flex-col items-center gap-6 shadow-sm"
        >
          {/* Header */}
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
                    online === null
                      ? "#B4B2A9"
                      : online
                        ? "#1D9E75"
                        : "#E24B4A",
                }}
              />
              <p
                className="text-xs text-stone-400 dark:text-stone-500"
                style={{ fontFamily: "'Space Mono',monospace" }}
              >
                {online === null
                  ? "Vérification…"
                  : online
                    ? `Backend connecté · ffmpeg ${health?.ffmpegAvailable ? "✓" : "✗"}`
                    : "Backend hors ligne"}
              </p>
            </div>
          </div>

          <AudioVisualizer canvasRef={canvasRef} />

          <StatusBar
            msg={msg}
            isRecording={isRec}
            isTranscribing={isTx}
            isBusy={busy}
            lang={lang}
            engine={engine}
            languages={languages}
            engines={engines}
            onLangChange={setLang}
            onEngineChange={setEngine}
          />

          <TranscriptBox
            isTranscribing={isTx}
            isRecording={isRec}
            transcript={transcript}
            result={result}
            apiError={apiError}
          />

          {/* Boutons */}
          <div className="flex gap-3 w-full justify-center">
            <button
              onClick={handleToggle}
              disabled={!canAct || busy || online === false}
              className={btnClass}
              style={{ fontFamily: "'Syne',sans-serif" }}
            >
              {isTx ? <SpinnerIcon /> : isRec ? <StopIcon /> : <MicIcon />}
              {isRec ? "Arrêter" : isTx ? "Transcription…" : "Démarrer"}
            </button>

            <button
              onClick={handleClear}
              disabled={isRec || isTx}
              className="px-4 py-2.5 rounded-xl text-sm border border-stone-200 dark:border-stone-700
                bg-transparent text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800
                hover:text-stone-700 dark:hover:text-stone-200 transition-all
                focus:outline-none disabled:opacity-30 disabled:cursor-not-allowed"
              style={{ fontFamily: "'Syne',sans-serif" }}
            >
              Effacer
            </button>
          </div>

          <FfmpegBanner show={!!(health?.online && !health.ffmpegAvailable)} />
          <OfflineBanner show={online === false} />
        </div>
      </div>
    </div>
  );
}
