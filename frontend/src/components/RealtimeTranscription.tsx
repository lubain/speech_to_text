// Interface de transcription temps réel via WebSocket.

import { useState, useCallback, useRef, useEffect } from "react";
import {
  useRealtimeTranscription,
  type RTMessage,
  type RTStatus,
} from "@/hooks/useRealtimeTranscription";
import { useRealtimeVisualizer } from "@/hooks/useRealtimeVisualizer";
import {
  fetchEngines,
  fetchLanguages,
  checkHealth,
  type Engine,
  type Language,
} from "@/lib/api";

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
    width={14}
    height={14}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    style={{ animation: "spin .9s linear infinite" }}
  >
    <circle cx="12" cy="12" r="10" strokeOpacity={0.25} />
    <path d="M12 2a10 10 0 0 1 10 10" />
  </svg>
);

// ─── StatusLabel ─────────────────────────────────────────────────────────────

function statusInfo(s: RTStatus): {
  label: string;
  color: string;
  pulse: boolean;
} {
  switch (s) {
    case "connecting":
      return { label: "Connexion WebSocket…", color: "#EF9F27", pulse: true };
    case "ready":
      return { label: "Prêt…", color: "#1D9E75", pulse: false };
    case "streaming":
      return { label: "Écoute en cours", color: "#1D9E75", pulse: true };
    case "finalizing":
      return { label: "Finalisation…", color: "#EF9F27", pulse: true };
    case "error":
      return { label: "Erreur", color: "#E24B4A", pulse: false };
    default:
      return { label: "Inactif", color: "#B4B2A9", pulse: false };
  }
}

// ─── TranscriptLine ──────────────────────────────────────────────────────────

type Line = {
  id: number;
  text: string;
  type: "interim" | "final";
  confidence?: number | null;
  ts: string;
};

function TranscriptLine({ line }: { line: Line }) {
  return (
    <div
      className={[
        "flex items-start gap-2 py-1.5 border-b border-stone-100 dark:border-stone-800 last:border-0",
        line.type === "interim" ? "opacity-60" : "",
      ].join(" ")}
    >
      <span
        className="text-[10px] shrink-0 mt-0.5 px-1.5 py-0.5 rounded"
        style={{
          fontFamily: "'Space Mono', monospace",
          background:
            line.type === "final"
              ? "rgba(29,158,117,0.12)"
              : "rgba(136,135,128,0.1)",
          color: line.type === "final" ? "#0F6E56" : "#a8a29e",
        }}
      >
        {line.type === "final" ? "✓" : "~"}
      </span>
      <span className="flex-1 text-[15px] leading-relaxed text-stone-800 dark:text-stone-200">
        {line.text}
        {line.type === "interim" && (
          <span
            className="inline-block w-0.5 h-[1em] bg-stone-400 ml-0.5 align-text-bottom"
            style={{ animation: "blink 1s step-end infinite" }}
          />
        )}
      </span>
      <span
        className="text-[10px] text-stone-300 dark:text-stone-600 shrink-0 mt-0.5"
        style={{ fontFamily: "'Space Mono', monospace" }}
      >
        {line.ts}
        {line.confidence != null && ` · ${Math.round(line.confidence * 100)}%`}
      </span>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function RealtimeTranscription() {
  const [languages, setLanguages] = useState<Language[]>([]);
  const [engines, setEngines] = useState<Engine[]>([]);
  const [lang, setLang] = useState("fr-FR");
  const [engine, setEngine] = useState("google");
  const [lines, setLines] = useState<Line[]>([]);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const lineIdRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // ── Handlers WebSocket ────────────────────────────────────────────────────
  const handleMessage = useCallback((msg: RTMessage) => {
    if (msg.type === "interim" || msg.type === "final") {
      const now = new Date();
      const ts = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;

      setLines((prev) => {
        // Remplace le dernier interim par le nouveau, ou ajoute une ligne
        if (msg.type === "interim") {
          const lastIdx = prev.findLastIndex((l) => l.type === "interim");
          if (lastIdx !== -1) {
            const updated = [...prev];
            updated[lastIdx] = {
              ...updated[lastIdx],
              text: msg.transcript!,
              ts,
            };
            return updated;
          }
        }
        return [
          ...prev,
          {
            id: ++lineIdRef.current,
            text: msg.transcript!,
            type: msg.type as "interim" | "final",
            confidence: msg.confidence,
            ts,
          },
        ];
      });
    }
  }, []);

  const { status, connect, disconnect, finalize, analyserRef } =
    useRealtimeTranscription({
      language: lang,
      engine,
      onMessage: handleMessage,
    });

  const { canvasRef } = useRealtimeVisualizer(
    status === "streaming",
    analyserRef,
  );

  // ── Scroll automatique ────────────────────────────────────────────────────
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [lines]);

  // ── Init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      const h = await checkHealth();
      setBackendOk(h.online);
      if (!h.online) return;
      const [langs, engs] = await Promise.all([
        fetchLanguages(),
        fetchEngines(),
      ]);
      setLanguages(langs);
      setEngines(engs);
      if (langs.length) setLang(langs[0].code);
      if (engs.length) setEngine(engs[0].id);
    })();
  }, []);

  const isStreaming = status === "streaming";
  const isBusy =
    status === "connecting" || status === "ready" || status === "finalizing";
  const si = statusInfo(status);

  const handleToggle = () => {
    if (isStreaming || isBusy) {
      finalize();
      setTimeout(disconnect, 800);
    } else {
      connect();
    }
  };

  const handleClear = () => setLines([]);

  const handleCopy = () => {
    const text = lines
      .filter((l) => l.type === "final")
      .map((l) => l.text)
      .join(" ");
    if (text) navigator.clipboard.writeText(text);
  };

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950 flex items-center justify-center p-4">
      {/* Grille de fond */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(136,135,128,0.08) 1px,transparent 1px)," +
            "linear-gradient(90deg,rgba(136,135,128,0.08) 1px,transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10 w-full max-w-2xl">
        <div
          className="bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700
          rounded-2xl shadow-sm overflow-hidden flex flex-col"
          style={{ height: "90vh", maxHeight: 720 }}
        >
          {/* ── Header ──────────────────────────────────────────────────── */}
          <div className="px-6 pt-5 pb-4 border-b border-stone-100 dark:border-stone-800">
            <div className="flex items-center justify-between">
              <div>
                <h1
                  className="text-xl font-black tracking-tight text-stone-900 dark:text-stone-100"
                  style={{ fontFamily: "'Syne', sans-serif" }}
                >
                  Transcription Temps Réel
                </h1>
                <div className="flex items-center gap-2 mt-1">
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{
                      background:
                        backendOk === null
                          ? "#B4B2A9"
                          : backendOk
                            ? "#1D9E75"
                            : "#E24B4A",
                    }}
                  />
                  <span
                    className="text-xs text-stone-400"
                    style={{ fontFamily: "'Space Mono', monospace" }}
                  >
                    {backendOk === null
                      ? "…"
                      : backendOk
                        ? "Backend connecté · WebSocket"
                        : "Backend hors ligne"}
                  </span>
                </div>
              </div>

              {/* Sélecteurs */}
              <div className="flex gap-2">
                {[
                  {
                    val: lang,
                    set: setLang,
                    opts: languages.map((l) => ({
                      v: l.code,
                      label: l.label.split(" ")[0],
                    })),
                  },
                  {
                    val: engine,
                    set: setEngine,
                    opts: engines.map((e) => ({
                      v: e.id,
                      label: e.label.split(" ")[0],
                    })),
                  },
                ].map((sel, idx) => (
                  <select
                    key={idx}
                    value={sel.val}
                    onChange={(e) => sel.set(e.target.value)}
                    disabled={isStreaming || isBusy}
                    className="text-xs border border-stone-200 dark:border-stone-600 rounded-lg
                      px-2 py-1.5 bg-transparent text-stone-500 dark:text-stone-400
                      focus:outline-none disabled:opacity-40 cursor-pointer"
                    style={{ fontFamily: "'Space Mono', monospace" }}
                  >
                    {sel.opts.map((o) => (
                      <option key={o.v} value={o.v}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                ))}
              </div>
            </div>

            {/* Visualiseur */}
            <div
              className="mt-3 h-16 bg-stone-50 dark:bg-stone-800/50 border border-stone-100
              dark:border-stone-700 rounded-xl overflow-hidden flex items-end px-2 pb-1.5"
            >
              <canvas
                ref={canvasRef}
                width={680}
                height={56}
                className="w-full h-full"
                style={{ display: "block" }}
              />
            </div>

            {/* Status */}
            <div className="flex items-center gap-2 mt-2.5">
              <span
                className="w-1.5 h-1.5 rounded-full transition-all"
                style={{
                  background: si.color,
                  boxShadow: si.pulse ? `0 0 0 3px ${si.color}26` : "none",
                  animation: si.pulse ? "pulse-dot 1.2s infinite" : "none",
                }}
              />
              <span
                className="text-xs text-stone-400"
                style={{ fontFamily: "'Space Mono', monospace" }}
              >
                {si.label}
                {isStreaming && (
                  <span className="ml-2 text-teal-600 dark:text-teal-400">
                    · {lines.filter((l) => l.type === "final").length} phrase(s)
                  </span>
                )}
              </span>
              {isBusy && <SpinnerIcon />}
            </div>
          </div>

          {/* ── Transcript scroll ────────────────────────────────────────── */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-3">
            {lines.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <p
                  className="text-sm text-stone-300 dark:text-stone-600 text-center"
                  style={{ fontFamily: "'Space Mono', monospace" }}
                >
                  {status === "idle"
                    ? "Appuyez sur Démarrer puis parlez\nLes résultats apparaissent en temps réel"
                    : "En attente de la parole…"}
                </p>
              </div>
            ) : (
              <div className="flex flex-col">
                {lines.map((line) => (
                  <TranscriptLine key={line.id} line={line} />
                ))}
              </div>
            )}
          </div>

          {/* ── Actions ──────────────────────────────────────────────────── */}
          <div
            className="px-6 py-4 border-t border-stone-100 dark:border-stone-800
            flex items-center gap-3 justify-between"
          >
            <div className="flex gap-2">
              <button
                onClick={handleClear}
                disabled={lines.length === 0 || isStreaming}
                className="px-3 py-2 rounded-lg text-sm border border-stone-200 dark:border-stone-700
                  text-stone-400 hover:bg-stone-50 dark:hover:bg-stone-800 transition-all
                  disabled:opacity-30 disabled:cursor-not-allowed focus:outline-none"
                style={{ fontFamily: "'Syne', sans-serif" }}
              >
                Effacer
              </button>
              <button
                onClick={handleCopy}
                disabled={lines.filter((l) => l.type === "final").length === 0}
                className="px-3 py-2 rounded-lg text-sm border border-stone-200 dark:border-stone-700
                  text-stone-400 hover:bg-stone-50 dark:hover:bg-stone-800 transition-all
                  disabled:opacity-30 disabled:cursor-not-allowed focus:outline-none"
                style={{ fontFamily: "'Syne', sans-serif" }}
              >
                Copier
              </button>
            </div>

            <button
              onClick={handleToggle}
              disabled={backendOk === false || backendOk === null}
              className={[
                "flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold",
                "transition-all duration-200 border focus:outline-none active:scale-[0.98]",
                "disabled:opacity-40 disabled:cursor-not-allowed",
                isStreaming || isBusy
                  ? "bg-teal-50 dark:bg-teal-950 border-teal-300 dark:border-teal-700 text-teal-700 dark:text-teal-300"
                  : "bg-stone-900 dark:bg-stone-100 border-stone-900 dark:border-stone-100 text-white dark:text-stone-900 hover:bg-stone-700 dark:hover:bg-stone-300",
              ].join(" ")}
              style={{ fontFamily: "'Syne', sans-serif" }}
            >
              {isStreaming || isBusy ? <StopIcon /> : <MicIcon />}
              {isStreaming ? "Arrêter" : isBusy ? "…" : "Démarrer"}
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@400;600;800&display=swap');
        @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes blink     { 0%,100%{opacity:1} 50%{opacity:0}   }
        @keyframes spin      { to{transform:rotate(360deg)}         }
      `}</style>
    </div>
  );
}
