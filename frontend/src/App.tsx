import { useState } from "react";
import VoiceRecognition from "@/components/VoiceRecognition";
import RealtimeTranscription from "@/components/RealtimeTranscription";

type Mode = "standard" | "realtime";

export default function App() {
  const [mode, setMode] = useState<Mode>("realtime");

  return (
    <div>
      {/* Sélecteur de mode */}
      <div
        className="fixed top-3 left-1/2 -translate-x-1/2 z-50 flex gap-1
        bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700
        rounded-xl p-1 shadow-sm"
      >
        {(["realtime", "standard"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={[
              "px-4 py-1.5 rounded-lg text-xs font-semibold transition-all",
              mode === m
                ? "bg-stone-900 dark:bg-stone-100 text-white dark:text-stone-900"
                : "text-stone-400 hover:text-stone-600 dark:hover:text-stone-300",
            ].join(" ")}
            style={{ fontFamily: "'Syne', sans-serif" }}
          >
            {m === "realtime" ? "⚡ Temps réel" : "📎 Standard"}
          </button>
        ))}
      </div>

      {mode === "realtime" ? <RealtimeTranscription /> : <VoiceRecognition />}
    </div>
  );
}
