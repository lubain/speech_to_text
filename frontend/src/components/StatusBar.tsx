import type { Language, Engine } from "@/lib/api";

type Props = {
  msg: string;
  isRecording: boolean;
  isTranscribing: boolean;
  isBusy: boolean;
  lang: string;
  engine: string;
  languages: Language[];
  engines: Engine[];
  onLangChange: (v: string) => void;
  onEngineChange: (v: string) => void;
};

export function StatusBar({
  msg,
  isRecording,
  isTranscribing,
  isBusy,
  lang,
  engine,
  languages,
  engines,
  onLangChange,
  onEngineChange,
}: Props) {
  const dotColor = isRecording
    ? "#1D9E75"
    : isTranscribing
      ? "#EF9F27"
      : "#B4B2A9";
  const dotShadow = isRecording
    ? "0 0 0 4px rgba(29,158,117,.15)"
    : isTranscribing
      ? "0 0 0 4px rgba(239,159,39,.15)"
      : "none";

  const selectCls = `text-xs border border-stone-200 dark:border-stone-600 rounded-md
    px-2 py-1 bg-transparent text-stone-500 dark:text-stone-400
    focus:outline-none disabled:opacity-40 cursor-pointer`;

  return (
    <div className="flex items-center gap-3 w-full">
      <span
        className="w-2 h-2 rounded-full flex-shrink-0 transition-all duration-300"
        style={{
          background: dotColor,
          boxShadow: dotShadow,
          animation:
            isRecording || isTranscribing ? "pulse-dot 1.2s infinite" : "none",
        }}
      />
      <span
        className="text-xs text-stone-400 dark:text-stone-500 flex-1 truncate"
        style={{ fontFamily: "'Space Mono', monospace" }}
      >
        {msg}
      </span>

      <div className="flex gap-1.5 flex-shrink-0">
        <select
          value={lang}
          onChange={(e) => onLangChange(e.target.value)}
          disabled={isRecording || isBusy}
          className={selectCls}
          style={{ fontFamily: "'Space Mono', monospace" }}
        >
          {(languages.length
            ? languages
            : [{ code: "fr-FR", label: "Français" }]
          ).map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>

        <select
          value={engine}
          onChange={(e) => onEngineChange(e.target.value)}
          disabled={isRecording || isBusy}
          className={selectCls}
          style={{ fontFamily: "'Space Mono', monospace" }}
        >
          {(engines.length
            ? engines
            : [{ id: "google", label: "Google Web Speech API" }]
          ).map((e) => (
            <option key={e.id} value={e.id}>
              {e.label.split(" ")[0]}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
