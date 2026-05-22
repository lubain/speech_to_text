import type { TranscriptionResult, ApiError } from "@/lib/api";
import { MetaBadge } from "./MetaBadge";
import { SpinnerIcon } from "./Icons";

type Props = {
  isTranscribing: boolean;
  isRecording: boolean;
  transcript: string;
  result: TranscriptionResult | null;
  apiError: ApiError | null;
};

export function TranscriptBox({
  isTranscribing,
  isRecording,
  transcript,
  result,
  apiError,
}: Props) {
  return (
    <div
      className="w-full min-h-24 bg-stone-50 dark:bg-stone-800/50 border
        border-stone-100 dark:border-stone-700 rounded-xl px-4 py-3 flex flex-col gap-2"
    >
      {isTranscribing ? (
        <div className="flex items-center gap-3">
          <SpinnerIcon />
          <span
            className="text-sm text-stone-400 italic"
            style={{ fontFamily: "'Space Mono', monospace" }}
          >
            Envoi au backend…
          </span>
        </div>
      ) : apiError ? (
        <div className="flex flex-col gap-1.5">
          <p className="text-sm text-red-600 dark:text-red-400 font-medium break-words">
            {apiError.message}
          </p>
          <span
            className="text-[11px] text-stone-400"
            style={{ fontFamily: "'Space Mono', monospace" }}
          >
            code : {apiError.code}
          </span>
        </div>
      ) : transcript ? (
        <>
          <p className="text-[15px] leading-relaxed text-stone-800 dark:text-stone-200 break-words">
            {transcript}
            {isRecording && (
              <span
                className="inline-block w-0.5 h-[1em] bg-stone-400 ml-0.5
                  align-text-bottom animate-blink"
              />
            )}
          </p>
          {result && (
            <div className="flex flex-wrap gap-1.5 mt-1">
              <MetaBadge label="moteur" value={result.engine} />
              <MetaBadge label="langue" value={result.language} />
              {result.confidence != null && (
                <MetaBadge
                  label="confiance"
                  value={`${Math.round(result.confidence * 100)}%`}
                />
              )}
              {result.audio_duration_seconds != null && (
                <MetaBadge
                  label="audio"
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
          style={{ fontFamily: "'Space Mono', monospace" }}
        >
          {isRecording
            ? "Enregistrement… appuyez sur Arrêter pour transcrire"
            : "La transcription apparaîtra ici après l'enregistrement"}
        </p>
      )}
    </div>
  );
}
