// ─── useMediaRecorder.ts ──────────────────────────────────────────────────────
// Hook qui gère MediaRecorder : capture audio → Blob prêt à envoyer.

import { useRef, useCallback } from "react";

type RecorderOptions = {
  /** Appelé quand l'enregistrement est terminé avec le Blob final. */
  onStop: (blob: Blob, mimeType: string) => void;
};

/** Choisit le premier mimeType supporté par le navigateur. */
function getSupportedMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) return mime;
  }
  return ""; // laisse le navigateur décider
}

export function useMediaRecorder({ onStop }: RecorderOptions) {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const mimeTypeRef = useRef<string>("");

  const start = useCallback(
    (stream: MediaStream) => {
      const mimeType = getSupportedMimeType();
      mimeTypeRef.current = mimeType;
      chunksRef.current = [];

      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined
      );
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: mimeTypeRef.current || "audio/webm",
        });
        onStop(blob, mimeTypeRef.current || "audio/webm");
      };

      // timeslice 250 ms → chunks réguliers même pour les longs enregistrements
      recorder.start(250);
    },
    [onStop]
  );

  const stop = useCallback(() => {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  return { start, stop };
}
