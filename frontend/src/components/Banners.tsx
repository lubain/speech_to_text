type FfmpegBannerProps = { show: boolean };

export function FfmpegBanner({ show }: FfmpegBannerProps) {
  if (!show) return null;
  return (
    <div
      className="w-full text-xs bg-amber-50 dark:bg-amber-950 border border-amber-200
        dark:border-amber-800 rounded-xl px-4 py-3 text-amber-700 dark:text-amber-300"
      style={{ fontFamily: "'Space Mono', monospace" }}
    >
      <p className="font-bold mb-1">⚠ ffmpeg non détecté sur le serveur</p>
      <p>
        L'audio est enregistré en WAV pur JavaScript (pas de ffmpeg requis).
      </p>
      <p className="mt-1 opacity-70">
        Pour activer WebM/OGG/MP3 :{" "}
        <a
          href="https://www.gyan.dev/ffmpeg/builds/"
          target="_blank"
          rel="noreferrer"
          className="underline"
        >
          gyan.dev/ffmpeg/builds
        </a>{" "}
        → ajoutez <code>ffmpeg\bin</code> au PATH puis relancez uvicorn.
      </p>
    </div>
  );
}

type OfflineBannerProps = { show: boolean };

export function OfflineBanner({ show }: OfflineBannerProps) {
  if (!show) return null;
  return (
    <div
      className="w-full text-xs text-center text-red-600 dark:text-red-400
        bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800
        rounded-xl px-4 py-3"
      style={{ fontFamily: "'Space Mono', monospace" }}
    >
      Backend inaccessible. Lancez :<br />
      <code className="font-bold">
        uvicorn app.main:app --reload --port 8000
      </code>
    </div>
  );
}
