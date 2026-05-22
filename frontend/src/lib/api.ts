export const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export type Engine = {
  id: string;
  label: string;
  requires_key: boolean;
  offline: boolean;
  description: string;
};

export type Language = {
  code: string;
  label: string;
};

export type TranscriptionResult = {
  transcript: string;
  language: string;
  engine: string;
  duration_seconds: number;
  confidence: number | null;
  audio_duration_seconds: number | null;
};

export type ApiError = {
  code: string;
  message: string;
};

export type HealthStatus = {
  online: boolean;
  ffmpegAvailable: boolean;
  enginesAvailable: string[];
};

// ─── Helper ───────────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let err: ApiError = {
      code: "unknown_error",
      message: `HTTP ${res.status}`,
    };
    try {
      const body = await res.json();
      if (body?.detail?.code) err = body.detail as ApiError;
      else if (typeof body?.detail === "string")
        err = { code: "error", message: body.detail };
    } catch {
      /* ignore parse error */
    }
    throw err;
  }
  return res.json() as Promise<T>;
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok)
      return { online: false, ffmpegAvailable: false, enginesAvailable: [] };
    const data = await res.json();
    return {
      online: true,
      ffmpegAvailable: data.ffmpeg_available ?? false,
      enginesAvailable: data.engines_available ?? [],
    };
  } catch {
    return { online: false, ffmpegAvailable: false, enginesAvailable: [] };
  }
}

export async function fetchEngines(): Promise<Engine[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/engines`);
  const data = await handleResponse<{ engines: Engine[]; default: string }>(
    res,
  );
  return data.engines;
}

export async function fetchLanguages(): Promise<Language[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/languages`);
  const data = await handleResponse<{ languages: Language[]; default: string }>(
    res,
  );
  return data.languages;
}

export async function transcribeAudio(
  blob: Blob,
  language: string,
  engine: string,
): Promise<TranscriptionResult> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  const audio_base64 = btoa(binary);

  const res = await fetch(`${API_BASE_URL}/api/v1/transcribe/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_base64,
      content_type: blob.type || "audio/wav",
      language,
      engine,
    }),
  });

  return handleResponse<TranscriptionResult>(res);
}
