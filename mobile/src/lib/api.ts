import * as FileSystem from "expo-file-system";

export const API_BASE_URL = "http://192.168.1.XX:8000";

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
  errorDetail?: string;
};

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
      /* ignore */
    }
    throw err;
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<HealthStatus> {
  const url = `${API_BASE_URL}/health`;
  console.log(`[API] checkHealth → ${url}`);
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);

    const res = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    clearTimeout(timeout);

    console.log(`[API] health status: ${res.status}`);
    if (!res.ok)
      return {
        online: false,
        ffmpegAvailable: false,
        enginesAvailable: [],
        errorDetail: `HTTP ${res.status}`,
      };

    const data = await res.json();
    console.log("[API] health:", JSON.stringify(data));
    return {
      online: true,
      ffmpegAvailable: data.ffmpeg_available ?? false,
      enginesAvailable: data.engines_available ?? [],
    };
  } catch (e: any) {
    const detail = e?.message ?? String(e);
    console.warn(`[API] health failed: ${detail}`);
    return {
      online: false,
      ffmpegAvailable: false,
      enginesAvailable: [],
      errorDetail: detail,
    };
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

/**
 * Lit le fichier audio via expo-file-system (base64 natif),
 * puis envoie au backend.
 * ✅ Compatible Hermes — pas de blob.arrayBuffer() qui n'existe pas en RN.
 */
export async function transcribeFile(
  fileUri: string,
  language: string,
  engine: string,
): Promise<TranscriptionResult> {
  console.log(`[API] transcribeFile: ${fileUri}`);

  // ── Lecture base64 via FileSystem (pas de blob, pas d'ArrayBuffer) ────────
  const audio_base64 = await FileSystem.readAsStringAsync(fileUri, {
    encoding: FileSystem.EncodingType.Base64,
  });

  console.log(`[API] base64 length: ${audio_base64.length}`);

  // ── Détection du format depuis l'extension ────────────────────────────────
  const ext = fileUri.split(".").pop()?.toLowerCase() ?? "m4a";
  const mimeMap: Record<string, string> = {
    m4a: "audio/mp4",
    mp4: "audio/mp4",
    wav: "audio/wav",
    webm: "audio/webm;codecs=opus",
    ogg: "audio/ogg",
    mp3: "audio/mpeg",
    flac: "audio/flac",
  };
  const content_type = mimeMap[ext] ?? "audio/mp4";
  console.log(`[API] content_type: ${content_type}, ext: ${ext}`);

  // ── Envoi au backend ──────────────────────────────────────────────────────
  const res = await fetch(`${API_BASE_URL}/api/v1/transcribe/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_base64, content_type, language, engine }),
  });

  return handleResponse<TranscriptionResult>(res);
}
