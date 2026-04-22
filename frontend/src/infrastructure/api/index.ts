// ─── api.ts — Client HTTP pour le backend FastAPI ─────────────────────────────

export const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";

// ─── Types miroir des schémas Pydantic ────────────────────────────────────────

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

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let err: ApiError = {
      code: "unknown_error",
      message: `HTTP ${res.status}`,
    };
    try {
      const body = await res.json();
      if (body?.detail?.code) err = body.detail as ApiError;
    } catch {}
    throw err;
  }
  return res.json() as Promise<T>;
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

/** Récupère la liste des moteurs disponibles. */
export async function fetchEngines(): Promise<Engine[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/engines`);
  const data = await handleResponse<{ engines: Engine[]; default: string }>(
    res
  );
  return data.engines;
}

/** Récupère la liste des langues supportées. */
export async function fetchLanguages(): Promise<Language[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/languages`);
  const data = await handleResponse<{ languages: Language[]; default: string }>(
    res
  );
  return data.languages;
}

/**
 * Envoie un Blob audio au backend via base64 et retourne la transcription.
 * On utilise base64 pour éviter les problèmes CORS liés au multipart.
 */
export async function transcribeAudio(
  blob: Blob,
  language: string,
  engine: string
): Promise<TranscriptionResult> {
  // Blob → ArrayBuffer → base64
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const audio_base64 = btoa(binary);

  const res = await fetch(`${API_BASE_URL}/api/v1/transcribe/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_base64,
      content_type: blob.type || "audio/webm",
      language,
      engine,
    }),
  });

  return handleResponse<TranscriptionResult>(res);
}

/** Vérifie que le backend est joignable. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}
