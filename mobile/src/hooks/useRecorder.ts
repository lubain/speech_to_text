import { useState, useRef, useCallback } from "react";
import { Audio } from "expo-av";

export type RecorderState = "idle" | "recording" | "stopped";

// Android renvoie des dBFS entre -160 et 0 (silence profond ~ -120 dB).
// On normalise sur [-120..0] et on amplifie pour que la voix soit visible.
export function dbToNorm(db: number): number {
  const FLOOR = -120; // plancher pratique Android
  const clamped = Math.max(FLOOR, Math.min(0, db ?? FLOOR));
  const raw = (clamped - FLOOR) / -FLOOR; // 0..1 linéaire
  // Amplification : racine carrée pour rendre les valeurs moyennes plus visibles
  return Math.sqrt(raw);
}

type RecorderOptions = {
  onMeteringUpdate?: (level: number) => void;
};

export function useRecorder({ onMeteringUpdate }: RecorderOptions = {}) {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const onMeteringRef = useRef(onMeteringUpdate);
  onMeteringRef.current = onMeteringUpdate;

  const requestPermission = useCallback(async (): Promise<boolean> => {
    const { status } = await Audio.requestPermissionsAsync();
    return status === "granted";
  }, []);

  const start = useCallback(async (): Promise<boolean> => {
    setError(null);
    try {
      const granted = await requestPermission();
      if (!granted) {
        setError("permission_denied");
        return false;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      });

      const { recording } = await Audio.Recording.createAsync(
        {
          isMeteringEnabled: true,
          android: {
            extension: ".m4a",
            outputFormat: Audio.AndroidOutputFormat.MPEG_4,
            audioEncoder: Audio.AndroidAudioEncoder.AAC,
            sampleRate: 16000,
            numberOfChannels: 1,
            bitRate: 64000,
          },
          ios: {
            extension: ".m4a",
            outputFormat: Audio.IOSOutputFormat.MPEG4AAC,
            audioQuality: Audio.IOSAudioQuality.HIGH,
            sampleRate: 16000,
            numberOfChannels: 1,
            bitRate: 64000,
            linearPCMBitDepth: 16,
            linearPCMIsBigEndian: false,
            linearPCMIsFloat: false,
          },
          web: { mimeType: "audio/webm;codecs=opus", bitsPerSecond: 64000 },
        },
        (status) => {
          if (!status.isRecording) return;
          const db = status.metering;
          if (db !== undefined && db !== null) {
            const norm = dbToNorm(db);
            console.log(
              `[METER] dB: ${db.toFixed(1)} → norm: ${norm.toFixed(3)}`,
            );
            onMeteringRef.current?.(norm);
          }
        },
        80,
      );

      recordingRef.current = recording;
      setState("recording");
      return true;
    } catch (e) {
      console.error("Recording start error:", e);
      setError("start_failed");
      return false;
    }
  }, [requestPermission]);

  const stop = useCallback(async (): Promise<string | null> => {
    if (!recordingRef.current) return null;
    try {
      await recordingRef.current.stopAndUnloadAsync();
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      setState("stopped");
      return uri ?? null;
    } catch (e) {
      console.error("Recording stop error:", e);
      setError("stop_failed");
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState("idle");
    setError(null);
  }, []);

  return { state, error, start, stop, reset };
}
