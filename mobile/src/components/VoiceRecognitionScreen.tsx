import { View, Text, Pressable, StyleSheet, Alert } from "react-native";
import * as Clipboard from "expo-clipboard";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { useState, useEffect, useCallback } from "react";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";

import {
  checkHealth,
  fetchEngines,
  fetchLanguages,
  transcribeFile,
  type Engine,
  type Language,
  type TranscriptionResult,
  type ApiError,
  type HealthStatus,
} from "@/lib/api";
import { useRecorder } from "@/hooks/useRecorder";
import { useVisualizer } from "@/hooks/useVisualizer";
import { AudioVisualizer } from "./AudioVisualizer";
import { MicButton } from "./MicButton";
import { TranscriptCard } from "./TranscriptCard";
import { LanguageEnginePicker } from "./LanguageEnginePicker";
import { StatusDot } from "./StatusDot";
import { DebugPanel } from "./DebugPanel";
import { COLORS, FONTS, RADIUS } from "@/constants/theme";

type AppStatus =
  | "loading"
  | "idle"
  | "recording"
  | "transcribing"
  | "done"
  | "error";

export default function VoiceRecognitionScreen() {
  const [appStatus, setAppStatus] = useState<AppStatus>("loading");
  const [msg, setMsg] = useState("Connexion au backend…");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | undefined>();
  const [languages, setLanguages] = useState<Language[]>([]);
  const [engines, setEngines] = useState<Engine[]>([]);
  const [lang, setLang] = useState("fr-FR");
  const [engine, setEngine] = useState("google");
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);

  const isRec = appStatus === "recording";
  const isTx = appStatus === "transcribing";

  // ── Visualiseur ──────────────────────────────────────────────────────────
  const { bars, updateFromMeter } = useVisualizer(isRec);

  // ── Recorder — onMeteringUpdate directement branché sur updateFromMeter ──
  // useRecorder stocke le callback dans un ref interne → pas de stale closure
  const recorder = useRecorder({ onMeteringUpdate: updateFromMeter });

  // ── Init / Retry ──────────────────────────────────────────────────────────
  const initBackend = useCallback(async () => {
    setAppStatus("loading");
    setMsg("Connexion au backend…");
    setErrorDetail(undefined);
    const h = await checkHealth();
    setHealth(h);
    if (!h.online) {
      setAppStatus("error");
      setErrorDetail(h.errorDetail);
      setMsg("Backend inaccessible");
      return;
    }
    try {
      const [langs, engs] = await Promise.all([
        fetchLanguages(),
        fetchEngines(),
      ]);
      setLanguages(langs);
      setEngines(engs);
      if (langs.length) setLang(langs[0].code);
      if (engs.length) setEngine(engs[0].id);
      setAppStatus("idle");
      setMsg("Prêt — appuyez sur le micro");
    } catch {
      setAppStatus("error");
      setMsg("Impossible de charger la config du backend");
    }
  }, []);

  useEffect(() => {
    initBackend();
  }, []);

  // ── Démarrer ──────────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    setApiError(null);
    setResult(null);
    setTranscript("");
    const ok = await recorder.start();
    if (!ok) {
      if (recorder.error === "permission_denied") {
        Alert.alert(
          "Permission refusée",
          "Autorisez l'accès au microphone dans les Réglages.",
          [{ text: "OK" }],
        );
      }
      setAppStatus("error");
      setMsg("Accès microphone refusé");
      return;
    }
    setAppStatus("recording");
    setMsg("Enregistrement en cours…");
  }, [recorder]);

  // ── Arrêter + Transcrire ──────────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    setAppStatus("transcribing");
    setMsg("Envoi au backend…");
    const uri = await recorder.stop();
    if (!uri) {
      setAppStatus("error");
      setMsg("Erreur lors de l'enregistrement");
      return;
    }
    try {
      const res = await transcribeFile(uri, lang, engine);
      setTranscript(res.transcript);
      setResult(res);
      setAppStatus("done");
      setMsg("Transcription terminée ✓");
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (err) {
      const e = err as ApiError;
      setApiError(e);
      setAppStatus("error");
      setMsg(`Erreur : ${e.message}`);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }
  }, [recorder, lang, engine]);

  const handleToggle = () => {
    if (isRec) handleStop();
    else if (["idle", "done", "error"].includes(appStatus)) {
      if (appStatus === "error" && !health?.online) initBackend();
      else handleStart();
    }
  };

  const handleClear = () => {
    setTranscript("");
    setResult(null);
    setApiError(null);
    recorder.reset();
    if (["done", "error"].includes(appStatus) && health?.online) {
      setAppStatus("idle");
      setMsg("Prêt — appuyez sur le micro");
    }
  };

  const handleCopy = async () => {
    if (!transcript) return;
    await Clipboard.setStringAsync(transcript);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    Alert.alert("Copié !", "Texte copié dans le presse-papiers.");
  };

  const online = health?.online ?? null;
  const canAct = ["idle", "recording", "done", "error"].includes(appStatus);

  return (
    <LinearGradient
      colors={["#0c0a09", "#111110", "#0c0a09"]}
      style={styles.gradient}
    >
      <SafeAreaView style={styles.safe} edges={["top", "bottom"]}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Reconnaissance{"\n"}Vocale</Text>
          <View
            style={[
              styles.onlineBadge,
              {
                borderColor:
                  online === true
                    ? "#1a4a35"
                    : online === false
                      ? "#4a1a1a"
                      : "#2c2825",
              },
            ]}
          >
            <View
              style={[
                styles.onlineDot,
                {
                  backgroundColor:
                    online === true
                      ? COLORS.teal500
                      : online === false
                        ? COLORS.red400
                        : "#57534e",
                },
              ]}
            />
            <Text style={styles.onlineText}>
              {online === null
                ? "…"
                : online
                  ? `online · ffmpeg ${health?.ffmpegAvailable ? "✓" : "✗"}`
                  : "offline"}
            </Text>
          </View>
        </View>

        {/* Visualiseur */}
        <AudioVisualizer bars={bars} isRecording={isRec} />

        {/* Status */}
        <StatusDot appStatus={appStatus} health={health} msg={msg} />

        {/* Debug panel si offline */}
        {online === false && (
          <DebugPanel errorDetail={errorDetail} onRetry={initBackend} />
        )}

        {/* Sélecteurs */}
        {online === true && (
          <LanguageEnginePicker
            languages={languages}
            engines={engines}
            selectedLang={lang}
            selectedEngine={engine}
            disabled={isRec || isTx}
            onLangChange={setLang}
            onEngineChange={setEngine}
          />
        )}

        {/* Transcription */}
        {online === true && (
          <View style={{ flex: 1 }}>
            <TranscriptCard
              isTranscribing={isTx}
              isRecording={isRec}
              transcript={transcript}
              result={result}
              apiError={apiError}
            />
          </View>
        )}

        {/* Bouton micro */}
        <View style={styles.micRow}>
          <MicButton
            isRecording={isRec}
            isTranscribing={isTx}
            disabled={!canAct || appStatus === "loading"}
            onPress={handleToggle}
          />
        </View>

        {/* Actions */}
        {online === true && (
          <View style={styles.actions}>
            <Pressable
              onPress={handleClear}
              disabled={isRec || isTx}
              style={({ pressed }) => [
                styles.actionBtn,
                pressed && { opacity: 0.6 },
                (isRec || isTx) && { opacity: 0.3 },
              ]}
            >
              <Ionicons
                name="trash-outline"
                size={17}
                color={COLORS.textMuted}
              />
              <Text style={styles.actionText}>Effacer</Text>
            </Pressable>

            <Pressable
              onPress={handleCopy}
              disabled={!transcript}
              style={({ pressed }) => [
                styles.actionBtn,
                pressed && { opacity: 0.6 },
                !transcript && { opacity: 0.3 },
              ]}
            >
              <Ionicons
                name="copy-outline"
                size={17}
                color={COLORS.textMuted}
              />
              <Text style={styles.actionText}>Copier</Text>
            </Pressable>
          </View>
        )}

        {/* Banner ffmpeg */}
        {health?.online && !health.ffmpegAvailable && (
          <View style={styles.bannerWarn}>
            <Ionicons
              name="warning-outline"
              size={13}
              color={COLORS.amber400}
            />
            <Text style={[styles.bannerText, { color: COLORS.amber400 }]}>
              ffmpeg absent — seul le WAV est accepté par le backend
            </Text>
          </View>
        )}
      </SafeAreaView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  safe: {
    flex: 1,
    paddingHorizontal: 18,
    paddingTop: 6,
    paddingBottom: 6,
    gap: 12,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  title: {
    fontSize: 26,
    fontWeight: "800",
    color: COLORS.text,
    letterSpacing: -0.5,
    lineHeight: 32,
  },
  onlineBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: RADIUS.full,
    borderWidth: StyleSheet.hairlineWidth,
    backgroundColor: "#161412",
    marginTop: 4,
  },
  onlineDot: { width: 6, height: 6, borderRadius: 3 },
  onlineText: { color: COLORS.textMuted, fontSize: 10, fontFamily: FONTS.mono },
  micRow: { alignItems: "center", paddingVertical: 4 },
  actions: { flexDirection: "row", justifyContent: "center", gap: 20 },
  actionBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: RADIUS.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#2c2825",
    backgroundColor: "#161412",
  },
  actionText: { color: COLORS.textMuted, fontSize: 13 },
  bannerWarn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 7,
    backgroundColor: "#1c1200",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#3d2e00",
    borderRadius: RADIUS.md,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  bannerText: { flex: 1, fontSize: 11, fontFamily: FONTS.mono, lineHeight: 16 },
});
