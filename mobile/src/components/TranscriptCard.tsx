import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import type { TranscriptionResult, ApiError } from "@/lib/api";
import { COLORS, FONTS, RADIUS } from "@/constants/theme";

function MetaBadge({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.badge}>
      <Text style={styles.badgeLabel}>{label} </Text>
      <Text style={styles.badgeValue}>{value}</Text>
    </View>
  );
}

type Props = {
  isTranscribing: boolean;
  isRecording: boolean;
  transcript: string;
  result: TranscriptionResult | null;
  apiError: ApiError | null;
};

export function TranscriptCard({
  isTranscribing,
  isRecording,
  transcript,
  result,
  apiError,
}: Props) {
  return (
    <View style={styles.card}>
      {isTranscribing ? (
        <View style={styles.row}>
          <ActivityIndicator color={COLORS.amber400} size="small" />
          <Text style={[styles.hint, { marginLeft: 10 }]}>
            Transcription en cours…
          </Text>
        </View>
      ) : apiError ? (
        <>
          <Text style={styles.errorText}>{apiError.message}</Text>
          <Text style={styles.errorCode}>code : {apiError.code}</Text>
        </>
      ) : transcript ? (
        <>
          <ScrollView
            style={{ maxHeight: 160 }}
            showsVerticalScrollIndicator={false}
          >
            <Text style={styles.transcript}>
              {transcript}
              {isRecording && (
                <Text style={{ color: COLORS.textMuted }}>|</Text>
              )}
            </Text>
          </ScrollView>
          {result && (
            <View style={styles.badges}>
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
            </View>
          )}
        </>
      ) : (
        <Text style={styles.hint}>
          {isRecording
            ? "Enregistrement… appuyez sur Stop pour transcrire"
            : "La transcription apparaîtra ici"}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    width: "100%",
    minHeight: 96,
    backgroundColor: "#111010",
    borderRadius: RADIUS.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#2c2825",
    padding: 14,
  },
  row: { flexDirection: "row", alignItems: "center" },
  hint: {
    color: COLORS.textFaint,
    fontSize: 13,
    fontFamily: FONTS.mono,
    lineHeight: 20,
  },
  transcript: {
    color: COLORS.text,
    fontSize: 16,
    lineHeight: 26,
    fontFamily: FONTS.sans,
  },
  errorText: { color: COLORS.red400, fontSize: 14, marginBottom: 4 },
  errorCode: { color: COLORS.textFaint, fontSize: 11, fontFamily: FONTS.mono },
  badges: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 10 },
  badge: {
    flexDirection: "row",
    backgroundColor: "#1c1917",
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#2c2825",
  },
  badgeLabel: { color: COLORS.textFaint, fontSize: 11, fontFamily: FONTS.mono },
  badgeValue: { color: COLORS.textMuted, fontSize: 11, fontFamily: FONTS.mono },
});
