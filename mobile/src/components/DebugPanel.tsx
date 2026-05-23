// Panneau de debug affiché quand le backend est inaccessible.
// Montre l'URL tentée + le message d'erreur exact.

import { View, Text, StyleSheet, Pressable } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { API_BASE_URL } from "@/lib/api";
import { COLORS, FONTS, RADIUS } from "@/constants/theme";

type Props = {
  errorDetail?: string;
  onRetry: () => void;
};

export function DebugPanel({ errorDetail, onRetry }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="warning" size={16} color={COLORS.red400} />
        <Text style={styles.title}>Backend inaccessible</Text>
      </View>

      <View style={styles.row}>
        <Text style={styles.label}>URL tentée :</Text>
        <Text style={styles.value}>{API_BASE_URL}/health</Text>
      </View>

      {errorDetail && (
        <View style={styles.row}>
          <Text style={styles.label}>Erreur :</Text>
          <Text style={[styles.value, { color: COLORS.red400 }]}>
            {errorDetail}
          </Text>
        </View>
      )}

      <View style={styles.divider} />

      <Text style={styles.hint}>Vérifiez que :</Text>
      <Text style={styles.check}>
        ✓ uvicorn tourne avec <Text style={styles.code}>--host 0.0.0.0</Text>
      </Text>
      <Text style={styles.check}>✓ Le téléphone est sur le même Wi-Fi</Text>
      <Text style={styles.check}>
        ✓ Le pare-feu Windows autorise le port 8000
      </Text>
      <Text style={styles.check}>
        ✓ L'IP dans api.ts correspond à{" "}
        <Text style={styles.code}>ipconfig</Text>
      </Text>

      <Pressable onPress={onRetry} style={styles.retryBtn}>
        <Ionicons name="refresh" size={14} color={COLORS.teal400} />
        <Text style={styles.retryText}>Réessayer</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    backgroundColor: "#1a0a0a",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#5a1a1a",
    borderRadius: RADIUS.lg,
    padding: 14,
    gap: 6,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 4,
  },
  title: { color: COLORS.red400, fontSize: 14, fontWeight: "700" },
  row: { gap: 2 },
  label: { color: COLORS.textFaint, fontSize: 11, fontFamily: FONTS.mono },
  value: { color: COLORS.textMuted, fontSize: 12, fontFamily: FONTS.mono },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "#3d1a1a",
    marginVertical: 4,
  },
  hint: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: "600",
    marginBottom: 2,
  },
  check: {
    color: COLORS.textFaint,
    fontSize: 12,
    fontFamily: FONTS.mono,
    lineHeight: 20,
  },
  code: { color: COLORS.teal300, fontFamily: FONTS.mono },
  retryBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    marginTop: 8,
    paddingVertical: 9,
    borderRadius: RADIUS.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: COLORS.teal700,
    backgroundColor: "#0b2a1e",
  },
  retryText: { color: COLORS.teal400, fontSize: 13, fontFamily: FONTS.sans },
});
