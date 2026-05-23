import { View, Text, ScrollView, Pressable, StyleSheet } from "react-native";
import type { Language, Engine } from "@/lib/api";
import { COLORS, FONTS, RADIUS } from "@/constants/theme";

type PillProps = {
  label: string;
  selected: boolean;
  disabled: boolean;
  onPress: () => void;
};

function Pill({ label, selected, disabled, onPress }: PillProps) {
  const short = label.split(" ")[0] ?? label;
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={[
        styles.pill,
        selected && styles.pillSelected,
        disabled && styles.pillDisabled,
      ]}
    >
      <Text style={[styles.pillText, selected && styles.pillTextSelected]}>
        {short}
      </Text>
    </Pressable>
  );
}

type Props = {
  languages: Language[];
  engines: Engine[];
  selectedLang: string;
  selectedEngine: string;
  disabled: boolean;
  onLangChange: (v: string) => void;
  onEngineChange: (v: string) => void;
};

export function LanguageEnginePicker({
  languages,
  engines,
  selectedLang,
  selectedEngine,
  disabled,
  onLangChange,
  onEngineChange,
}: Props) {
  const langs = languages.length
    ? languages
    : [{ code: "fr-FR", label: "Français" }];
  const engs = engines.length
    ? engines
    : [
        {
          id: "google",
          label: "Google",
          requires_key: false,
          offline: false,
          description: "",
        },
      ];

  return (
    <View style={styles.container}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        {langs.map((l) => (
          <Pill
            key={l.code}
            label={l.label}
            selected={selectedLang === l.code}
            disabled={disabled}
            onPress={() => onLangChange(l.code)}
          />
        ))}
      </ScrollView>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={{ marginTop: 6 }}
      >
        {engs.map((e) => (
          <Pill
            key={e.id}
            label={e.label}
            selected={selectedEngine === e.id}
            disabled={disabled}
            onPress={() => onEngineChange(e.id)}
          />
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: "100%", gap: 0 },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: RADIUS.full,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#3d3530",
    backgroundColor: "#1c1917",
    marginRight: 8,
  },
  pillSelected: { borderColor: COLORS.teal600, backgroundColor: "#0b2a1e" },
  pillDisabled: { opacity: 0.35 },
  pillText: { color: COLORS.textMuted, fontSize: 12, fontFamily: FONTS.mono },
  pillTextSelected: { color: COLORS.teal300 },
});
