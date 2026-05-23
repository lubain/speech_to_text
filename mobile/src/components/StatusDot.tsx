import { View, Text, StyleSheet } from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  cancelAnimation,
} from "react-native-reanimated";
import { useEffect } from "react";
import { COLORS, FONTS } from "@/constants/theme";
import type { HealthStatus } from "@/lib/api";

type AppStatus =
  | "loading"
  | "idle"
  | "recording"
  | "transcribing"
  | "done"
  | "error";

type Props = {
  appStatus: AppStatus;
  health: HealthStatus | null;
  msg: string;
};

export function StatusDot({ appStatus, health, msg }: Props) {
  const opacity = useSharedValue(1);
  const isActive = appStatus === "recording" || appStatus === "transcribing";

  useEffect(() => {
    if (isActive) {
      opacity.value = withRepeat(
        withSequence(
          withTiming(0.25, { duration: 550 }),
          withTiming(1.0, { duration: 550 }),
        ),
        -1,
      );
    } else {
      cancelAnimation(opacity);
      opacity.value = withTiming(1, { duration: 200 });
    }
  }, [isActive]);

  const dotStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

  const dotColor =
    appStatus === "recording"
      ? COLORS.teal500
      : appStatus === "transcribing"
        ? COLORS.amber500
        : appStatus === "error"
          ? COLORS.red400
          : health?.online === false
            ? COLORS.red400
            : health?.online === true
              ? COLORS.teal500
              : "#57534e";

  return (
    <View style={styles.row}>
      <Animated.View
        style={[styles.dot, { backgroundColor: dotColor }, dotStyle]}
      />
      <Text style={styles.msg} numberOfLines={2}>
        {msg}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "center", gap: 8, width: "100%" },
  dot: { width: 7, height: 7, borderRadius: 4, flexShrink: 0 },
  msg: {
    flex: 1,
    color: COLORS.textFaint,
    fontSize: 12,
    fontFamily: FONTS.mono,
    lineHeight: 17,
  },
});
