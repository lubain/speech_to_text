import { View, StyleSheet } from "react-native";
import Animated, {
  useAnimatedStyle,
  interpolateColor,
} from "react-native-reanimated";
import type { SharedValue } from "react-native-reanimated";
import { COLORS } from "@/constants/theme";

type BarProps = {
  value: SharedValue<number>;
  isRecording: boolean;
};

function Bar({ value, isRecording }: BarProps) {
  const animStyle = useAnimatedStyle(() => {
    const h = Math.max(4, value.value * 92);
    const color = isRecording
      ? interpolateColor(value.value, [0, 0.25, 0.5, 0.75, 1], COLORS.viz)
      : "rgba(120,113,108,0.28)";
    return { height: h, backgroundColor: color };
  });

  return <Animated.View style={[styles.bar, animStyle]} />;
}

type Props = {
  bars: SharedValue<number>[];
  isRecording: boolean;
};

export function AudioVisualizer({ bars, isRecording }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.inner}>
        {bars.map((bar, i) => (
          <Bar key={i} value={bar} isRecording={isRecording} />
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    height: 96,
    backgroundColor: "#111010",
    borderRadius: 16,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#2c2825",
    overflow: "hidden",
    justifyContent: "flex-end",
    paddingHorizontal: 6,
    paddingBottom: 6,
  },
  inner: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 2,
    height: "100%",
  },
  bar: {
    flex: 1,
    borderRadius: 3,
    minHeight: 4,
  },
});
