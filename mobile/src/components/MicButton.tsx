import { Pressable, StyleSheet, View, ActivityIndicator } from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  cancelAnimation,
  Easing,
} from "react-native-reanimated";
import { useEffect } from "react";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { COLORS } from "@/constants/theme";

const BTN = 88;
const RING = BTN + 28;

type Props = {
  isRecording: boolean;
  isTranscribing: boolean;
  disabled: boolean;
  onPress: () => void;
};

export function MicButton({
  isRecording,
  isTranscribing,
  disabled,
  onPress,
}: Props) {
  const scale = useSharedValue(1);
  const ringS1 = useSharedValue(1);
  const ringO1 = useSharedValue(0);
  const ringS2 = useSharedValue(1);
  const ringO2 = useSharedValue(0);

  useEffect(() => {
    if (isRecording) {
      scale.value = withRepeat(
        withSequence(
          withTiming(1.07, {
            duration: 550,
            easing: Easing.inOut(Easing.ease),
          }),
          withTiming(1.0, { duration: 550, easing: Easing.inOut(Easing.ease) }),
        ),
        -1,
      );
      ringS1.value = withRepeat(
        withTiming(1.85, { duration: 1400, easing: Easing.out(Easing.quad) }),
        -1,
      );
      ringO1.value = withRepeat(
        withSequence(
          withTiming(0.55, { duration: 80 }),
          withTiming(0, { duration: 1320 }),
        ),
        -1,
      );
      setTimeout(() => {
        ringS2.value = withRepeat(
          withTiming(1.85, { duration: 1400, easing: Easing.out(Easing.quad) }),
          -1,
        );
        ringO2.value = withRepeat(
          withSequence(
            withTiming(0.4, { duration: 80 }),
            withTiming(0, { duration: 1320 }),
          ),
          -1,
        );
      }, 700);
    } else {
      cancelAnimation(scale);
      cancelAnimation(ringS1);
      cancelAnimation(ringO1);
      cancelAnimation(ringS2);
      cancelAnimation(ringO2);
      scale.value = withTiming(1, { duration: 250 });
      ringS1.value = withTiming(1, { duration: 250 });
      ringO1.value = withTiming(0, { duration: 250 });
      ringS2.value = withTiming(1, { duration: 250 });
      ringO2.value = withTiming(0, { duration: 250 });
    }
  }, [isRecording]);

  const btnAnim = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));
  const ring1Anim = useAnimatedStyle(() => ({
    transform: [{ scale: ringS1.value }],
    opacity: ringO1.value,
  }));
  const ring2Anim = useAnimatedStyle(() => ({
    transform: [{ scale: ringS2.value }],
    opacity: ringO2.value,
  }));

  const handlePress = () => {
    if (disabled || isTranscribing) return;
    Haptics.impactAsync(
      isRecording
        ? Haptics.ImpactFeedbackStyle.Medium
        : Haptics.ImpactFeedbackStyle.Heavy,
    );
    onPress();
  };

  const bg = isRecording
    ? COLORS.teal600
    : isTranscribing
      ? "#78350f"
      : "#1c1917";

  return (
    <View style={styles.wrapper}>
      <Animated.View style={[styles.ring, ring1Anim]} />
      <Animated.View style={[styles.ring, ring2Anim]} />
      <Pressable onPress={handlePress} disabled={disabled || isTranscribing}>
        <Animated.View style={[styles.btn, { backgroundColor: bg }, btnAnim]}>
          {isTranscribing ? (
            <ActivityIndicator color={COLORS.amber400} size="large" />
          ) : isRecording ? (
            <Ionicons name="stop" size={34} color="#fff" />
          ) : (
            <Ionicons name="mic" size={34} color={COLORS.teal400} />
          )}
        </Animated.View>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    alignItems: "center",
    justifyContent: "center",
    width: RING + 20,
    height: RING + 20,
  },
  ring: {
    position: "absolute",
    width: RING,
    height: RING,
    borderRadius: RING / 2,
    borderWidth: 1.5,
    borderColor: COLORS.teal500,
  },
  btn: {
    width: BTN,
    height: BTN,
    borderRadius: BTN / 2,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#3d3530",
    elevation: 8,
    shadowColor: COLORS.teal500,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 20,
  },
});
