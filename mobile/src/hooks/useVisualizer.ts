// Visualiseur réactif au vrai signal micro via metering expo-av.

import { useEffect, useRef, useCallback } from "react";
import {
  useSharedValue,
  withTiming,
  withRepeat,
  withSequence,
  cancelAnimation,
  Easing,
  type SharedValue,
} from "react-native-reanimated";
import { NUM_BARS } from "@/constants/theme";

// Enveloppe spectrale gaussienne centrée sur les fréquences vocales
const ENVELOPE = Array.from({ length: NUM_BARS }, (_, i) => {
  const center = NUM_BARS * 0.38;
  const sigma = NUM_BARS * 0.3;
  return Math.exp(-0.5 * Math.pow((i - center) / sigma, 2));
});

// Facteur aléatoire fixe par barre (aspect spectre réaliste)
const BAR_RAND = Array.from(
  { length: NUM_BARS },
  () => 0.6 + Math.random() * 0.4,
);

// Niveau de bruit de fond Android (~-107 dB → norm ~0.33)
// On soustrait ce plancher pour que le silence donne des barres à zéro
const NOISE_FLOOR = 0.3;
const SIGNAL_GAIN = 1.8; // amplification du signal net

export function useVisualizer(isRecording: boolean) {
  // ── 40 SharedValues au top-level ─────────────────────────────────────────
  const b0 = useSharedValue(0.04);
  const b1 = useSharedValue(0.04);
  const b2 = useSharedValue(0.04);
  const b3 = useSharedValue(0.04);
  const b4 = useSharedValue(0.04);
  const b5 = useSharedValue(0.04);
  const b6 = useSharedValue(0.04);
  const b7 = useSharedValue(0.04);
  const b8 = useSharedValue(0.04);
  const b9 = useSharedValue(0.04);
  const b10 = useSharedValue(0.04);
  const b11 = useSharedValue(0.04);
  const b12 = useSharedValue(0.04);
  const b13 = useSharedValue(0.04);
  const b14 = useSharedValue(0.04);
  const b15 = useSharedValue(0.04);
  const b16 = useSharedValue(0.04);
  const b17 = useSharedValue(0.04);
  const b18 = useSharedValue(0.04);
  const b19 = useSharedValue(0.04);
  const b20 = useSharedValue(0.04);
  const b21 = useSharedValue(0.04);
  const b22 = useSharedValue(0.04);
  const b23 = useSharedValue(0.04);
  const b24 = useSharedValue(0.04);
  const b25 = useSharedValue(0.04);
  const b26 = useSharedValue(0.04);
  const b27 = useSharedValue(0.04);
  const b28 = useSharedValue(0.04);
  const b29 = useSharedValue(0.04);
  const b30 = useSharedValue(0.04);
  const b31 = useSharedValue(0.04);
  const b32 = useSharedValue(0.04);
  const b33 = useSharedValue(0.04);
  const b34 = useSharedValue(0.04);
  const b35 = useSharedValue(0.04);
  const b36 = useSharedValue(0.04);
  const b37 = useSharedValue(0.04);
  const b38 = useSharedValue(0.04);
  const b39 = useSharedValue(0.04);

  const bars: SharedValue<number>[] = [
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
    b16,
    b17,
    b18,
    b19,
    b20,
    b21,
    b22,
    b23,
    b24,
    b25,
    b26,
    b27,
    b28,
    b29,
    b30,
    b31,
    b32,
    b33,
    b34,
    b35,
    b36,
    b37,
    b38,
    b39,
  ];

  const levelRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── updateFromMeter : stocke le niveau net (signal - bruit de fond) ───────
  const updateFromMeter = useCallback((rawLevel: number) => {
    // Soustrait le bruit de fond, amplifie le signal net, clamp [0..1]
    const net = Math.max(0, (rawLevel - NOISE_FLOOR) * SIGNAL_GAIN);
    levelRef.current = Math.min(1, net);
  }, []);

  // ── Idle : onde sinusoïdale douce ─────────────────────────────────────────
  const startIdle = useCallback(() => {
    bars.forEach((bar, i) => {
      const base = 0.03 + Math.sin((i / NUM_BARS) * Math.PI) * 0.05;
      const dur = 1200 + i * 25;
      bar.value = withRepeat(
        withSequence(
          withTiming(base + 0.05, {
            duration: dur,
            easing: Easing.inOut(Easing.sin),
          }),
          withTiming(base, { duration: dur, easing: Easing.inOut(Easing.sin) }),
        ),
        -1,
        true,
      );
    });
  }, []);

  // ── Boucle recording : lit levelRef et anime les barres ───────────────────
  const startRecordingLoop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      const level = levelRef.current;
      bars.forEach((bar, i) => {
        const env = ENVELOPE[i];
        const rand = BAR_RAND[i];
        const micro = 0.8 + Math.random() * 0.2;
        const target = 0.04 + level * env * rand * micro;
        bar.value = withTiming(Math.min(0.96, target), { duration: 80 });
      });
    }, 80);
  }, []);

  const stopLoop = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // ── Réagit à isRecording ──────────────────────────────────────────────────
  useEffect(() => {
    if (isRecording) {
      bars.forEach((bar) => cancelAnimation(bar));
      levelRef.current = 0;
      startRecordingLoop();
    } else {
      stopLoop();
      levelRef.current = 0;
      bars.forEach((bar) => {
        cancelAnimation(bar);
        bar.value = withTiming(0.04, { duration: 300 });
      });
      const t = setTimeout(startIdle, 350);
      return () => clearTimeout(t);
    }
  }, [isRecording]);

  // ── Idle au montage ───────────────────────────────────────────────────────
  useEffect(() => {
    startIdle();
    return () => {
      stopLoop();
      bars.forEach((bar) => cancelAnimation(bar));
    };
  }, []);

  return { bars, updateFromMeter };
}
