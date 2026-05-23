# Voice Recognition — Mobile (Expo SDK 52)

Application mobile Expo / React Native connectée au backend FastAPI.

> **Expo SDK 52** — Compatible avec **Expo Go** sur Android et iOS.

## Installation

```bash
cd mobile
npm install
```

## ⚠️ Configuration IP du backend

`localhost` ne fonctionne **pas** sur appareil physique ou émulateur Android.

Éditez `src/lib/api.ts` :

```ts
// Android émulateur (AVD)
export const API_BASE_URL = "http://10.0.2.2:8000";

// Appareil physique Android ou iOS — votre IP LAN
export const API_BASE_URL = "http://192.168.1.XX:8000";

// iOS Simulator uniquement
export const API_BASE_URL = "http://localhost:8000";
```

Trouver votre IP :

- **Windows** : `ipconfig` → "Adresse IPv4"
- **macOS/Linux** : `ifconfig | grep "inet "` → `en0` ou `wlan0`

## Lancement

```bash
# Expo Go (scanner le QR avec l'app Expo Go sur le téléphone)
npx expo start --clear

# Émulateur Android
npx expo start --android

# Simulateur iOS
npx expo start --ios
```

## Architecture

```
mobile/
├── app/
│   ├── _layout.tsx              Root layout (Gesture + SafeArea)
│   └── index.tsx                Page principale
├── src/
│   ├── components/
│   │   ├── VoiceRecognitionScreen.tsx   Écran principal
│   │   ├── AudioVisualizer.tsx          40 bâtonnets Reanimated
│   │   ├── MicButton.tsx                Bouton animé + anneaux
│   │   ├── TranscriptCard.tsx           Zone transcription
│   │   ├── LanguageEnginePicker.tsx     Pills scrollables
│   │   └── StatusDot.tsx                Point de statut animé
│   ├── hooks/
│   │   ├── useRecorder.ts              expo-av — enregistrement m4a 16kHz
│   │   └── useVisualizer.ts            40 SharedValues Reanimated
│   ├── lib/
│   │   └── api.ts                      Client HTTP FastAPI
│   └── constants/
│       └── theme.ts                    Couleurs, polices, constantes
├── metro.config.js                     Alias @/ → src/
├── babel.config.js
├── tsconfig.json
└── app.json
```

## Fonctionnalités

- 🎙 Enregistrement m4a via `expo-av` (16kHz mono, qualité optimisée ASR)
- 📊 Visualiseur 40 bâtonnets animés (`react-native-reanimated`)
- ✨ Bouton micro avec anneaux de pulsation et haptics
- 📋 Copie dans le presse-papiers (`expo-clipboard`)
- 🌍 Sélection langue + moteur (pills scrollables)
- 🟢 Indicateur connexion backend temps réel
- 🔔 Retours haptiques sur les actions clés
- 🌙 Thème sombre complet
