# Speech to Text — Fullstack

Reconnaissance vocale fullstack : **React 19 + Vite** (frontend) et **FastAPI + SpeechRecognition** (backend).

```
speech_to_text/
├── frontend/          React 19, Vite, Tailwind CSS 4, TypeScript
└── backend/           FastAPI, SpeechRecognition, Python 3.10+
```

## Démarrage rapide

### 1. Backend

```bash
cd backend

# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer
cp .env.example .env

# Lancer
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **Windows sans ffmpeg** : seul le format WAV est supporté.  
> Le frontend bascule automatiquement en mode WAV dans ce cas.  
> Pour activer WebM/OGG/MP3 : [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)

API disponible sur `http://localhost:8000` · Docs : `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend

# Installer les dépendances
npm install

# Lancer en développement
npm run dev
```

Frontend disponible sur `http://localhost:5173`

> Le proxy Vite redirige `/api` et `/health` vers le backend automatiquement.  
> Pas besoin de configurer `VITE_API_URL` en développement local.

### 3. Tests backend

```bash
cd backend
pytest tests/ -v
```

## Architecture

```
frontend/src/
├── components/
│   ├── VoiceRecognition.tsx   Composant principal
│   ├── AudioVisualizer.tsx    Canvas animé (bâtonnets)
│   ├── TranscriptBox.tsx      Zone de transcription + métadonnées
│   ├── StatusBar.tsx          Statut + sélecteurs langue/moteur
│   ├── Banners.tsx            Alertes ffmpeg / backend offline
│   ├── MetaBadge.tsx          Badge confiance/durée/moteur
│   └── Icons.tsx              SVG icons
├── hooks/
│   ├── useAudioVisualizer.ts  Canvas + WebAudio API
│   └── useMediaRecorder.ts    Capture micro → Blob audio
└── lib/
    └── api.ts                 Client HTTP FastAPI

backend/app/
├── main.py                    FastAPI + CORS + error handler
├── config.py                  Settings (.env)
├── schemas.py                 Modèles Pydantic
├── routers/
│   ├── health.py              GET /health
│   └── transcription.py      POST /transcribe, /transcribe/base64
└── services/
    └── transcription.py       SpeechRecognition + conversion audio
```

## Endpoints API

| Méthode | URL                         | Description                      |
| ------- | --------------------------- | -------------------------------- |
| `GET`   | `/health`                   | État du backend + dispo ffmpeg   |
| `POST`  | `/api/v1/transcribe`        | Upload fichier audio (multipart) |
| `POST`  | `/api/v1/transcribe/base64` | Audio base64 (JSON)              |
| `GET`   | `/api/v1/engines`           | Moteurs disponibles              |
| `GET`   | `/api/v1/languages`         | Langues supportées               |
