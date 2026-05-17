# Voice Recognition API — FastAPI Backend

API REST de reconnaissance vocale construite avec **FastAPI** et le package Python **SpeechRecognition**.

## Architecture

```
voice-backend/
├── app/
│   ├── main.py                  # Point d'entrée FastAPI + CORS
│   ├── config.py                # Settings via pydantic-settings (.env)
│   ├── schemas.py               # Modèles Pydantic (requête / réponse)
│   ├── routers/
│   │   ├── health.py            # GET /health
│   │   └── transcription.py     # POST /transcribe, /transcribe/base64, GET /engines, /languages
│   └── services/
│       └── transcription.py     # Logique métier : conversion audio + reconnaissance
├── tests/
│   └── test_api.py              # Suite de tests pytest
├── requirements.txt
├── .env.example
└── README.md
```

## Prérequis système

- Python 3.11+
- **ffmpeg** (requis par pydub pour la conversion audio)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Télécharger sur https://ffmpeg.org/download.html et ajouter au PATH
```

## Installation

```bash
# 1. Cloner / copier le projet
cd voice-backend

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
cp .env.example .env
# Éditez .env selon vos besoins
```

## Lancement

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera disponible sur `http://localhost:8000`.  
Documentation interactive : `http://localhost:8000/docs`

## Endpoints

| Méthode | URL | Description |
|---------|-----|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/transcribe` | Upload fichier audio (multipart) |
| `POST` | `/api/v1/transcribe/base64` | Audio encodé en base64 (JSON) |
| `GET` | `/api/v1/engines` | Moteurs disponibles |
| `GET` | `/api/v1/languages` | Langues supportées |

## Utilisation

### Upload fichier audio

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@enregistrement.wav;type=audio/wav" \
  -F "language=fr-FR" \
  -F "engine=google"
```

### Audio en base64 (depuis le frontend)

```typescript
const blob = new Blob(chunks, { type: "audio/webm" });
const buffer = await blob.arrayBuffer();
const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));

const res = await fetch("http://localhost:8000/api/v1/transcribe/base64", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    audio_base64: base64,
    content_type: "audio/webm",
    language: "fr-FR",
    engine: "google",
  }),
});
const data = await res.json();
console.log(data.transcript);
```

### Réponse type

```json
{
  "transcript": "Bonjour, ceci est un test de reconnaissance vocale.",
  "language": "fr-FR",
  "engine": "google",
  "duration_seconds": 1.243,
  "confidence": 0.9821,
  "audio_duration_seconds": 3.5
}
```

## Moteurs disponibles

| ID | Nom | Clé API | Hors-ligne |
|----|-----|---------|------------|
| `google` | Google Web Speech API | Non (gratuit) | Non |
| `sphinx` | CMU Sphinx | Non | Oui |

Pour activer Sphinx : `pip install pocketsphinx` puis `DEFAULT_ENGINE=sphinx` dans `.env`.

## Tests

```bash
pytest tests/ -v
```

## Formats audio supportés

WAV, WebM, OGG, MP3, MP4, FLAC (conversion automatique via pydub/ffmpeg).

## Intégration avec le frontend React

Dans `VoiceRecognition.tsx`, utilisez `MediaRecorder` pour capturer l'audio du micro,
puis envoyez les chunks en base64 à `POST /api/v1/transcribe/base64` à la fin de l'enregistrement :

```typescript
const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
const chunks: Blob[] = [];

mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
mediaRecorder.onstop = async () => {
  const blob = new Blob(chunks, { type: "audio/webm" });
  // → envoyer au backend
};
```
