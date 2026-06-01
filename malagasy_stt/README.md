# 🎙️ Speech-to-Text Malgache — Pipeline Complet

Système de reconnaissance vocale pour la langue malgache, construit
étape par étape : collecte des données → prétraitement → extraction
des caractéristiques → modèle → entraînement → test.

---

## 📁 Structure du projet

```
malagasy_stt/
│
├── etape1_collecte_donnees.py      # Enregistrement + téléchargement Common Voice
├── etape2_pretraitement_audio.py   # Nettoyage, débruitage, augmentation
├── etape3_extraction_features.py   # Log-Mel, MFCC, features prosodiques
├── etape4_modele.py                # Whisper (fine-tuning)
├── etape5_entrainement.py          # HuggingFace Trainer
├── etape6_test_evaluation.py       # WER, CER, analyse d'erreurs, rapport HTML
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

---

## ⚡ Démarrage rapide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Vérification des dépendances

```bash
python main_pipeline.py --step check
```

### 3. Pipeline complet

```bash
python main_pipeline.py --step all
```

### 4. Étapes individuelles

```bash
python main_pipeline.py --step 1    # Collecte
python main_pipeline.py --step 2    # Prétraitement
python main_pipeline.py --step 3    # Features
python main_pipeline.py --step 4    # Modèle
python main_pipeline.py --step 5    # Entraînement
python main_pipeline.py --step 6 --model-path ./whisper-malagasy-final
```

### 5. Transcription rapide

```bash
python main_pipeline.py --transcribe mon_audio.wav --model-path ./whisper-malagasy-final
```

---

## 🗺️ Description des étapes

### Étape 1 — Collecte des données

- Enregistrement via microphone avec countdown
- 15 phrases malgaches prêtes à l'emploi
- Téléchargement automatique depuis Mozilla Common Voice (code langue : `mg`)
- Export CSV avec métadonnées (locuteur, durée, timestamp)

**Volume recommandé :**
| Objectif | Volume |
|---|---|
| Prototype | 10 heures |
| Bon modèle | 100 heures |
| Modèle robuste | 1 000+ heures |

---

### Étape 2 — Prétraitement audio

- Rééchantillonnage à 16 kHz (standard ASR)
- Suppression des silences (début, fin, inter-mots)
- Réduction de bruit par soustraction spectrale
- Normalisation du volume (LUFS cible)
- **Augmentation de données** (×6) :
  - Bruit gaussien léger/fort
  - Vitesse ±10%
  - Pitch ±2 demi-tons
  - Reverb simulé

---

### Étape 3 — Extraction des caractéristiques

| Feature             | Dimensions | Usage              |
| ------------------- | ---------- | ------------------ |
| Log-Mel Spectrogram | (T, 80)    | Whisper, Conformer |
| MFCC + Δ + ΔΔ       | (T, 120)   | CTC classique      |
| F0 + RMS + ZCR      | (T, 3)     | Analyse prosodique |

Normalisation CMVN appliquée (robustesse micro/environnement).

---

### Étape 4 — Modèle

**Fine-tuning Whisper (recommandé)**

- Architecture : Encoder-Decoder Transformer
- Entrée : Log-Mel 80 canaux (fenêtre 30s)
- Génération auto-régressive token par token
- Supporte les phrases de longueur quelconque
- Tailles : `tiny` (39M) → `large` (1.5B paramètres)

---

### Étape 5 — Entraînement

**Whisper fine-tuning :**

```
learning_rate = 1e-5
warmup_steps  = 500
max_steps     = 5000
batch_size    = 8
fp16          = True (si GPU)
metric        = WER (minimisé)
```

---

### Étape 6 — Test et évaluation

**Métriques :**
| Métrique | Description | Cible |
|---|---|---|
| WER | Word Error Rate | < 15% |
| CER | Character Error Rate | < 5% |

**Interprétation WER :**

- `< 5%` → Excellent (niveau commercial)
- `5–15%` → Bon (utilisable en production)
- `15–30%` → Acceptable (à améliorer)
- `> 30%` → Insuffisant

**Sorties :**

- Rapport HTML interactif (`rapport_evaluation.html`)
- JSON détaillé (`evaluation_results.json`)
- Analyse des erreurs (substitutions, insertions, suppressions)

---

## 💡 Conseils pour le malgache

1. **Diversité des locuteurs** : variez les âges, régions (Merina, Côtier, etc.) et genres
2. **Environnements variés** : enregistrez en intérieur, extérieur, avec bruit de fond
3. **Phonèmes spécifiques** : couvrez les groupes consonantiques `tr`, `dr`, `ntr`, `nd`
4. **Voyelles longues** : incluez des mots avec `aa`, `ee`, `oo`
5. **Code-switching** : le malgache moderne mélange souvent avec le français

---

## 🔗 Ressources

- [Mozilla Common Voice — Malgache](https://commonvoice.mozilla.org/mg)
- [Whisper OpenAI](https://github.com/openai/whisper)
- [HuggingFace ASR Guide](https://huggingface.co/docs/transformers/tasks/asr)
- [Alphabet malgache officiel](https://fr.wikipedia.org/wiki/Langue_malgache)

---

## 📄 Licence

MIT License — Libre d'utilisation, modification et distribution.
