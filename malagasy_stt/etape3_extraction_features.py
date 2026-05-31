import librosa
import numpy as np
import os
import pickle
from pathlib import Path

# ─────────────────────────────────────────────
# Installation des dépendances :
#   pip install librosa numpy
# ─────────────────────────────────────────────


class FeatureExtractor:
    """
    Extraction des caractéristiques acoustiques pour l'ASR.

    Méthodes disponibles :
      - Log-Mel Spectrogram  → recommandé pour Whisper / Transformers
      - MFCC + Delta         → classique, compatible CTC
      - Features combinées   → pour analyse et visualisation
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,          # taille FFT
        hop_length=160,     # pas de 10ms (16000 * 0.01)
        win_length=400,     # fenêtre de 25ms (16000 * 0.025)
        n_mels=80,          # canaux Mel — standard Whisper
        n_mfcc=40,          # coefficients MFCC
    ):
        self.sr         = sample_rate
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels     = n_mels
        self.n_mfcc     = n_mfcc

    # ──────────────────────────────────────────
    # LOG-MEL SPECTROGRAM (recommandé)
    # ──────────────────────────────────────────

    def extract_log_mel(self, audio):
        """
        Log-Mel Spectrogram — utilisé par Whisper, wav2vec2, Conformer.
        C'est la représentation de référence pour les modèles modernes.

        Shape sortie : (T, n_mels) = (frames, 80)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=80,    # fréquence minimale (filtre la rumeur)
            fmax=8000,  # fréquence maximale
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel.T  # → (T, 80)

    # ──────────────────────────────────────────
    # MFCC + DELTA (classique)
    # ──────────────────────────────────────────

    def extract_mfcc_with_deltas(self, audio):
        """
        MFCC avec dérivées 1er et 2ème ordre.
        Capture la dynamique temporelle du signal vocal.

        Shape sortie : (T, n_mfcc * 3)
        """
        mfcc        = librosa.feature.mfcc(
            y=audio, sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        delta       = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)

        # Concaténation : (n_mfcc*3, T)
        features = np.concatenate([mfcc, delta, delta_delta], axis=0)
        return features.T  # → (T, n_mfcc*3)

    # ──────────────────────────────────────────
    # FEATURES PROSODIQUES
    # ──────────────────────────────────────────

    def extract_prosodic(self, audio):
        """
        Features prosodiques : F0 (pitch), énergie, ZCR.
        Utiles pour capturer l'intonation malgache.
        """
        # Fréquence fondamentale (F0 / pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            hop_length=self.hop_length
        )
        f0 = np.nan_to_num(f0)  # remplace NaN par 0 (non voisé)

        # Énergie RMS par frame
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )

        # Centroïde spectral
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Harmonicité (HNR — Harmonic to Noise Ratio)
        harmonic, percussive = librosa.effects.hpss(audio)
        hnr = np.sqrt(np.mean(harmonic**2)) / (np.sqrt(np.mean(percussive**2)) + 1e-8)

        return {
            "f0"       : f0,
            "rms"      : rms.flatten(),
            "zcr"      : zcr.flatten(),
            "centroid" : centroid.flatten(),
            "hnr"      : hnr,
        }

    # ──────────────────────────────────────────
    # NORMALISATION CMVN
    # ──────────────────────────────────────────

    def normalize_cmvn(self, features):
        """
        Cepstral Mean and Variance Normalization (CMVN).
        Standard en ASR : rend le modèle robuste aux différents
        microphones et conditions d'enregistrement.
        """
        mean = np.mean(features, axis=0)
        std  = np.std(features, axis=0) + 1e-8
        return (features - mean) / std

    def normalize_global(self, features, global_mean=None, global_std=None):
        """
        Normalisation globale sur tout le dataset.
        À utiliser si vous avez calculé mean/std sur le corpus entier.
        """
        if global_mean is None or global_std is None:
            global_mean = np.mean(features, axis=0)
            global_std  = np.std(features, axis=0) + 1e-8
        return (features - global_mean) / global_std, global_mean, global_std

    # ──────────────────────────────────────────
    # EXTRACTION COMPLÈTE
    # ──────────────────────────────────────────

    def extract_for_whisper(self, audio):
        """
        Extraction optimisée pour le fine-tuning de Whisper.
        Retourne le Log-Mel normalisé.
        """
        log_mel = self.extract_log_mel(audio)
        log_mel = self.normalize_cmvn(log_mel)
        return log_mel  # (T, 80)

    def extract_all(self, audio):
        """Extraction complète pour analyse et visualisation."""
        return {
            "log_mel"  : self.extract_log_mel(audio),
            "mfcc"     : self.extract_mfcc_with_deltas(audio),
            "prosodic" : self.extract_prosodic(audio),
        }

    # ──────────────────────────────────────────
    # TRAITEMENT PAR LOT (DATASET)
    # ──────────────────────────────────────────

    def process_dataset(self, audio_dir, output_dir, method="log_mel"):
        """
        Extrait les features de tous les fichiers WAV d'un répertoire.

        method : "log_mel" | "mfcc" | "all"
        Sauvegarde les features en .npy pour chaque fichier.
        """
        os.makedirs(output_dir, exist_ok=True)
        wav_files = list(Path(audio_dir).glob("*.wav"))

        print(f"\n{'='*55}")
        print(f" Extraction des features — méthode : {method}")
        print(f"  {len(wav_files)} fichiers à traiter")
        print(f"{'='*55}\n")

        global_stats = {"mean": [], "std": []}

        for i, wav_path in enumerate(wav_files):
            try:
                audio, _ = librosa.load(str(wav_path), sr=self.sr, mono=True)

                if method == "log_mel":
                    features = self.extract_for_whisper(audio)
                elif method == "mfcc":
                    features = self.normalize_cmvn(
                        self.extract_mfcc_with_deltas(audio)
                    )
                else:
                    features = self.extract_log_mel(audio)

                # Sauvegarde en numpy
                out_path = os.path.join(output_dir, wav_path.stem + ".npy")
                np.save(out_path, features)

                global_stats["mean"].append(np.mean(features, axis=0))
                global_stats["std"].append(np.std(features, axis=0))

                if (i + 1) % 20 == 0 or i == 0:
                    print(f"  [{i+1}/{len(wav_files)}] {wav_path.name}"
                          f"  →  shape {features.shape}")

            except Exception as e:
                print(f"  ❌ Erreur sur {wav_path.name} : {e}")

        # Sauvegarde des statistiques globales
        global_mean = np.mean(global_stats["mean"], axis=0)
        global_std  = np.mean(global_stats["std"],  axis=0)
        stats_path  = os.path.join(output_dir, "global_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump({"mean": global_mean, "std": global_std}, f)

        print(f"\n{'='*55}")
        print(f"  Extraction terminée !")
        print(f"  Features sauvegardées dans : {output_dir}/")
        print(f"  Statistiques globales      : {stats_path}")
        print(f"{'='*55}\n")

        return global_mean, global_std

    def describe_features(self, audio):
        """Affiche un résumé des features extraites."""
        log_mel = self.extract_log_mel(audio)
        mfcc    = self.extract_mfcc_with_deltas(audio)
        dur_s   = len(audio) / self.sr
        n_frames = log_mel.shape[0]
        frame_dur_ms = self.hop_length / self.sr * 1000

        print(f"""
  ┌─────────────────────────────────────────────┐
  │  RÉSUMÉ DES FEATURES EXTRAITES              │
  ├─────────────────────────────────────────────┤
  │  Durée audio         : {dur_s:.2f}s               │
  │  Frames totales      : {n_frames}              │
  │  Durée par frame     : {frame_dur_ms:.1f}ms             │
  ├─────────────────────────────────────────────┤
  │  Log-Mel             : {log_mel.shape}      │
  │  MFCC + Δ + ΔΔ       : {mfcc.shape}     │
  │  Fréquence d'éch.    : {self.sr} Hz            │
  │  FFT size            : {self.n_fft}                  │
  └─────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import librosa

    extractor = FeatureExtractor(sample_rate=16000)

    # Test sur un fichier synthétique (440 Hz, 2 secondes)
    sr        = 16000
    duration  = 2.0
    t         = np.linspace(0, duration, int(sr * duration))
    audio_test = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("Test d'extraction sur signal synthétique...")
    log_mel = extractor.extract_for_whisper(audio_test)
    mfcc    = extractor.extract_mfcc_with_deltas(audio_test)
    extractor.describe_features(audio_test)

    # Traitement du dataset complet
    # extractor.process_dataset(
    #     audio_dir  = "dataset/processed",
    #     output_dir = "dataset/features",
    #     method     = "log_mel"
    # )

    print("Étape 3 terminée — Features extraites dans ./dataset/features/")
