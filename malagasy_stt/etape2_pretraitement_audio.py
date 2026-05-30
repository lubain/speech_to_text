import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import os
import csv
from pathlib import Path

# ─────────────────────────────────────────────
# Installation des dépendances :
#   pip install librosa soundfile scipy numpy
# ─────────────────────────────────────────────


class AudioPreprocessor:
    """
    Pipeline complet de prétraitement audio :
      1. Chargement + rééchantillonnage
      2. Suppression des silences
      3. Réduction de bruit
      4. Normalisation
      5. Augmentation de données
    """

    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    # ──────────────────────────────────────────
    # 1. CHARGEMENT
    # ──────────────────────────────────────────

    def load_audio(self, filepath):
        """Charge un fichier audio et le rééchantillonne à target_sr."""
        audio, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        return audio, sr

    # ──────────────────────────────────────────
    # 2. SUPPRESSION DES SILENCES
    # ──────────────────────────────────────────

    def remove_silence(self, audio, threshold_db=-40, frame_length=2048):
        """
        Supprime les silences en début et fin, et entre les mots.
        threshold_db : seuil en dB sous lequel on considère silence
        """
        intervals = librosa.effects.split(
            audio,
            top_db=abs(threshold_db),
            frame_length=frame_length,
            hop_length=frame_length // 4
        )

        if len(intervals) == 0:
            return audio

        # Recolle uniquement les segments de parole
        audio_trimmed = np.concatenate([
            audio[start:end] for start, end in intervals
        ])
        return audio_trimmed

    # ──────────────────────────────────────────
    # 3. RÉDUCTION DE BRUIT
    # ──────────────────────────────────────────

    def reduce_noise_spectral(self, audio, sr, noise_duration=0.5):
        """
        Réduction de bruit par soustraction spectrale.
        Estime le bruit sur les premières noise_duration secondes.
        """
        noise_samples = int(sr * noise_duration)
        if len(audio) <= noise_samples:
            return audio  # signal trop court

        noise_sample = audio[:noise_samples]
        noise_stft   = librosa.stft(noise_sample)
        noise_mag    = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

        # STFT du signal complet
        stft      = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase     = np.angle(stft)

        # Soustraction spectrale avec plancher à 0
        magnitude_denoised = np.maximum(magnitude - noise_mag * 1.5, 0)

        # Reconstruction
        stft_denoised  = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, length=len(audio))
        return audio_denoised

    def apply_wiener_filter(self, audio):
        """
        Filtre de Wiener pour réduction de bruit plus agressive.
        Utile pour les enregistrements bruités.
        """
        from scipy.signal import wiener
        return wiener(audio, mysize=29).astype(np.float32)

    # ──────────────────────────────────────────
    # 4. NORMALISATION
    # ──────────────────────────────────────────

    def normalize_audio(self, audio, target_db=-20.0):
        """
        Normalise le volume audio à un niveau cible en dB.
        target_db : niveau RMS cible (défaut -20 dBFS)
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio

        target_rms = 10 ** (target_db / 20.0)
        gain = target_rms / rms

        # Évite l'écrêtage
        audio_normalized = audio * gain
        max_val = np.max(np.abs(audio_normalized))
        if max_val > 1.0:
            audio_normalized = audio_normalized / max_val

        return audio_normalized

    # ──────────────────────────────────────────
    # 5. AUGMENTATION DE DONNÉES
    # (critique pour les langues peu dotées !)
    # ──────────────────────────────────────────

    def augment_audio(self, audio, sr):
        """
        Génère plusieurs versions augmentées d'un même enregistrement.
        Multiplie la taille du dataset par ~6x.

        Retourne : liste de (nom_augmentation, audio_augmenté)
        """
        augmented = []

        # A. Bruit gaussien léger (simule bruit de fond)
        noise = np.random.normal(0, 0.004, audio.shape)
        augmented.append(("noise_light", audio + noise))

        # B. Bruit gaussien modéré (environnement bruyant)
        noise_heavy = np.random.normal(0, 0.012, audio.shape)
        augmented.append(("noise_heavy", audio + noise_heavy))

        # C. Vitesse +10% (locuteur rapide)
        try:
            audio_fast = librosa.effects.time_stretch(audio, rate=1.1)
            augmented.append(("speed_fast", audio_fast))
        except Exception:
            pass

        # D. Vitesse -10% (locuteur lent)
        try:
            audio_slow = librosa.effects.time_stretch(audio, rate=0.9)
            augmented.append(("speed_slow", audio_slow))
        except Exception:
            pass

        # E. Pitch +2 demi-tons (voix plus aiguë)
        try:
            audio_higher = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            augmented.append(("pitch_high", audio_higher))
        except Exception:
            pass

        # F. Pitch -2 demi-tons (voix plus grave)
        try:
            audio_lower = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
            augmented.append(("pitch_low", audio_lower))
        except Exception:
            pass

        # G. Simulation de reverb léger (pièce fermée)
        try:
            impulse = np.zeros(int(sr * 0.15))
            impulse[0] = 1.0
            impulse[int(sr * 0.025)] = 0.35
            impulse[int(sr * 0.05)]  = 0.15
            audio_reverb = signal.fftconvolve(audio, impulse, mode='full')[:len(audio)]
            augmented.append(("reverb", audio_reverb))
        except Exception:
            pass

        return augmented

    # ──────────────────────────────────────────
    # PIPELINE COMPLET POUR UN FICHIER
    # ──────────────────────────────────────────

    def preprocess_file(self, input_path, output_dir, augment=True, verbose=True):
        """
        Applique toutes les étapes de prétraitement à un fichier audio.
        Retourne la liste des fichiers générés.
        """
        if verbose:
            print(f"Traitement : {os.path.basename(input_path)}")

        audio, sr = self.load_audio(input_path)
        audio     = self.remove_silence(audio)
        audio     = self.reduce_noise_spectral(audio, sr)
        audio     = self.normalize_audio(audio)

        os.makedirs(output_dir, exist_ok=True)
        basename = Path(input_path).stem
        saved_files = []

        # Sauvegarde du fichier nettoyé
        clean_path = f"{output_dir}/{basename}_clean.wav"
        sf.write(clean_path, audio, sr)
        saved_files.append(clean_path)

        # Augmentations
        if augment:
            versions = self.augment_audio(audio, sr)
            for aug_name, aug_audio in versions:
                aug_audio = self.normalize_audio(aug_audio)
                aug_path  = f"{output_dir}/{basename}_{aug_name}.wav"
                sf.write(aug_path, aug_audio, sr)
                saved_files.append(aug_path)

        if verbose:
            print(f"{len(saved_files)} fichiers générés "
                  f"(1 clean + {len(saved_files)-1} augmentations)\n")
        return saved_files

    # ──────────────────────────────────────────
    # TRAITEMENT PAR LOT (BATCH)
    # ──────────────────────────────────────────

    def preprocess_dataset(self, metadata_csv, output_dir, augment=True):
        """
        Traite tout un dataset décrit dans un fichier metadata.csv.
        Génère un nouveau metadata_processed.csv avec les fichiers créés.
        """
        print(f"\n{'='*55}")
        print(f"Prétraitement du dataset")
        print(f"{'='*55}\n")

        processed_rows = []
        errors         = []

        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows   = list(reader)

        input_dir = os.path.dirname(metadata_csv)

        for i, row in enumerate(rows):
            audio_path = os.path.join(input_dir, "audio", row["filename"])
            print(f"  [{i+1}/{len(rows)}] {row['filename']}")

            if not os.path.exists(audio_path):
                print(f"         ⚠️  Fichier introuvable, ignoré.")
                errors.append(row["filename"])
                continue

            try:
                saved = self.preprocess_file(audio_path, output_dir, augment)
                for path in saved:
                    aug_type = Path(path).stem.split("_")[-1]
                    processed_rows.append({
                        "filename"     : os.path.basename(path),
                        "transcription": row["transcription"],
                        "speaker_id"   : row["speaker_id"],
                        "augmentation" : aug_type,
                        "original_file": row["filename"],
                    })
            except Exception as e:
                print(f"         ❌ Erreur : {e}")
                errors.append(row["filename"])

        # Sauvegarde du nouveau metadata
        out_csv = os.path.join(os.path.dirname(output_dir), "metadata_processed.csv")
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=processed_rows[0].keys())
            writer.writeheader()
            writer.writerows(processed_rows)

        print(f"\n{'='*55}")
        print(f"  Prétraitement terminé !")
        print(f"  Fichiers générés : {len(processed_rows)}")
        print(f"  Erreurs          : {len(errors)}")
        print(f"  Metadata         : {out_csv}")
        print(f"{'='*55}\n")
        return processed_rows, errors


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    preprocessor = AudioPreprocessor(target_sr=16000)

    # Traitement d'un fichier unique
    # preprocessor.preprocess_file(
    #     "dataset/audio/spk001_utt001.wav",
    #     "dataset/processed",
    #     augment=True
    # )

    # Traitement de tout le dataset
    # preprocessor.preprocess_dataset(
    #     "dataset/metadata.csv",
    #     "dataset/processed",
    #     augment=True
    # )

    print("Étape 2 terminée — Audio prétraité et augmenté dans ./dataset/processed/")
