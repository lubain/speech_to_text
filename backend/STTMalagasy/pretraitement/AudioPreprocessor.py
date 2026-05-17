import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import os

class AudioPreprocessor:
    def __init__(self, target_sr=16000, target_duration=None):
        self.target_sr = target_sr
        self.target_duration = target_duration

    def load_audio(self, filepath):
        """Charge et resample l'audio"""
        audio, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        return audio, sr

    def remove_silence(self, audio, threshold_db=-40, frame_length=2048):
        """Supprime les silences au début et à la fin"""
        # Calcul de l'énergie par frame
        intervals = librosa.effects.split(
            audio, 
            top_db=abs(threshold_db),
            frame_length=frame_length
        )
        
        if len(intervals) == 0:
            return audio
            
        # Recolle les segments non-silencieux
        audio_trimmed = np.concatenate([
            audio[start:end] for start, end in intervals
        ])
        return audio_trimmed

    def normalize_audio(self, audio):
        """Normalise le volume entre -1 et 1"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def reduce_noise(self, audio, sr):
        """Réduction de bruit simple par spectral subtraction"""
        # Estimation du bruit sur les 0.5 premières secondes
        noise_sample = audio[:int(sr * 0.5)]
        noise_power = np.mean(noise_sample ** 2)
        
        # STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Soustraction spectrale
        noise_mag = np.sqrt(noise_power) * np.ones_like(magnitude)
        magnitude_denoised = np.maximum(magnitude - noise_mag * 2, 0)
        
        # Reconstruction
        phase = np.angle(stft)
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised)
        
        return audio_denoised

    def augment_audio(self, audio, sr):
        """
        Augmentation de données pour enrichir le dataset
        Crucial pour les langues peu dotées !
        """
        augmented = []
        
        # 1. Bruit gaussien léger
        noise = np.random.normal(0, 0.005, audio.shape)
        augmented.append(("noisy", audio + noise))
        
        # 2. Changement de vitesse (±10%)
        audio_fast = librosa.effects.time_stretch(audio, rate=1.1)
        audio_slow = librosa.effects.time_stretch(audio, rate=0.9)
        augmented.append(("fast", audio_fast))
        augmented.append(("slow", audio_slow))
        
        # 3. Changement de pitch (±2 demi-tons)
        audio_higher = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        audio_lower  = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
        augmented.append(("higher", audio_higher))
        augmented.append(("lower",  audio_lower))
        
        # 4. Simulation d'environnement (reverb léger)
        impulse = np.zeros(int(sr * 0.1))
        impulse[0] = 1.0
        impulse[int(sr * 0.02)] = 0.3
        audio_reverb = signal.fftconvolve(audio, impulse, mode='same')
        augmented.append(("reverb", audio_reverb))
        
        return augmented

    def preprocess_file(self, input_path, output_dir, augment=True):
        """Pipeline complet de prétraitement pour un fichier"""
        print(f"Traitement: {input_path}")
        
        audio, sr = self.load_audio(input_path)
        audio = self.remove_silence(audio)
        audio = self.reduce_noise(audio, sr)
        audio = self.normalize_audio(audio)
        
        # Sauvegarde du fichier nettoyé
        basename = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{basename}_clean.wav"
        sf.write(output_path, audio, sr)
        
        saved_files = [output_path]
        
        # Augmentation si demandée
        if augment:
            augmented_versions = self.augment_audio(audio, sr)
            for aug_name, aug_audio in augmented_versions:
                aug_path = f"{output_dir}/{basename}_{aug_name}.wav"
                # Normalise avant de sauvegarder
                aug_audio = self.normalize_audio(aug_audio)
                sf.write(aug_path, aug_audio, sr)
                saved_files.append(aug_path)
        
        print(f"{len(saved_files)} fichiers générés (original + augmentations)")
        return saved_files

# Exemple d'utilisation
preprocessor = AudioPreprocessor(target_sr=16000)
preprocessor.preprocess_file("dataset/audio/sample.wav", "dataset/processed")