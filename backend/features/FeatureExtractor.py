import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=80,
                 n_fft=512, hop_length=160, win_length=400):
        self.sr         = sample_rate
        self.n_mfcc     = n_mfcc
        self.n_fft      = n_fft
        self.hop_length = hop_length  # 10ms
        self.win_length = win_length  # 25ms — standard ASR

    def extract_mfcc(self, audio):
        """
        MFCC — Mel-Frequency Cepstral Coefficients
        Représentation compacte du spectre vocal
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        # Ajout des dérivées (delta et delta-delta)
        delta       = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        
        # Concaténation: shape (n_mfcc*3, T)
        features = np.concatenate([mfcc, delta, delta_delta], axis=0)
        return features.T  # (T, features)

    def extract_log_mel_spectrogram(self, audio):
        """
        Log-Mel Spectrogram — utilisé par Whisper et wav2vec2
        Meilleur pour les modèles modernes basés sur les transformers
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=80  # 80 canaux mel — standard Whisper
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel.T  # (T, 80)

    def extract_all_features(self, audio):
        """Extraction complète pour visualisation et analyse"""
        return {
            "mfcc"         : self.extract_mfcc(audio),
            "log_mel"      : self.extract_log_mel_spectrogram(audio),
            "zcr"          : librosa.feature.zero_crossing_rate(audio).T,
            "spectral_centroid": librosa.feature.spectral_centroid(
                                     y=audio, sr=self.sr).T,
        }

    def normalize_features(self, features):
        """Normalisation CMVN — standard en ASR"""
        mean = np.mean(features, axis=0)
        std  = np.std(features, axis=0) + 1e-8
        return (features - mean) / std

# Exemple
extractor = FeatureExtractor()
audio, sr = librosa.load("dataset/processed/sample_clean.wav", sr=16000)
log_mel   = extractor.extract_log_mel_spectrogram(audio)
log_mel   = extractor.normalize_features(log_mel)
print(f"✅ Features extraites: shape = {log_mel.shape}")
# → (T_frames, 80)