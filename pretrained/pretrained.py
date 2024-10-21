import numpy as np
import librosa
from pydub import AudioSegment
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io import wavfile

class Pretrained():
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.target_length = self.get_target_length(audio_dir)
    
    def get_target_legth(self, audio_dir):
        # Charger les fichiers audio et calculer leurs longueurs
        audio_files = []
        for filename in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, filename)
            audio_files.append(audio_path)

        audio_lengths = [librosa.get_duration(filename=file) for file in tqdm(audio_files)]

        # Déterminer la longueur cible (par exemple, la plus grande longueur)
        target_length = np.mean(audio_lengths)
        return target_length
    
    def remove_silence(self, audio_path, output):
        # Charger le fichier audio
        signal, sr = librosa.load(audio_path, sr=None)

        # Détection des régions actives dans le signal
        non_silent_intervals = librosa.effects.split(signal, top_db=30)

        # Fusionner les intervalles non silencieux
        non_silent_signal = librosa.effects.remix(signal, non_silent_intervals)

        # Sauvegarder le signal audio sans les silences
        sf.write(output, non_silent_signal, sr)

    def normalize_audio_volume(self, audio_path, output_path, target_dBFS=-20.0):
        # Chargement de l'enregistrement audio
        audio = AudioSegment.from_file(audio_path)

        # Calcul du facteur de normalisation pour atteindre le niveau cible
        current_dBFS = audio.dBFS
        normalization_factor = (target_dBFS - current_dBFS)

        # Normalisation du volume de l'audio
        normalized_audio = audio + normalization_factor

        # Export de l'audio normalisé
        normalized_audio.export(output_path, format="wav")

    def filtrage_du_bruit(self, audio_path, output, noise_threshold=40.0):
        # Chargement de l'enregistrement audio
        audio = AudioSegment.from_file(audio_path)

        # Détection du bruit de fond
        background_noise = audio.dBFS

        # Filtrer le bruit de fond
        if background_noise > noise_threshold:
            audio = audio - noise_threshold
        else:
            audio = audio - background_noise

        # Export de l'audio filtré
        audio.export(output, format="wav")

    def segmentation_parole(self, audio_path, output_file, silence_threshold=-45):
        # Charger le fichier audio
        audio = AudioSegment.from_file(audio_path, format="wav")

        # Détection des silences
        non_silent_audio = audio.strip_silence(silence_thresh=silence_threshold)

        # Exporter le fichier audio sans les silences
        non_silent_audio.export(output_file, format="wav")

    def remove_artifacts(self, audio_path, output_path):
        # Chargement de l'enregistrement audio
        audio = AudioSegment.from_file(audio_path)

        # Suppression d'artefacts basée sur la fréquence ou l'amplitude
        # Par exemple, supprimer les fréquences inférieures à 1000 Hz
        audio_filtered = audio.low_pass_filter(1000)

        # Export de l'audio filtré
        audio_filtered.export(output_path, format="wav")

    def preaccentuation(self, audio_file, output):
        # Charger le signal vocal (remplacer "audio.wav" par votre propre fichier audio)
        sample_rate, audio_data = wavfile.read(audio_file)

        # Paramètres de la préaccentuation
        alpha = 0.95  # Facteur de préaccentuation (typiquement entre 0.9 et 1)

        # Appliquer la préaccentuation
        preemphasis_audio = np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])

        # Enregistrer le signal filtré en tant que fichier WAV
        wavfile.write(output, sample_rate, np.int16(preemphasis_audio))

    def time_stretch_audio(self, input_file, output_file, target_duration):
        # Charger l'audio
        audio, sr = librosa.load(input_file)

        # Calculer la durée actuelle
        current_duration = len(audio)/44100

        # Calculer le facteur de normalisation
        speed_factor = current_duration / target_duration

        # Normaliser l'audio en modifiant la vitesse
        normalized_audio = librosa.effects.time_stretch(y=audio, rate=speed_factor)

        # Sauvegarder l'audio normalisé
        sf.write(output_file, normalized_audio, sr)

    def run(self):
        for filename in tqdm(os.listdir(self.audio_dir)):
            audio_path = os.path.join(self.audio_dir, filename)
            self.normalize_audio_volume(input_path, "normalize_audio_volume.wav")
            self.remove_silence("normalize_audio_volume.wav", "remove_silence.wav")
            self.time_stretch_audio("remove_silence.wav", "time_stretch_audio.wav", self.target_length)
            self.filtrage_du_bruit("time_stretch_audio.wav", "filtrage_du_bruit.wav")
            self.segmentation_parole("filtrage_du_bruit.wav", "segmentation_parole.wav")
            self.preaccentuation("segmentation_parole.wav", "audio_pretrained/"+input_path)

if __name__ == "__main__":
    audio_dir = "audio_wav/"
    audios = Pretrained(audio_dir)
    audios.run()