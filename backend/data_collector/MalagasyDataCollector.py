import sounddevice as sd
import soundfile as sf
import numpy as np
import csv
import os
from datetime import datetime

class MalagasyDataCollector:
    def __init__(self, output_dir="dataset", sample_rate=16000):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/transcriptions", exist_ok=True)

        # Phrases malgaches pour l'enregistrement
        self.sentences = [
            "Manao ahoana ianao",           # Comment allez-vous
            "Misaotra betsaka",              # Merci beaucoup
            "Aiza ny toeram-pisakafoanana",  # Où est le restaurant
            "Ny andro anio dia tsara",       # Le temps est beau aujourd'hui
            "Mila fanampiana aho",           # J'ai besoin d'aide
        ]

    def record_sentence(self, sentence, speaker_id, duration=5):
        """Enregistre une phrase avec countdown"""
        print(f"\n📢 Préparez-vous à lire: '{sentence}'")
        for i in range(3, 0, -1):
            print(f"  {i}...")
        
        print("🔴 ENREGISTREMENT...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("✅ Enregistrement terminé")
        return audio.flatten()

    def save_recording(self, audio, sentence, speaker_id, utt_id):
        """Sauvegarde l'audio et la transcription"""
        filename = f"spk{speaker_id:03d}_utt{utt_id:03d}.wav"
        audio_path = f"{self.output_dir}/audio/{filename}"
        
        sf.write(audio_path, audio, self.sample_rate)
        
        # Sauvegarde métadonnées
        with open(f"{self.output_dir}/metadata.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, sentence, speaker_id, 
                           datetime.now().isoformat()])
        
        print(f"💾 Sauvegardé: {audio_path}")

    def collect_session(self, speaker_id):
        """Session complète d'enregistrement"""
        print(f"\n🎙️ Session pour le locuteur {speaker_id}")
        
        for i, sentence in enumerate(self.sentences):
            audio = self.record_sentence(sentence, speaker_id)
            self.save_recording(audio, sentence, speaker_id, i)
            
        print(f"\n✅ Session terminée! {len(self.sentences)} phrases enregistrées")

# Utilisation
collector = MalagasyDataCollector()
collector.collect_session(speaker_id=1)