
import sounddevice as sd
import soundfile as sf
import numpy as np
import csv
import os
from datetime import datetime

# ─────────────────────────────────────────────
# Installation des dépendances :
#   pip install sounddevice soundfile numpy datasets
# ─────────────────────────────────────────────


class MalagasyDataCollector:
    """
    Collecte des enregistrements audio malgaches
    avec transcriptions associées.
    """

    def __init__(self, output_dir="dataset", sample_rate=16000):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/transcriptions", exist_ok=True)

        # Fichier métadonnées
        metadata_path = f"{output_dir}/metadata.csv"
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "transcription", "speaker_id", "timestamp", "duration_s"])

        # Phrases malgaches pour l'enregistrement
        self.sentences = [
            "Manao ahoana ianao",
            "Misaotra betsaka",
            "Aiza ny toeram-pisakafoanana",
            "Ny andro anio dia tsara",
            "Mila fanampiana aho",
            "Faly mifankahita aminao",
            "Aiza ny hopitaly akaiky indrindra",
            "Toy ny ahoana ny vidiny",
            "Azafady, tsy azoko ny hevitr'izany",
            "Ny firenena malagasy dia be harena",
            "Mahita ny ranomasina aho isan'andro",
            "Ny tanàna Antananarivo dia renivohitra",
            "Tia mozika sy dihy aho",
            "Omeo ahy mofo iray azafady",
            "Handeha any an-tsekoly ny ankizy",
        ]

    def record_sentence(self, sentence, duration=5):
        """Enregistre une phrase avec countdown visuel."""
        print(f"\n  📢 Lisez cette phrase :")
        print(f"     ➜  « {sentence} »")
        print()
        for i in range(3, 0, -1):
            print(f"     {i}...", end="\r")
            import time; time.sleep(1)

        print("  🔴 ENREGISTREMENT EN COURS...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("  ✅ Terminé !\n")
        return audio.flatten()

    def save_recording(self, audio, sentence, speaker_id, utt_id):
        """Sauvegarde audio (.wav) + entrée dans metadata.csv"""
        filename = f"spk{speaker_id:03d}_utt{utt_id:03d}.wav"
        audio_path = f"{self.output_dir}/audio/{filename}"

        sf.write(audio_path, audio, self.sample_rate)

        duration = len(audio) / self.sample_rate
        with open(f"{self.output_dir}/metadata.csv", 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, sentence, speaker_id,
                             datetime.now().isoformat(), f"{duration:.2f}"])

        print(f"  💾 Sauvegardé : {audio_path}  ({duration:.1f}s)")
        return audio_path

    def collect_session(self, speaker_id, sentence_indices=None):
        """
        Session complète d'enregistrement pour un locuteur.
        sentence_indices : liste d'indices à enregistrer (None = toutes)
        """
        sentences = self.sentences if sentence_indices is None \
            else [self.sentences[i] for i in sentence_indices]

        print(f"\n{'='*55}")
        print(f"  🎙️  Session — Locuteur n°{speaker_id}")
        print(f"  {len(sentences)} phrase(s) à enregistrer")
        print(f"{'='*55}")

        saved = []
        for i, sentence in enumerate(sentences):
            print(f"\n  [{i+1}/{len(sentences)}]")
            audio = self.record_sentence(sentence)
            path  = self.save_recording(audio, sentence, speaker_id, i)
            saved.append(path)

        print(f"\n{'='*55}")
        print(f"  ✅ Session terminée ! {len(saved)} fichiers enregistrés.")
        print(f"{'='*55}\n")
        return saved


def download_common_voice_malagasy(output_dir="dataset"):
    """
    Télécharge les données malgaches depuis Mozilla Common Voice.
    Le code langue malgache est 'mg'.
    """
    try:
        from datasets import load_dataset
        import soundfile as sf

        print("⬇️  Téléchargement Common Voice — langue malgache (mg)...")
        dataset = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "mg",
            split="train",
            trust_remote_code=True
        )
        print(f"✅ {len(dataset)} échantillons trouvés !")

        # Sauvegarde locale
        audio_dir = f"{output_dir}/audio"
        os.makedirs(audio_dir, exist_ok=True)

        metadata_path = f"{output_dir}/metadata.csv"
        with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i, sample in enumerate(dataset):
                filename = f"cv_mg_{i:05d}.wav"
                audio_path = f"{audio_dir}/{filename}"

                audio_array = np.array(sample['audio']['array'], dtype=np.float32)
                sr = sample['audio']['sampling_rate']
                sf.write(audio_path, audio_array, sr)

                writer.writerow([filename, sample['sentence'], f"cv_{i}",
                                 datetime.now().isoformat(),
                                 f"{len(audio_array)/sr:.2f}"])
                if i % 50 == 0:
                    print(f"  Sauvegarde... {i}/{len(dataset)}")

        print(f"\n✅ Dataset Common Voice sauvegardé dans {output_dir}/")
        return dataset

    except Exception as e:
        print(f"⚠️  Erreur lors du téléchargement : {e}")
        print("   → Collectez manuellement avec MalagasyDataCollector")
        return None


def estimate_data_requirements():
    print("""
╔══════════════════════════════════════════════════════╗
║         VOLUME DE DONNÉES RECOMMANDÉ                 ║
╠══════════════════════════════════════════════════════╣
║  Prototype viable       →   10 heures d'audio        ║
║  Bon modèle             →  100 heures d'audio        ║
║  Modèle robuste         → 1 000+ heures d'audio      ║
╠══════════════════════════════════════════════════════╣
║  Stratégie pour le malgache (langue peu dotée) :     ║
║  → Collecter 10-50h et fine-tuner Whisper            ║
║  → Diversifier les locuteurs (âge, région, accent)   ║
║  → Inclure différents environnements sonores         ║
╚══════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    estimate_data_requirements()

    # Option A : Enregistrement manuel
    collector = MalagasyDataCollector(output_dir="dataset")
    # collector.collect_session(speaker_id=1)          # décommentez pour enregistrer
    # collector.collect_session(speaker_id=2)

    # Option B : Téléchargement Common Voice
    # download_common_voice_malagasy(output_dir="dataset")

    print("✅ Étape 1 terminée — Données collectées dans ./dataset/")
