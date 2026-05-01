import requests
from datasets import load_dataset

def download_malagasy_data():
    """
    Télécharge les données malgaches disponibles en ligne
    Common Voice de Mozilla contient quelques données malgaches
    """
    try:
        # Mozilla Common Voice — langue malgache (mg)
        dataset = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "mg",  # code langue malgache
            split="train",
            trust_remote_code=True
        )
        print(f"{len(dataset)} échantillons trouvés dans Common Voice")
        return dataset
    except Exception as e:
        print(f"Peu de données disponibles: {e}")
        print("Il faudra collecter manuellement")
        return None

# Estimation du volume nécessaire
print("""
    Volume de données recommandé:
   - Minimum viable : 10 heures (prototype)
   - Bon modèle     : 100 heures
   - Modèle robuste : 1000+ heures
   
   Pour le malgache (langue peu dotée) :
   → Commencer avec 10-50 heures et fine-tuner Whisper
""")
