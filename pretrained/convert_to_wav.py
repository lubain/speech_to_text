from pydub import AudioSegment
import os
from datetime import datetime

# Fonction pour extraire les parties numériques du nom du fichier
def extraire_numeros_avec_texte(fichier):
    # Découpe en parties texte et nombres avec regex (textes et numéros séparés)
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', fichier)]

# Fonction pour lister et trier les fichiers audio
def lister_fichiers_audio(dossier, audio_extensions, output_dir):
    try:
        # Lire le contenu du dossier
        fichiers = os.listdir(dossier)

        # Filtrer les fichiers avec des extensions audio
        fichiers_audio = [f for f in fichiers if os.path.splitext(f)[1].lower() in audio_extensions]

        # Trier les fichiers en tenant compte des numéros dans les noms de fichiers
        fichiers_audio.sort(key=extraire_numeros_avec_texte)
        
        i = 0
        # Obtenir la date et l'heure actuelles
        date_actuelle = datetime.now()

        # Convertir la date en secondes depuis l'époque (01/01/1970)
        secondes = (date_actuelle - datetime(2024, 4, 1)).total_seconds()
        secondes = str(secondes)
        secondes = secondes.replace(".", "_")

        # Afficher les fichiers audio triés
        for audio_file in fichiers_audio:
            # Spécifier le chemin de sortie
            output_file = output_dir+secondes+str(i)+".wav"
            # Nos donnees
            file_path = os.path.join(audio_file)
            convert_to_wav(file_path,output_dir,output_file)
            print(file_path)
            i += 1

    except FileNotFoundError:
        print(f"Le dossier {dossier} n'existe pas.")

def convert_to_wav(input_file, output_dir, output_file):
    try:
        # Charger le fichier audio
        audio = AudioSegment.from_file(input_file)

        # Exporter en format WAV
        audio.export(output_file, format="wav")

        print(f"Conversion réussie ! Le fichier WAV a été sauvegardé à : {output_file}")

    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")

if __name__ == "__main__":
    output_dir = "audio_wav/"
    input_dir = "audio_data/"
    
    # Extensions de fichiers audio à rechercher
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac']

    # Appelle la fonction pour lister les fichiers
    lister_fichiers_audio(input_dir, audio_extensions, output_dir)