import json
with open("dataset/audio_path.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

with open("gasy.txt", 'w', encoding="utf-8") as f:
    for transcription in transcriptions:
        f.write(f"{transcription['key']} | {transcription['text']}\n")