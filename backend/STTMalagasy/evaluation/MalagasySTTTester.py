import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

class MalagasySTTTester:
    def __init__(self, model_path="./whisper-malagasy-final"):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model     = WhisperForConditionalGeneration.from_pretrained(
            model_path
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def transcribe(self, audio_path):
        """Transcrit un fichier audio"""
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language="malagasy",
                task="transcribe",
                num_beams=5,            # Beam search pour meilleure qualité
                no_repeat_ngram_size=3
            )
        
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription

    def evaluate_dataset(self, test_files, references):
        """Évalue sur un ensemble de test"""
        predictions = []
        
        print(" Évaluation en cours...\n")
        for i, (audio_path, reference) in enumerate(
            zip(test_files, references)
        ):
            prediction = self.transcribe(audio_path)
            predictions.append(prediction)
            
            print(f"[{i+1}/{len(test_files)}]")
            print(f"  Référence  : {reference}")
            print(f"  Prédiction : {prediction}")
            print()

        # Calcul des métriques
        wer = self.wer_metric.compute(
            predictions=predictions, references=references
        )
        cer = self.cer_metric.compute(
            predictions=predictions, references=references
        )
        
        print("=" * 50)
        print(" RÉSULTATS FINAUX")
        print("=" * 50)
        print(f"  WER (Word Error Rate)      : {wer * 100:.2f}%")
        print(f"  CER (Character Error Rate) : {cer * 100:.2f}%")
        print(f"\n  Interprétation WER:")
        print(f"  < 10% → Excellent  | 10-20% → Bon")
        print(f"  20-30% → Acceptable | > 30% → À améliorer")
        
        return {"wer": wer, "cer": cer, "predictions": predictions}

# Test final
tester = MalagasySTTTester()

# Test rapide sur un fichier
transcription = tester.transcribe("test_audio.wav")
print(f"Transcription: {transcription}")

# Évaluation complète
test_files = ["test1.wav", "test2.wav", "test3.wav"]
references = [
    "Manao ahoana ianao",
    "Misaotra betsaka",
    "Ny andro anio dia tsara"
]
results = tester.evaluate_dataset(test_files, references)