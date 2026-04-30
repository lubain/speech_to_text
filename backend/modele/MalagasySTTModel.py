from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset, DatasetDict
import torch
import numpy as np

class MalagasySTTModel:
    def __init__(self, model_size="small"):
        """
        Tailles disponibles: tiny, base, small, medium, large
        → 'small' est le meilleur compromis pour le malgache
        """
        self.model_name = f"openai/whisper-{model_size}"
        self.processor  = None
        self.model      = None

    def load_pretrained(self):
        """Charge Whisper pré-entraîné"""
        print(f"⬇️  Chargement de {self.model_name}...")
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name,
            language="Malagasy",
            task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name
        )
        # Force la langue malgache
        self.model.config.forced_decoder_ids = (
            self.processor.get_decoder_prompt_ids(
                language="malagasy", task="transcribe"
            )
        )
        print("✅ Modèle chargé!")
        return self.processor, self.model

    def prepare_dataset(self, batch):
        """Prépare un batch pour l'entraînement"""
        audio = batch["audio"]
        
        # Extraction features audio
        batch["input_features"] = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        # Tokenisation de la transcription
        batch["labels"] = self.processor.tokenizer(
            batch["sentence"]
        ).input_ids
        
        return batch

# Architecture du modèle (pour référence / from-scratch)
import torch.nn as nn

class SimpleCTCModel(nn.Module):
    """
    Modèle CTC simple si vous voulez entraîner from scratch
    (nécessite beaucoup plus de données: 100h+)
    """
    def __init__(self, n_features=80, n_hidden=256, n_classes=50):
        super().__init__()
        
        # Encodeur convolutif
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2,1), padding=1),
            nn.ReLU(),
        )
        
        # Encodeur récurrent
        self.rnn = nn.GRU(
            input_size=64 * (n_features // 2),
            hidden_size=n_hidden,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Couche de classification
        self.fc = nn.Linear(n_hidden * 2, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: (batch, time, features)
        x = x.unsqueeze(1)                  # (B, 1, T, F)
        x = self.conv(x)                    # (B, C, T', F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return self.log_softmax(x)