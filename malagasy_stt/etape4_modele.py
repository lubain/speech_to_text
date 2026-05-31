"""
=============================================================
 Fine-tuning de Whisper (recommandé — peu de données nécessaires)
=============================================================
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────
# Installation des dépendances :
#   pip install torch transformers
# ─────────────────────────────────────────────

class WhisperMalagasyLoader:
    """
    Charge et configure Whisper d'OpenAI pour le malgache.
    Architecture : Encoder-Decoder Transformer

    Fonctionnement :
      Audio → Log-Mel (80 canaux) → Encoder Transformer
           → Decoder Transformer  → Tokens → Texte malgache
    """

    # Tailles disponibles et leurs paramètres
    MODEL_SPECS = {
        "tiny"  : {"params": "39M",  "vram": "~1 GB",  "speed": "32x"},
        "base"  : {"params": "74M",  "vram": "~1 GB",  "speed": "16x"},
        "small" : {"params": "244M", "vram": "~2 GB",  "speed": "6x"},
        "medium": {"params": "769M", "vram": "~5 GB",  "speed": "2x"},
        "large" : {"params": "1.5B", "vram": "~10 GB", "speed": "1x"},
    }

    def __init__(self, model_size="small"):
        """
        model_size : "tiny" | "base" | "small" | "medium" | "large"
        → "small" est le meilleur compromis pour le malgache
        """
        self.model_size = model_size
        self.model_name = f"openai/whisper-{model_size}"
        self.processor  = None
        self.model      = None
        self._print_specs()

    def _print_specs(self):
        spec = self.MODEL_SPECS.get(self.model_size, {})
        print(f"""
  ┌─────────────────────────────────────────────┐
  │  MODÈLE SÉLECTIONNÉ : Whisper-{self.model_size:<7}       │
  ├─────────────────────────────────────────────┤
  │  Paramètres  : {spec.get('params','?'):<30} │
  │  VRAM req.   : {spec.get('vram','?'):<30} │
  │  Vitesse rel.: {spec.get('speed','?'):<30} │
  └─────────────────────────────────────────────┘
""")

    def load(self):
        """Charge le processeur et le modèle Whisper pré-entraîné."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"  Chargement de {self.model_name}...")
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name,
            language="Malagasy",
            task="transcribe"
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name
        )

        # Force la langue et la tâche
        self.model.config.forced_decoder_ids = (
            self.processor.get_decoder_prompt_ids(
                language="malagasy",
                task="transcribe"
            )
        )

        # Désactive suppress_tokens pour éviter les suppressions incorrectes
        self.model.config.suppress_tokens = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f" Modèle chargé sur {device.upper()} "
              f"({n_params/1e6:.1f}M paramètres)\n")

        return self.processor, self.model

    def freeze_encoder(self):
        """
        Gèle l'encodeur pour un fine-tuning rapide avec peu de données.
        Seul le décodeur est mis à jour → 4x moins de mémoire.
        """
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        n_trainable = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        print(f" Encodeur gelé — Paramètres entraînables : {n_trainable/1e6:.1f}M")

    def unfreeze_all(self):
        """Dégèle tous les paramètres pour un fine-tuning complet."""
        for param in self.model.parameters():
            param.requires_grad = True
        print(" Tous les paramètres sont entraînables")

    def get_model_info(self):
        """Affiche les informations détaillées sur le modèle."""
        if self.model is None:
            print(" Modèle non chargé. Appelez .load() d'abord.")
            return

        total   = sum(p.numel() for p in self.model.parameters())
        trainab = sum(p.numel() for p in self.model.parameters()
                      if p.requires_grad)
        frozen  = total - trainab

        print(f"""
  Architecture Whisper-{self.model_size} :
  ──────────────────────────────────────
  • Encodeur   : Transformer (Conv1D + Attention)
  • Décodeur   : Transformer auto-régressif
  • Input      : Log-Mel Spectrogram (80 canaux, 30s max)
  • Output     : Tokens de texte (multilangue)

  Paramètres :
  ──────────────────────────────────────
  • Total      : {total/1e6:.1f}M
  • Entraîn.   : {trainab/1e6:.1f}M
  • Gelés      : {frozen/1e6:.1f}M
""")

# ══════════════════════════════════════════════
# VOCABULAIRE MALGACHE
# ══════════════════════════════════════════════

class MalagasyVocabulary:
    """
    Vocabulaire pour la tokenisation au niveau caractère.
    Le malgache utilise l'alphabet latin avec quelques spécificités.
    """

    # Caractères du malgache standard
    CHARS = list("abdefghijklmnoprstvwxyz") + \
            ["à", "â", "é", "è", "ê", "î", "ô", "û"] + \
            [" ", "'", "-"]

    # Tokens spéciaux
    BLANK_TOKEN = "<blank>"  # requis par CTC
    UNK_TOKEN   = "<unk>"
    BOS_TOKEN   = "<s>"
    EOS_TOKEN   = "</s>"
    PAD_TOKEN   = "<pad>"

    def __init__(self):
        special = [self.PAD_TOKEN, self.BLANK_TOKEN,
                   self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        all_tokens = special + self.CHARS

        self.token2id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        self.blank_id = self.token2id[self.BLANK_TOKEN]
        self.pad_id   = self.token2id[self.PAD_TOKEN]
        self.bos_id   = self.token2id[self.BOS_TOKEN]
        self.eos_id   = self.token2id[self.EOS_TOKEN]
        self.unk_id   = self.token2id[self.UNK_TOKEN]

    def encode(self, text):
        """Convertit une chaîne en liste d'IDs."""
        text = text.lower().strip()
        return [self.token2id.get(c, self.unk_id) for c in text]

    def decode(self, ids, remove_special=True):
        """Convertit une liste d'IDs en texte."""
        tokens = [self.id2token.get(i, self.UNK_TOKEN) for i in ids]
        if remove_special:
            tokens = [t for t in tokens
                      if t not in [self.PAD_TOKEN, self.BLANK_TOKEN,
                                   self.BOS_TOKEN, self.EOS_TOKEN]]
        return "".join(tokens)

    def __len__(self):
        return len(self.token2id)

    def info(self):
        print(f"""
  Vocabulaire Malgache :
  ─────────────────────
  • Taille totale : {len(self)}
  • Caractères    : {len(self.CHARS)}
  • Tokens spéc.  : 5 (pad, blank, bos, eos, unk)
  • BLANK index   : {self.blank_id}  (requis CTC)
""")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ÉTAPE 4 — DÉFINITION DU MODÈLE")
    print("="*55)

    # ──  Whisper ───────────────── 
    print("\n Approche A — Whisper (fine-tuning) :")
    loader = WhisperMalagasyLoader(model_size="small")
    # processor, model = loader.load()
    # loader.freeze_encoder()
    # loader.get_model_info()

    # Test forward pass
    batch_size, time_steps = 2, 200
    dummy_input  = torch.randn(batch_size, time_steps, config.n_features)
    dummy_output = model(dummy_input)
    print(f"\n  Test forward pass : {dummy_input.shape} → {dummy_output.shape}")
    # → (2, 100, 50)  [temps divisé par 2 après conv stride=2]

    # Test du vocabulaire
    text    = "Manao ahoana ianao"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)
    print(f"\n  Encodage : '{text}'")
    print(f"  IDs      : {encoded}")
    print(f"  Décodage : '{decoded}'")

    print("\n Étape 4 terminée — Modèle défini et testé.")
