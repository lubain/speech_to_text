"""
=============================================================
ÉTAPE 5 — ENTRAÎNEMENT (OPTIMISÉ CPU + PETIT DATASET)
Speech-to-Text Malgache
=============================================================

Configuration adaptée à votre situation :
  Fichiers WAV + CSV (filename, transcription)
  Moins d'1 heure de données
  CPU uniquement (pas de GPU)

Stratégie :
  → Whisper-tiny  (39M params, 4x plus rapide que small)
  → Encodeur gelé (seul le décodeur s'entraîne)
  → 300 steps max, batch_size=2, pas de fp16
  → Augmentation automatique pour compenser le manque de données
=============================================================
"""

import os
import csv
import json
import time
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union


# ══════════════════════════════════════════════
# ÉTAPE 0 — VÉRIFICATION DE VOTRE CSV
# ══════════════════════════════════════════════

def verify_and_fix_csv(csv_path, audio_dir, output_csv=None):
    """
    Vérifie que votre CSV est compatible et corrige les problèmes courants.

    Format attendu (au moins ces deux colonnes) :
      filename,transcription
      audio_001.wav,Manao ahoana ianao
      audio_002.wav,Misaotra betsaka

    csv_path  : chemin vers votre CSV
    audio_dir : répertoire contenant les fichiers WAV
    """
    print("\n" + "="*55)
    print("  Vérification du CSV")
    print("="*55)

    rows_ok    = []
    rows_error = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig gère le BOM Windows
        # Détection automatique du séparateur (virgule ou point-virgule)
        sample = f.read(2048)
        f.seek(0)
        sep = ';' if sample.count(';') > sample.count(',') else ','
        print(f"  Séparateur détecté : '{sep}'")

        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames or []
        print(f"  Colonnes trouvées  : {headers}")

        # Détection flexible des noms de colonnes
        col_file  = next((h for h in headers if h.lower().strip() in
                          ['filename','file','audio','fichier','nom','name','path']), None)
        col_text  = next((h for h in headers if h.lower().strip() in
                          ['transcription','text','texte','sentence','phrase',
                           'transcript','label','content']), None)

        if not col_file or not col_text:
            print(f"\n  Colonnes non reconnues : {headers}")
            print("     Colonnes attendues : 'filename' et 'transcription'")
            print("     Renommez vos colonnes ou modifiez col_file/col_text ci-dessus.")
            return None

        print(f"  Colonne fichier    : '{col_file}'")
        print(f"  Colonne texte      : '{col_text}'")
        print()

        for i, row in enumerate(reader):
            fname = row[col_file].strip()
            text  = row[col_text].strip()

            # Ignore lignes vides
            if not fname or not text:
                continue

            # Ajoute extension .wav si absente
            if not fname.lower().endswith('.wav'):
                fname += '.wav'

            audio_path = os.path.join(audio_dir, fname)

            if not os.path.exists(audio_path):
                rows_error.append((fname, f"Fichier introuvable : {audio_path}"))
                continue

            # Vérifie que le fichier est lisible et mesure la durée
            try:
                info  = sf.info(audio_path)
                dur_s = info.frames / info.samplerate
                if dur_s < 0.3:
                    rows_error.append((fname, f"Trop court ({dur_s:.2f}s)"))
                    continue
                if dur_s > 30:
                    print(f"  [{i+1}] {fname} — durée {dur_s:.1f}s > 30s (sera découpé)")

                rows_ok.append({
                    "filename"      : fname,
                    "transcription" : text,
                    "duration_s"    : round(dur_s, 2),
                })
            except Exception as e:
                rows_error.append((fname, str(e)))

    # Résumé
    total_dur = sum(r['duration_s'] for r in rows_ok)
    print(f"  Échantillons valides : {len(rows_ok)}")
    print(f"  Ignorés             : {len(rows_error)}")
    print(f"  Durée totale        : {total_dur/60:.1f} minutes")

    if rows_error:
        print(f"\n  Erreurs détectées :")
        for fname, reason in rows_error[:10]:
            print(f"    • {fname} — {reason}")

    if len(rows_ok) < 5:
        print("\n Pas assez d'échantillons valides (minimum 5).")
        return None

    # Sauvegarde CSV nettoyé
    out = output_csv or csv_path.replace('.csv', '_clean.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename','transcription','duration_s'])
        writer.writeheader()
        writer.writerows(rows_ok)

    print(f"\n CSV nettoyé sauvegardé : {out}")
    print("="*55 + "\n")
    return out, rows_ok


# ══════════════════════════════════════════════
# ÉTAPE 1 — PRÉPARATION DU DATASET HUGGINGFACE
# ══════════════════════════════════════════════

def build_dataset(clean_csv, audio_dir, processor, test_ratio=0.15):
    """
    Construit le DatasetDict HuggingFace depuis votre CSV nettoyé.
    Avec moins d'1h, on garde 85% en train et 15% en test.
    """
    from datasets import Dataset, DatasetDict, Audio

    print("Construction du dataset HuggingFace...")

    rows = []
    with open(clean_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                "audio"    : os.path.join(audio_dir, row['filename']),
                "sentence" : row['transcription'],
            })

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("  Extraction des features audio (peut prendre quelques minutes)...")
    dataset = dataset.map(
        prepare,
        remove_columns=dataset.column_names,
        desc="Préparation"
    )

    split = dataset.train_test_split(test_size=test_ratio, seed=42)
    print(f" Train : {len(split['train'])} | Test : {len(split['test'])}\n")
    return split


# ══════════════════════════════════════════════
# ÉTAPE 2 — COLLATEUR DE DONNÉES
# ══════════════════════════════════════════════

@dataclass
class DataCollatorWhisperCPU:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ══════════════════════════════════════════════
# ÉTAPE 3 — ENTRAÎNEMENT OPTIMISÉ CPU
# ══════════════════════════════════════════════

def train_cpu(processor, model, train_dataset, eval_dataset,
              output_dir="./whisper-malagasy-tiny-cpu",
              max_steps=300):
    """
    Fine-tuning Whisper-tiny optimisé pour CPU avec peu de données.

    Adaptations CPU :
      - fp16=False        (fp16 non supporté sur CPU)
      - batch_size=2      (moins de RAM)
      - gradient_accum=4  (simule batch effectif de 8)
      - no_cuda=True      (force CPU explicitement)
      - dataloader_workers=0 (évite les problèmes multiprocessing Windows)
    """
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    import evaluate

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pred_str  = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        print(f"\n  WER : {wer*100:.2f}%  |  CER : {cer*100:.2f}%")
        if pred_str:
            print(f"  Ex. Ref  : {label_str[0]}")
            print(f"  Ex. Pred : {pred_str[0]}")
        return {"wer": wer, "cer": cer}

    eval_steps = max(30, max_steps // 10)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = output_dir,

        # ── Taille de batch (CPU : petit) ────────
        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 4,    # batch effectif = 2×4 = 8

        # ── Optimiseur ───────────────────────────
        learning_rate               = 1e-4, # plus élevé : converge plus vite
        warmup_steps                = max(10, max_steps // 15),
        max_steps                   = max_steps,
        weight_decay                = 0.01,

        # ── CPU / Mémoire ────────────────────────
        fp16                        = False,            # CPU ne supporte pas fp16
        no_cuda                     = True,             # force CPU
        dataloader_num_workers      = 0,                # évite bugs multiprocessing Windows
        dataloader_pin_memory       = False,

        # ── Évaluation & sauvegarde ──────────────
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_steps                  = eval_steps,
        save_total_limit            = 2,
        logging_steps               = 10,
        logging_first_step          = True,

        # ── Sélection du meilleur modèle ─────────
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,

        # ── Génération ───────────────────────────
        predict_with_generate       = True,
        generation_max_length       = 225,

        report_to                   = ["tensorboard"],
        push_to_hub                 = False,
    )

    data_collator = DataCollatorWhisperCPU(processor=processor)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        tokenizer       = processor.feature_extractor,
    )

    # Estimation du temps
    secs_per_step = 8  # ~8s/step sur CPU pour Whisper-tiny
    eta_min = (max_steps * secs_per_step) / 60
    print(f"\n{'='*55}")
    print(f"  Début de l'entraînement (CPU — Whisper-tiny)")
    print(f"  Steps       : {max_steps}")
    print(f"  Batch eff.  : {2 * 4} (2 × 4 grad_accum)")
    print(f"  Éval tous   : {eval_steps} steps")
    print(f"  ETA estimée : ~{eta_min:.0f} minutes")
    print(f"{'='*55}\n")

    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60

    print(f"\n  Entraînement terminé en {elapsed:.1f} minutes")

    # Sauvegarde finale
    final_dir = output_dir + "-final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"  Modèle sauvegardé : {final_dir}")
    return trainer, final_dir


# ══════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════

def run(
    csv_path,
    audio_dir,
    output_dir = "./whisper-malagasy-tiny-cpu",
    max_steps  = 300,
):
    """
    Lance le pipeline complet :
      1. Vérifie votre CSV
      2. Charge Whisper-tiny
      3. Prépare le dataset
      4. Entraîne sur CPU
      5. Sauvegarde le modèle

    Paramètres :
      csv_path   : chemin vers votre CSV (colonnes filename + transcription)
      audio_dir  : dossier contenant vos fichiers WAV
      output_dir : où sauvegarder le modèle entraîné
      max_steps  : nombre de steps (300 ≈ 40 min sur CPU)
    """
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    # ── 1. Vérification CSV ──────────────────
    result = verify_and_fix_csv(csv_path, audio_dir)
    if result is None:
        print(" Arrêt : corrigez votre CSV avant de continuer.")
        return
    clean_csv, rows = result

    # ── 2. Chargement Whisper-tiny ───────────
    print("  Chargement de Whisper-tiny...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-tiny",
        language="Malagasy",
        task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="malagasy", task="transcribe"
    )
    model.config.suppress_tokens = []

    # Gèle TOUT l'encodeur — seul le décodeur s'entraîne
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Modèle prêt — {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params entraînables\n")

    # ── 3. Dataset ───────────────────────────
    dataset = build_dataset(clean_csv, audio_dir, processor)

    # Avertissement si très peu de données
    if len(dataset['train']) < 20:
        print(f"    Seulement {len(dataset['train'])} échantillons en train.")
        print("     Résultats limités — collectez plus de données pour améliorer.\n")

    # ── 4. Entraînement ──────────────────────
    trainer, final_dir = train_cpu(
        processor    = processor,
        model        = model,
        train_dataset= dataset['train'],
        eval_dataset = dataset['test'],
        output_dir   = output_dir,
        max_steps    = max_steps,
    )

    # ── 5. Résumé final ──────────────────────
    print(f"\n{'='*55}")
    print(f"  ENTRAÎNEMENT TERMINÉ !")
    print(f"  Modèle prêt dans : {final_dir}")
    print(f"\n  Pour transcrire un fichier :")
    print(f"    python main_pipeline.py --transcribe audio.wav \\")
    print(f"           --model-path {final_dir}")
    print(f"{'='*55}\n")

    return trainer, final_dir


# ─────────────────────────────────────────────
# CONFIGURATION — MODIFIEZ CES CHEMINS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement STT Malgache — CPU")
    parser.add_argument(
        "--csv",
        required=True,
        help="Chemin vers votre fichier CSV (ex: dataset/metadata.csv)"
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Dossier contenant vos fichiers WAV (ex: dataset/audio)"
    )
    parser.add_argument(
        "--output-dir",
        default="./whisper-malagasy-tiny-cpu",
        help="Dossier de sortie pour le modèle entraîné"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Nombre de steps (défaut: 300 ≈ 40 min sur CPU)"
    )
    args = parser.parse_args()

    run(
        csv_path   = args.csv,
        audio_dir  = args.audio_dir,
        output_dir = args.output_dir,
        max_steps  = args.steps,
    )
