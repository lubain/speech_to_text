import os
import torch
import librosa
import numpy as np
import json
import csv
from pathlib import Path

# ─────────────────────────────────────────────
# Installation des dépendances :
#   pip install torch transformers librosa evaluate jiwer
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# TRANSCRIPTEUR WHISPER
# ══════════════════════════════════════════════

class WhisperMalagasyTester:
    """
    Évaluateur complet pour le modèle Whisper fine-tuné sur le malgache.
    Supporte : transcription simple, évaluation batch, analyse d'erreurs.
    """

    def __init__(self, model_path="./whisper-malagasy-final"):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import evaluate

        print(f" Chargement du modèle depuis : {model_path}")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model     = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = self.model.to(self.device)
        self.model.eval()

        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

        print(f"Modèle prêt sur {self.device.upper()}\n")

    # ──────────────────────────────────────────
    # TRANSCRIPTION D'UN FICHIER
    # ──────────────────────────────────────────

    def transcribe(self, audio_path, num_beams=5, verbose=True):
        """
        Transcrit un fichier audio en texte malgache.

        audio_path : chemin vers un fichier .wav (16kHz recommandé)
        num_beams  : taille du beam search (plus grand = meilleur mais plus lent)
        """
        audio, sr = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language            = "malagasy",
                task                = "transcribe",
                num_beams           = num_beams,
                no_repeat_ngram_size= 3,
                early_stopping      = True,
            )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        if verbose:
            duration = len(audio) / sr
            print(f"  Fichier   : {os.path.basename(audio_path)}")
            print(f"  Durée     : {duration:.2f}s")
            print(f"  Résultat  : {transcription}\n")

        return transcription

    def transcribe_with_timestamps(self, audio_path):
        """
        Transcrit avec timestamps par mot — utile pour sous-titrage.
        """
        from transformers import pipeline

        pipe = pipeline(
            task             = "automatic-speech-recognition",
            model            = self.model,
            tokenizer        = self.processor.tokenizer,
            feature_extractor= self.processor.feature_extractor,
            chunk_length_s   = 30,
            stride_length_s  = (5, 5),
            return_timestamps= "word",
            device           = 0 if self.device == "cuda" else -1,
        )

        result = pipe(
            audio_path,
            generate_kwargs={
                "language": "malagasy",
                "task"    : "transcribe",
            }
        )

        print(f"Transcription : {result['text']}\n")
        if "chunks" in result:
            print("Timestamps par mot :")
            for chunk in result["chunks"]:
                t_start, t_end = chunk["timestamp"]
                print(f"     [{t_start:6.2f}s → {t_end:6.2f}s]  {chunk['text']}")

        return result

    def transcribe_long_audio(self, audio_path):
        """
        Transcrit des fichiers audio longs (> 30s) en les découpant
        automatiquement en segments avec chevauchement.
        """
        audio, sr = librosa.load(audio_path, sr=16000)
        total_dur = len(audio) / sr

        print(f"Audio long détecté : {total_dur:.1f}s")
        print("Découpage en segments de 30s...\n")

        chunk_size   = 30 * sr  # 30 secondes
        overlap_size = 5  * sr  # 5s de chevauchement
        step         = chunk_size - overlap_size

        transcriptions = []
        offset         = 0

        while offset < len(audio):
            chunk = audio[offset : offset + chunk_size]
            if len(chunk) < sr:  # < 1s → ignore
                break

            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                ids = self.model.generate(
                    inputs.input_features.to(self.device),
                    language="malagasy", task="transcribe"
                )
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            transcriptions.append(text.strip())
            print(f"  [{offset/sr:.1f}s] → {text}")
            offset += step

        full_text = " ".join(transcriptions)
        print(f"\n Transcription complète :\n  {full_text}\n")
        return full_text

    # ──────────────────────────────────────────
    # ÉVALUATION SUR UN DATASET DE TEST
    # ──────────────────────────────────────────

    def evaluate_dataset(self, test_csv, audio_dir, max_samples=None):
        """
        Évalue le modèle sur un ensemble de test complet.
        Calcule WER, CER et affiche les pires erreurs.

        test_csv   : CSV avec colonnes "filename" et "transcription"
        audio_dir  : répertoire contenant les fichiers WAV
        """
        rows = []
        with open(test_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if max_samples:
            rows = rows[:max_samples]

        print(f"\n{'='*55}")
        print(f"Évaluation — {len(rows)} échantillons")
        print(f"{'='*55}\n")

        predictions = []
        references  = []
        errors      = []  # (wer_individuel, ref, pred, fichier)

        for i, row in enumerate(rows):
            audio_path = os.path.join(audio_dir, row["filename"])
            reference  = row["transcription"].strip()

            if not os.path.exists(audio_path):
                print(f"Fichier introuvable : {row['filename']}")
                continue

            try:
                prediction = self.transcribe(audio_path, verbose=False)
                predictions.append(prediction)
                references.append(reference)

                # WER individuel
                ind_wer = self.wer_metric.compute(
                    predictions=[prediction], references=[reference]
                )
                errors.append((ind_wer, reference, prediction, row["filename"]))

                print(f"  [{i+1:3d}/{len(rows)}] WER={ind_wer*100:5.1f}%")
                print(f"          Ref  : {reference}")
                print(f"          Pred : {prediction}\n")

            except Exception as e:
                print(f"Erreur sur {row['filename']} : {e}\n")

        if not predictions:
            print("Aucune prédiction générée.")
            return {}

        # ── Métriques globales ───────────────
        global_wer = self.wer_metric.compute(
            predictions=predictions, references=references
        )
        global_cer = self.cer_metric.compute(
            predictions=predictions, references=references
        )

        # ── Analyse des erreurs ──────────────
        errors_sorted = sorted(errors, key=lambda x: x[0], reverse=True)

        print(f"\n{'='*55}")
        print(f"RÉSULTATS FINAUX")
        print(f"{'='*55}")
        print(f"  WER (Word Error Rate)      : {global_wer*100:.2f}%")
        print(f"  CER (Character Error Rate) : {global_cer*100:.2f}%")
        print(f"  Échantillons évalués       : {len(predictions)}")
        print(f"\n  Interprétation WER :")
        print(f"    < 5%  → Excellent  (niveau commercial)")
        print(f"    5-15% → Bon        (utilisable)")
        print(f"   15-30% → Acceptable (à améliorer)")
        print(f"    > 30% → Insuffisant")

        print(f"\n10 pires prédictions :")
        for wer_i, ref, pred, fname in errors_sorted[:10]:
            print(f"    [{wer_i*100:.0f}%] {fname}")
            print(f"      Ref  : {ref}")
            print(f"      Pred : {pred}")

        results = {
            "wer"        : global_wer,
            "cer"        : global_cer,
            "n_samples"  : len(predictions),
            "predictions": predictions,
            "references" : references,
        }

        # Sauvegarde des résultats
        out_path = "evaluation_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "wer": global_wer,
                "cer": global_cer,
                "n_samples": len(predictions),
                "details": [
                    {"file": e[3], "wer": e[0], "ref": e[1], "pred": e[2]}
                    for e in errors_sorted
                ]
            }, f, ensure_ascii=False, indent=2)
        print(f"\nRésultats sauvegardés : {out_path}")
        print(f"{'='*55}\n")

        return results

    # ──────────────────────────────────────────
    # ANALYSE D'ERREURS DÉTAILLÉE
    # ──────────────────────────────────────────

    def analyze_errors(self, predictions, references):
        """
        Analyse les types d'erreurs les plus fréquents :
        - Substitutions (mauvais mot)
        - Insertions   (mot ajouté)
        - Suppressions (mot manqué)
        """
        substitutions = {}
        insertions    = []
        deletions     = []

        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words  = ref.lower().split()

            # Alignement simple (Levenshtein)
            for i, (p, r) in enumerate(zip(pred_words, ref_words)):
                if p != r:
                    key = f"'{r}' → '{p}'"
                    substitutions[key] = substitutions.get(key, 0) + 1

            # Insertions / suppressions
            if len(pred_words) > len(ref_words):
                insertions.extend(pred_words[len(ref_words):])
            elif len(pred_words) < len(ref_words):
                deletions.extend(ref_words[len(pred_words):])

        # Top substitutions
        top_subs = sorted(substitutions.items(), key=lambda x: x[1], reverse=True)

        print("\nAnalyse des erreurs :")
        print(f"     Substitutions fréquentes :")
        for sub, count in top_subs[:10]:
            print(f"       {sub}  (×{count})")

        if insertions:
            from collections import Counter
            top_ins = Counter(insertions).most_common(5)
            print(f"     Mots souvent insérés : {top_ins}")

        if deletions:
            from collections import Counter
            top_del = Counter(deletions).most_common(5)
            print(f"     Mots souvent supprimés : {top_del}")

        return {"substitutions": top_subs, "insertions": insertions,
                "deletions": deletions}


# ══════════════════════════════════════════════
# TRANSCRIPTEUR CTC (modèle from scratch)
# ══════════════════════════════════════════════

class CTCMalagasyTester:
    """Évaluateur pour le modèle CTC entraîné from scratch."""

    def __init__(self, model, vocab, checkpoint_path=None, device=None):
        self.model  = model
        self.vocab  = vocab
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"Checkpoint chargé (époque {ckpt['epoch']}, "
                  f"WER {ckpt['wer']*100:.2f}%)")

        self.model.eval()

    def transcribe(self, features_path_or_array, beam_size=1):
        """Transcrit depuis un fichier .npy de features ou un array numpy."""
        if isinstance(features_path_or_array, str):
            features = np.load(features_path_or_array).astype(np.float32)
        else:
            features = features_path_or_array.astype(np.float32)

        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.model(x)

        # Décodage greedy
        ids   = torch.argmax(log_probs[0], dim=-1).cpu().numpy()
        decoded = []
        prev    = None
        for token_id in ids:
            if token_id != self.vocab.blank_id and token_id != prev:
                decoded.append(int(token_id))
            prev = token_id

        text = self.vocab.decode(decoded)
        return text

    def evaluate(self, test_loader):
        """Évalue sur un DataLoader de test."""
        import evaluate as ev
        wer_metric = ev.load("wer")
        cer_metric = ev.load("cer")

        predictions = []
        references  = []

        for batch in test_loader:
            features = batch["features"].to(self.device)
            labels   = batch["labels"]
            lab_len  = batch["lab_len"]

            with torch.no_grad():
                log_probs = self.model(features)

            for i in range(features.shape[0]):
                pred = self.transcribe(features[i].cpu().numpy())
                ref  = self.vocab.decode(labels[i][:lab_len[i]].numpy())
                predictions.append(pred)
                references.append(ref)

        wer = wer_metric.compute(predictions=predictions, references=references)
        cer = cer_metric.compute(predictions=predictions, references=references)

        print(f"  WER : {wer*100:.2f}%  |  CER : {cer*100:.2f}%")
        return {"wer": wer, "cer": cer,
                "predictions": predictions, "references": references}


# ──────────────────────────────────────────────
# UTILITAIRE : GÉNÉRATION D'UN RAPPORT HTML
# ──────────────────────────────────────────────

def generate_html_report(results, output_path="rapport_evaluation.html"):
    """Génère un rapport HTML lisible à partir des résultats d'évaluation."""
    wer     = results.get("wer", 0) * 100
    cer     = results.get("cer", 0) * 100
    n       = results.get("n_samples", 0)
    preds   = results.get("predictions", [])
    refs    = results.get("references",  [])

    rows_html = ""
    for i, (p, r) in enumerate(zip(preds[:50], refs[:50])):
        match = "✅" if p.strip().lower() == r.strip().lower() else "❌"
        rows_html += f"""
        <tr>
          <td>{i+1}</td>
          <td>{r}</td>
          <td>{p}</td>
          <td>{match}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport STT Malgache</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #222; }}
    h1   {{ color: #1a1a2e; }}
    .kpi {{ display: flex; gap: 20px; margin: 20px 0; }}
    .kpi-box {{ background: #f0f4ff; border-radius: 10px; padding: 20px;
                text-align: center; flex: 1; }}
    .kpi-val  {{ font-size: 2.5em; font-weight: bold; color: #3a5bc7; }}
    table  {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th     {{ background: #1a1a2e; color: white; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
  </style>
</head>
<body>
  <h1>🎙️ Rapport d'Évaluation — STT Malgache</h1>

  <div class="kpi">
    <div class="kpi-box">
      <div class="kpi-val">{wer:.1f}%</div>
      <div>WER (Word Error Rate)</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-val">{cer:.1f}%</div>
      <div>CER (Char Error Rate)</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-val">{n}</div>
      <div>Échantillons évalués</div>
    </div>
  </div>

  <h2>Détail des transcriptions (50 premiers)</h2>
  <table>
    <thead><tr><th>#</th><th>Référence</th><th>Prédiction</th><th>OK?</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Rapport HTML généré : {output_path}")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ÉTAPE 6 — TEST ET ÉVALUATION")
    print("="*55)

    # ── Test Whisper fine-tuné ───────────────
    """
    tester = WhisperMalagasyTester("./whisper-malagasy-final")

    # Test sur un fichier unique
    tester.transcribe("mon_audio.wav")

    # Test avec timestamps
    tester.transcribe_with_timestamps("mon_audio.wav")

    # Évaluation complète sur dataset de test
    results = tester.evaluate_dataset(
        test_csv  = "dataset/metadata_test.csv",
        audio_dir = "dataset/audio",
    )

    # Analyse des erreurs
    tester.analyze_errors(results["predictions"], results["references"])

    # Rapport HTML
    generate_html_report(results, "rapport_evaluation.html")
    """

    print("\n💡 Décommentez le bloc souhaité pour lancer l'évaluation.")
    print("✅ Étape 6 prête.")
