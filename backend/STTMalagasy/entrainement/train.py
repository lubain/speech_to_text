from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collateur de données avec padding dynamique"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # Sépare inputs et labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        
        # Remplace le padding par -100 (ignoré dans la loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Retire le token de début si présent
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def train_malagasy_stt(processor, model, train_dataset, eval_dataset):
    """Pipeline d'entraînement complet"""
    
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids     = pred.predictions
        label_ids    = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        print(f"\n WER: {wer * 100:.2f}%")
        return {"wer": wer}

    # Configuration d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir          = "./whisper-malagasy",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 4,
        gradient_accumulation_steps = 2,        # Simule batch_size=16
        learning_rate       = 1e-5,
        warmup_steps        = 500,
        max_steps           = 5000,
        gradient_checkpointing = True,
        fp16                = torch.cuda.is_available(),
        eval_strategy       = "steps",
        eval_steps          = 500,
        save_steps          = 500,
        logging_steps       = 100,
        load_best_model_at_end  = True,
        metric_for_best_model   = "wer",
        greater_is_better   = False,            # WER: plus bas = meilleur
        predict_with_generate   = True,
        generation_max_length   = 225,
        report_to           = ["tensorboard"],
        push_to_hub         = False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        model          = model,
        args           = training_args,
        train_dataset  = train_dataset,
        eval_dataset   = eval_dataset,
        data_collator  = data_collator,
        compute_metrics= compute_metrics,
        tokenizer      = processor.feature_extractor,
    )

    print("Début de l'entraînement...")
    trainer.train()
    
    # Sauvegarde
    trainer.save_model("./whisper-malagasy-final")
    processor.save_pretrained("./whisper-malagasy-final")
    print("Modèle sauvegardé dans ./whisper-malagasy-final")
    return trainer