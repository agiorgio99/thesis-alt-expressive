#!/usr/bin/env python3
"""
finetune_whisper.py
───────────────────
Fine-tune openai/whisper-large-v3 on the WORLD-augmented GTSinger technique
groups produced by build_augmented_dataset.py.

Training data : GTSinger_Augmented technique group WAVs  (~4 000 files)
Validation    : random 10% holdout from the same pool (stratified by technique)
Best model    : saved to  results/finetune_whisper/best_model/
               (loadable with --set asr.extra.model_id=... in the pipeline)

After training run the evaluation step:
  python scripts/run_pipeline.py \\
      --config configs/finetuned_eval.yaml --stage asr

Usage
─────
  python scripts/finetune_whisper.py                       # defaults
  python scripts/finetune_whisper.py --epochs 5 --lr 5e-6
  python scripts/finetune_whisper.py --batch-size 4 --grad-accum 2
  python scripts/finetune_whisper.py --dry-run             # list items only
  python scripts/finetune_whisper.py \\
      --train-src data/GTSinger_Augmented/English          # custom data root
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ── stdlib path setup ─────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# ── Constants ─────────────────────────────────────────────────────────────────
SR = 16_000
MODEL_ID = "openai/whisper-large-v3"

# Folder names that carry augmented technique audio (not control/speech).
TECHNIQUE_GROUPS = {
    "Vibrato_Group", "Breathy_Group", "Glissando_Group",
    "Pharyngeal_Group", "Falsetto_Group", "Mixed_Voice_Group",
}


# ── Transcript extraction ─────────────────────────────────────────────────────

def lyrics_from_json(json_path: Path) -> str:
    """Join non-silence words from a GTSinger JSON annotation."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return " ".join(
            w["word"] for w in data
            if isinstance(w, dict) and w.get("word") not in ("<SP>", None, "")
        )
    except Exception:
        return ""


# ── Data collection ───────────────────────────────────────────────────────────

def collect_items(aug_root: Path) -> list[tuple[str, str]]:
    """Return (wav_path, transcript) pairs from augmented technique groups."""
    items: list[tuple[str, str]] = []
    for wav in sorted(aug_root.rglob("*.wav")):
        if wav.parent.name not in TECHNIQUE_GROUPS:
            continue
        json_path = wav.with_suffix(".json")
        if not json_path.exists():
            continue
        text = lyrics_from_json(json_path).strip()
        if not text:
            continue
        items.append((str(wav), text))
    return items


def train_val_split(
    items: list[tuple[str, str]], val_frac: float = 0.1, seed: int = 42
) -> tuple[list, list]:
    """Stratified train/val split by technique (parent-parent folder name)."""
    rng = random.Random(seed)
    by_tech: dict[str, list] = {}
    for item in items:
        tech = Path(item[0]).parent.parent.parent.name   # {Singer}/{Tech}/{Song}/GRP/file
        by_tech.setdefault(tech, []).append(item)

    train, val = [], []
    for group in by_tech.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class WhisperDataset:
    """Lazy-loading dataset: reads audio + tokenises transcript on demand."""

    def __init__(self, items: list[tuple[str, str]], processor: Any) -> None:
        self.items = items
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        wav_path, text = self.items[idx]

        audio, sr = sf.read(wav_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            text, return_tensors="pt"
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}


# ── Data collator ─────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads a batch of (input_features, labels) to the same length.

    Standard HuggingFace recipe for Whisper fine-tuning:
    https://huggingface.co/blog/fine-tune-whisper
    """
    processor: Any
    decoder_start_token_id: int
    input_dtype: Any = None   # torch dtype — cast features so generate() sees right dtype

    def __call__(self, features: list[dict]) -> dict:
        import torch

        # Pad log-mel features (all already [80, T] from the feature extractor).
        input_batch = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_batch, return_tensors="pt"
        )
        # Cast to model dtype so encoder sees matching types in both train and
        # generate() (which runs outside the bf16 autocast context).
        if self.input_dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.input_dtype)

        # Pad token-id labels; mask padding with -100 so loss ignores it.
        label_batch = [{"input_ids": f["labels"]} for f in features]
        labels = self.processor.tokenizer.pad(label_batch, return_tensors="pt")
        label_ids = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        # Strip leading BOS token added by the tokenizer (Whisper adds its own).
        if (label_ids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            label_ids = label_ids[:, 1:]

        batch["labels"] = label_ids
        return batch


# ── WER metric ────────────────────────────────────────────────────────────────

def make_compute_metrics(processor: Any):
    """Return a compute_metrics function compatible with Seq2SeqTrainer."""
    try:
        import evaluate
        metric = evaluate.load("wer")
    except Exception:
        metric = None

    def compute_metrics(pred) -> dict:
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str  = [s.strip().lower() for s in pred_str]
        label_str = [s.strip().lower() for s in label_str]

        if metric is None:
            from jiwer import wer as jiwer_wer
            wer = float(np.mean([jiwer_wer(r, h) for r, h in zip(label_str, pred_str)]))
        else:
            wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(float(wer), 4)}

    return compute_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--train-src",
        default=str(REPO / "data/GTSinger_Augmented/English"),
        help="Root of augmented dataset (default: data/GTSinger_Augmented/English)",
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO / "results/finetune_whisper"),
        help="Output directory for checkpoints (default: results/finetune_whisper)",
    )
    p.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"HuggingFace model id to fine-tune (default: {MODEL_ID})",
    )
    p.add_argument("--epochs",     type=int,   default=3,    help="Training epochs (default: 3)")
    p.add_argument("--batch-size", type=int,   default=8,    help="Per-device batch size (default: 8)")
    p.add_argument("--grad-accum", type=int,   default=1,    help="Gradient accumulation steps (default: 1)")
    p.add_argument("--lr",         type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    p.add_argument("--warmup",     type=int,   default=500,  help="Warmup steps (default: 500)")
    p.add_argument("--eval-steps", type=int,   default=500,  help="Eval + save every N steps (default: 500)")
    p.add_argument("--val-frac",   type=float, default=0.10, help="Validation fraction (default: 0.10)")
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed (default: 42)")
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing to save GPU memory (disables KV-cache)",
    )
    p.add_argument(
        "--freeze-encoder", action="store_true",
        help="Freeze all encoder parameters — only decoder weights are updated. "
             "Cuts trainable params and optimizer memory by ~50%%. "
             "Academically justified: the acoustic encoder already generalises; "
             "the decoder benefits most from domain adaptation.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print data stats and exit without training",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_src = Path(args.train_src)
    out_dir   = Path(args.out_dir)
    best_dir  = out_dir / "best_model"

    if not train_src.exists():
        print(f"[ERROR] Training data not found: {train_src}")
        print("  Run build_augmented_dataset.py first.")
        sys.exit(1)

    # ── Collect items ─────────────────────────────────────────────────────────
    print(f"Scanning {train_src} …")
    all_items = collect_items(train_src)
    if not all_items:
        print("[ERROR] No WAV+JSON pairs found in technique groups.")
        sys.exit(1)

    train_items, val_items = train_val_split(all_items, args.val_frac, args.seed)
    print(f"  Total items : {len(all_items)}")
    print(f"  Train       : {len(train_items)}")
    print(f"  Val         : {len(val_items)}")

    # Technique breakdown
    def _tech_counts(items):
        from collections import Counter
        return Counter(Path(p).parent.parent.parent.name for p, _ in items)
    print(f"  Train breakdown: {dict(_tech_counts(train_items))}")
    print(f"  Val   breakdown: {dict(_tech_counts(val_items))}")

    if args.dry_run:
        print("[DRY RUN] Exiting without training.")
        return

    # ── Load model + processor ────────────────────────────────────────────────
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    # Detect precision BEFORE loading so we can pass torch_dtype to from_pretrained.
    # This ensures the model lives in one dtype throughout (train + eval/generate).
    # Without this, the Trainer casts weights to bf16 during training but the
    # generation call in eval sees float32 inputs against half-precision encoder
    # weights → dtype mismatch RuntimeError.
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    print(f"\nLoading processor + model: {args.model_id}")
    print(f"  Mixed precision  : {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    device = "cuda" if has_cuda else "cpu"

    processor = WhisperProcessor.from_pretrained(args.model_id, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch_dtype
    ).to(device)

    # Standard Whisper fine-tuning setup: disable forced decoder tokens so the
    # model learns to predict the language/task tokens from data.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — only decoder weights will be updated.")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params / 1e6:.0f} M")
    print(f"  Trainable params : {trainable_params / 1e6:.0f} M  ({100*trainable_params/total_params:.1f}%)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = WhisperDataset(train_items, processor)
    val_ds   = WhisperDataset(val_items,   processor)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_dtype=torch_dtype,
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(out_dir),
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup,
        num_train_epochs            = args.epochs,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_strategy               = "steps",
        save_steps                  = args.eval_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = 225,
        logging_steps               = 50,
        report_to                   = "none",   # disable wandb/tensorboard
        seed                        = args.seed,
        dataloader_num_workers      = min(4, os.cpu_count() or 1),
        remove_unused_columns       = False,    # our Dataset returns custom keys
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model              = model,
        args               = training_args,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        data_collator      = collator,
        compute_metrics    = make_compute_metrics(processor),
        processing_class   = processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training — output: {out_dir}")
    trainer.train()

    # ── Save best model ───────────────────────────────────────────────────────
    print(f"\nSaving best model to {best_dir}")
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    # ── Final validation WER ──────────────────────────────────────────────────
    print("\nFinal evaluation on validation set …")
    metrics = trainer.evaluate()
    wer = metrics.get("eval_wer", "N/A")
    print(f"  Validation WER: {wer}")

    print(f"""
═══════════════════════════════════════════════════════
Fine-tuning complete.
Best model saved to: {best_dir}

To evaluate on the original GTSinger test set, run:
  python scripts/run_pipeline.py \\
      --config configs/finetuned_eval.yaml --stage asr

To compare with the baseline:
  results/baseline_english/asr_whisper_largev3_by_technique.csv
  results/finetuned_eval/asr_whisper_finetuned_by_technique.csv
═══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
