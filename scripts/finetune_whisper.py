#!/usr/bin/env python3
"""
finetune_whisper.py
───────────────────
Fine-tune openai/whisper-large-v3 on a controlled train/test split designed
for two comparative experiments:

  Experiment 1 (--mode mixed)
      Train: ALL augmented technique WAVs  +  20 % of original technique WAVs
      Test : held-out 30 % of original technique WAVs  (saved as manifest)

  Experiment 2 (--mode aug_only)
      Train: ALL augmented technique WAVs  (no original samples)
      Test : the SAME held-out 30 % manifest as experiment 1

Both experiments evaluate against the identical test split, so WER differences
are solely attributable to whether original samples were included in training.

The test-set manifest is saved to:
    results/shared_test_manifest.json
and is referenced by the evaluation configs:
    configs/exp1_mixed_eval.yaml
    configs/exp2_aug_only_eval.yaml

Usage
─────
  # Experiment 1 — mixed original + augmented
  python scripts/finetune_whisper.py \\
      --mode mixed \\
      --aug-src  data/GTSinger_Augmented/English \\
      --orig-src data/GTSinger/English

  # Experiment 2 — augmented only (re-uses the manifest written by exp 1)
  python scripts/finetune_whisper.py \\
      --mode aug_only \\
      --aug-src  data/GTSinger_Augmented/English \\
      --orig-src data/GTSinger/English

  # Quick sanity check (no training)
  python scripts/finetune_whisper.py --mode mixed --dry-run \\
      --aug-src data/GTSinger_Augmented/English \\
      --orig-src data/GTSinger/English
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

TECHNIQUE_GROUPS = {
    "Vibrato_Group", "Breathy_Group", "Glissando_Group",
    "Pharyngeal_Group", "Falsetto_Group", "Mixed_Voice_Group",
}

# Shared manifest path — written by either experiment run, consumed by both.
DEFAULT_MANIFEST = str(REPO / "results/shared_test_manifest.json")


# ── Transcript extraction ─────────────────────────────────────────────────────

def lyrics_from_json(json_path: Path) -> str:
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

def collect_augmented_items(aug_root: Path) -> list[tuple[str, str]]:
    """(wav_path, transcript) pairs from augmented technique groups."""
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


def collect_original_items(orig_root: Path) -> list[tuple[str, str, str, str]]:
    """(wav_path, transcript, singer_id, technique) from original technique groups."""
    items: list[tuple[str, str, str, str]] = []
    for wav in sorted(orig_root.rglob("*.wav")):
        if wav.parent.name not in TECHNIQUE_GROUPS:
            continue
        json_path = wav.with_suffix(".json")
        if not json_path.exists():
            continue
        text = lyrics_from_json(json_path).strip()
        if not text:
            continue
        # Layout: <orig_root>/<singer>/<technique_folder>/<song>/<group>/<file>
        singer_id = wav.parent.parent.parent.parent.name
        technique = wav.parent.parent.parent.name
        items.append((str(wav), text, singer_id, technique))
    return items


# ── Splits ────────────────────────────────────────────────────────────────────

def split_for_experiments(
    orig_items: list[tuple],
    test_frac: float = 0.30,
    orig_train_frac: float = 0.20,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratified split of original items by technique.

    Returns:
        train_orig — ~orig_train_frac of total, used in exp1 training
        test       — ~test_frac of total, held out for evaluation in both exps
    The remaining ~(1 - test_frac - orig_train_frac) of original items are
    intentionally discarded so that exp1 trains on less-than-all original data.
    """
    rng = random.Random(seed)
    by_tech: dict[str, list] = {}
    for item in orig_items:
        tech = item[3]  # technique folder name
        by_tech.setdefault(tech, []).append(item)

    train_orig: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []

    for group in by_tech.values():
        rng.shuffle(group)
        n_test  = round(len(group) * test_frac)
        n_train = round(len(group) * orig_train_frac)
        test.extend((p, t) for p, t, *_ in group[:n_test])
        train_orig.extend((p, t) for p, t, *_ in group[n_test: n_test + n_train])

    rng.shuffle(train_orig)
    rng.shuffle(test)
    return train_orig, test


def aug_val_split(
    aug_items: list[tuple[str, str]], val_frac: float = 0.10, seed: int = 42
) -> tuple[list, list]:
    """Stratified train/val split of augmented items by technique group folder."""
    rng = random.Random(seed)
    by_tech: dict[str, list] = {}
    for item in aug_items:
        tech = Path(item[0]).parent.parent.parent.name  # technique folder
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
    processor: Any
    decoder_start_token_id: int
    input_dtype: Any = None

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_batch = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_batch, return_tensors="pt"
        )
        if self.input_dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.input_dtype)

        label_batch = [{"input_ids": f["labels"]} for f in features]
        labels = self.processor.tokenizer.pad(label_batch, return_tensors="pt")
        label_ids = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        if (label_ids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            label_ids = label_ids[:, 1:]

        batch["labels"] = label_ids
        return batch


# ── WER metric ────────────────────────────────────────────────────────────────

def make_compute_metrics(processor: Any):
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["mixed", "aug_only", "aug_matched", "orig_only"],
        required=True,
        help=(
            "mixed       — train on ALL augmented + 20%% of original        (exp 1)\n"
            "aug_only    — train on ALL augmented only                      (exp 2)\n"
            "aug_matched — ALL augmented + N extra aug items, N=len(train_orig), "
            "same total size as mixed  (exp A: fair comparison)\n"
            "orig_only   — 20%% original only, no augmented                 (exp B)"
        ),
    )
    p.add_argument(
        "--aug-src",
        default=str(REPO / "data/GTSinger_Augmented/English"),
        help="Root of augmented dataset (default: data/GTSinger_Augmented/English)",
    )
    p.add_argument(
        "--orig-src",
        default=str(REPO / "data/GTSinger/English"),
        help="Root of ORIGINAL GTSinger English (default: data/GTSinger/English). "
             "Used to build the shared test split and (in mixed mode) train_orig.",
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Where to write/read the shared test manifest (default: {DEFAULT_MANIFEST})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for checkpoints. Defaults to "
             "results/finetune_whisper_<mode>/",
    )
    p.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"HuggingFace model id (default: {MODEL_ID})",
    )
    p.add_argument("--test-frac",       type=float, default=0.30,
                   help="Fraction of original data held out as test (default: 0.30)")
    p.add_argument("--orig-train-frac", type=float, default=0.20,
                   help="Fraction of original data used for training in mixed mode (default: 0.20)")
    p.add_argument("--val-frac",        type=float, default=0.10,
                   help="Fraction of augmented data used for validation (default: 0.10)")
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--grad-accum",      type=int,   default=1)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--warmup",          type=int,   default=500)
    p.add_argument("--eval-steps",      type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing (saves GPU memory, disables KV-cache)",
    )
    p.add_argument(
        "--freeze-encoder", action="store_true",
        help="Freeze encoder weights — only decoder is updated",
    )
    p.add_argument(
        "--optim",
        default="adamw_torch",
        help=(
            "HuggingFace optimizer name passed to Seq2SeqTrainingArguments. "
            "Use 'adamw_8bit' (requires bitsandbytes) to cut optimizer-state "
            "GPU memory from ~12 GB to ~3 GB for Whisper large-v3. "
            "(default: adamw_torch)"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print data stats and exit without training",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    aug_root  = Path(args.aug_src)
    orig_root = Path(args.orig_src)
    out_dir   = Path(args.out_dir) if args.out_dir else (
        REPO / f"results/finetune_whisper_{args.mode}"
    )
    best_dir      = out_dir / "best_model"
    manifest_path = Path(args.manifest)

    # ── Validate sources ──────────────────────────────────────────────────────
    for p, label in [(aug_root, "--aug-src"), (orig_root, "--orig-src")]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    # ── Collect augmented items ───────────────────────────────────────────────
    print(f"Scanning augmented data: {aug_root}")
    aug_items = collect_augmented_items(aug_root)
    if not aug_items:
        print("[ERROR] No WAV+JSON pairs found in augmented technique groups.")
        sys.exit(1)
    print(f"  Augmented items: {len(aug_items)}")

    # ── Collect original items and build shared split ─────────────────────────
    print(f"\nScanning original data: {orig_root}")
    orig_items = collect_original_items(orig_root)
    if not orig_items:
        print("[ERROR] No WAV+JSON pairs found in original technique groups.")
        sys.exit(1)
    print(f"  Original items: {len(orig_items)}")

    train_orig, test_items = split_for_experiments(
        orig_items,
        test_frac=args.test_frac,
        orig_train_frac=args.orig_train_frac,
        seed=args.seed,
    )
    print(f"  → Test set   : {len(test_items):4d}  ({args.test_frac:.0%} of original)")
    print(f"  → Train orig : {len(train_orig):4d}  ({args.orig_train_frac:.0%} of original, exp1 only)")
    print(f"  → Unused orig: {len(orig_items) - len(test_items) - len(train_orig):4d}  (intentionally discarded)")

    # Save test manifest (overwrite — deterministic with same seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump([p for p, _ in test_items], fh, indent=2)
    print(f"\nTest manifest saved → {manifest_path}")

    # ── Build training set based on mode ─────────────────────────────────────
    aug_train, aug_val = aug_val_split(aug_items, val_frac=args.val_frac, seed=args.seed)

    if args.mode == "mixed":
        train_items = aug_train + train_orig
        random.Random(args.seed).shuffle(train_items)
        val_items   = aug_val
        mode_label  = "ALL augmented + 20% original"
    elif args.mode == "aug_only":
        train_items = aug_train
        val_items   = aug_val
        mode_label  = "ALL augmented only (no original)"
    elif args.mode == "aug_matched":
        # Top up aug_train with N aug_val items so total == len(mixed train).
        # N = len(train_orig) so the only variable vs exp1 is data quality.
        n_orig      = len(train_orig)
        extra_aug   = aug_val[:n_orig]
        val_items   = aug_val[n_orig:] or aug_val   # keep at least some val
        train_items = aug_train + extra_aug
        random.Random(args.seed).shuffle(train_items)
        mode_label  = (f"ALL augmented + {n_orig} extra aug items "
                       f"(size-matched to mixed, total={len(train_items)})")
    else:  # orig_only
        train_items = train_orig
        val_items   = aug_val   # proxy val set for checkpoint selection
        mode_label  = "20% original only (no augmented data)"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nMode            : {args.mode}  ({mode_label})")
    print(f"Train           : {len(train_items)}")
    print(f"Val             : {len(val_items)}")
    print(f"Test (held-out) : {len(test_items)}  [not used during training]")

    def _tech_counts(items):
        from collections import Counter
        return Counter(Path(p).parent.parent.parent.name for p, _ in items)

    print(f"Train breakdown : {dict(_tech_counts(train_items))}")
    print(f"Val   breakdown : {dict(_tech_counts(val_items))}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without training.")
        return

    # ── Load model + processor ────────────────────────────────────────────────
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    has_cuda   = torch.cuda.is_available()
    use_bf16   = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16   = has_cuda and not use_bf16
    torch_dtype = (torch.bfloat16 if use_bf16 else
                   torch.float16  if use_fp16 else torch.float32)
    device = "cuda" if has_cuda else "cpu"

    print(f"\nLoading processor + model: {args.model_id}")
    print(f"  Mixed precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    processor = WhisperProcessor.from_pretrained(args.model_id, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch_dtype
    ).to(device)

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
    print(f"  Total params    : {total_params / 1e6:.0f} M")
    print(f"  Trainable params: {trainable_params / 1e6:.0f} M  ({100*trainable_params/total_params:.1f}%)")

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
        optim                       = args.optim,
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
        report_to                   = "none",
        seed                        = args.seed,
        dataloader_num_workers      = min(4, os.cpu_count() or 1),
        remove_unused_columns       = False,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = make_compute_metrics(processor),
        processing_class= processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training — output: {out_dir}")
    trainer.train()

    print(f"\nSaving best model → {best_dir}")
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    print("\nFinal evaluation on validation set …")
    metrics = trainer.evaluate()
    print(f"  Validation WER: {metrics.get('eval_wer', 'N/A')}")

    print(f"""
═══════════════════════════════════════════════════════════════
Fine-tuning complete  [{args.mode}]
Best model   → {best_dir}
Test manifest→ {manifest_path}

Evaluate on the shared test set:
  python scripts/run_pipeline.py \\
      --config configs/exp1_mixed_eval.yaml --stage asr   # exp 1
  python scripts/run_pipeline.py \\
      --config configs/exp2_aug_only_eval.yaml --stage asr  # exp 2
═══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
