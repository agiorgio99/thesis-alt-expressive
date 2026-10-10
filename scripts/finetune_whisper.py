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

Song-disjoint split
    GTSinger records the same song several times — once per technique folder,
    and often by more than one singer. An utterance-level split therefore puts
    lines of the same song (same lyrics) in both train and test, inflating
    test scores. The split is instead made over *songs* (normalised song-folder
    title, pooled across singers and techniques): every utterance of a test
    song — real or augmented, any singer, any technique — is kept out of every
    training and validation pool.

Recipe mode (--mode recipe) — the ablation used from 2026-10 on
    Three song-disjoint sets per language: the fixed test songs (above), a
    REAL validation set (--val-song-frac of the corpus, whole songs, technique
    takes only; drives checkpoint selection) and the training songs (the
    rest). The training set is composed explicitly, so every effect can be
    isolated:
      R   real technique takes        --real-frac (share of the corpus; 1 = all)
      C   real control takes          --with-control
      A   augmented takes             --with-aug  (from --aug-src; --aug-max caps)
    No real training-song clip is discarded unless --real-frac < 1. Validation
    songs are removed from R, C and A, like test songs.

The test-set manifest is saved per language (default):
    results/splits/test_manifest_<lang>.json      (list of audio paths)
    results/splits/test_manifest_<lang>.songs.json (test songs + split stats)
and is referenced by the evaluation configs (configs/exp*_eval.yaml).

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
import re
import sys
from collections import Counter
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

from alt.augment.pairing import song_key  # noqa: E402  (split unit, shared with augmentation)
from alt.text import (  # noqa: E402  (needs the sys.path tweak above)
    ISO_TO_LANGUAGE, SKIP_LABELS, join_words, primary_metric,
)

# Shared manifest path template — written by every experiment run of a given
# language (deterministic for a fixed seed), consumed by all eval configs.
DEFAULT_MANIFEST_TEMPLATE = str(REPO / "results/splits/test_manifest_{lang}.json")


# ── Transcript extraction ─────────────────────────────────────────────────────

def lyrics_from_json(json_path: Path, language: str = "english") -> str:
    """Build the training/eval transcript from a GTSinger JSON annotation.

    Silence and breath tokens (``<SP>``, ``<AP>``, ... — see ``SKIP_LABELS``)
    are dropped so they never become Whisper training targets. Words are
    joined per language: without separators for character-level languages
    (zh/ja/ko, whose "words" are single characters/syllables), with spaces
    otherwise.

    Args:
        json_path: Path to the ``.json`` annotation next to a WAV.
        language:  Lowercase language name (e.g. "english", "chinese").

    Returns:
        The transcript, or "" if the file is unreadable or has no lyrics.
    """
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        words = [
            str(w["word"]) for w in data
            if isinstance(w, dict) and w.get("word") is not None
            and str(w["word"]).strip() not in SKIP_LABELS
            and not re.fullmatch(r"<[^>]*>", str(w["word"]).strip())
        ]
        return join_words(words, language)
    except Exception:
        return ""


# ── Data collection ───────────────────────────────────────────────────────────

def collect_augmented_items(aug_root: Path,
                            language: str = "english") -> list[tuple[str, str]]:
    """(wav_path, transcript) pairs from augmented technique groups."""
    items: list[tuple[str, str]] = []
    for wav in sorted(aug_root.rglob("*.wav")):
        if wav.parent.name not in TECHNIQUE_GROUPS:
            continue
        json_path = wav.with_suffix(".json")
        if not json_path.exists():
            continue
        text = lyrics_from_json(json_path, language).strip()
        if not text:
            continue
        items.append((str(wav), text))
    return items


def collect_original_items(orig_root: Path, language: str = "english",
                           ) -> list[tuple[str, str, str, str, str]]:
    """(wav_path, transcript, singer_id, technique, song_key) from original
    technique groups."""
    items: list[tuple[str, str, str, str, str]] = []
    for wav in sorted(orig_root.rglob("*.wav")):
        if wav.parent.name not in TECHNIQUE_GROUPS:
            continue
        json_path = wav.with_suffix(".json")
        if not json_path.exists():
            continue
        text = lyrics_from_json(json_path, language).strip()
        if not text:
            continue
        # Layout: <orig_root>/<singer>/<technique_folder>/<song>/<group>/<file>
        singer_id = wav.parent.parent.parent.parent.name
        technique = wav.parent.parent.parent.name
        items.append((str(wav), text, singer_id, technique, song_key(wav)))
    return items


def collect_control_items(orig_root: Path, language: str = "english",
                          ) -> list[tuple[str, str, str, str, str]]:
    """(wav_path, transcript, singer_id, technique, song_key) of the real
    Control_Group takes (same phrases sung without the technique)."""
    items: list[tuple[str, str, str, str, str]] = []
    for wav in sorted(orig_root.rglob("Control_Group/*.wav")):
        json_path = wav.with_suffix(".json")
        if not json_path.exists():
            continue
        text = lyrics_from_json(json_path, language).strip()
        if not text:
            continue
        items.append((str(wav), text, wav.parents[3].name, wav.parents[2].name,
                      song_key(wav)))
    return items


def select_val_songs(pool_items: list[tuple], n_total: int, val_song_frac: float,
                     seed: int) -> set[str]:
    """Pick whole validation songs from the non-test pool.

    Reuses the technique-balanced song selection of the test split, aiming at
    ``val_song_frac`` of the WHOLE corpus (so the share is comparable across
    languages), with a different seed stream than the test split.

    Args:
        pool_items:    Original technique items of the non-test songs.
        n_total:       Number of original technique items in the corpus.
        val_song_frac: Target share of the corpus held out for validation.
        seed:          Base seed.

    Returns:
        The validation song keys.
    """
    frac_of_pool = min(0.5, val_song_frac * n_total / max(len(pool_items), 1))
    _, _, val_songs = split_for_experiments(pool_items, test_frac=frac_of_pool,
                                            orig_train_frac=0.0, seed=seed + 1)
    return val_songs


def stratified_take(items: list[tuple], n: int, seed: int) -> list[tuple]:
    """Up to ``n`` items, stratified by technique (item[3]), seeded."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    by_tech: dict[str, list] = {}
    for it in items:
        by_tech.setdefault(it[3], []).append(it)
    out: list[tuple] = []
    for tech in sorted(by_tech):
        g = by_tech[tech][:]
        rng.shuffle(g)
        out += g[:round(n * len(g) / len(items))]
    rng.shuffle(out)
    return out[:n]


# ── Splits ────────────────────────────────────────────────────────────────────

def split_for_experiments(
    orig_items: list[tuple],
    test_frac: float = 0.30,
    orig_train_frac: float = 0.20,
    seed: int = 42,
    n_trials: int = 500,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], set[str]]:
    """Song-disjoint, technique-balanced split of original items.

    Whole songs (``song_key``, pooled across singers and techniques) are
    assigned to the test set, so no lyrics are shared between test and any
    training data. Songs are visited in a seeded random order and a song is
    accepted into the test set only if doing so brings the per-technique
    utterance counts closer to their targets (``test_frac`` × technique size),
    which keeps the test set's technique mix close to the corpus mix even
    though songs span several techniques. The best of ``n_trials`` such
    passes (smallest worst-case per-technique deviation) is kept.

    Args:
        orig_items:      ``collect_original_items`` output.
        test_frac:       Target fraction of original utterances held out.
        orig_train_frac: Fraction of original utterances (per technique) used
            as ``train_orig``; drawn only from non-test songs.
        seed:            RNG seed — the split is deterministic for a seed.
        n_trials:        Number of random greedy passes to choose from.

    Returns:
        train_orig — ~orig_train_frac of total, used in exp1 training
        test       — ~test_frac of total, held out for evaluation in all exps
        test_songs — song keys in the test set (used to filter augmented data)
    The remaining original items are intentionally discarded so that exp1
    trains on less-than-all original data.
    """
    rng = random.Random(seed)

    # Per-song technique counts, and per-technique targets for the test set.
    by_song: dict[str, list[tuple]] = {}
    for item in orig_items:
        by_song.setdefault(item[4], []).append(item)
    tech_total = Counter(item[3] for item in orig_items)
    target = {t: test_frac * n for t, n in tech_total.items()}

    def _deficit(counts: Counter) -> float:
        """Total absolute distance from the per-technique test targets."""
        return sum(abs(target[t] - counts[t]) for t in target)

    def _one_pass(order: list[str]) -> tuple[set[str], Counter]:
        """Greedy pass: accept a song if it moves the test mix toward target."""
        chosen: set[str] = set()
        current: Counter = Counter()
        for song in order:
            song_counts = Counter(item[3] for item in by_song[song])
            if _deficit(current + song_counts) < _deficit(current):
                chosen.add(song)
                current += song_counts
        return chosen, current

    def _worst_ratio_error(counts: Counter) -> float:
        """Largest per-technique deviation of the held-out fraction from
        ``test_frac`` (e.g. 0.45 vs 0.30 → 0.15)."""
        return max(abs(counts[t] / tech_total[t] - test_frac) for t in tech_total)

    # A single greedy pass can overshoot one technique when a few songs carry
    # most of its utterances. Run several seeded random passes and keep the
    # one whose worst per-technique fraction is closest to ``test_frac`` —
    # still random over songs, deterministic for a given seed.
    songs = sorted(by_song)          # sort first so the shuffles are reproducible
    best: tuple[float, set[str]] | None = None
    for _ in range(n_trials):
        order = songs[:]
        rng.shuffle(order)
        chosen, counts = _one_pass(order)
        err = _worst_ratio_error(counts)
        if best is None or err < best[0]:
            best = (err, chosen)
    test_songs = best[1]

    test = [(p, t) for song in test_songs for p, t, *_ in by_song[song]]

    # train_orig: stratified by technique, only from songs NOT in the test set.
    by_tech: dict[str, list] = {}
    for item in orig_items:
        if item[4] not in test_songs:
            by_tech.setdefault(item[3], []).append(item)
    train_orig: list[tuple[str, str]] = []
    for tech in sorted(by_tech):
        group = by_tech[tech]
        rng.shuffle(group)
        n_train = min(len(group), round(tech_total[tech] * orig_train_frac))
        train_orig.extend((p, t) for p, t, *_ in group[:n_train])

    rng.shuffle(train_orig)
    test.sort()                      # stable, readable manifest order
    return train_orig, test, test_songs


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


# ── WER / CER metric ──────────────────────────────────────────────────────────

def make_compute_metrics(processor: Any, language: str = "english"):
    """Build the Seq2SeqTrainer ``compute_metrics`` callback.

    Scores validation predictions with the project's shared ``score_pair``
    (same text normalisation as the evaluation pipeline) and reports corpus
    means of both WER and CER. Checkpoint selection uses the language's
    headline metric (``primary_metric``: CER for zh/ja/ko, WER otherwise).

    Args:
        processor: Whisper processor used to decode token ids.
        language:  Lowercase language name (e.g. "english", "chinese").

    Returns:
        A callable mapping an ``EvalPrediction`` to ``{"wer": .., "cer": ..}``.
    """
    from alt.metrics import score_pair

    def compute_metrics(pred) -> dict:
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        scores = [score_pair(r, h, language) for r, h in zip(label_str, pred_str)]
        wers = [x["wer"] for x in scores if x["wer"] is not None]
        cers = [x["cer"] for x in scores if x["cer"] is not None]
        return {
            "wer": round(float(np.mean(wers)), 4) if wers else 1.0,
            "cer": round(float(np.mean(cers)), 4) if cers else 1.0,
        }

    return compute_metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["mixed", "aug_only", "aug_matched", "orig_only", "recipe"],
        default=None,
        help=(
            "mixed       — train on ALL augmented + 20%% of original        (exp 1)\n"
            "aug_only    — train on ALL augmented only                      (exp 2)\n"
            "aug_matched — ALL augmented + N extra aug items, N=len(train_orig), "
            "same total size as mixed  (exp A: fair comparison)\n"
            "orig_only   — 20%% original only, no augmented                 (exp B)\n"
            "recipe      — explicit R / C / A composition with a real, "
            "song-disjoint validation set (see --real-frac, --with-control, --with-aug)"
        ),
    )
    p.add_argument(
        "--aug-src",
        default=None,
        help="Root of augmented dataset (default: "
             "data/GTSinger_Augmented/<Language> for --language)",
    )
    p.add_argument(
        "--orig-src",
        default=None,
        help="Root of ORIGINAL GTSinger data (default: "
             "data/GTSinger/<Language> for --language). "
             "Used to build the shared test split and (in mixed mode) train_orig.",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Where to write the shared test manifest (default: "
             "results/splits/test_manifest_<language>.json). A sibling "
             "'.songs.json' file records the test songs and split stats.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for checkpoints. Defaults to "
             "results/finetune_whisper_<mode>/ for English and "
             "results/finetune_whisper_<mode>_<lang>/ otherwise.",
    )
    p.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"HuggingFace model id (default: {MODEL_ID})",
    )
    p.add_argument(
        "--language", default="en",
        choices=sorted(ISO_TO_LANGUAGE),
        help="Whisper ISO-639-1 language code for the processor's decoder "
             "prompt (default: en). Must match the dataset's actual language "
             "— e.g. zh/ja/ko/ru/es/fr/de/it — or the fine-tuned model's "
             "language token will be wrong. Also selects transcript joining, "
             "text normalisation and the checkpoint-selection metric "
             "(CER for zh/ja/ko, WER otherwise).",
    )
    p.add_argument("--test-frac",       type=float, default=0.30,
                   help="Target fraction of original data held out as test, "
                        "assigned by whole songs (default: 0.30)")
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
    # ── Recipe mode ──────────────────────────────────────────────────────────
    p.add_argument("--run-name", default=None,
                   help="Recipe run name; output defaults to "
                        "results/finetune/<lang>/<run-name>/")
    p.add_argument("--real-frac", type=float, default=1.0,
                   help="Recipe: real technique takes used for training, as a share "
                        "of the WHOLE corpus (1.0 = every training-song take; "
                        "0.2 = the low-resource setting)")
    p.add_argument("--with-control", action="store_true",
                   help="Recipe: also train on the real Control_Group takes")
    p.add_argument("--with-aug", action="store_true",
                   help="Recipe: also train on the augmented takes of --aug-src")
    p.add_argument("--aug-max", type=int, default=None,
                   help="Recipe: cap the augmented items (random, stratified) — "
                        "e.g. to match two augmentation engines in size")
    p.add_argument("--val-song-frac", type=float, default=0.10,
                   help="Recipe: share of the corpus held out as REAL validation "
                        "songs (default 0.10)")
    p.add_argument("--val-max", type=int, default=300,
                   help="Recipe: cap on validation clips (generation is slow)")
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="Warmup as a share of all optimiser steps (recipe default "
                        "0.1; overrides --warmup)")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="Keep intermediate checkpoint-* folders (default: removed "
                        "after training in recipe mode; best_model is kept)")
    p.add_argument(
        "--split-only", action="store_true",
        help="Only build and save the song-disjoint test manifest for "
             "--orig-src, then exit (no augmented data needed; --mode is "
             "ignored). Use it to fix each language's test set before any "
             "augmentation or training run.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print data stats and exit without training",
    )
    args = p.parse_args()
    if args.mode is None and not args.split_only:
        p.error("--mode is required unless --split-only is given")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    language  = ISO_TO_LANGUAGE[args.language]   # e.g. "zh" → "chinese"
    metric    = primary_metric(language)          # "cer" or "wer"
    lang_dir  = language.capitalize()             # GTSinger folder name
    aug_root  = Path(args.aug_src or REPO / "data/GTSinger_Augmented" / lang_dir)
    orig_root = Path(args.orig_src or REPO / "data/GTSinger" / lang_dir)
    # English keeps the historical output folder names (used by the configs);
    # other languages get a language suffix so runs never overwrite each other.
    suffix    = "" if args.language == "en" else f"_{args.language}"
    if args.mode == "recipe" and not args.run_name:
        sys.exit("--run-name is required with --mode recipe")
    out_dir   = Path(args.out_dir) if args.out_dir else (
        REPO / "results" / "finetune" / args.language / args.run_name
        if args.mode == "recipe" else
        REPO / f"results/finetune_whisper_{args.mode}{suffix}"
    )
    best_dir      = out_dir / "best_model"
    manifest_path = Path(args.manifest or
                         DEFAULT_MANIFEST_TEMPLATE.format(lang=args.language))

    # ── Validate sources ──────────────────────────────────────────────────────
    # --split-only needs only the original data (the test set is built from it).
    sources = [(orig_root, "--orig-src")]
    need_aug = not args.split_only and (args.mode != "recipe" or args.with_aug)
    if need_aug:
        sources.insert(0, (aug_root, "--aug-src"))
    for p, label in sources:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    # ── Collect augmented items ───────────────────────────────────────────────
    aug_items: list[tuple[str, str]] = []
    if need_aug:
        print(f"Scanning augmented data: {aug_root}")
        aug_items = collect_augmented_items(aug_root, language)
        if not aug_items:
            print("[ERROR] No WAV+JSON pairs found in augmented technique groups.")
            sys.exit(1)
        print(f"  Augmented items: {len(aug_items)}")

    # ── Collect original items and build shared split ─────────────────────────
    print(f"\nScanning original data: {orig_root}")
    orig_items = collect_original_items(orig_root, language)
    if not orig_items:
        print("[ERROR] No WAV+JSON pairs found in original technique groups.")
        sys.exit(1)
    print(f"  Original items: {len(orig_items)}")

    train_orig, test_items, test_songs = split_for_experiments(
        orig_items,
        test_frac=args.test_frac,
        orig_train_frac=args.orig_train_frac,
        seed=args.seed,
    )
    n_orig = len(orig_items)
    all_songs = {item[4] for item in orig_items}
    print(f"  → Test set   : {len(test_items):4d}  ({len(test_items) / n_orig:.1%} of original, "
          f"{len(test_songs)}/{len(all_songs)} songs; target {args.test_frac:.0%})")
    print(f"  → Train orig : {len(train_orig):4d}  ({len(train_orig) / n_orig:.1%} of original, exp1 only)")
    print(f"  → Unused orig: {n_orig - len(test_items) - len(train_orig):4d}  (intentionally discarded)")

    # Technique mix of the test set vs. the whole corpus (balance check).
    tech_all  = Counter(item[3] for item in orig_items)
    tech_test = Counter(item[3] for item in orig_items if item[4] in test_songs)
    for tech in sorted(tech_all):
        print(f"      {tech:28s} test {tech_test[tech]:4d} / {tech_all[tech]:4d} "
              f"({tech_test[tech] / tech_all[tech]:.1%})")

    # Save test manifest (overwrite — deterministic with same seed), plus a
    # sidecar recording which songs are held out and the split statistics.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        # Repo-relative paths: portable across machines and identical to the
        # paths the evaluation pipeline lists (it runs from the repo root).
        json.dump([os.path.relpath(p, REPO) for p, _ in test_items], fh,
                  indent=2, ensure_ascii=False)
    songs_path = manifest_path.with_suffix(".songs.json")
    with open(songs_path, "w", encoding="utf-8") as fh:
        json.dump({
            "language": language,
            "seed": args.seed,
            "test_frac_target": args.test_frac,
            "n_original": n_orig,
            "n_test": len(test_items),
            "test_songs": sorted(test_songs),
            "test_by_technique": dict(sorted(tech_test.items())),
            "all_by_technique": dict(sorted(tech_all.items())),
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nTest manifest saved → {manifest_path}")
    print(f"Test songs saved    → {songs_path}")
    if args.split_only:
        print("\n[SPLIT ONLY] Exiting without building training data.")
        return

    if args.mode == "recipe":
        train_items, val_items, mode_label = build_recipe(
            args, orig_items, orig_root, language, aug_items, test_songs)
    else:
        train_items, val_items, mode_label = build_legacy_modes(
            args, aug_items, train_orig, test_songs)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nMode            : {args.mode}  ({mode_label})")
    print(f"Train           : {len(train_items)}")
    print(f"Val             : {len(val_items)}")
    print(f"Test (held-out) : {len(test_items)}  [not used during training]")

    def _tech_counts(items):
        return Counter(Path(p).parent.parent.parent.name for p, _ in items)

    print(f"Train breakdown : {dict(_tech_counts(train_items))}")
    print(f"Val   breakdown : {dict(_tech_counts(val_items))}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without training.")
        return
    train_and_save(args, train_items, val_items, out_dir, best_dir, manifest_path,
                   language, metric)


def build_recipe(args, orig_items, orig_root, language, aug_items, test_songs):
    """Compose train / real validation sets for ``--mode recipe``.

    Returns:
        ``(train_items, val_items, label)`` — items are (wav, text) pairs.
    """
    n_total = len(orig_items)
    pool = [it for it in orig_items if it[4] not in test_songs]
    val_songs = select_val_songs(pool, n_total, args.val_song_frac, args.seed)
    held = test_songs | val_songs
    val_all = [it for it in pool if it[4] in val_songs]
    val_items = [(p, t) for p, t, *_ in stratified_take(val_all, args.val_max, args.seed)]
    train_pool = [it for it in pool if it[4] not in val_songs]
    print(f"  → Val songs  : {len(val_songs)} songs, {len(val_all)} real technique takes "
          f"(using {len(val_items)})")
    print(f"  → Train pool : {len(train_pool)} real technique takes "
          f"(nothing discarded at --real-frac 1)")

    parts, label = [], []
    n_real = min(len(train_pool), round(n_total * args.real_frac))
    real = stratified_take(train_pool, n_real, args.seed)
    parts += [(p, t) for p, t, *_ in real]
    label.append(f"R{args.real_frac:g}={len(real)}")
    if args.with_control:
        ctrl = [it for it in collect_control_items(orig_root, language) if it[4] not in held]
        parts += [(p, t) for p, t, *_ in ctrl]
        label.append(f"C={len(ctrl)}")
    if args.with_aug:
        aug = [(p, t) for p, t in aug_items if song_key(Path(p)) not in held]
        if args.aug_max is not None and len(aug) > args.aug_max:
            tagged = [(p, t, "", Path(p).parent.name, "") for p, t in aug]
            aug = [(p, t) for p, t, *_ in stratified_take(tagged, args.aug_max, args.seed)]
        parts += aug
        label.append(f"A={len(aug)}")

    # Leakage guards: no test or validation song on the training side.
    train_songs = {song_key(Path(p)) for p, _ in parts}
    assert not (train_songs & test_songs), "song leakage train ↔ test"
    assert not (train_songs & val_songs), "song leakage train ↔ validation"
    random.Random(args.seed).shuffle(parts)
    # Record the validation songs next to the test manifest.
    val_path = REPO / "results" / "splits" / f"val_songs_{args.language}.json"
    with open(val_path, "w", encoding="utf-8") as fh:
        json.dump(sorted(val_songs), fh, indent=2, ensure_ascii=False)
    return parts, val_items, " + ".join(label) + f"  (real val: {len(val_items)})"


def build_legacy_modes(args, aug_items, train_orig, test_songs):
    """Training / validation sets of the original four experiment modes.

    Returns:
        ``(train_items, val_items, label)``.
    """
    # ── Remove test songs from the augmented pool ─────────────────────────────
    # Augmented variants are synthesised from each song's Control_Group, so a
    # test song's variants carry the test lyrics. Drop them from BOTH training
    # and validation (validation drives checkpoint selection) to keep the
    # held-out songs completely unseen.
    n_aug_before = len(aug_items)
    aug_items = [(p, t) for p, t in aug_items if song_key(Path(p)) not in test_songs]
    print(f"\nAugmented items after removing test songs: {len(aug_items)} "
          f"(dropped {n_aug_before - len(aug_items)})")
    if not aug_items:
        print("[ERROR] No augmented items left after removing test songs.")
        sys.exit(1)

    # Sanity check: no song may appear on both sides of the split.
    train_songs = {song_key(Path(p)) for p, _ in train_orig + aug_items}
    assert not (train_songs & test_songs), "song leakage between train and test"

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

    return train_items, val_items, mode_label


def train_and_save(args, train_items, val_items, out_dir, best_dir, manifest_path,
                   language, metric) -> None:
    """Fine-tune Whisper on ``train_items``, select on ``val_items``, save best."""
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

    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task="transcribe")
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

    # Step budget: warm up over a share of all steps (fixed 500 steps would
    # exceed a whole small run), evaluate / save twice per epoch.
    steps_per_epoch = max(1, -(-len(train_items) // (args.batch_size * args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else \
        (0.1 if args.mode == "recipe" else None)
    warmup_steps = int(warmup_ratio * total_steps) if warmup_ratio is not None else args.warmup
    eval_steps = max(25, steps_per_epoch // 2) if args.mode == "recipe" else args.eval_steps
    print(f"  Steps: {total_steps} ({steps_per_epoch}/epoch) · warmup {warmup_steps} · "
          f"eval every {eval_steps}")

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(out_dir),
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = warmup_steps,
        num_train_epochs            = args.epochs,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        optim                       = args.optim,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        save_total_limit            = 2,        # best + latest (disk: ~9 GB each)
        load_best_model_at_end      = True,
        metric_for_best_model       = metric,   # CER for zh/ja/ko, else WER
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
        compute_metrics = make_compute_metrics(processor, language),
        processing_class= processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training — output: {out_dir}")
    trainer.train()

    print(f"\nSaving best model → {best_dir}")
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    # Trainer state (loss / validation curves) next to the model.
    trainer.state.save_to_json(str(out_dir / "trainer_state.json"))

    print("\nFinal evaluation on validation set …")
    metrics = trainer.evaluate()
    print(f"  Validation WER: {metrics.get('eval_wer', 'N/A')}  "
          f"CER: {metrics.get('eval_cer', 'N/A')}  (selection metric: {metric.upper()})")

    if args.mode == "recipe" and not args.keep_checkpoints:
        import shutil
        for ck in out_dir.glob("checkpoint-*"):
            shutil.rmtree(ck, ignore_errors=True)
        with open(out_dir / "val_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Removed intermediate checkpoints; kept {best_dir}")

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
