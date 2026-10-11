#!/usr/bin/env python3
"""
run_finetune_matrix.py
──────────────────────
Run the whole fine-tuning ablation (``finetune_whisper.py --mode recipe``) for
every language, one run after the other on the single GPU, then evaluate each
model on the fixed song-disjoint test set and update the statistics.

Runs per language (same test set, same real validation songs, same seed):

  E0      no fine-tuning (whisper-large-v3 as is)              reference
  R       real technique takes of the training songs           fine-tuning effect
  RC      R + real control takes                               "more real singing"
  RCA     RC + augmented (world_v2 / PSOLA, sampled)           does synthesis add?
  RCA_v1  RC + augmented with v1, size-capped to RCA's A       calibrated vs hand-set
  R20C    20 % of the corpus as real technique takes + C       low-resource setting
  R20CA   R20C + augmented                                     low-resource + synthesis
  C       real control takes only (no real technique take)     zero-resource reference
  CA      C + augmented (v2)                                   can synthesis replace
                                                               missing technique data?
  CA_v1   C + augmented with v1, size-capped to CA's A         calibrated vs hand-set

Resumable: a finished training (best_model/) or evaluation (test predictions)
is never redone, so the script can simply be restarted after an interruption.
Augmented data is produced by a separate job; a run that needs it waits for
the marker file ``<aug dir>/.complete``.

Outputs
  results/finetune/<iso>/<run>/best_model/            model (not committed)
  results/finetune/<iso>/<run>/test_eval/asr_*.csv    per-utterance test scores
  results/finetune/summary.md                         (finetune_stats.py)
  results/logs/finetune_<iso>_<run>.log               console logs

Usage
─────
  python scripts/run_finetune_matrix.py                       # everything
  python scripts/run_finetune_matrix.py --languages en --runs E0 R
  python scripts/run_finetune_matrix.py --dry-run             # print the plan
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from alt.text import ISO_TO_LANGUAGE  # noqa: E402

# Smallest languages first after English, so complete results arrive early.
LANG_ORDER = ["en", "ru", "it", "ja", "ko", "fr", "es", "de", "zh"]
RUN_ORDER = ["E0", "R", "RC", "RCA", "RCA_v1", "R20C", "R20CA", "C", "CA", "CA_v1"]
# Common training settings (fit a 16 GB GPU with whisper-large-v3).
TRAIN_FLAGS = ["--optim", "adamw_8bit", "--gradient-checkpointing",
               "--epochs", "3", "--batch-size", "8", "--seed", "42"]


def aug_dirs(lang_dir: str) -> dict[str, Path]:
    """Augmented-data folders of one language (v2 = engine auto, v1 = legacy)."""
    return {"v2": REPO / "data" / "GTSinger_aug_auto" / lang_dir,
            "v1": REPO / "data" / "GTSinger_Augmented" / lang_dir}


def recipe_flags(run: str, aug: dict[str, Path]) -> tuple[list[str], str | None]:
    """finetune_whisper.py flags of one run, and the augmented set it needs."""
    if run == "R":
        return [], None
    if run == "RC":
        return ["--with-control"], None
    if run == "RCA":
        return ["--with-control", "--with-aug", "--aug-src", str(aug["v2"])], "v2"
    if run == "RCA_v1":
        return ["--with-control", "--with-aug", "--aug-src", str(aug["v1"])], "v1"
    if run == "R20C":
        return ["--real-frac", "0.2", "--with-control"], None
    if run == "R20CA":
        return ["--real-frac", "0.2", "--with-control", "--with-aug",
                "--aug-src", str(aug["v2"])], "v2"
    # Zero-resource runs: no real technique take at all (--real-frac 0).
    if run == "C":
        return ["--real-frac", "0", "--with-control"], None
    if run == "CA":
        return ["--real-frac", "0", "--with-control", "--with-aug",
                "--aug-src", str(aug["v2"])], "v2"
    if run == "CA_v1":
        return ["--real-frac", "0", "--with-control", "--with-aug",
                "--aug-src", str(aug["v1"])], "v1"
    raise ValueError(run)


def n_aug_for_training(aug_root: Path, iso: str) -> int:
    """Augmented items a recipe run would use (test and validation songs removed)."""
    from finetune_whisper import collect_augmented_items
    from alt.augment.pairing import song_key

    test = set(json.load(open(REPO / f"results/splits/test_manifest_{iso}.songs.json",
                              encoding="utf-8"))["test_songs"])
    val_path = REPO / f"results/splits/val_songs_{iso}.json"
    val = set(json.load(open(val_path, encoding="utf-8"))) if val_path.exists() else set()
    items = collect_augmented_items(aug_root, ISO_TO_LANGUAGE[iso])
    return sum(song_key(Path(p)) not in test | val for p, _ in items)


def wait_for(marker: Path, log) -> None:
    """Block until an augmentation job has written its completion marker."""
    while not marker.exists():
        log(f"waiting for {marker} …")
        time.sleep(300)


def run_cmd(cmd: list[str], log_path: Path) -> int:
    """Run a command, streaming its output to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.call(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)


def evaluate(iso: str, model_id: str, run_dir: Path, log_path: Path) -> int:
    """Score a model on the fixed test manifest with the evaluation pipeline.

    Uses the same ``WhisperHF`` wrapper, text normalisation and metrics as the
    baselines (scripts/run_pipeline.py, ASR stage).
    """
    language = ISO_TO_LANGUAGE[iso]
    cfg = {
        "experiment_name": "test_eval",
        "paths": {"dataset_root": f"data/GTSinger/{language.capitalize()}",
                  "results_dir": str(run_dir.relative_to(REPO)),
                  "work_dir": "data/_work"},
        "data": {"name": "gtsinger", "language": language, "limit": None,
                 "manifest": f"results/splits/test_manifest_{iso}.json"},
        "asr": {"enabled": True, "model_names": ["whisper_finetuned"], "device": "cuda",
                "batch_size": 8, "language": iso, "extra": {"model_id": model_id}},
        "alignment": {"enabled": False},
        "pitch": {"enabled": False},
        "evaluation": {"stratify_by": ["technique", "singer_id"],
                       "hallucination_threshold": 1.0},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    return run_cmd([PY, "scripts/run_pipeline.py", "--config", str(cfg_path),
                    "--stage", "asr"], log_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", default=LANG_ORDER, choices=LANG_ORDER)
    ap.add_argument("--runs", nargs="+", default=RUN_ORDER, choices=RUN_ORDER)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan only")
    args = ap.parse_args()

    status_path = REPO / "results" / "finetune" / "status.log"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        if not args.dry_run:
            with open(status_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    for iso in args.languages:
        lang_dir = ISO_TO_LANGUAGE[iso].capitalize()
        aug = aug_dirs(lang_dir)
        for run in [r for r in RUN_ORDER if r in args.runs]:
            run_dir = REPO / "results" / "finetune" / iso / run
            preds = run_dir / "test_eval" / "asr_whisper_finetuned.csv"
            log_path = REPO / "results" / "logs" / f"finetune_{iso}_{run}.log"
            if preds.exists():
                log(f"[{iso}/{run}] done — skip")
                continue
            if run == "E0":
                log(f"[{iso}/{run}] evaluate whisper-large-v3 (no fine-tuning)")
                if not args.dry_run:
                    rc = evaluate(iso, "openai/whisper-large-v3", run_dir, log_path)
                    log(f"[{iso}/{run}] evaluation exit {rc}")
                continue

            flags, needs = recipe_flags(run, aug)
            best = run_dir / "best_model"
            if not (best / "config.json").exists():
                if needs and not args.dry_run:
                    wait_for(aug[needs] / ".complete", log)
                if run in ("RCA_v1", "CA_v1") and not args.dry_run:
                    # Same amount of augmented data as RCA: only quality differs.
                    n_v2 = n_aug_for_training(aug["v2"], iso)
                    flags += ["--aug-max", str(n_v2)]
                cmd = [PY, "scripts/finetune_whisper.py", "--mode", "recipe",
                       "--language", iso, "--run-name", run, *flags, *TRAIN_FLAGS]
                log(f"[{iso}/{run}] train: {' '.join(cmd[2:])}")
                if args.dry_run:
                    continue
                t0 = time.time()
                rc = run_cmd(cmd, log_path)
                log(f"[{iso}/{run}] training exit {rc} after {(time.time() - t0) / 3600:.2f} h")
                if rc != 0 or not (best / "config.json").exists():
                    log(f"[{iso}/{run}] FAILED — see {log_path}; continuing")
                    continue
            if args.dry_run:
                continue
            rc = evaluate(iso, str(best), run_dir, log_path)
            log(f"[{iso}/{run}] evaluation exit {rc}")
        if not args.dry_run:
            run_cmd([PY, "scripts/finetune_stats.py"],
                    REPO / "results" / "logs" / "finetune_stats.log")
            log(f"[{iso}] statistics updated → results/finetune/summary.md")
    log("matrix finished")


if __name__ == "__main__":
    main()
