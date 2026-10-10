#!/usr/bin/env python3
"""
build_all_augmentations.py
──────────────────────────
Generate the augmented training data of every language for the fine-tuning
matrix, in the order run_finetune_matrix.py needs them:

  v2  data/GTSinger_aug_auto/<Lang>       --engine auto --param-mode sampled
  v1  data/GTSinger_Augmented/<Lang>      --engine legacy (original presets)

Test songs are excluded (validation songs are filtered at training time).
After each successful build a marker ``<dir>/.complete`` is written; the
fine-tuning runner waits for it. Already-complete folders are skipped.

Usage
─────
  python scripts/build_all_augmentations.py --workers 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANGS = ["english", "russian", "italian", "japanese", "korean", "french",
         "spanish", "german", "chinese"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", default=LANGS)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    for lang in args.languages:
        lang_dir = lang.capitalize()
        jobs = [("v2", REPO / "data" / "GTSinger_aug_auto" / lang_dir,
                 ["--engine", "auto", "--param-mode", "sampled", "--targets", "language"]),
                ("v1", REPO / "data" / "GTSinger_Augmented" / lang_dir,
                 ["--engine", "legacy"])]
        for tag, dst, flags in jobs:
            marker = dst / ".complete"
            if marker.exists():
                print(f"[{lang}/{tag}] complete — skip", flush=True)
                continue
            log = REPO / "results" / "logs" / f"augment_{tag}_{lang}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "scripts/build_augmented_dataset.py", "--language", lang,
                   "--dst", str(dst), "--exclude-test-songs", "--workers",
                   str(args.workers), *flags]
            t0 = time.time()
            print(f"[{lang}/{tag}] building → {dst}", flush=True)
            with open(log, "a", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
            if rc == 0:
                marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            print(f"[{lang}/{tag}] exit {rc} after {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
