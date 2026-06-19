#!/usr/bin/env python3
"""
run_pipeline.py — command-line entry point for the ALT baseline pipeline.

Usage
-----
Run the full baseline from a YAML config::

    python scripts/run_pipeline.py --config configs/baseline.yaml

Override any config field without editing the YAML (dotted keys)::

    python scripts/run_pipeline.py --config configs/baseline.yaml \
        --set asr.model_name=whisper_small asr.device=cpu data.limit=20

Run only one stage (handy while iterating)::

    python scripts/run_pipeline.py --config configs/baseline.yaml --stage asr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the local ``src/`` package importable without installing the project.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from alt.config import apply_overrides, load_config   # noqa: E402
from alt.pipeline import Pipeline                     # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed ``argparse.Namespace`` with ``config``, ``set`` and
        ``stage`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Run the ALT baseline pipeline for expressive singing.")
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML experiment config (see configs/baseline.yaml).")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="key=value",
        help="Override config fields, e.g. --set asr.device=cpu data.limit=10")
    parser.add_argument(
        "--stage", choices=["all", "dataset", "asr", "alignment", "pitch", "report"],
        default="all", help="Run only one stage instead of the full pipeline.")
    return parser.parse_args()


def main() -> None:
    """Load the config, apply overrides, and run the requested stage(s)."""
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)

    pipeline = Pipeline(cfg)
    if args.stage == "all":
        pipeline.run()
        return

    from alt.report import make_report

    if args.stage == "report":
        make_report(pipeline.out_dir)
        print(f"\n=== Report written to: {pipeline.out_dir} ===")
        return

    # Single-stage runs: the dataset must always be loaded first.
    pipeline.load_dataset()
    if args.stage == "asr":
        pipeline.run_asr()
    elif args.stage == "alignment":
        pipeline.run_alignment()
    elif args.stage == "pitch":
        pipeline.run_pitch()
        make_report(pipeline.out_dir)
    print(f"\n=== Stage '{args.stage}' done. Results in: {pipeline.out_dir} ===")


if __name__ == "__main__":
    main()
