#!/usr/bin/env python3
"""
measure_technique_features.py
─────────────────────────────
Measure what each singing technique actually changes in the real GTSinger
data, per language, using the paired design: every technique take has a
control take of the same phrase by the same singer.

For every pair (train songs only, unless --include-test) and every phone span
flagged with the pair's technique, the same span is measured in:

  source = "real"     the real technique take
  source = "control"  the matching span of the control take
  source = "legacy_vK" (optional, --legacy-sample) the v1 augmentation of the
                      control take with preset K, measured on the control span

Measurements (``alt.augment.features``): vibrato rate/extent/regularity/onset,
glide size/duration/slope, band aperiodicity, spectral tilt, H1–H2, level,
median F0 and the LTAS vector.

Output: results/calibration/<language>/features.parquet (long format, one row
per span × source).

Usage
─────
  python scripts/measure_technique_features.py --language english
  python scripts/measure_technique_features.py --language all --workers 24 \\
         --legacy-sample 25
  python scripts/measure_technique_features.py --language russian --max-pairs 20   # quick test
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alt.augment.engines import ENGINES  # noqa: E402
from alt.augment.features import (  # noqa: E402
    analyse, match_control_span, span_features, technique_spans,
)
from alt.augment.io import VOWELS_BY_LANGUAGE, load_audio, load_meta  # noqa: E402
from alt.augment.pairing import (  # noqa: E402
    LANGUAGE_ISO, ClipPair, iter_pairs, load_test_songs, voice_type,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def measure_pair(pair: ClipPair, legacy_variants: bool) -> list[dict]:
    """Measure one (technique, control) pair; runs inside a worker process.

    Args:
        pair:            The paired takes.
        legacy_variants: Also measure the v1 augmentation of the control take.

    Returns:
        Feature rows (one per flagged span × source).
    """
    voice = voice_type(pair.singer_id)
    vowels = VOWELS_BY_LANGUAGE[pair.language]
    tech_meta = load_meta(pair.tech_wav.with_suffix(".json"))
    ctrl_meta = load_meta(pair.control_wav.with_suffix(".json"))
    spans = technique_spans(tech_meta, pair.flag, vowels)
    if not spans:
        return []
    # Keep only spans that exist (same word + phone) in the control take.
    matched = [(s, c) for s in spans
               if (c := match_control_span(s, ctrl_meta)) is not None]
    if not matched:
        return []

    base = {"language": pair.language, "singer": pair.singer_id, "voice": voice,
            "group": pair.group, "flag": pair.flag, "song_key": pair.song_key,
            "stem": pair.stem}
    wav_t, sr = load_audio(pair.tech_wav)
    wav_c, _ = load_audio(pair.control_wav)
    a_t, a_c = analyse(wav_t, sr, voice), analyse(wav_c, sr, voice)

    rows = []
    for s, c in matched:
        key = {**base, "word_idx": s["word_idx"], "ph_idx": s["ph_idx"],
               "ph": s["ph"]}
        rows.append({**key, "source": "real", "flag_in_control": c[pair.flag],
                     **span_features(a_t, s["start"], s["end"])})
        rows.append({**key, "source": "control", "flag_in_control": c[pair.flag],
                     **span_features(a_c, c["start"], c["end"])})

    if legacy_variants and pair.group in ENGINES["legacy"].groups:
        eng = ENGINES["legacy"]
        for k, params in enumerate(eng.presets(pair.group, pair.language)):
            res = eng.run(pair.control_wav, pair.control_wav.with_suffix(".json"),
                          pair.group, params, pair.language)
            a_l = analyse(res.wav, res.sr, voice)
            for s, c in matched:
                rows.append({**base, "word_idx": s["word_idx"], "ph_idx": s["ph_idx"],
                             "ph": s["ph"], "source": f"legacy_v{k}",
                             "flag_in_control": c[pair.flag],
                             **span_features(a_l, c["start"], c["end"])})
    return rows


def _safe_measure(pair: ClipPair, legacy_variants: bool) -> tuple[list[dict], str | None]:
    """Worker wrapper: never let one bad file kill the pool."""
    try:
        return measure_pair(pair, legacy_variants), None
    except Exception as e:  # report and continue
        return [], f"{pair.tech_wav}: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="all",
                    help="Language name or 'all' (default: all)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Cap pairs per (language, group) — random, seeded (quick runs)")
    ap.add_argument("--legacy-sample", type=int, default=0,
                    help="Pairs per (language, group) on which the v1 presets are "
                         "also applied and measured (default: 0 = off)")
    ap.add_argument("--include-test", action="store_true",
                    help="Also measure test songs (default: train songs only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "results" / "calibration"))
    args = ap.parse_args()

    languages = list(LANGUAGE_ISO) if args.language == "all" else [args.language.lower()]
    rng = random.Random(args.seed)
    for lang in languages:
        root = REPO / "data" / "GTSinger" / lang.capitalize()
        exclude = set() if args.include_test else load_test_songs(lang, REPO)
        by_group: dict[str, list[ClipPair]] = defaultdict(list)
        for p in iter_pairs(root, lang, exclude_songs=exclude):
            by_group[p.group].append(p)

        # Seeded sampling per group; the first --legacy-sample pairs of each
        # (shuffled) group also get the v1 variants measured.
        jobs: list[tuple[ClipPair, bool]] = []
        for group in sorted(by_group):
            pairs = by_group[group][:]
            rng.shuffle(pairs)
            if args.max_pairs:
                pairs = pairs[:args.max_pairs]
            jobs += [(p, i < args.legacy_sample) for i, p in enumerate(pairs)]
        print(f"[{lang}] {len(jobs)} pairs "
              f"({', '.join(f'{g}={len(v)}' for g, v in sorted(by_group.items()))})",
              flush=True)

        rows, errors, t0 = [], [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_safe_measure, p, leg) for p, leg in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r, err = fut.result()
                rows += r
                if err:
                    errors.append(err)
                if i % 200 == 0:
                    print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

        out_dir = Path(args.out) / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "features.parquet", index=False)
        print(f"[{lang}] {len(df)} rows, {len(errors)} errors, "
              f"{time.time() - t0:.0f}s → {out_dir / 'features.parquet'}", flush=True)
        for e in errors[:5]:
            print("   ERR", e)


if __name__ == "__main__":
    main()
