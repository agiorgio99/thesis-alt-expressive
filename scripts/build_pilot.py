#!/usr/bin/env python3
"""
build_pilot.py
──────────────
Generate the augmentation pilot: a small, stratified set of clips on which the
three engines are compared before any full-scale generation.

For every language × technique group, N control takes from TRAIN songs (that
have a paired real technique take) are augmented by every engine that
supports the group, with its three variants plus one **passthrough** (the
engine's resynthesis with no modification — the artefact reference that
isolates "sounds processed" from "sounds like the technique"):

  legacy    v1, 16 kHz, whole-clip WORLD (variants v0–v2; passthrough = plain
            WORLD analysis/resynthesis)
  world_v2  data-driven, 48 kHz, local resynthesis (variants = q25/q50/q75)
  psola     Praat overlap-add, pitch techniques only (same variants as v2)

Output: data/pilot/<lang>/<group>/<engine>/<singer>__<song>__<stem>__<variant>.wav
(float WAV, no clipping) + results/pilot/manifest.csv.

Usage
─────
  python scripts/build_pilot.py --n-clips 8 --workers 20
  python scripts/build_pilot.py --languages russian --n-clips 2   # quick test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alt.augment.engines import ENGINES, AugmentResult, augment  # noqa: E402
from alt.augment.pairing import (  # noqa: E402
    LANGUAGE_ISO, ClipPair, iter_pairs, load_test_songs,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

PILOT_AUDIO = REPO / "data" / "pilot"
PILOT_RESULTS = REPO / "results" / "pilot"
N_VARIANTS = 3   # same number of variants for every engine/technique


def passthrough(engine: str, pair: ClipPair) -> AugmentResult:
    """Engine resynthesis with no modification (the artefact reference).

    Args:
        engine: Engine name.
        pair:   The pair whose control take is resynthesised.

    Returns:
        The unmodified resynthesis.
    """
    if engine == "legacy":
        from alt.augment import legacy
        from alt.augment.io import load_utterance
        wav, meta = load_utterance(pair.control_wav, pair.control_wav.with_suffix(".json"),
                                   sr=legacy.SR)
        f0, sp, ap = legacy.world_decompose(wav)
        return AugmentResult(legacy.world_synthesize(f0, sp, ap), legacy.SR, meta,
                             "legacy", {})
    # v2 / psola: the same code path with all effects set to zero, so the
    # passthrough carries exactly the engine's own analysis/splice artefacts.
    zero = {"Vibrato_Group": {"extent_cents": 0.0, "rate_hz": 5.5, "onset_s": 0.0,
                              "am_db": 0.0, "jitter": 0.0},
            "Glissando_Group": {"glide_cents": 0.0, "share_up": 0.5, "steepness": 8.0},
            }.get(pair.group, {"f0_cents": 0.0, "h1h2_db": 0.0, "level_db": 0.0,
                               "ap_delta": [0.0] * 5, "ltas_scale": 0.0})
    if engine == "psola" and pair.group == "Falsetto_Group":
        zero = {"f0_cents": 0.0}
    res = augment(engine, pair.control_wav, pair.group, {**zero, "seed": 0}, pair.language)
    return res


def generate(pair: ClipPair, engine: str) -> list[dict]:
    """All variants + passthrough of one engine for one pair (worker process).

    Args:
        pair:   The (technique, control) pair.
        engine: Engine name.

    Returns:
        Manifest rows (one per written clip).
    """
    out_dir = PILOT_AUDIO / pair.language / pair.group / engine
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{pair.singer_id}__{pair.song.replace(' ', '_')}__{pair.stem}"
    presets = ENGINES[engine].presets(pair.group, pair.language)[:N_VARIANTS]
    jobs = [(f"v{k}", p) for k, p in enumerate(presets)] + [("pt", None)]
    rows = []
    for variant, params in jobs:
        res = passthrough(engine, pair) if params is None else \
            augment(engine, pair.control_wav, pair.group, params, pair.language)
        wav_out = out_dir / f"{base}__{variant}.wav"
        sf.write(wav_out, res.wav, res.sr, subtype="FLOAT")
        with open(wav_out.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(res.meta, fh, ensure_ascii=False)
        rows.append({
            "language": pair.language, "group": pair.group, "engine": engine,
            "variant": variant, "singer": pair.singer_id, "song": pair.song,
            "song_key": pair.song_key, "stem": pair.stem,
            "control_wav": str(pair.control_wav.relative_to(REPO)),
            "real_wav": str(pair.tech_wav.relative_to(REPO)),
            "out_wav": str(wav_out.relative_to(REPO)),
            "sr": res.sr, "params": json.dumps(res.params),
        })
    return rows


def _safe(pair: ClipPair, engine: str) -> tuple[list[dict], str | None]:
    """Worker wrapper: report errors instead of crashing the pool."""
    try:
        return generate(pair, engine), None
    except Exception as e:
        return [], f"{engine} {pair.tech_wav}: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", default=list(LANGUAGE_ISO))
    ap.add_argument("--n-clips", type=int, default=8,
                    help="Control takes per language × group (default: 8)")
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    jobs: list[tuple[ClipPair, str]] = []
    for lang in args.languages:
        root = REPO / "data" / "GTSinger" / lang.capitalize()
        test_songs = load_test_songs(lang, REPO)
        by_group: dict[str, list[ClipPair]] = defaultdict(list)
        for p in iter_pairs(root, lang, exclude_songs=test_songs):
            by_group[p.group].append(p)
        for group in sorted(by_group):
            pairs = by_group[group][:]
            rng.shuffle(pairs)
            # Prefer distinct songs so the pilot is not dominated by one song.
            chosen, seen = [], set()
            for p in pairs:
                if p.song_key not in seen:
                    chosen.append(p)
                    seen.add(p.song_key)
                if len(chosen) == args.n_clips:
                    break
            for p in pairs:
                if len(chosen) >= args.n_clips:
                    break
                if p not in chosen:
                    chosen.append(p)
            for p in chosen:
                assert p.song_key not in test_songs, "test song leaked into the pilot"
                for eng in args.engines:
                    if group in ENGINES[eng].groups:
                        jobs.append((p, eng))
    print(f"{len(jobs)} (clip, engine) jobs", flush=True)

    rows, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_safe, p, e) for p, e in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r, err = fut.result()
            rows += r
            if err:
                errors.append(err)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    PILOT_RESULTS.mkdir(parents=True, exist_ok=True)
    man = pd.DataFrame(rows).sort_values(["language", "group", "engine", "stem", "variant"])
    man.to_csv(PILOT_RESULTS / "manifest.csv", index=False)
    print(f"{len(man)} clips, {len(errors)} errors, {time.time() - t0:.0f}s "
          f"→ {PILOT_RESULTS / 'manifest.csv'}")
    for e in errors[:10]:
        print("  ERR", e)


if __name__ == "__main__":
    main()
