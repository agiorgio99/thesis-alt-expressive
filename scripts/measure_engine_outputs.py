#!/usr/bin/env python3
"""
measure_engine_outputs.py
─────────────────────────
Measure what every augmentation engine produces, with exactly the same span
definition as the calibration (``measure_technique_features.py``): the phone
spans flagged with the technique in the real take, matched to the control
take. The resulting rows can be plotted directly against the real
technique-vs-control distributions (``distribution_analysis.py``).

Pairs are the pilot ones (results/pilot/manifest.csv: train songs, stratified
per language × technique). Engine outputs are regenerated in memory
(deterministic seeds), so modes that were never written to disk can be
measured too:

  quartile  the q25 / q50 / q75 presets (= the pilot clips)
  sampled   per-clip targets drawn from the whole real distribution
            (``world_v2.sample_presets``, tail_gamma = 1)
  tails     same, with draws pushed toward the tails (tail_gamma = 0.6)

Rows: one per span × source, where source is ``real``, ``control`` or
``<engine>_<mode>_v<k>`` (legacy only has its fixed v1 presets → mode "preset").

Output: results/distributions/engine_features.parquet

Usage
─────
  python scripts/measure_engine_outputs.py --workers 24
  python scripts/measure_engine_outputs.py --languages russian --max-pairs 2   # quick test
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
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
from alt.augment.pairing import GROUP_FLAG, voice_type  # noqa: E402

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

OUT_DIR = REPO / "results" / "distributions"
MODES = {"quartile": None, "sampled": 1.0, "tails": 0.6}   # mode → tail_gamma


def variants(engine: str, mode: str, group: str, language: str, clip_key: str) -> list[dict]:
    """Parameter variants of one engine in one mode.

    Args:
        engine:   "legacy", "world_v2" or "psola".
        mode:     Key of ``MODES``.
        group:    Target technique group.
        language: Lowercase language name.
        clip_key: Control-take path (seeds the sampled modes).

    Returns:
        The parameter dicts (empty when the combination does not exist).
    """
    if engine == "legacy":
        return ENGINES["legacy"].presets(group, language) if mode == "quartile" else []
    if mode == "quartile":
        return ENGINES[engine].presets(group, language)
    from alt.augment import psola, world_v2
    mod = world_v2 if engine == "world_v2" else psola
    return mod.sample_presets(group, language, clip_key, 3, MODES[mode])


def measure_pair(row: dict, engines: list[str], modes: list[str]) -> list[dict]:
    """Measure real, control and every engine output of one pilot pair.

    Runs inside a worker process.

    Args:
        row:     One unique pair of the pilot manifest.
        engines: Engines to run.
        modes:   Parameter modes to run.

    Returns:
        Feature rows (one per span × source).
    """
    lang, group = row["language"], row["group"]
    flag = GROUP_FLAG[group]
    voice = voice_type(row["singer"])
    ctrl_wav, real_wav = REPO / row["control_wav"], REPO / row["real_wav"]
    ctrl_meta = load_meta(ctrl_wav.with_suffix(".json"))
    spans = technique_spans(load_meta(real_wav.with_suffix(".json")), flag,
                            VOWELS_BY_LANGUAGE[lang])
    matched = [(s, c) for s in spans if (c := match_control_span(s, ctrl_meta)) is not None]
    if not matched:
        return []
    base = {"language": lang, "singer": row["singer"], "voice": voice, "group": group,
            "song_key": row["song_key"], "stem": row["stem"]}

    def rows_for(source: str, a, use_real_timing: bool, extra: dict | None = None) -> list[dict]:
        out = []
        for s, c in matched:
            sp = s if use_real_timing else c
            out.append({**base, "word_idx": s["word_idx"], "ph_idx": s["ph_idx"],
                        "ph": s["ph"], "source": source, "flag_in_control": c[flag],
                        **(extra or {}), **span_features(a, sp["start"], sp["end"])})
        return out

    wav_r, sr = load_audio(real_wav)
    wav_c, _ = load_audio(ctrl_wav)
    rows = rows_for("real", analyse(wav_r, sr, voice), True)
    rows += rows_for("control", analyse(wav_c, sr, voice), False)

    json_path = ctrl_wav.with_suffix(".json")
    for eng in engines:
        if group not in ENGINES[eng].groups:
            continue
        for mode in modes:
            for k, params in enumerate(variants(eng, mode, group, lang, str(row["control_wav"]))):
                res = ENGINES[eng].run(ctrl_wav, json_path, group, params, lang)
                tag = "preset" if eng == "legacy" else mode
                rows += rows_for(f"{eng}_{tag}_v{k}", analyse(res.wav, res.sr, voice), False,
                                 {"engine": eng, "mode": tag, "variant": k,
                                  "quantile": params.get("quantile")})
    return rows


def _safe(row: dict, engines: list[str], modes: list[str]) -> tuple[list[dict], str | None]:
    """Worker wrapper: never let one bad file kill the pool."""
    try:
        return measure_pair(row, engines, modes), None
    except Exception as e:  # report and continue
        return [], f"{row['control_wav']}: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "results" / "pilot" / "manifest.csv"))
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--engines", nargs="+", default=["legacy", "world_v2", "psola"])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Cap pairs per (language, group) (quick runs)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT_DIR / "engine_features.parquet"))
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, dtype={"stem": str})
    pairs = (man.drop_duplicates(["language", "group", "control_wav"])
             [["language", "group", "singer", "song_key", "stem", "control_wav", "real_wav"]])
    if args.languages:
        pairs = pairs[pairs.language.isin([lg.lower() for lg in args.languages])]
    if args.max_pairs:
        pairs = pairs.groupby(["language", "group"]).head(args.max_pairs)
    jobs = pairs.to_dict("records")
    print(f"{len(jobs)} pairs · engines {args.engines} · modes {args.modes}", flush=True)

    rows, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_safe, r, args.engines, args.modes) for r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r, err = fut.result()
            rows += r
            if err:
                errors.append(err)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows, {len(errors)} errors, {time.time() - t0:.0f}s → {out}", flush=True)
    for e in errors[:5]:
        print("   ERR", e)


if __name__ == "__main__":
    main()
