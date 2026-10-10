#!/usr/bin/env python3
"""
evaluate_pilot.py
─────────────────
Compare the augmentation engines on the pilot (results/pilot/manifest.csv)
along three independent axes, per language × technique group × engine ×
variant:

1. objective — re-measure the generated clips with alt.augment.features on the
   modified spans; the effect (output − control) is compared with the REAL
   effect distribution of the same language (calibration data): median,
   whether it falls inside the real IQR, and distance in IQR units.
2. clap — CLAP (laion/larger_clap_music_and_speech) displacement of the
   augmented clip from its control, *after removing the engine's passthrough
   displacement* (pure resynthesis artefact), compared with the real
   technique-minus-control displacement: direction_score (cosine, > 0 =
   right direction) and magnitude_ratio (≈ 1 ideal). artefact_ratio =
   |passthrough displacement| / |real displacement| measures how "processed"
   the engine sounds relative to the size of the technique itself.
3. asr — Whisper-large-v3 error (CER for zh/ja/ko, else WER) on control, real
   and augmented takes: does augmentation change recognition like the real
   technique does (Δaug vs Δreal)?

Outputs in results/pilot/: objective.csv, clap.csv, asr.csv, summary.csv,
report.md.

Usage
─────
  python scripts/evaluate_pilot.py                       # all stages
  python scripts/evaluate_pilot.py --stages objective    # CPU only
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from alt.augment.features import analyse, span_features  # noqa: E402
from alt.augment.io import load_audio, load_meta  # noqa: E402
from alt.augment.pairing import LANGUAGE_ISO  # noqa: E402

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

PILOT = REPO / "results" / "pilot"
CALIB = REPO / "results" / "calibration"
CELL = ["language", "group", "engine", "variant"]


# ── 1. Objective re-measurement ──────────────────────────────────────────────

def _measure_row(row: dict) -> list[dict]:
    """Features of one generated clip and of its control on the target spans.

    Runs in a worker process.

    Args:
        row: Manifest row.

    Returns:
        One dict per span with ``out_<feat>`` and ``ctrl_<feat>`` values.
    """
    from alt.augment import world_v2
    from alt.augment.pairing import voice_type

    ctrl = REPO / row["control_wav"]
    meta = load_meta(ctrl.with_suffix(".json"))
    spans, _ = world_v2.target_spans(ctrl, row["group"], row["language"], meta)
    if not spans:
        return []
    voice = voice_type(row["singer"])
    w_out, sr_out = load_audio(REPO / row["out_wav"])
    # Compare at a common rate: legacy output is 16 kHz, so its control is
    # analysed at 16 kHz too (bands above 8 kHz are then missing for both).
    w_ctrl, sr_ctrl = load_audio(ctrl, sr=sr_out)
    a_out, a_ctrl = analyse(w_out, sr_out, voice), analyse(w_ctrl, sr_ctrl, voice)
    rows = []
    for i, (s, e) in enumerate(spans):
        fo, fc = span_features(a_out, s, e), span_features(a_ctrl, s, e)
        r = {k: row[k] for k in CELL + ["singer", "stem"]}
        r["span"] = i
        for k, v in fo.items():
            if k != "ltas":
                r[f"out_{k}"] = v
        for k, v in fc.items():
            if k != "ltas":
                r[f"ctrl_{k}"] = v
        rows.append(r)
    return rows


def objective(man: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Re-measure every pilot clip and score it against the real effects.

    Args:
        man:     Pilot manifest.
        workers: Parallel processes.

    Returns:
        One row per cell × key feature with the augmented and real medians,
        ``in_iqr`` and ``iqr_distance``.
    """
    from calibration_report import KEY_FEATURES, effects, load_all

    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_measure_row, man.to_dict("records"), chunksize=4))
    spans = pd.DataFrame([r for p in parts for r in p])
    spans.to_parquet(PILOT / "objective_spans.parquet", index=False)

    real_all = load_all(CALIB)
    real_all["glide_abs_cents"] = real_all["glide_cents"].abs()
    out = []
    for (lang, group), cell in spans.groupby(["language", "group"]):
        for feat, mode, label in KEY_FEATURES.get(group, []):
            eff = effects(real_all[real_all.language == lang], group, feat, mode)
            real = eff[eff.source == "real"]["value"] if not eff.empty else pd.Series(dtype=float)
            if real.empty:
                continue
            q25, q50, q75 = real.quantile([0.25, 0.5, 0.75])
            iqr = max(q75 - q25, 1e-9)
            for (eng, var), c in cell.groupby(["engine", "variant"]):
                aug = c[f"out_{feat}"] - (c[f"ctrl_{feat}"] if mode == "delta" else 0)
                if feat == "glide_cents":   # direction is random per span: compare size
                    aug, real_cmp = aug.abs(), real.abs()
                    q25, q50, q75 = real_cmp.quantile([0.25, 0.5, 0.75])
                    iqr = max(q75 - q25, 1e-9)
                m = aug.median()
                out.append({"language": lang, "group": group, "engine": eng,
                            "variant": var, "feature": label, "n_spans": int(aug.notna().sum()),
                            "aug_median": m, "real_q25": q25, "real_median": q50,
                            "real_q75": q75, "in_iqr": bool(q25 <= m <= q75),
                            "iqr_distance": abs(m - q50) / iqr})
    return pd.DataFrame(out)


# ── 2. CLAP ──────────────────────────────────────────────────────────────────

def clap(man: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """CLAP displacement metrics per cell (artefact-corrected).

    Args:
        man:        Pilot manifest.
        batch_size: CLAP windows per forward pass.

    Returns:
        One row per cell: direction_score, magnitude_ratio, artefact_ratio.
    """
    from clap_embed import ClapEmbedder, embed_dataframe

    paths = sorted(set(man["control_wav"]) | set(man["real_wav"]) | set(man["out_wav"]))
    df = pd.DataFrame({"audio_path": [str(REPO / p) for p in paths]})
    emb = embed_dataframe(df, ClapEmbedder(), batch_size=batch_size)
    E = dict(zip(paths, emb))
    np.savez_compressed(PILOT / "clap_embeddings.npz", paths=np.array(paths), emb=emb)

    rows = []
    for (lang, group, eng), cell in man.groupby(["language", "group", "engine"]):
        pairs = cell.drop_duplicates("control_wav")
        d_real = np.mean([E[r] - E[c] for r, c in zip(pairs.real_wav, pairs.control_wav)], axis=0)
        pt = cell[cell.variant == "pt"]
        d_pt = np.mean([E[o] - E[c] for o, c in zip(pt.out_wav, pt.control_wav)], axis=0) \
            if len(pt) else np.zeros_like(d_real)
        for var, c in cell.groupby("variant"):
            d_aug = np.mean([E[o] - E[k] for o, k in zip(c.out_wav, c.control_wav)], axis=0)
            d_tech = d_aug - d_pt if var != "pt" else d_aug
            nr = np.linalg.norm(d_real) + 1e-9
            rows.append({"language": lang, "group": group, "engine": eng, "variant": var,
                         "direction_score": float(d_tech @ d_real / (np.linalg.norm(d_tech) * nr + 1e-9)),
                         "magnitude_ratio": float(np.linalg.norm(d_tech) / nr),
                         "artefact_ratio": float(np.linalg.norm(d_pt) / nr),
                         "cos_to_real": float(np.mean([E[o] @ E[r] for o, r in zip(c.out_wav, c.real_wav)]))})
    return pd.DataFrame(rows)


# ── 3. ASR ───────────────────────────────────────────────────────────────────

def asr(man: pd.DataFrame, model: str) -> pd.DataFrame:
    """Whisper error on control / real / augmented takes, per cell.

    Args:
        man:   Pilot manifest.
        model: ASR registry name (default whisper_largev3).

    Returns:
        One row per cell with median error of control, real, augmented and
        the two deltas.
    """
    from alt.asr import get_asr_model
    from alt.metrics import score_pair
    from alt.text import join_words, primary_metric

    per_clip = []
    for lang, cell in man.groupby("language"):
        metric = primary_metric(lang)
        paths = sorted(set(cell.control_wav) | set(cell.real_wav) | set(cell.out_wav))
        m = get_asr_model(model, device="cuda", batch_size=8, language=LANGUAGE_ISO[lang])
        hyps = dict(zip(paths, m.transcribe([str(REPO / p) for p in paths])))
        m.unload()
        # Reference = lyrics of the control take (all takes sing the same text).
        refs = {}
        for c in set(cell.control_wav):
            words = [w["word"] for w in load_meta((REPO / c).with_suffix(".json"))
                     if not str(w["word"]).startswith("<")]
            refs[c] = join_words(words, lang)
        for p in paths:
            ctrl = p if p in refs else None
            if ctrl is None:
                hit = cell[(cell.real_wav == p) | (cell.out_wav == p)].iloc[0]
                ctrl = hit.control_wav
            per_clip.append({"language": lang, "path": p, "metric": metric,
                             "error": score_pair(refs[ctrl], hyps[p], lang)[metric],
                             "hypothesis": hyps[p]})
    err = pd.DataFrame(per_clip).set_index("path")["error"].to_dict()
    pd.DataFrame(per_clip).to_csv(PILOT / "asr_clips.csv", index=False)

    rows = []
    for (lang, group, eng, var), c in man.groupby(CELL):
        e_c = c.control_wav.map(err)
        e_r = c.real_wav.map(err)
        e_a = c.out_wav.map(err)
        rows.append({"language": lang, "group": group, "engine": eng, "variant": var,
                     "err_control": e_c.median(), "err_real": e_r.median(),
                     "err_aug": e_a.median(),
                     "delta_real": (e_r - e_c).mean(), "delta_aug": (e_a - e_c).mean()})
    return pd.DataFrame(rows)


# ── Summary ──────────────────────────────────────────────────────────────────

def summarise(obj: pd.DataFrame | None, clp: pd.DataFrame | None,
              asr_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join the three axes per cell and apply the automatic gate criteria.

    Gate (per language × group × engine × variant, passthrough excluded):
      objective  ≥ 50 % of key features within 1 real IQR of the real median
      clap       direction_score > 0 and 0.5 ≤ magnitude_ratio ≤ 2
      asr        Δaug has the sign of Δreal or |Δaug − Δreal| ≤ 0.05
    """
    parts = []
    if obj is not None and not obj.empty:
        o = (obj.assign(close=obj.iqr_distance <= 1.0)
             .groupby(CELL).agg(obj_close_share=("close", "mean"),
                                obj_mean_iqr_dist=("iqr_distance", "mean")))
        parts.append(o)
    if clp is not None and not clp.empty:
        parts.append(clp.set_index(CELL))
    if asr_df is not None and not asr_df.empty:
        parts.append(asr_df.set_index(CELL))
    if not parts:
        return pd.DataFrame()
    s = pd.concat(parts, axis=1).reset_index()
    s = s[s.variant != "pt"]
    if "obj_close_share" in s:
        s["pass_objective"] = s.obj_close_share >= 0.5
    if "direction_score" in s:
        s["pass_clap"] = (s.direction_score > 0) & s.magnitude_ratio.between(0.5, 2.0)
    if "delta_aug" in s:
        s["pass_asr"] = (np.sign(s.delta_aug) == np.sign(s.delta_real)) | \
            ((s.delta_aug - s.delta_real).abs() <= 0.05)
    return s


def write_report(s: pd.DataFrame, obj, clp, asr_df) -> None:
    """Markdown report with engine-level and per-cell tables."""
    md = ["# Augmentation pilot — engine comparison", "",
          "Per language × technique group, N control takes (train songs) were "
          "augmented by each engine (3 variants weak→strong + a passthrough). "
          "See scripts/evaluate_pilot.py for the metric definitions.", ""]
    passes = [c for c in s.columns if c.startswith("pass_")]
    if passes:
        eng = s.groupby("engine")[passes + [c for c in ("obj_mean_iqr_dist",
                                                       "direction_score", "magnitude_ratio",
                                                       "artefact_ratio") if c in s]].mean()
        md += ["## Engines overall (share of cells passing; mean metrics)", "",
               eng.round(3).to_markdown(), ""]
        by_group = s.groupby(["group", "engine"])[passes].mean()
        md += ["## By technique", "", by_group.round(2).to_markdown(), ""]
        by_lang = s.groupby(["language", "engine"])[passes].mean()
        md += ["## By language", "", by_lang.round(2).to_markdown(), ""]
    if obj is not None and not obj.empty:
        piv = obj[obj.variant == "v1"].pivot_table(
            index=["group", "feature"], columns="engine", values="iqr_distance",
            aggfunc="median")
        md += ["## Objective: distance from the real median (IQR units, typical "
               "variant v1, median over languages; ≤ 1 = realistic)", "",
               piv.round(2).to_markdown(), ""]
    (PILOT / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["objective", "clap", "asr"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--asr-model", default="whisper_largev3")
    ap.add_argument("--clap-batch", type=int, default=16)
    args = ap.parse_args()

    man = pd.read_csv(PILOT / "manifest.csv")
    obj = clp = asr_df = None
    if "objective" in args.stages:
        obj = objective(man, args.workers)
        obj.to_csv(PILOT / "objective.csv", index=False)
        print(f"objective: {len(obj)} rows")
    elif (PILOT / "objective.csv").exists():
        obj = pd.read_csv(PILOT / "objective.csv")
    if "clap" in args.stages:
        clp = clap(man, args.clap_batch)
        clp.to_csv(PILOT / "clap.csv", index=False)
        print(f"clap: {len(clp)} rows")
    elif (PILOT / "clap.csv").exists():
        clp = pd.read_csv(PILOT / "clap.csv")
    if "asr" in args.stages:
        asr_df = asr(man, args.asr_model)
        asr_df.to_csv(PILOT / "asr.csv", index=False)
        print(f"asr: {len(asr_df)} rows")
    elif (PILOT / "asr.csv").exists():
        asr_df = pd.read_csv(PILOT / "asr.csv")

    s = summarise(obj, clp, asr_df)
    s.to_csv(PILOT / "summary.csv", index=False)
    write_report(s, obj, clp, asr_df)
    print(f"report → {PILOT / 'report.md'}")


if __name__ == "__main__":
    main()
