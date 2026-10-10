#!/usr/bin/env python3
"""
finetune_stats.py
─────────────────
Test-set results of the fine-tuning ablation (run_finetune_matrix.py) with
uncertainty and paired significance tests.

Per language and run: mean and median error (CER for zh/ja/ko, else WER) on
the fixed song-disjoint test set, with a 95 % confidence interval from a
bootstrap over test SONGS (clips of one song are not independent).

Per language, the planned comparisons (same test clips, paired):

  R      vs E0      does fine-tuning on real singing help?
  RC     vs R       does adding real control takes help?
  RCA    vs RC      does augmented technique data add anything?  ← key question
  RCA    vs RCA_v1  calibrated (v2) vs hand-set (v1) augmentation
  R20CA  vs R20C    the same in the low-resource setting

reported as mean difference (song-bootstrap 95 % CI) and a Wilcoxon
signed-rank p-value on per-clip errors, Holm-corrected within each language.

Outputs: results/finetune/summary.md, summary.csv, comparisons.csv

Usage
─────
  python scripts/finetune_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alt.augment.pairing import song_key  # noqa: E402
from alt.text import ISO_TO_LANGUAGE, primary_metric  # noqa: E402

FT = REPO / "results" / "finetune"
RUNS = ["E0", "R", "RC", "RCA", "RCA_v1", "R20C", "R20CA"]
COMPARISONS = [("R", "E0"), ("RC", "R"), ("RCA", "RC"), ("RCA", "RCA_v1"),
               ("R20CA", "R20C")]
N_BOOT = 2000


def load_run(iso: str, run: str) -> pd.DataFrame | None:
    """Per-clip test scores of one run, with the clip's song."""
    d = FT / iso / run / "test_eval"
    preds, inv = d / "asr_whisper_finetuned.csv", d / "inventory.csv"
    if not preds.exists() or not inv.exists():
        return None
    df = pd.read_csv(preds)
    songs = pd.read_csv(inv)[["utt_id", "audio_path"]]
    songs["song"] = songs["audio_path"].map(lambda p: song_key(Path(p)))
    return df.merge(songs[["utt_id", "song"]], on="utt_id", how="left")


def song_bootstrap(values: pd.Series, songs: pd.Series, rng: np.random.Generator) -> tuple:
    """95 % CI of the mean, resampling whole songs with replacement."""
    by_song = [g.values for _, g in values.groupby(songs.values)]
    means = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(by_song), len(by_song))
        means.append(np.concatenate([by_song[i] for i in pick]).mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(0)
    rows, comps = [], []
    for iso in ISO_TO_LANGUAGE:
        metric = primary_metric(ISO_TO_LANGUAGE[iso])
        runs = {r: df for r in RUNS if (df := load_run(iso, r)) is not None}
        for run, df in runs.items():
            v = df[metric].astype(float)
            lo, hi = song_bootstrap(v, df["song"], rng)
            rows.append({"language": iso, "run": run, "metric": metric, "n": len(df),
                         "mean": v.mean(), "ci_low": lo, "ci_high": hi,
                         "median": v.median(), "halluc_rate": (v > 1.0).mean()})
        lang_comps = []
        for a, b in COMPARISONS:
            if a not in runs or b not in runs:
                continue
            m = runs[a][["utt_id", "song", metric]].merge(
                runs[b][["utt_id", metric]], on="utt_id", suffixes=("_a", "_b"))
            diff = m[f"{metric}_a"] - m[f"{metric}_b"]
            lo, hi = song_bootstrap(diff, m["song"], rng)
            p = wilcoxon(m[f"{metric}_a"], m[f"{metric}_b"], zero_method="zsplit").pvalue \
                if (diff != 0).any() else 1.0
            lang_comps.append({"language": iso, "comparison": f"{a} vs {b}",
                               "metric": metric, "n": len(m), "mean_diff": diff.mean(),
                               "ci_low": lo, "ci_high": hi,
                               "a_better_share": (diff < 0).mean(),
                               "b_better_share": (diff > 0).mean(), "p_wilcoxon": p})
        for c, p_adj in zip(lang_comps, holm([c["p_wilcoxon"] for c in lang_comps])):
            c["p_holm"] = p_adj
        comps += lang_comps

    summ, comp = pd.DataFrame(rows), pd.DataFrame(comps)
    FT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(FT / "summary.csv", index=False)
    comp.to_csv(FT / "comparisons.csv", index=False)

    md = ["# Fine-tuning ablation — test-set results", "",
          "Fixed song-disjoint test set per language (real technique takes only); "
          "metric = CER for zh/ja/ko, WER otherwise (lower is better). CI = 95 % "
          "bootstrap over test songs. Runs: E0 no fine-tuning · R real technique "
          "takes · RC + real control takes · RCA + augmented (v2) · RCA_v1 + "
          "augmented (v1, same amount) · R20C / R20CA low-resource (20 % real).", ""]
    if not summ.empty:
        summ["cell"] = summ.apply(
            lambda r: f"{r['mean']:.3f} [{r.ci_low:.3f}–{r.ci_high:.3f}]", axis=1)
        t = summ.pivot(index="language", columns="run", values="cell")
        md += ["## Mean error per run", "",
               t.reindex(columns=[r for r in RUNS if r in t.columns]).to_markdown(), ""]
        med = summ.pivot(index="language", columns="run", values="median")
        md += ["## Median error per run", "",
               med.reindex(columns=[r for r in RUNS if r in med.columns]).round(3)
               .to_markdown(), ""]
    if not comp.empty:
        show = comp.assign(
            diff=comp.apply(lambda r: f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                            axis=1),
            p=comp["p_holm"].map(lambda p: f"{p:.3g}"))
        md += ["## Paired comparisons (A − B; negative = A better)", "",
               "p = Wilcoxon signed-rank on per-clip errors, Holm-corrected within "
               "each language.", "",
               show[["language", "comparison", "n", "diff", "a_better_share",
                     "b_better_share", "p"]].round(2).to_markdown(index=False), ""]
    (FT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"→ {FT / 'summary.md'}")


if __name__ == "__main__":
    main()
