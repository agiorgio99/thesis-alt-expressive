#!/usr/bin/env python3
"""
listening_stats.py
──────────────────
Analysis of the blind listening test (app tab "Listening test",
results/listening/ratings.csv).

Per source (real take, control take, v1 = legacy, world_v2, psola):
  - naturalness 1–5: mean with a 95 % bootstrap CI over clips, and a
    Mann–Whitney U test of each engine against the real takes (Holm);
  - technique recognition: share of clips where the heard technique is the
    intended one; for control takes the correct answer is "none".
Also: confusion matrix (intended → heard) and naturalness of the engine
actually used for fine-tuning per technique (engine "auto": PSOLA for
vibrato / glissando, world_v2 for the rest).

Outputs: results/listening/report.md, results/listening/figures/*.png

Usage
─────
  python scripts/listening_stats.py                      # default raters
  python scripts/listening_stats.py --raters antonello_en antonello_it 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from finetune_stats import holm  # noqa: E402

LIS = REPO / "results" / "listening"
FIG = LIS / "figures"
SOURCES = ["real", "control", "legacy", "world_v2", "psola"]
LABELS = {"real": "real take", "control": "control take", "legacy": "v1",
          "world_v2": "world_v2", "psola": "psola"}
COLORS = {"real": "#2a78d6", "control": "#8c8b87", "legacy": "#eb6834",
          "world_v2": "#1baf7a", "psola": "#eda100"}
TECHS = ["vibrato", "glissando", "breathy", "pharyngeal", "falsetto", "mixed voice"]
HEARD = TECHS + ["none"]
# Engine used for the fine-tuning data per technique (--engine auto).
USED = {"vibrato": "psola", "glissando": "psola"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(raters: list[str]) -> pd.DataFrame:
    """Ratings of the chosen raters with intended / heard technique labels."""
    r = pd.read_csv(LIS / "ratings.csv", dtype={"rater": str})
    r = r[r["rater"].isin(raters)].copy()
    r["intended"] = (r["group"].str.replace("_Group", "").str.replace("_", " ")
                     .str.lower())
    r["heard"] = r["technique_heard"].str.lower().str.replace(" (plain singing)", "",
                                                              regex=False)
    # Correct answer: the technique for real / augmented clips, "none" for
    # the control take (same phrase sung plainly).
    r["correct"] = np.where(r["source"] == "control", r["heard"] == "none",
                            r["heard"] == r["intended"])
    return r


def boot_ci(v: np.ndarray, rng: np.random.Generator, n: int = 5000) -> tuple:
    """95 % bootstrap CI of the mean."""
    m = rng.choice(v, (n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def per_source(r: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Naturalness (mean, CI, test vs real) and recognition per source."""
    rows = []
    real = r[r["source"] == "real"]["naturalness"].values
    for s in SOURCES:
        g = r[r["source"] == s]
        if g.empty:
            continue
        v = g["naturalness"].values.astype(float)
        lo, hi = boot_ci(v, rng)
        p = (mannwhitneyu(v, real, alternative="two-sided").pvalue
             if s not in ("real", "control") and len(real) else np.nan)
        rows.append({"source": s, "n": len(g), "naturalness": v.mean(), "ci_low": lo,
                     "ci_high": hi, "p_vs_real": p, "correct": g["correct"].mean(),
                     "heard_none": (g["heard"] == "none").mean()})
    out = pd.DataFrame(rows)
    mask = out["p_vs_real"].notna()
    out.loc[mask, "p_holm"] = holm(out.loc[mask, "p_vs_real"].tolist())
    return out


def plot_naturalness(by_rater: dict[str, pd.DataFrame], out: Path) -> None:
    """Mean naturalness per source with CI, one marker per session."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    marks = ["o", "s", "^"]
    off = np.linspace(-0.18, 0.18, len(by_rater))
    for k, (name, t) in enumerate(by_rater.items()):
        t = t.set_index("source").reindex(SOURCES)
        for i, s in enumerate(SOURCES):
            if pd.isna(t.loc[s, "naturalness"]):
                continue
            m = t.loc[s, "naturalness"]
            ax.errorbar(i + off[k], m, yerr=[[m - t.loc[s, "ci_low"]],
                                             [t.loc[s, "ci_high"] - m]],
                        fmt=marks[k % 3], color=COLORS[s], ms=7, capsize=3,
                        mfc=COLORS[s] if k == 0 else "white", lw=1.3)
    for k, name in enumerate(by_rater):
        ax.plot([], [], marks[k % 3], color=MUTED, mfc=MUTED if k == 0 else "white",
                label=name)
    ax.set_xticks(range(len(SOURCES)), [LABELS[s] for s in SOURCES])
    ax.set_ylim(1, 5.3)
    ax.set_ylabel("naturalness (1 = robotic … 5 = real singer)")
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title("Blind test: naturalness per source (mean, 95 % bootstrap CI)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confusion(r: pd.DataFrame, out: Path) -> None:
    """Intended technique (rows) → heard technique (columns), counts.

    Control takes are shown as their own row ("control → should be none").
    """
    rows = {f"{LABELS[s]}": r[r["source"] == s] for s in ["real", "legacy", "world_v2",
                                                         "psola"]}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, (name, g) in zip(axes, rows.items()):
        m = pd.crosstab(g["intended"], g["heard"]).reindex(index=TECHS, columns=HEARD,
                                                         fill_value=0)
        ax.imshow(m.values, cmap="Blues", vmin=0, vmax=max(1, r.groupby(
            ["source", "intended", "heard"]).size().max()))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                            color="white" if v >= 3 else INK,
                            fontweight="bold" if TECHS[i] == HEARD[j] else "normal")
        ax.set_xticks(range(len(HEARD)), HEARD, rotation=40, ha="right")
        ax.set_yticks(range(len(TECHS)), TECHS)
        ax.set_title(name, color=INK)
        ax.set_xlabel("heard")
    axes[0].set_ylabel("intended technique")
    fig.suptitle("What was heard (counts; diagonal = recognised)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raters", nargs="+", default=["antonello_en", "antonello_it"])
    ap.add_argument("--extra", nargs="*", default=["0"],
                    help="Sessions shown only as a consistency check")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    main_r = load(args.raters)
    sessions = {f"{name} ({load([name])['language'].nunique()} lang.)": per_source(load([name]), rng)
                for name in args.raters + args.extra if not load([name]).empty}
    pooled = per_source(main_r, rng)
    plot_naturalness(sessions, FIG / "naturalness.png")
    plot_confusion(main_r, FIG / "confusion.png")

    # Engine actually used for fine-tuning, per technique.
    aug = main_r[main_r["source"].isin(["legacy", "world_v2", "psola"])]
    used = aug[aug.apply(lambda x: x["source"] == USED.get(x["intended"], "world_v2"),
                         axis=1)]
    tech = (pd.concat([aug[aug["source"] == "legacy"].assign(kind="v1"),
                       used.assign(kind="v2 as used")])
            .groupby(["intended", "kind"])
            .agg(n=("naturalness", "size"), naturalness=("naturalness", "mean"),
                 recognised=("correct", "mean"))
            .round(2).reset_index())
    # Overall: v1 vs v2 as used (Mann–Whitney on clip ratings).
    v1 = aug[aug["source"] == "legacy"]["naturalness"].values
    v2 = used["naturalness"].values
    p_v = mannwhitneyu(v2, v1, alternative="two-sided").pvalue
    overall = (f"**v2 as used for fine-tuning: naturalness {v2.mean():.2f} (n = {len(v2)}, "
               f"recognised {used['correct'].mean():.2f}) vs v1 {v1.mean():.2f} "
               f"(n = {len(v1)}, recognised "
               f"{aug[aug['source'] == 'legacy']['correct'].mean():.2f}); "
               f"Mann–Whitney p = {p_v:.2g}.**")

    show = pooled.assign(
        natural=pooled.apply(lambda x: f"{x.naturalness:.2f} [{x.ci_low:.2f}, {x.ci_high:.2f}]",
                             axis=1),
        p=pooled["p_holm"].map(lambda p: "" if pd.isna(p) else f"{p:.2g}"))
    md = ["# Listening test — results", "",
          f"Raters/sessions pooled: {', '.join(args.raters)} "
          f"({len(main_r)} clips; one listener). Correct = intended technique heard "
          "(for control takes: 'none'). Chance level ≈ 1/7 = 0.14. p = Mann–Whitney "
          "vs real takes, Holm-corrected.", "",
          show[["source", "n", "natural", "p", "correct", "heard_none"]]
          .round(2).to_markdown(index=False), "",
          "![naturalness](figures/naturalness.png)", "",
          "![confusion](figures/confusion.png)", "",
          "## v1 vs the engine used for fine-tuning, per technique", "", overall, "",
          tech.to_markdown(index=False), ""]
    for name, t in sessions.items():
        md += [f"### Session {name}", "",
               t[["source", "n", "naturalness", "correct", "heard_none"]].round(2)
               .to_markdown(index=False), ""]
    (LIS / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"→ {LIS / 'report.md'}")


if __name__ == "__main__":
    main()
