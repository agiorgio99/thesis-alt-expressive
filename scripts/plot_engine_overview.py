#!/usr/bin/env python3
"""
plot_engine_overview.py
───────────────────────
Compact slide/paper figures: for every technique, its defining feature per
language, real effect vs v1 vs the v2 augmentation actually used for
fine-tuning (engine "auto" with sampled parameters: PSOLA for vibrato and
glissando, world_v2 for the rest). Same style as the September vibrato
figure, but with Tukey whiskers, every outlier and the real 1–99 % band
("human range"), and for all six techniques.

Outputs (results/distributions/figures/):
  overview_pitch.png    vibrato extent · glissando size · pharyngeal ΔF0
  overview_quality.png  breathy Δ aperiodicity · falsetto ΔH1–H2 · mixed ΔH1–H2

Usage
─────
  python scripts/plot_engine_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from calibration_report import LANG_ORDER, load_all  # noqa: E402
from distribution_analysis import (  # noqa: E402
    CALIB, COLORS, GRID, INK, OUT, _boxes, _style, all_effects, human_range,
)

# One defining feature per technique: (group, feature, mode, axis label).
PITCH = [("Vibrato_Group", "vib_extent_cents", "abs", "Vibrato: extent (cents)"),
         ("Glissando_Group", "glide_cents", "abs", "Glissando: size (cents, + = up)"),
         ("Pharyngeal_Group", "f0_cents", "delta", "Pharyngeal: Δ F0 (cents)")]
QUALITY = [("Breathy_Group", "ap_5k_8k", "delta", "Breathy: Δ aperiodicity 5–8 kHz"),
           ("Falsetto_Group", "h1h2_db", "delta", "Falsetto: Δ H1–H2 (dB)"),
           ("Mixed_Voice_Group", "h1h2_db", "delta", "Mixed voice: Δ H1–H2 (dB)")]
# Engine actually used per group in the fine-tuning data (--engine auto).
USED = {"Vibrato_Group": "psola sampled", "Glissando_Group": "psola sampled"}
V2_LABEL = "v2 (as used for fine-tuning)"
FAMS = ["real", "v1", V2_LABEL]
FAM_COLORS = {"real": COLORS["real"], "v1": COLORS["v1"], V2_LABEL: COLORS["world_v2"]}
SHORT = {"english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
         "russian": "ru", "italian": "it", "french": "fr", "spanish": "es",
         "german": "de"}


def plot_overview(eff: dict, panels: list[tuple], out: Path) -> None:
    """Three stacked panels (one technique each), languages on the x axis.

    Args:
        eff:    ``all_effects`` output.
        panels: (group, feature, mode, label) per panel.
        out:    PNG path.
    """
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.5 * len(panels) + 0.6),
                             sharex=True)
    width = 0.8 / len(FAMS)
    for ax, (group, feat, mode, label) in zip(axes, panels):
        e = eff[(group, feat)].copy()
        # Map the engine used for this group onto one "v2" family.
        e.loc[e["family"] == USED.get(group, "world_v2 sampled"), "family"] = V2_LABEL
        langs = [lg for lg in LANG_ORDER if lg in set(e.language)]
        for i, lg in enumerate(langs):   # human range behind the boxes
            r = e[(e.language == lg) & (e.family == "real")]["value"]
            if len(r):
                lo, hi = human_range(r)
                ax.fill_between([i - 0.45, i + 0.45], lo, hi, color=GRID, alpha=0.7,
                                lw=0, zorder=0)
        for j, fam in enumerate(FAMS):
            data = [e[(e.language == lg) & (e.family == fam)]["value"].values for lg in langs]
            pos = [i - 0.4 + width * (j + 0.5) for i in range(len(langs))]
            _boxes(ax, data, pos, FAM_COLORS[fam], width * 0.85, True)
        _style(ax, "", mode == "delta")
        ax.set_title(label, loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([SHORT[lg] for lg in langs], fontsize=10)
        # Extreme outliers would squash the boxes: clip the axis at 0.2/99.8 %.
        v = e[e.family.isin(FAMS)]["value"]
        lo, hi = v.quantile([0.002, 0.998])
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    handles = [Patch(facecolor=FAM_COLORS[f] + "40", edgecolor=FAM_COLORS[f], label=f)
               for f in FAMS]
    handles.append(Patch(facecolor=GRID, label="real 1–99 % (human range)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    eng_path = OUT / "engine_features.parquet"
    eff = all_effects(load_all(CALIB), pd.read_parquet(eng_path))
    fig_dir = OUT / "figures"
    plot_overview(eff, PITCH, fig_dir / "overview_pitch.png")
    plot_overview(eff, QUALITY, fig_dir / "overview_quality.png")
    print(f"→ {fig_dir}/overview_*.png")


if __name__ == "__main__":
    main()
