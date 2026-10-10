#!/usr/bin/env python3
"""
distribution_analysis.py
────────────────────────
Distribution-level analysis of real vs augmented singing techniques (action
items of the 2026-09-28 supervisor meeting):

  A1  one figure per technique, same layout as the vibrato one
  A2  the same figures with the full distribution (outliers + human range)
  A3  the new engines (world_v2, psola) overlaid next to v1
  A4  diversity: is augmented data as spread as real data, per language and
      across languages?
  A5  coverage of the tails / human range, and sampled-parameter modes that
      target under-represented regions (``world_v2.sample_presets``)
  A6  does the technique depend on register / singer / voice type, not only
      on language?
  A7  do the calibrated targets rest on a single performer? (leave-one-
      singer-out stability)

Inputs
  results/calibration/<lang>/features.parquet   real + control (all train songs)
  results/distributions/engine_features.parquet engine outputs on the pilot
                                                pairs (measure_engine_outputs.py)

Outputs (results/distributions/)
  figures/<group>_iqr.png      A1/A3: IQR boxes (as the vibrato figure)
  figures/<group>_full.png     A2/A3: Tukey boxes + outliers + 1–99 % band
  figures/<group>_modes.png    A4/A5: pooled distributions, parameter modes
  figures/<group>_register.png A6: real effect by register tercile
  diversity.csv, coverage.csv, dependence.csv, singer_stability.csv
  report.md

Usage
─────
  python scripts/distribution_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from calibration_report import (  # noqa: E402
    KEY, KEY_FEATURES, LANG_ORDER, effects, load_all,
)

CALIB = REPO / "results" / "calibration"
OUT = REPO / "results" / "distributions"

# Source families: engine-file source prefix → display name. Order and colours
# are fixed (categorical palette slots in order, colour follows the entity).
FAMILIES = {
    "real": "real",
    "legacy_preset": "v1",
    "world_v2_quartile": "world_v2",
    "psola_quartile": "psola",
    "world_v2_sampled": "world_v2 sampled",
    "world_v2_tails": "world_v2 tails",
    "psola_sampled": "psola sampled",
}
COLORS = {"real": "#2a78d6", "v1": "#eb6834", "world_v2": "#1baf7a", "psola": "#eda100",
          "world_v2 sampled": "#e87ba4", "world_v2 tails": "#008300",
          "psola sampled": "#4a3aa7"}
PER_LANG = ["real", "v1", "world_v2", "psola"]                 # A1–A3 figures
MODE_SET = ["real", "v1", "world_v2", "world_v2 sampled", "world_v2 tails",
            "psola", "psola sampled"]                            # A4/A5 figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REGISTERS = ["low", "mid", "high"]
REG_COLORS = ["#a9c8f0", "#5b98e0", "#1a5aa6"]   # one hue, light → dark (ordinal)


# ── Data ─────────────────────────────────────────────────────────────────────

def family(source: str) -> str | None:
    """Display family of an engine-file source (``world_v2_sampled_v1`` → ...)."""
    if source == "real":
        return "real"
    prefix = source.rsplit("_v", 1)[0]
    return FAMILIES.get(prefix)


def all_effects(real_df: pd.DataFrame, eng_df: pd.DataFrame | None) -> dict:
    """Per group × feature: long frame of effects with a ``family`` column.

    Real effects come from the whole calibration set (all train songs); engine
    effects from the pilot pairs, measured on the same phone spans.

    Args:
        real_df: Calibration rows (real / control / legacy).
        eng_df:  Engine rows (or None).

    Returns:
        ``{(group, feature): DataFrame[language, singer, family, value, ...]}``
    """
    out = {}
    for group, feats in KEY_FEATURES.items():
        for feat, mode, _ in feats:
            r = effects(real_df, group, feat, mode)
            if r.empty:
                continue
            parts = [r[r.source == "real"].assign(family="real")]
            if eng_df is not None and not eng_df.empty:
                e = effects(eng_df, group, feat, mode)
                if not e.empty:
                    e = e[e.source != "real"].copy()
                    e["family"] = e["source"].map(family)
                    parts.append(e.dropna(subset=["family"]))
            out[(group, feat)] = pd.concat(parts, ignore_index=True)
    return out


def human_range(real: pd.Series) -> tuple[float, float]:
    """1st–99th percentile of the real effect: the 'human range'."""
    lo, hi = real.quantile([0.01, 0.99])
    return float(lo), float(hi)


# ── Figures ──────────────────────────────────────────────────────────────────

def _style(ax, label: str, zero: bool) -> None:
    """Recessive grid/axes and a zero line for signed effects."""
    if zero:
        ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_ylabel(label, fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _boxes(ax, data: list[np.ndarray], pos: list[float], color: str, width: float,
           full: bool) -> None:
    """Draw one family's boxes (IQR only, or Tukey whiskers + outlier dots)."""
    keep = [(d, p) for d, p in zip(data, pos) if len(d)]
    if not keep:
        return
    bp = ax.boxplot([d for d, _ in keep], positions=[p for _, p in keep], widths=width,
                    showfliers=full, whis=1.5 if full else 0, patch_artist=True,
                    flierprops={"marker": "o", "markersize": 2.2, "alpha": 0.35,
                                "markerfacecolor": color, "markeredgewidth": 0})
    for b in bp["boxes"]:
        b.set(facecolor=color + "40", edgecolor=color, linewidth=1.1)
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set(color=color, linewidth=0.9, visible=full)
    for ln in bp["medians"]:
        ln.set(color=color, linewidth=2)


def plot_per_language(eff: dict, group: str, out: Path, full: bool) -> None:
    """A1–A3: per language, real vs v1 / world_v2 / psola (quartile presets).

    Args:
        eff:   ``all_effects`` output.
        group: Technique group.
        out:   PNG path.
        full:  False = IQR boxes only (as the original vibrato figure);
               True = Tukey whiskers, every outlier, and the real 1–99 % band.
    """
    feats = [(f, m, lb) for f, m, lb in KEY_FEATURES[group] if (group, f) in eff]
    if not feats:
        return
    fams = [f for f in PER_LANG
            if any((eff[(group, ft)]["family"] == f).any() for ft, _, _ in feats)]
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, 2.8 * len(feats)),
                             sharex=True, squeeze=False)
    width = 0.8 / len(fams)
    for ax, (feat, mode, label) in zip(axes[:, 0], feats):
        e = eff[(group, feat)]
        langs = [lg for lg in LANG_ORDER if lg in set(e.language)]
        if full:   # human range of every language, behind the boxes
            for i, lg in enumerate(langs):
                r = e[(e.language == lg) & (e.family == "real")]["value"]
                if len(r):
                    lo, hi = human_range(r)
                    ax.fill_between([i - 0.45, i + 0.45], lo, hi, color=GRID, alpha=0.7,
                                    lw=0, zorder=0)
        for j, fam in enumerate(fams):
            data = [e[(e.language == lg) & (e.family == fam)]["value"].values for lg in langs]
            pos = [i - 0.4 + width * (j + 0.5) for i in range(len(langs))]
            _boxes(ax, data, pos, COLORS[fam], width * 0.85, full)
        _style(ax, label, mode == "delta")
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels(langs, fontsize=9)
        if full:   # outliers can be extreme: keep the bulk readable
            v = e["value"]
            lo, hi = v.quantile([0.002, 0.998])
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
    handles = [Patch(facecolor=COLORS[f] + "40", edgecolor=COLORS[f], label=f) for f in fams]
    if full:
        handles.append(Patch(facecolor=GRID, label="real 1–99 % (human range)"))
    name = group.replace("_Group", "").replace("_", " ")
    what = ("box = IQR, whiskers = 1.5 IQR, dots = outliers (axis clipped at 0.2/99.8 %)"
            if full else "box = IQR, line = median")
    top = 1 - 1.0 / fig.get_figheight()
    fig.suptitle(f"{name}: real effect vs augmentation engines ({what})",
                 y=0.995, fontsize=11, color=INK)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, top + 0.5 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_modes(eff: dict, group: str, out: Path) -> None:
    """A4/A5: pooled over languages, every source incl. sampled parameter modes."""
    feats = [(f, m, lb) for f, m, lb in KEY_FEATURES[group] if (group, f) in eff]
    if not feats:
        return
    fig, axes = plt.subplots(1, len(feats), figsize=(3.4 * len(feats) + 1, 4.2),
                             squeeze=False)
    for ax, (feat, mode, label) in zip(axes[0], feats):
        e = eff[(group, feat)]
        fams = [f for f in MODE_SET if (e.family == f).any()]
        real = e[e.family == "real"]["value"]
        lo, hi = human_range(real)
        ax.axhspan(lo, hi, color=GRID, alpha=0.7, lw=0, zorder=0)
        for j, fam in enumerate(fams):
            _boxes(ax, [e[e.family == fam]["value"].values], [j], COLORS[fam], 0.6, True)
        _style(ax, label, mode == "delta")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, rotation=40, ha="right", fontsize=8)
        v = e["value"]
        vlo, vhi = v.quantile([0.002, 0.998])
        pad = 0.08 * (vhi - vlo)
        ax.set_ylim(vlo - pad, vhi + pad)
    name = group.replace("_Group", "").replace("_", " ")
    fig.suptitle(f"{name}: pooled over languages — parameter modes "
                 "(grey band = real 1–99 %)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_register(dep_rows: pd.DataFrame, group: str, out: Path) -> None:
    """A6: real effect by register tercile (pitch within the singer's range)."""
    d = dep_rows[dep_rows.group == group]
    feats = [(f, m, lb) for f, m, lb in KEY_FEATURES[group] if f in set(d.feature)]
    if not feats:
        return
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats) + 1, 3.6),
                             squeeze=False)
    for ax, (feat, mode, label) in zip(axes[0], feats):
        e = d[d.feature == feat]
        for j, reg in enumerate(REGISTERS):
            _boxes(ax, [e[e.register == reg]["value"].values], [j], REG_COLORS[j], 0.6, False)
        _style(ax, label, mode == "delta")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{r}\nregister" for r in REGISTERS], fontsize=8)
    name = group.replace("_Group", "").replace("_", " ")
    fig.suptitle(f"{name}: real effect by register (tercile of the control pitch "
                 "within each singer; box = IQR)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Tables ───────────────────────────────────────────────────────────────────

def _spread(v: pd.Series) -> dict:
    """Location and spread summary of one distribution."""
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
    return {"n": int(len(v)), "median": q[2], "iqr": q[3] - q[1], "range90": q[4] - q[0]}


def diversity(eff: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A4: spread of every source relative to real, within and across languages.

    Returns:
        ``(per_language, summary)``. Summary columns: ``spread_ratio`` = median
        over languages of range90(source) / range90(real) (1 = as diverse as
        real); ``between_lang_ratio`` = std of the per-language medians,
        source / real (1 = languages differ as much as in real data).
    """
    rows = []
    for (group, feat), e in eff.items():
        for (lg, fam), v in e.groupby(["language", "family"])["value"]:
            if len(v) >= 5:
                rows.append({"group": group, "feature": feat, "language": lg,
                             "family": fam, **_spread(v)})
    per = pd.DataFrame(rows)
    real = per[per.family == "real"].set_index(["group", "feature", "language"])
    per = per.join(real[["range90", "median"]].rename(
        columns={"range90": "real_range90", "median": "real_median"}),
        on=["group", "feature", "language"])
    per["spread_ratio"] = per["range90"] / per["real_range90"].replace(0, np.nan)
    summ = []
    for (group, feat, fam), g in per.groupby(["group", "feature", "family"]):
        r = per[(per.group == group) & (per.feature == feat) & (per.family == "real")
                & per.language.isin(g.language)]
        summ.append({"group": group, "feature": feat, "family": fam,
                     "n_languages": len(g), "spread_ratio": g["spread_ratio"].median(),
                     "between_lang_ratio": g["median"].std() / r["median"].std()
                     if len(g) > 2 and r["median"].std() > 0 else np.nan})
    return per, pd.DataFrame(summ)


def coverage(eff: dict) -> pd.DataFrame:
    """A5: how augmented values sit in the real distribution of their language.

    Columns: ``outside_human`` = share outside the real 1–99 % range;
    ``tail_share`` = share in the real outer deciles (< p10 or > p90; real
    data has 0.20 by definition); ``ks`` = Kolmogorov–Smirnov distance to
    the real distribution (0 = identical), median over languages.
    """
    from scipy.stats import ks_2samp

    rows = []
    for (group, feat), e in eff.items():
        for fam in [f for f in e.family.unique() if f != "real"]:
            outs, tails, kss, n = [], [], [], 0
            for lg, el in e.groupby("language"):
                real = el[el.family == "real"]["value"]
                aug = el[el.family == fam]["value"]
                if len(real) < 20 or len(aug) < 5:
                    continue
                lo, hi = human_range(real)
                p10, p90 = real.quantile([0.1, 0.9])
                outs.append(((aug < lo) | (aug > hi)).mean())
                tails.append(((aug < p10) | (aug > p90)).mean())
                kss.append(ks_2samp(aug, real).statistic)
                n += len(aug)
            if outs:
                rows.append({"group": group, "feature": feat, "family": fam, "n": n,
                             "outside_human": np.mean(outs), "tail_share": np.mean(tails),
                             "ks": np.median(kss)})
    return pd.DataFrame(rows)


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Share of variance explained by a grouping (between-group SS / total SS)."""
    ok = values.notna() & groups.notna()
    v, g = values[ok], groups[ok]
    if len(v) < 10 or g.nunique() < 2:
        return np.nan
    total = ((v - v.mean()) ** 2).sum()
    between = v.groupby(g).agg(lambda x: len(x) * (x.mean() - v.mean()) ** 2).sum()
    return float(between / total) if total > 0 else np.nan


def dependence(real_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A6: what explains the real effect — language, singer, voice type, register?

    Register = tercile of the control-span pitch within each singer (low /
    mid / high part of that singer's own range in the data).

    Returns:
        ``(rows, table)``: per-span rows with ``register`` (for the figure) and
        one table row per group × feature with eta² per factor and the
        Spearman correlation with the relative pitch.
    """
    from scipy.stats import spearmanr

    ctrl = real_df[real_df.source == "control"][KEY + ["voice", "f0_cents"]]
    ctrl = ctrl.rename(columns={"f0_cents": "ctrl_pitch"}).dropna(subset=["ctrl_pitch"])
    ctrl["rel_pitch"] = ctrl["ctrl_pitch"] - ctrl.groupby("singer")["ctrl_pitch"].transform("median")
    ctrl["register"] = ctrl.groupby("singer")["ctrl_pitch"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=REGISTERS))
    rows, table = [], []
    for group, feats in KEY_FEATURES.items():
        for feat, mode, label in feats:
            e = effects(real_df, group, feat, mode)
            if e.empty:
                continue
            e = e[e.source == "real"].merge(ctrl, on=KEY, how="inner")
            if len(e) < 30:
                continue
            e["group"], e["feature"] = group, feat
            rows.append(e[["group", "feature", "language", "singer", "voice", "register",
                           "rel_pitch", "value"]])
            rho = spearmanr(e["rel_pitch"], e["value"], nan_policy="omit").statistic
            table.append({"group": group, "feature": label, "n": len(e),
                          "eta2_language": eta_squared(e["value"], e["language"]),
                          "eta2_voice": eta_squared(e["value"], e["voice"]),
                          "eta2_singer": eta_squared(e["value"], e["singer"]),
                          "eta2_register": eta_squared(e["value"], e["register"].astype(str)),
                          "spearman_rel_pitch": rho,
                          **{f"median_{r}": e.loc[e.register == r, "value"].median()
                             for r in REGISTERS}})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(table)


def singer_stability(eff: dict) -> pd.DataFrame:
    """A7: do the calibrated targets rest on a single performer?

    For every language × group × feature: number of singers, share of spans
    of the largest contributor, and the leave-one-singer-out (LOSO) shift of
    the median target, in IQR units (0 = no single singer moves the target).
    """
    rows = []
    for (group, feat), e in eff.items():
        r = e[e.family == "real"]
        for lg, el in r.groupby("language"):
            if len(el) < 20:
                continue
            q25, q50, q75 = el["value"].quantile([0.25, 0.5, 0.75])
            iqr = max(q75 - q25, 1e-9)
            share = el["singer"].value_counts(normalize=True)
            loso = [abs(el[el.singer != s]["value"].median() - q50) / iqr
                    for s in share.index if (el.singer != s).sum() >= 10]
            rows.append({"group": group, "feature": feat, "language": lg,
                         "n": len(el), "n_singers": int(share.size),
                         "top_singer_share": float(share.iloc[0]),
                         "loso_max_shift_iqr": max(loso) if loso else np.nan})
    return pd.DataFrame(rows)


# ── Report ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-features", default=str(OUT / "engine_features.parquet"))
    args = ap.parse_args()
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    real_df = load_all(CALIB)
    eng_path = Path(args.engine_features)
    eng_df = pd.read_parquet(eng_path) if eng_path.exists() else None
    if eng_df is None:
        print(f"[warn] {eng_path} missing — engines not overlaid "
              "(run measure_engine_outputs.py)")
    eff = all_effects(real_df, eng_df)

    for group in KEY_FEATURES:
        plot_per_language(eff, group, fig_dir / f"{group}_iqr.png", full=False)
        plot_per_language(eff, group, fig_dir / f"{group}_full.png", full=True)
        if eng_df is not None:
            plot_modes(eff, group, fig_dir / f"{group}_modes.png")

    per_lang, div = diversity(eff)
    cov = coverage(eff)
    dep_rows, dep = dependence(real_df)
    for group in KEY_FEATURES:
        plot_register(dep_rows, group, fig_dir / f"{group}_register.png")
    stab = singer_stability(eff)
    per_lang.to_csv(OUT / "diversity_per_language.csv", index=False)
    div.to_csv(OUT / "diversity.csv", index=False)
    cov.to_csv(OUT / "coverage.csv", index=False)
    dep.to_csv(OUT / "dependence.csv", index=False)
    stab.to_csv(OUT / "singer_stability.csv", index=False)

    def pivot(df: pd.DataFrame, col: str) -> str:
        if df.empty:
            return "_(no data)_"
        t = df.pivot_table(index=["group", "feature"], columns="family", values=col)
        t = t[[c for c in MODE_SET if c in t.columns and c != "real"]]
        return t.round(2).to_markdown()

    md = ["# Distribution analysis — real vs augmented techniques", "",
          "Real effects: all train songs (calibration). Engine effects: the pilot "
          "pairs, measured on the same phone spans (technique − control for "
          "voice-quality features; the gesture itself for vibrato / glissando).",
          "Engines: **v1** (current presets), **world_v2** / **psola** with the "
          "quartile presets (= pilot), and the new per-clip *sampled* modes "
          "(`world_v2.sample_presets`): **sampled** draws targets from the whole "
          "real distribution (2–98 %), **tails** pushes draws toward both tails.", ""]
    md += ["## A1–A3. Per-language figures", "",
           "`*_iqr.png` = same style as the vibrato figure (IQR boxes); "
           "`*_full.png` = whiskers, every outlier and the real 1–99 % band "
           "(\"human range\").", ""]
    for group in KEY_FEATURES:
        for kind in ("iqr", "full"):
            if (fig_dir / f"{group}_{kind}.png").exists():
                md.append(f"![{group} {kind}](figures/{group}_{kind}.png)")
        md.append("")
    md += ["## A4. Diversity", "",
           "**Spread ratio** = width of the central 90 % of the augmented values / "
           "the same width in real data, per language (median over languages). "
           "1 = as diverse as real; below 1 = too uniform.", "",
           pivot(div, "spread_ratio"), "",
           "**Between-language ratio** = how much the per-language medians differ, "
           "augmented / real (1 = languages differ as much as in real data).", "",
           pivot(div, "between_lang_ratio"), ""]
    md += ["## A5. Coverage of the tails and of the human range", "",
           "**Outside human range** = share of augmented values outside the real "
           "1–99 % of their language (should be ≈ 0.02).", "",
           pivot(cov, "outside_human"), "",
           "**Tail share** = share of augmented values in the real outer deciles "
           "(< p10 or > p90). Real data: 0.20. Higher = more weight on "
           "under-represented values.", "", pivot(cov, "tail_share"), "",
           "**KS distance** to the real distribution (0 = same distribution).", "",
           pivot(cov, "ks"), ""]
    for group in KEY_FEATURES:
        if (fig_dir / f"{group}_modes.png").exists():
            md.append(f"![{group} modes](figures/{group}_modes.png)")
    md += ["", "## A6. Language, singer, voice type or register?", "",
           "η² = share of the variance of the real effect explained by each factor "
           "(singer is nested in language, so η²(singer) ≥ η²(language)). Register = "
           "tercile of the control pitch within each singer.", "",
           dep.round(3).to_markdown(index=False) if not dep.empty else "_(no data)_", ""]
    for group in KEY_FEATURES:
        if (fig_dir / f"{group}_register.png").exists():
            md.append(f"![{group} register](figures/{group}_register.png)")
    md += ["", "## A7. Do the targets rest on one performer?", "",
           "Leave-one-singer-out: largest shift of the median target when one "
           "singer is removed, in IQR units (> 0.5 = the target depends "
           "noticeably on one performer). Languages with one singer cannot be "
           "checked (NaN).", ""]
    if not stab.empty:
        t = stab.pivot_table(index=["group", "feature"], columns="language",
                             values="loso_max_shift_iqr")
        md += [t.reindex(columns=[c for c in LANG_ORDER if c in t.columns])
               .round(2).to_markdown(), "",
               "Largest single-singer share of spans per language: "
               + ", ".join(f"{lg} {v:.0%}" for lg, v in
                           stab.groupby("language")["top_singer_share"].max()
                           .reindex(LANG_ORDER).dropna().items()), ""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"report → {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
