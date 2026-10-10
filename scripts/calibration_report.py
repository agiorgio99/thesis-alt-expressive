#!/usr/bin/env python3
"""
calibration_report.py
─────────────────────
Answer "do the v1 (English-tuned) augmentation presets reproduce what real
singers do, in every language?" from the measurements written by
measure_technique_features.py, and derive per-language target parameters for
the improved engines.

For every language × technique group it compares, on the SAME phone spans:

  real effect   = real technique take − control take      (what singers do)
  legacy effect = v1 variant of the control − control     (what v1 does)

on the features that define each technique (e.g. vibrato extent, glide size and
direction, high-band aperiodicity, H1–H2, F0 shift), and classifies every v1
preset as: within the real IQR, too weak, too strong, or wrong direction.

Outputs (results/calibration/):
  report.md                  tables + verdicts + figures
  summary.csv                one row per language × group × feature × source
  figures/<group>.png        real distribution per language vs v1 presets
  <language>/params.json     q25/q50/q75 targets for the v2 engines

Usage
─────
  python scripts/calibration_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alt.augment.features import LTAS_CENTRES, QGRID  # noqa: E402

# Span identity shared by the real / control / legacy rows of one measurement.
KEY = ["language", "singer", "group", "song_key", "stem", "word_idx", "ph_idx"]

# Defining features per technique group. "delta" = measured as the change vs
# the paired control span; "abs" = the technique take's own value (vibrato
# rate/extent and glide size are properties of the gesture itself).
KEY_FEATURES: dict[str, list[tuple[str, str, str]]] = {
    # (feature, mode, human label)
    "Vibrato_Group": [("vib_extent_cents", "abs", "vibrato extent (cents, peak)"),
                      ("vib_rate_hz", "abs", "vibrato rate (Hz)"),
                      ("vib_onset_s", "abs", "vibrato onset delay (s)")],
    "Glissando_Group": [("glide_cents", "abs", "glide size (cents, + = up)"),
                        ("glide_dur_s", "abs", "glide duration (s)")],
    "Breathy_Group": [("ap_5k_8k", "delta", "Δ aperiodicity 5–8 kHz"),
                      ("ap_1k_3k", "delta", "Δ aperiodicity 1–3 kHz"),
                      ("h1h2_db", "delta", "Δ H1–H2 (dB)"),
                      ("tilt_db_oct", "delta", "Δ spectral tilt (dB/oct)"),
                      ("level_db", "delta", "Δ level (dB)")],
    "Pharyngeal_Group": [("ap_5k_8k", "delta", "Δ aperiodicity 5–8 kHz"),
                         ("tilt_db_oct", "delta", "Δ spectral tilt (dB/oct)"),
                         ("h1h2_db", "delta", "Δ H1–H2 (dB)"),
                         ("f0_cents", "delta", "Δ F0 (cents)"),
                         ("level_db", "delta", "Δ level (dB)")],
    "Falsetto_Group": [("f0_cents", "delta", "Δ F0 (cents)"),
                       ("h1h2_db", "delta", "Δ H1–H2 (dB)"),
                       ("tilt_db_oct", "delta", "Δ spectral tilt (dB/oct)"),
                       ("ap_5k_8k", "delta", "Δ aperiodicity 5–8 kHz")],
    "Mixed_Voice_Group": [("f0_cents", "delta", "Δ F0 (cents)"),
                          ("h1h2_db", "delta", "Δ H1–H2 (dB)"),
                          ("tilt_db_oct", "delta", "Δ spectral tilt (dB/oct)"),
                          ("ap_5k_8k", "delta", "Δ aperiodicity 5–8 kHz")],
}

# Everything the v2 engine consumes, per group (superset of KEY_FEATURES; not
# all of these are tabulated in the report). All voice-quality ones are deltas.
_VQ = [("ap_0k_1k", "delta"), ("ap_1k_3k", "delta"), ("ap_3k_5k", "delta"),
       ("ap_5k_8k", "delta"), ("ap_8k_12k", "delta"), ("h1h2_db", "delta"),
       ("tilt_db_oct", "delta"), ("level_db", "delta"), ("f0_cents", "delta")]
PARAM_FEATURES: dict[str, list[tuple[str, str]]] = {
    "Vibrato_Group": [("vib_extent_cents", "abs"), ("vib_rate_hz", "abs"),
                      ("vib_onset_s", "abs"), ("vib_regularity", "abs")],
    "Glissando_Group": [("glide_cents", "abs"), ("glide_abs_cents", "abs"),
                        ("glide_dur_s", "abs")],
    "Breathy_Group": _VQ, "Pharyngeal_Group": _VQ,
    "Falsetto_Group": _VQ, "Mixed_Voice_Group": _VQ,
}

LANG_ORDER = ["english", "chinese", "japanese", "korean", "russian",
              "italian", "french", "spanish", "german"]
MIN_N = 20   # below this many spans a language cell falls back to pooled targets

# Colours (dataviz reference palette, light mode): real data = categorical
# slot 1 (blue); v1 presets are ordered light → heavy, so they use a single
# sequential hue (orange) from light to dark.
REAL_COLOR = "#2a78d6"
LEGACY_COLORS = ["#f5b08f", "#eb6834", "#b54414", "#7a2a08"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ── Data shaping ─────────────────────────────────────────────────────────────

def load_all(calib_dir: Path) -> pd.DataFrame:
    """Concatenate every language's features.parquet."""
    frames = [pd.read_parquet(p) for p in sorted(calib_dir.glob("*/features.parquet"))]
    if not frames:
        sys.exit(f"No features found under {calib_dir} — run "
                 "measure_technique_features.py first.")
    return pd.concat(frames, ignore_index=True)


def effects(df: pd.DataFrame, group: str, feature: str, mode: str) -> pd.DataFrame:
    """Per-span effect of every source (real, legacy_vK) on one feature.

    Args:
        df:      All feature rows.
        group:   Technique group.
        feature: Feature column.
        mode:    "delta" (source − control on the same span) or "abs"
                 (source value; control kept as its own source for reference).

    Returns:
        Long frame: KEY columns + ``source`` + ``value``.
    """
    d = df[(df["group"] == group) & df[feature].notna()]
    wide = d.pivot_table(index=KEY, columns="source", values=feature, aggfunc="first")
    out = []
    for src in wide.columns:
        if mode == "delta":
            if src == "control" or "control" not in wide:
                continue
            v = wide[src] - wide["control"]
        else:
            v = wide[src]
        s = v.dropna().rename("value").reset_index()
        s["source"] = src
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def verdict(real: pd.Series, preset: float) -> str:
    """Classify a v1 preset's median effect against the real distribution.

    Args:
        real:   Real per-span effects.
        preset: Median effect of one v1 preset.

    Returns:
        "within IQR", "too weak", "too strong" or "wrong direction".
    """
    q25, q50, q75 = real.quantile([0.25, 0.5, 0.75])
    if q25 <= preset <= q75:
        return "within IQR"
    # Direction only matters when the real effect has a clear sign (IQR does
    # not straddle zero).
    if (q25 > 0 or q75 < 0) and np.sign(preset) != np.sign(q50) and preset != 0:
        return "wrong direction"
    return "too strong" if abs(preset) > abs(q50) else "too weak"


def quantile_spec(values: pd.Series, mode: str | None = None, nd: int = 4) -> dict:
    """Quartiles + full quantile grid of one real effect distribution.

    Args:
        values: Real per-span effects.
        mode:   "delta" / "abs" (stored when given).
        nd:     Rounding digits.

    Returns:
        ``{"mode"?, "n", "q25", "q50", "q75", "qgrid"}`` (``qgrid`` aligned with
        ``QGRID``).
    """
    q = values.quantile([0.25, 0.5, 0.75]).values
    spec = {"mode": mode} if mode else {}
    spec.update({"n": int(len(values)), "q25": round(float(q[0]), nd),
                 "q50": round(float(q[1]), nd), "q75": round(float(q[2]), nd),
                 "qgrid": [round(float(x), nd) for x in values.quantile(QGRID).values]})
    return spec


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_group(eff: dict[str, pd.DataFrame], group: str, out: Path) -> None:
    """One panel per key feature: real distribution per language (boxes) with
    the median effect of each v1 preset overlaid (markers).

    Args:
        eff:   feature → long effects frame (from ``effects``).
        group: Technique group (title).
        out:   PNG path.
    """
    feats = KEY_FEATURES[group]
    fig, axes = plt.subplots(len(feats), 1, figsize=(10, 2.5 * len(feats)),
                             sharex=True, squeeze=False)
    for ax, (feat, mode, label) in zip(axes[:, 0], feats):
        e = eff.get(feat)
        if e is None or e.empty:
            ax.set_visible(False)
            continue
        langs = [lg for lg in LANG_ORDER if lg in set(e["language"])]
        real = [e[(e.language == lg) & (e.source == "real")]["value"].values
                for lg in langs]
        bp = ax.boxplot(real, positions=range(len(langs)), widths=0.5,
                        showfliers=False, patch_artist=True)
        for b in bp["boxes"]:
            b.set(facecolor=REAL_COLOR + "33", edgecolor=REAL_COLOR, linewidth=1.2)
        for part in ("whiskers", "caps"):
            for ln in bp[part]:
                ln.set(color=REAL_COLOR, linewidth=1)
        for ln in bp["medians"]:
            ln.set(color=REAL_COLOR, linewidth=2)
        legacy_src = sorted(s for s in e["source"].unique() if s.startswith("legacy"))
        for k, src in enumerate(legacy_src):
            meds = [e[(e.language == lg) & (e.source == src)]["value"].median()
                    for lg in langs]
            ax.scatter(np.arange(len(langs)) + 0.3, meds, s=46, marker="D",
                       color=LEGACY_COLORS[k % len(LEGACY_COLORS)],
                       edgecolor="white", linewidth=1.5, zorder=3,
                       label=f"v1 preset {src.split('_')[1]}")
        if mode == "delta":
            ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_ylabel(label, fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels(langs, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.insert(0, plt.Rectangle((0, 0), 1, 1, facecolor=REAL_COLOR + "33",
                                    edgecolor=REAL_COLOR))
    labels.insert(0, "real (IQR box, median line)")
    # Title on top, legend in its own band below it, panels underneath.
    top = 1 - 0.9 / fig.get_figheight()
    fig.suptitle(f"{group.replace('_Group', '')}: real effect per language vs v1 presets",
                 y=0.995, fontsize=11, color=INK)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, top + 0.45 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_ltas(df: pd.DataFrame, group: str, out: Path) -> dict[str, list[float]]:
    """Median LTAS difference (technique − control) per language.

    Args:
        df:    All feature rows.
        group: Technique group.
        out:   PNG path.

    Returns:
        language → median ΔLTAS (dB) on ``LTAS_CENTRES`` (for params.json),
        plus a ``"pooled"`` curve over all languages.
    """
    d = df[df["group"] == group]
    curves: dict[str, list[float]] = {}
    all_diffs: list[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = plt.get_cmap("tab10")
    for i, lg in enumerate([lg for lg in LANG_ORDER if lg in set(d["language"])]):
        dl = d[d.language == lg]
        real = dl[dl.source == "real"].set_index(KEY)["ltas"]
        ctrl = dl[dl.source == "control"].set_index(KEY)["ltas"]
        idx = real.index.intersection(ctrl.index)
        if len(idx) < 5:
            continue
        # Spans with too few voiced frames have no LTAS (missing value, not a
        # 40-band vector): skip them.
        n_bands = len(LTAS_CENTRES)
        diffs = [np.asarray(real[k], float) - np.asarray(ctrl[k], float)
                 for k in idx
                 if np.ndim(real[k]) == 1 and np.ndim(ctrl[k]) == 1
                 and len(real[k]) == n_bands and len(ctrl[k]) == n_bands]
        if len(diffs) < 5:
            continue
        diff = np.stack(diffs)
        all_diffs.append(diff)
        med = np.nanmedian(diff, axis=0)
        curves[lg] = np.round(med, 2).tolist()
        ax.semilogx(LTAS_CENTRES, med, lw=2, color=cmap(i), label=lg)
    if all_diffs:   # all languages together (targets pooled across languages)
        med = np.nanmedian(np.concatenate(all_diffs), axis=0)
        curves["pooled"] = np.round(med, 2).tolist()
        ax.semilogx(LTAS_CENTRES, med, lw=2.5, color=INK, ls="--", label="all")
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Δ LTAS (dB)")
    ax.set_title(f"{group.replace('_Group', '')}: median spectral change vs control",
                 fontsize=11)
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.legend(ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return curves


# ── Report ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", default=str(REPO / "results" / "calibration"))
    args = ap.parse_args()
    calib = Path(args.calib_dir)
    fig_dir = calib / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(calib)
    df["glide_abs_cents"] = df["glide_cents"].abs()   # glide magnitude, any direction
    rows, md = [], []
    params: dict[str, dict] = {lg: {} for lg in df["language"].unique()}
    pooled_params: dict[str, dict] = {}

    md += ["# Technique calibration report", "",
           "Paired design: every technique take is compared with the control take "
           "of the same phrase (same singer, same lyrics), phone span by phone span, "
           "on train songs only. *Real effect* = technique − control; *v1 effect* = "
           "v1 augmentation of the control − control, measured on the same spans.",
           "", "Verdicts compare the median effect of each v1 preset with the real "
           "per-span distribution of that language: **within IQR**, **too weak**, "
           "**too strong**, or **wrong direction** (real effect has a clear sign "
           "and the preset goes the other way).", ""]

    # Contamination: how often the control span already carries the flag.
    cont = (df[df.source == "real"].groupby(["group", "language"])["flag_in_control"]
            .mean().unstack().reindex(columns=LANG_ORDER))
    md += ["## Control contamination", "",
           "Share of measured spans whose *control* take is already annotated with "
           "the same technique (the baseline is not technique-free there).", "",
           (cont * 100).round(1).to_markdown(), ""]

    for group, feats in KEY_FEATURES.items():
        md += [f"## {group.replace('_Group', '')}", "",
               f"![{group}](figures/{group}.png)", ""]
        eff_by_feat = {}
        table = []
        for feat, mode, label in feats:
            e = effects(df, group, feat, mode)
            if e.empty:
                continue
            eff_by_feat[feat] = e
            for lg in [lg for lg in LANG_ORDER if lg in set(e.language)]:
                el = e[e.language == lg]
                real = el[el.source == "real"]["value"]
                if real.empty:
                    continue
                q = real.quantile([0.25, 0.5, 0.75]).values
                cell = {"technique": group, "feature": label, "language": lg,
                        "n": len(real), "real_q25": q[0], "real_median": q[1],
                        "real_q75": q[2]}
                if mode == "abs":
                    ctrl = el[el.source == "control"]["value"]
                    cell["control_median"] = ctrl.median() if len(ctrl) else np.nan
                for src in sorted(s for s in el.source.unique() if s.startswith("legacy")):
                    m = el[el.source == src]["value"].median()
                    cell[f"{src}_median"] = m
                    cell[f"{src}_verdict"] = verdict(real, m)
                table.append(cell)
                rows.append(cell)
        # Targets for the v2 engines: q25/q50/q75 of the real effect, per
        # language and pooled over languages (fallback for sparse cells).
        for feat, mode in PARAM_FEATURES[group]:
            e = eff_by_feat.get(feat)
            if e is None:
                e = effects(df, group, feat, mode)
            if e.empty:
                continue
            real_all = e[e.source == "real"]
            for lg, v in real_all.groupby("language")["value"]:
                params[lg].setdefault(group, {})[feat] = quantile_spec(v, mode)
            pooled_params.setdefault(group, {})[feat] = quantile_spec(
                real_all["value"], mode)
        if eff_by_feat:
            plot_group(eff_by_feat, group, fig_dir / f"{group}.png")
        if table:
            t = pd.DataFrame(table)
            legacy_cols = [c for c in t.columns if c.endswith("_verdict")]
            show = ["feature", "language", "n", "real_q25", "real_median", "real_q75"] \
                + (["control_median"] if "control_median" in t else []) \
                + [c for c in t.columns if c.startswith("legacy") and c.endswith("_median")] \
                + legacy_cols
            md += [t[show].round(2).to_markdown(index=False), ""]
        # Glissando direction share (v1 only glides downward).
        if group == "Glissando_Group" and "glide_cents" in eff_by_feat:
            e = eff_by_feat["glide_cents"]
            up = (e[e.source == "real"].groupby("language")["value"]
                  .apply(lambda v: (v > 0).mean()).reindex(LANG_ORDER).dropna())
            md += ["Share of real glides that go **up**: "
                   + ", ".join(f"{lg} {v:.0%}" for lg, v in up.items())
                   + " (v1 glides are always downward).", ""]
            for lg, v in up.items():
                params[lg].setdefault(group, {})["share_up"] = round(float(v), 3)
            pooled_params.setdefault(group, {})["share_up"] = round(
                float((e[e.source == "real"]["value"] > 0).mean()), 3)
        if group in ("Pharyngeal_Group", "Falsetto_Group", "Mixed_Voice_Group",
                     "Breathy_Group"):
            curves = plot_ltas(df, group, fig_dir / f"{group}_ltas.png")
            md += [f"![{group} LTAS](figures/{group}_ltas.png)", ""]
            for lg, c in curves.items():
                target = pooled_params if lg == "pooled" else params[lg]
                target.setdefault(group, {})["delta_ltas_db"] = c

    # F0 change of falsetto / mixed voice depends on the *voice type*, not the
    # language: e.g. basses sing their falsetto takes an octave above the
    # control (scores +12 semitones) while sopranos stay at the same pitch.
    # Store per-voice quantiles (language × voice when ≥ MIN_N spans, else
    # the voice pooled over languages).
    for group in ("Falsetto_Group", "Mixed_Voice_Group"):
        e = effects(df, group, "f0_cents", "delta")
        if e.empty:
            continue
        e = e[e.source == "real"].merge(
            df[df.source == "real"][KEY + ["voice"]].drop_duplicates(KEY), on=KEY)
        pooled_voice = {v: g["value"] for v, g in e.groupby("voice") if len(g) >= MIN_N}
        pooled_params.setdefault(group, {})["f0_cents_by_voice"] = {
            v: {**quantile_spec(vals, nd=2), "pooled": True} for v, vals in pooled_voice.items()}
        for lg in params:
            by_voice = {}
            for v, pooled_vals in pooled_voice.items():
                cell = e[(e.language == lg) & (e.voice == v)]["value"]
                src = cell if len(cell) >= MIN_N else pooled_vals
                by_voice[v] = {**quantile_spec(src, nd=2), "n": int(len(cell)),
                               "pooled": bool(len(cell) < MIN_N)}
            params[lg].setdefault(group, {})["f0_cents_by_voice"] = by_voice
        tab = (e.groupby(["voice", "language"])["value"].median().unstack()
               .reindex(columns=LANG_ORDER).round(0))
        md += [f"### {group.replace('_Group', '')}: Δ F0 (cents) by voice type", "",
               "Median per voice × language — the F0 change is a property of the "
               "voice type (falsetto register), so the v2 engine uses these.", "",
               tab.to_markdown(), ""]

    # Mark sparse cells and write params.json per language.
    for lg, groups in params.items():
        for group, feats in groups.items():
            for feat, spec in feats.items():
                if isinstance(spec, dict) and spec.get("n", MIN_N) < MIN_N:
                    spec["fallback_pooled"] = pooled_params.get(group, {}).get(feat)
        out = calib / lg / "params.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"language": lg, "ltas_centres_hz": LTAS_CENTRES.round(1).tolist(),
                   "groups": groups}, open(out, "w"), indent=1)

    # Pooled targets (all languages together, F0 change per voice type): a
    # complete params file, so the engines can use language="pooled" as the
    # target source (see build_augmented_dataset.py --targets pooled).
    pooled_doc = {"language": "pooled", "ltas_centres_hz": LTAS_CENTRES.round(1).tolist(),
                  "groups": pooled_params}
    (calib / "pooled").mkdir(exist_ok=True)
    for out in (calib / "params_pooled.json", calib / "pooled" / "params.json"):
        json.dump(pooled_doc, open(out, "w"), indent=1)

    summary = pd.DataFrame(rows)
    summary.to_csv(calib / "summary.csv", index=False)

    # Headline: verdict counts per technique over languages and presets.
    vcols = [c for c in summary.columns if c.endswith("_verdict")]
    long = summary.melt(id_vars=["technique", "feature", "language"],
                        value_vars=vcols, value_name="verdict").dropna()
    head = (long.groupby(["technique", "feature", "verdict"]).size()
            .unstack(fill_value=0))
    md[1:1] = ["", "## Headline — v1 preset verdicts (count over languages × presets)",
               "", head.reset_index().to_markdown(index=False), ""]
    (calib / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"report → {calib / 'report.md'}")


if __name__ == "__main__":
    main()
