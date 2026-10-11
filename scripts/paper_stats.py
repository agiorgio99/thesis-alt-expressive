#!/usr/bin/env python3
"""
paper_stats.py
──────────────
Dataset description, technique-vs-control statistics and fine-tuning figures
for the TISMIR paper (supervisor meeting 2026-09-28, list B + ablation plots).

B9  Dataset composition: clips / hours per language × technique, singers and
    voice types per language (how uneven the corpus is).
B8  Technique vs control, paired on the same phrase by the same singer:
      - Whisper-large-v3 error (baseline, no fine-tuning): is the technique
        take harder to transcribe than its control take?
      - Defining acoustic features (calibration spans): is the technique
        measurably different from its control?
    Paired Wilcoxon signed-rank, song-bootstrap 95 % CI, Holm correction
    within each family of tests, and a per-singer sign check (few singers per
    language, so an effect carried by one singer is flagged).
FT  Fine-tuning ablation figures from results/finetune (finetune_stats.py):
    error per run with CIs, relative change vs no fine-tuning, and the
    per-technique effect of fine-tuning / augmentation.

Outputs: results/paper/{report.md, *.csv, figures/*.png}

Usage
─────
  python scripts/paper_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from alt.augment.pairing import song_key  # noqa: E402
from alt.text import ISO_TO_LANGUAGE, primary_metric  # noqa: E402
from calibration_report import KEY, KEY_FEATURES, load_all  # noqa: E402
from finetune_stats import RUNS, holm, load_run, song_bootstrap  # noqa: E402

OUT = REPO / "results" / "paper"
FIG = OUT / "figures"
LANG_TO_ISO = {v: k for k, v in ISO_TO_LANGUAGE.items()}
ISO_ORDER = ["en", "zh", "ja", "ko", "ru", "it", "fr", "es", "de"]
# Group folder → display name; Falsetto and Mixed Voice are separate takes
# that share one technique label ("mixed_falsetto") in the inventories.
GROUPS = {"Vibrato_Group": "Vibrato", "Glissando_Group": "Glissando",
          "Breathy_Group": "Breathy", "Pharyngeal_Group": "Pharyngeal",
          "Falsetto_Group": "Falsetto", "Mixed_Voice_Group": "Mixed voice"}
TECH_ORDER = list(GROUPS.values())

# Colours (dataviz reference palette, light mode, as in calibration_report.py).
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RUN_COLORS = {"E0": "#8c8b87", "R": "#2a78d6", "RC": "#1f9e89", "RCA": "#eb6834",
              "RCA_v1": "#b54414", "R20C": "#7e57c2", "R20CA": "#d6409f"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def group_of(audio_path: str) -> str:
    """Group folder of a clip (``.../<song>/<X>_Group/0000.wav``)."""
    return Path(audio_path).parent.name


def stars(p: float) -> str:
    """Significance marker for an adjusted p-value."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ── B9. Dataset composition ──────────────────────────────────────────────────

def load_inventories() -> pd.DataFrame:
    """All baseline inventories (every clip of the corpus) with durations.

    Returns:
        One row per clip: language (ISO), singer, voice type, group folder,
        role (technique / control / speech), song, duration in seconds.
    """
    frames = []
    for lang, iso in LANG_TO_ISO.items():
        inv = pd.read_csv(REPO / "results" / f"baseline_{lang}" / "inventory.csv",
                          keep_default_na=False)
        inv["language"] = iso
        frames.append(inv)
    df = pd.concat(frames, ignore_index=True)
    df["group_folder"] = df["audio_path"].map(group_of)
    df["song"] = df["audio_path"].map(lambda p: song_key(Path(p)))
    # Voice type is encoded in the singer id, e.g. "EN-Alto-1".
    df["voice"] = df["singer_id"].str.split("-").str[1].str.lower()
    # Duration from the WAV header only (no decoding), cached: reading ~38k
    # headers is slow while a training job is using the disk.
    cache = OUT / "clip_durations.csv"
    known = (pd.read_csv(cache).set_index("audio_path")["dur_s"].to_dict()
             if cache.exists() else {})
    df["dur_s"] = [known[p] if p in known else sf.info(str(REPO / p)).duration
                   for p in df["audio_path"]]
    df[["audio_path", "dur_s"]].drop_duplicates("audio_path").to_csv(cache, index=False)
    return df


def dataset_tables(inv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hours per language × technique and singer roster per language.

    Returns:
        (hours table [technique takes], singer table, markdown lines)
    """
    tech = inv[inv["group"] == "technique"].copy()
    tech["technique_name"] = tech["group_folder"].map(GROUPS)
    hours = (tech.pivot_table(index="language", columns="technique_name", values="dur_s",
                              aggfunc="sum").fillna(0) / 3600)
    hours = hours.reindex(index=ISO_ORDER, columns=TECH_ORDER)
    role_h = (inv.pivot_table(index="language", columns="group", values="dur_s",
                              aggfunc="sum").fillna(0) / 3600).reindex(ISO_ORDER)
    clips = inv.pivot_table(index="language", columns="group", values="utt_id",
                            aggfunc="count").reindex(ISO_ORDER)
    singers = (inv.groupby("language")
               .agg(singers=("singer_id", "nunique"),
                    voices=("voice", lambda v: ", ".join(sorted(set(v)))),
                    songs=("song", "nunique"))
               .reindex(ISO_ORDER))
    # Largest singer's share of technique hours: how much one performer
    # dominates a language.
    share = (tech.groupby(["language", "singer_id"])["dur_s"].sum()
             .groupby(level=0).apply(lambda s: s.max() / s.sum())).reindex(ISO_ORDER)
    singers["largest_singer_share"] = share.round(2)
    singers = singers.join(role_h.round(2).add_suffix("_h")).join(clips.add_suffix("_clips"))

    md = ["## B9. Dataset composition", "",
          f"Total: {inv['dur_s'].sum() / 3600:.1f} h, {len(inv)} clips, "
          f"{inv['singer_id'].nunique()} singers. Technique takes only below "
          "(hours); every technique take has a control take of the same phrase.", "",
          hours.round(2).to_markdown(), "",
          "Singers, voice types and hours per role:", "",
          singers.to_markdown(), "",
          "![dataset](figures/dataset_hours.png)", ""]
    return hours, singers, md


def plot_dataset(hours: pd.DataFrame, singers: pd.DataFrame, out: Path) -> None:
    """Heatmap of technique hours per language × technique, singers annotated."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    v = hours.values
    im = ax.imshow(v, cmap="Blues", aspect="auto", vmin=0)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v[i, j] > 0.6 * v.max() else INK)
    ax.set_xticks(range(len(hours.columns)), hours.columns)
    ax.set_yticks(range(len(hours.index)),
                  [f"{l}  ({int(n)} singer{'s' if n > 1 else ''})"
                   for l, n in singers.loc[hours.index, "singers"].items()])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="hours (technique takes)", shrink=0.8)
    ax.set_title("GTSinger: technique hours per language", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ── B8a. Technique vs control: Whisper error ────────────────────────────────

def load_baseline(iso: str) -> pd.DataFrame:
    """Whisper-large-v3 per-clip errors on the whole corpus of one language.

    Rows are matched to the inventory by position when the two files are
    aligned (utt_id is not unique: Falsetto and Mixed Voice takes share it);
    otherwise (older English baseline) by utt_id, and Falsetto / Mixed Voice
    are then not distinguishable and reported together.
    """
    lang = ISO_TO_LANGUAGE[iso]
    base = REPO / "results" / f"baseline_{lang}"
    df = pd.read_csv(base / "asr_whisper_largev3.csv", keep_default_na=False)
    inv = pd.read_csv(base / "inventory.csv", keep_default_na=False)
    metric = primary_metric(lang)
    if metric not in df.columns:
        raise SystemExit(f"{base}: no {metric} column")
    if len(df) == len(inv) and (df["utt_id"].values == inv["utt_id"].values).all():
        df["group_folder"] = inv["audio_path"].map(group_of).values
        df["song"] = inv["audio_path"].map(lambda p: song_key(Path(p))).values
    else:
        df = df.drop_duplicates("utt_id")
        inv = inv.drop_duplicates("utt_id")
        df = df.merge(inv[["utt_id", "audio_path"]], on="utt_id", how="left")
        df["group_folder"] = df["audio_path"].map(group_of)
        df["song"] = df["audio_path"].map(lambda p: song_key(Path(p)))
        df.loc[df["technique"] == "mixed_falsetto", "group_folder"] = "Mixed_Falsetto"
    df["metric_value"] = pd.to_numeric(df[metric], errors="coerce")
    df["metric"] = metric
    return df


def technique_vs_control_error(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired technique − control Whisper error per language × technique.

    Returns:
        (per language × technique results, per-singer differences)
    """
    rows, per_singer = [], []
    for iso in ISO_ORDER:
        df = load_baseline(iso)
        tech = df[df["group"] == "technique"]
        ctrl = df[df["group"] == "control"]
        # The control take of a phrase has the same utt_id with "technique"
        # replaced by "control" (singer__technique__role__song__index).
        ctrl_map = ctrl.drop_duplicates("utt_id").set_index("utt_id")["metric_value"]
        tech = tech.assign(ctrl_value=tech["utt_id"].str.replace(
            "__technique__", "__control__", regex=False).map(ctrl_map))
        tech = tech.dropna(subset=["metric_value", "ctrl_value"])
        lang_rows = []
        for gf, g in tech.groupby("group_folder"):
            diff = g["metric_value"] - g["ctrl_value"]
            lo, hi = song_bootstrap(diff, g["song"], rng)
            # Capped at 1 per clip: robust to repetition loops (CER up to 40).
            diff_c = g["metric_value"].clip(upper=1.0) - g["ctrl_value"].clip(upper=1.0)
            lo_c, hi_c = song_bootstrap(diff_c, g["song"], rng)
            p = wilcoxon(g["metric_value"], g["ctrl_value"], zero_method="zsplit").pvalue \
                if (diff != 0).any() else 1.0
            sing = diff.groupby(g["singer_id"]).mean()
            for s, v in sing.items():
                per_singer.append({"language": iso, "technique": GROUPS.get(gf, "Falsetto+Mixed"),
                                   "singer": s, "mean_diff": v, "n": int((g["singer_id"] == s).sum())})
            lang_rows.append({
                "language": iso, "technique": GROUPS.get(gf, "Falsetto+Mixed"),
                "metric": g["metric"].iloc[0], "n_pairs": len(g),
                "technique_err": g["metric_value"].mean(), "control_err": g["ctrl_value"].mean(),
                "mean_diff": diff.mean(), "ci_low": lo, "ci_high": hi,
                "mean_diff_clip": diff_c.mean(), "ci_low_clip": lo_c, "ci_high_clip": hi_c,
                "tech_worse_share": (diff > 0).mean(), "tech_better_share": (diff < 0).mean(),
                "p_wilcoxon": p,
                "singers_tech_worse": f"{int((sing > 0).sum())}/{len(sing)}"})
        for r, p_adj in zip(lang_rows, holm([r["p_wilcoxon"] for r in lang_rows])):
            r["p_holm"] = p_adj
        rows += lang_rows
    return pd.DataFrame(rows), pd.DataFrame(per_singer)


def plot_tc_error(res: pd.DataFrame, out: Path) -> None:
    """Heatmap of technique − control error, significance stars per cell."""
    t = res.pivot(index="language", columns="technique", values="mean_diff_clip")
    p = res.pivot(index="language", columns="technique", values="p_holm")
    cols = [c for c in TECH_ORDER + ["Falsetto+Mixed"] if c in t.columns]
    t, p = t.reindex(index=ISO_ORDER, columns=cols), p.reindex(index=ISO_ORDER, columns=cols)
    lim = np.nanmax(np.abs(t.values))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(t.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.3f}{stars(p.values[i, j])}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=20, ha="right")
    ax.set_yticks(range(len(ISO_ORDER)),
                  [f"{l} ({'CER' if l in ('zh', 'ja', 'ko') else 'WER'})" for l in ISO_ORDER])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="technique − control error\n(red = technique harder)",
                 shrink=0.8)
    ax.set_title("Whisper-large-v3: technique vs its own control take "
                 "(* p<.05, ** <.01, *** <.001, Holm)", loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ── B8b. Technique vs control: acoustic features ────────────────────────────

def technique_vs_control_features() -> pd.DataFrame:
    """Paired real − control test of each defining feature per language × group.

    Uses the calibration spans (train songs only, same phone in the technique
    and the control take). Glide size is compared in absolute cents (a glide
    can go either way). Spans are nested in clips and singers, so p-values
    are optimistic; the per-singer sign column is the conservative check.
    """
    df = load_all(REPO / "results" / "calibration")
    df = df[df["source"].isin(["real", "control"])]
    rows = []
    for group, feats in KEY_FEATURES.items():
        g = df[df["group"] == group]
        for feat, _, label in feats:
            if feat == "glide_dur_s":   # fixed by the span; not a contrast
                continue
            w = g.pivot_table(index=KEY, columns="source", values=feat, aggfunc="first")
            if not {"real", "control"} <= set(w.columns):
                continue
            w = w.dropna(subset=["real", "control"]).reset_index()
            if feat == "glide_cents":
                w[["real", "control"]] = w[["real", "control"]].abs()
                label = "|glide size| (cents)"
            for lang, c in w.groupby("language"):
                if len(c) < 20:
                    continue
                d = c["real"] - c["control"]
                sing = d.groupby(c["singer"]).median()
                med = d.median()
                rows.append({
                    "language": LANG_TO_ISO[lang], "technique": GROUPS[group],
                    "feature": label, "n_spans": len(c), "median_diff": med,
                    "iqr_low": d.quantile(0.25), "iqr_high": d.quantile(0.75),
                    "p_wilcoxon": wilcoxon(c["real"], c["control"],
                                           zero_method="zsplit").pvalue,
                    "singers_same_sign": f"{int((np.sign(sing) == np.sign(med)).sum())}/{len(sing)}"})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_wilcoxon"].tolist())
    return res


# ── FT. Fine-tuning ablation figures ─────────────────────────────────────────

def plot_ft_runs(summ: pd.DataFrame, out: Path) -> None:
    """Mean test error per run with song-bootstrap CI, one panel per language."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.2), sharex=True)
    for ax, iso in zip(axes.flat, ISO_ORDER):
        # Only the runs that have finished (the matrix may still be running).
        runs = [r for r in RUNS if r in set(summ["run"])]
        s = summ[summ["language"] == iso].set_index("run").reindex(runs)
        x = np.arange(len(runs))
        for xi, run in zip(x, runs):
            r = s.loc[run]
            if np.isnan(r["mean_clip"]):
                continue
            ax.errorbar(xi, r["mean_clip"], yerr=[[r["mean_clip"] - r.ci_low_clip],
                                                  [r.ci_high_clip - r["mean_clip"]]],
                        fmt="o", color=RUN_COLORS[run], ms=5, capsize=2.5, lw=1.2)
        ax.axhline(s.loc["E0", "mean_clip"], color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"{iso} ({s['metric'].iloc[0].upper()})", loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xticks(x, runs, rotation=45)
    fig.suptitle("Fine-tuning ablation: test error per run, per-clip error capped at 1 "
                 "(95 % song-bootstrap CI; dashed = no fine-tuning)", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_ft_relative(summ: pd.DataFrame, comp: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Heatmap of relative change of mean error vs E0 (%), per language × run."""
    m = summ.pivot(index="language", columns="run", values="mean_clip").reindex(ISO_ORDER)
    rel = (m.div(m["E0"], axis=0) - 1).mul(100)[[r for r in RUNS if r != "E0" and r in m]]
    lim = np.nanmax(np.abs(rel.values))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(rel.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(rel.shape[1]), rel.columns)
    ax.set_yticks(range(len(ISO_ORDER)), ISO_ORDER)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="change of mean error vs no fine-tuning (%)", shrink=0.8)
    ax.set_title("Relative change vs E0 (blue = better)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return rel


def plot_ft_comparisons(comp: pd.DataFrame, out: Path) -> None:
    """Forest plot of the planned paired comparisons, per language."""
    order = [c for c in ["R vs E0", "RC vs R", "RCA vs RC", "RCA vs RCA_v1", "R20CA vs R20C",
                         "CA vs C", "CA vs CA_v1", "RC vs CA"] if c in set(comp["comparison"])]
    fig, axes = plt.subplots(1, len(order), figsize=(2.4 * len(order), 3.8), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, order):
        c = comp[comp["comparison"] == name].set_index("language").reindex(ISO_ORDER)
        y = np.arange(len(ISO_ORDER))[::-1]
        for yi, (iso, r) in zip(y, c.iterrows()):
            if np.isnan(r["mean_diff_clip"]):
                continue
            sig = r["p_holm"] < 0.05
            col = (BLUE if r["mean_diff_clip"] < 0 else ORANGE) if sig else MUTED
            ax.errorbar(r["mean_diff_clip"], yi,
                        xerr=[[r["mean_diff_clip"] - r.ci_low_clip],
                              [r.ci_high_clip - r["mean_diff_clip"]]],
                        fmt="o", color=col, ms=4.5, capsize=2, lw=1.1,
                        mfc=col if sig else "white")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(name, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_yticks(np.arange(len(ISO_ORDER))[::-1], ISO_ORDER)
        ax.set_xlabel("A − B error (capped at 1)")
    fig.suptitle("Paired comparisons (negative = A better; filled = p<.05 Holm, "
                 "blue better / orange worse)", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ft_by_technique() -> pd.DataFrame:
    """Mean test error per technique for every run, all languages.

    Errors are relative to each language's E0 mean on that technique, so
    WER and CER languages can be pooled.
    """
    rows = []
    for iso in ISO_ORDER:
        metric = primary_metric(ISO_TO_LANGUAGE[iso])
        for run in RUNS:
            df = load_run(iso, run)
            if df is None:
                continue
            df["technique"] = df["audio_path"].map(group_of).map(GROUPS)
            for tech, g in df.groupby("technique"):
                rows.append({"language": iso, "run": run, "technique": tech,
                             "n": len(g), "mean": g[metric].clip(upper=1.0).mean()})
    d = pd.DataFrame(rows)
    e0 = d[d["run"] == "E0"].set_index(["language", "technique"])["mean"]
    d["rel_vs_E0"] = d.apply(lambda r: r["mean"] / e0[(r.language, r.technique)] - 1, axis=1)
    return d


def plot_ft_by_technique(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Median (over languages) relative change vs E0 per technique and run."""
    runs = ["R", "RC", "RCA", "RCA_v1"]
    t = (d[d["run"].isin(runs)].groupby(["technique", "run"])["rel_vs_E0"].median()
         .unstack().reindex(index=TECH_ORDER, columns=runs) * 100)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.2
    for k, run in enumerate(runs):
        ax.bar(np.arange(len(t)) + (k - 1.5) * w, t[run], w, color=RUN_COLORS[run], label=run)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(t)), t.index)
    ax.set_ylabel("change vs E0 (%, median over languages)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.set_title("Where fine-tuning helps, per technique (negative = better)",
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return t


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    md = ["# Paper statistics (dataset, technique vs control, fine-tuning)", "",
          "Generated by `scripts/paper_stats.py`. Metric = CER for zh/ja/ko, WER "
          "otherwise. CI = 95 % bootstrap over songs. p = Wilcoxon signed-rank "
          "(paired), Holm-corrected. "
          "\"Capped\" = per-clip error limited to 1, so that a Whisper repetition "
          "loop (error up to 40 on one clip) counts as one wrong phrase.", ""]

    print("B9 dataset composition (reading WAV headers)…")
    inv = load_inventories()
    hours, singers, lines = dataset_tables(inv)
    hours.to_csv(OUT / "dataset_hours.csv")
    singers.to_csv(OUT / "dataset_singers.csv")
    plot_dataset(hours, singers, FIG / "dataset_hours.png")
    md += lines

    print("B8a technique vs control: Whisper error…")
    tc, tc_singer = technique_vs_control_error(rng)
    tc.to_csv(OUT / "tech_vs_control_error.csv", index=False)
    tc_singer.to_csv(OUT / "tech_vs_control_error_per_singer.csv", index=False)
    plot_tc_error(tc, FIG / "tech_vs_control_error.png")
    show = tc.assign(
        diff=tc.apply(lambda r: f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1),
        diff_capped=tc.apply(lambda r: f"{r.mean_diff_clip:+.3f} "
                             f"[{r.ci_low_clip:+.3f}, {r.ci_high_clip:+.3f}]", axis=1),
        p=tc["p_holm"].map(lambda p: f"{p:.2g}"))
    # Direction from the capped mean (raw means can flip sign on one loop).
    n_sig_worse = int(((tc["p_holm"] < 0.05) & (tc["mean_diff_clip"] > 0)).sum())
    n_sig_better = int(((tc["p_holm"] < 0.05) & (tc["mean_diff_clip"] < 0)).sum())
    md += ["## B8a. Is the technique take harder for Whisper than its control?", "",
           "Whisper-large-v3 (no fine-tuning), whole corpus, technique take paired "
           "with the control take of the same phrase and singer. "
           f"Significantly harder: {n_sig_worse}/{len(tc)} cells; significantly "
           f"easier: {n_sig_better}/{len(tc)}. English baseline predates the "
           "Falsetto/Mixed split → reported together.", "",
           "![tech vs control](figures/tech_vs_control_error.png)", "",
           show[["language", "technique", "n_pairs", "technique_err", "control_err", "diff",
                 "diff_capped",
                 "tech_worse_share", "singers_tech_worse", "p"]]
           .round(3).to_markdown(index=False), ""]

    print("B8b technique vs control: acoustic features…")
    feat = technique_vs_control_features()
    feat.to_csv(OUT / "tech_vs_control_features.csv", index=False)
    fshow = feat.assign(
        median_diff=feat["median_diff"].map(lambda v: f"{v:+.3g}"),
        iqr=feat.apply(lambda r: f"[{r.iqr_low:+.3g}, {r.iqr_high:+.3g}]", axis=1),
        p=feat["p_holm"].map(lambda p: f"{p:.2g}"))
    md += ["## B8b. Is the technique acoustically different from its control?", "",
           "Calibration spans (train songs; same phone in technique and control "
           "take). Spans are nested in clips/singers, so p-values are optimistic: "
           "read them with the per-singer sign column. "
           f"{int((feat['p_holm'] < 0.05).sum())}/{len(feat)} tests significant.", ""]
    for tech in TECH_ORDER:
        sub = fshow[fshow["technique"] == tech]
        if sub.empty:
            continue
        md += [f"### {tech}", "",
               sub[["feature", "language", "n_spans", "median_diff", "iqr",
                    "singers_same_sign", "p"]].to_markdown(index=False), ""]

    print("FT fine-tuning figures…")
    ft = REPO / "results" / "finetune"
    summ, comp = pd.read_csv(ft / "summary.csv"), pd.read_csv(ft / "comparisons.csv")
    plot_ft_runs(summ, FIG / "ft_runs.png")
    rel = plot_ft_relative(summ, comp, FIG / "ft_relative.png")
    plot_ft_comparisons(comp, FIG / "ft_comparisons.png")
    by_tech = ft_by_technique()
    by_tech.to_csv(OUT / "ft_by_technique.csv", index=False)
    t = plot_ft_by_technique(by_tech, FIG / "ft_by_technique.png")
    md += ["## Fine-tuning ablation", "",
           "Numbers: results/finetune/summary.md.", "",
           "![runs](figures/ft_runs.png)", "",
           "Relative change of mean error vs E0 (%):", "",
           rel.round(1).to_markdown(), "",
           "![relative](figures/ft_relative.png)", "",
           "![comparisons](figures/ft_comparisons.png)", "",
           "Per technique (median over languages of the change vs E0, %):", "",
           t.round(1).to_markdown(), "",
           "![by technique](figures/ft_by_technique.png)", ""]

    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"→ {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
