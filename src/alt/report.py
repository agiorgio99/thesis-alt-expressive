"""
report.py — generate a self-contained HTML report from pipeline result CSVs.

Reads every CSV in a results directory, produces tables and matplotlib figures
(embedded as base64 PNG), and writes a single ``report.html`` next to them.

Can be called directly::

    python -m alt.report results/baseline_whisper_largev3

or is invoked automatically at the end of ``Pipeline.run()``.
"""

from __future__ import annotations

import base64
import io
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── colour palette (consistent with the probe notebook) ──────────────────────
TECH_COLORS: dict[str, str] = {
    "breathy":       "#4C72B0",
    "glissando":     "#DD8452",
    "vibrato":       "#55A868",
    "mixed_falsetto":"#C44E52",
    "pharyngeal":    "#8172B2",
}
GROUP_COLORS: dict[str, str] = {
    "technique": "#2196F3",
    "control":   "#4CAF50",
    "speech":    "#FF9800",
}


# ── small helpers ─────────────────────────────────────────────────────────────
def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'


def _df_html(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    s = df.style.set_table_attributes('class="data-table"')
    if fmt:
        s = s.format(fmt, na_rep="—")
    return s.to_html()


def _section(title: str, content: str) -> str:
    return f'<section><h2>{title}</h2>{content}</section>\n'


def _subsection(title: str, content: str) -> str:
    return f'<div class="subsection"><h3>{title}</h3>{content}</div>\n'


def _tech_color(tech: str) -> str:
    return TECH_COLORS.get(tech, "#888888")


# ── figure builders ───────────────────────────────────────────────────────────

def _pie(sizes: list[float], labels: list[str], colors: list[str],
         title: str) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(linewidth=0.6, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontsize=11, pad=10)
    return _img(_fig_to_b64(fig), title)


def _bar(x: list, y: list, colors: list[str] | None, title: str,
         xlabel: str, ylabel: str, *, fmt: str = ".3f",
         ylim: tuple | None = None) -> str:
    fig, ax = plt.subplots(figsize=(max(5, len(x) * 0.9), 4))
    bars = ax.bar(x, y, color=colors or "#4C72B0", edgecolor="white", linewidth=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] * 0.015 if ylim else 0.005),
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    return _img(_fig_to_b64(fig), title)


def _grouped_bar(df: pd.DataFrame, group_col: str, metric_col: str,
                 title: str, ylabel: str, color_map: dict[str, str]) -> str:
    groups = df[group_col].unique()
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 0.9), 4))
    bars = ax.bar(x, df.set_index(group_col)[metric_col].reindex(groups),
                  color=[color_map.get(g, "#888") for g in groups],
                  edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, df.set_index(group_col)[metric_col].reindex(groups)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    return _img(_fig_to_b64(fig), title)


def _hist(values: list[float], title: str, xlabel: str, color: str = "#4C72B0",
          bins: int = 20) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    fig.tight_layout()
    return _img(_fig_to_b64(fig), title)


def _boxplot(df: pd.DataFrame, group_col: str, value_col: str,
             title: str, ylabel: str, color_map: dict[str, str]) -> str:
    groups = [g for g in df[group_col].unique() if pd.notna(g)]
    data = [df[df[group_col] == g][value_col].dropna().tolist() for g in groups]
    colors = [color_map.get(g, "#888") for g in groups]
    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.1), 4))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    fig.tight_layout()
    return _img(_fig_to_b64(fig), title)


def _within_ms_bar(summary: pd.Series, title: str) -> str:
    cols = ["within_20ms", "within_50ms", "within_100ms"]
    labels = ["≤ 20 ms", "≤ 50 ms", "≤ 100 ms"]
    vals = [float(summary.get(c, 0)) for c in cols]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Fraction of boundaries", fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _img(_fig_to_b64(fig), title)


# ── section builders ──────────────────────────────────────────────────────────

def _build_inventory(out_dir: Path) -> str:
    p = out_dir / "inventory.csv"
    if not p.exists():
        return ""
    df = pd.read_csv(p)
    total = len(df)

    tech_counts = df["technique"].value_counts()
    group_counts = df["group"].value_counts() if "group" in df.columns else pd.Series()

    html = f"<p><strong>Total utterances:</strong> {total}</p>\n"
    html += '<div class="flex-row">\n'

    # Pie: by technique
    tc_colors = [_tech_color(t) for t in tech_counts.index]
    html += _pie(tech_counts.values.tolist(), tech_counts.index.tolist(),
                 tc_colors, "Utterances by technique")

    # Pie: by group
    if not group_counts.empty:
        gc_colors = [GROUP_COLORS.get(g, "#888") for g in group_counts.index]
        html += _pie(group_counts.values.tolist(), group_counts.index.tolist(),
                     gc_colors, "Utterances by group")

    html += "</div>\n"

    if "group" in df.columns:
        pivot = df.groupby(["technique", "group"]).size().unstack(fill_value=0)
        html += _subsection("Count by technique × group",
                             _df_html(pivot))
    return _section("Dataset overview", html)


def _build_asr(out_dir: Path) -> str:
    # Discover every ASR model by scanning for asr_<model>.csv
    models = sorted({
        re.sub(r"^asr_", "", p.stem)
        for p in out_dir.glob("asr_*.csv")
        if "by_" not in p.stem and "summary" not in p.stem
    })
    if not models:
        return ""

    content = ""
    for model in models:
        per_utt_path = out_dir / f"asr_{model}.csv"
        summary_path = out_dir / f"asr_{model}_summary.csv"
        by_tech_path = out_dir / f"asr_{model}_by_technique.csv"
        by_singer_path = out_dir / f"asr_{model}_by_singer_id.csv"

        inner = ""

        # Summary tile
        if summary_path.exists():
            s = pd.read_csv(summary_path).iloc[0]
            inner += f"""
<div class="metric-tiles">
  <div class="tile"><span class="tile-val">{s['wer']:.3f}</span><span class="tile-lbl">WER</span></div>
  <div class="tile"><span class="tile-val">{s['per']:.3f}</span><span class="tile-lbl">PER</span></div>
  <div class="tile"><span class="tile-val">{s.get('hallucination_rate', 0):.1%}</span><span class="tile-lbl">Hallucination rate</span></div>
  <div class="tile"><span class="tile-val">{int(s['n'])}</span><span class="tile-lbl">Utterances</span></div>
</div>\n"""

        inner += '<div class="flex-row">\n'

        # Bar chart: WER by technique
        if by_tech_path.exists():
            bt = pd.read_csv(by_tech_path)
            inner += _grouped_bar(bt, "technique", "wer",
                                  "WER by technique", "WER",
                                  TECH_COLORS)

        # Bar chart: WER by singer
        if by_singer_path.exists():
            bs = pd.read_csv(by_singer_path).sort_values("wer")
            inner += _bar(bs["singer_id"].tolist(), bs["wer"].tolist(),
                          None, "WER by singer", "Singer", "WER",
                          ylim=(0, 1.05))

        # Histogram: per-utterance WER
        if per_utt_path.exists():
            du = pd.read_csv(per_utt_path)
            inner += _hist(du["wer"].dropna().tolist(),
                           "WER distribution (per utterance)", "WER",
                           color="#4C72B0")

        inner += "</div>\n"

        # Per-utterance table (truncated columns)
        if per_utt_path.exists():
            du = pd.read_csv(per_utt_path)
            show = [c for c in ["utt_id", "singer_id", "technique", "group",
                                "text", "hypothesis", "wer", "per"] if c in du.columns]
            inner += _subsection(
                "Per-utterance results",
                _df_html(du[show], {"wer": "{:.3f}", "per": "{:.3f}"}),
            )

        content += _subsection(f"Model: {model}", inner)

    return _section("ASR (speech recognition)", content)


def _build_alignment(out_dir: Path) -> str:
    summary_path = out_dir / "alignment_mfa_summary.csv"
    tbe_path = out_dir / "alignment_mfa_tbe.csv"
    if not summary_path.exists():
        return ""

    summary = pd.read_csv(summary_path)
    content = ""

    # Summary table
    content += _subsection("Overall TBE summary", _df_html(
        summary,
        {"mean_tbe": "{:.4f}", "median_tbe": "{:.4f}",
         "within_20ms": "{:.1%}", "within_50ms": "{:.1%}", "within_100ms": "{:.1%}"},
    ))

    # Within-Xms bar charts side by side
    content += '<div class="flex-row">\n'
    for _, row in summary.iterrows():
        content += _within_ms_bar(row, f"Boundary accuracy — {row['level']} level")
    content += "</div>\n"

    # Per-utterance table + TBE distribution
    if tbe_path.exists():
        tbe = pd.read_csv(tbe_path)
        content += '<div class="flex-row">\n'
        content += _hist(tbe["word_mean_tbe"].dropna().tolist(),
                         "Word TBE distribution", "Mean TBE (s)", "#2196F3")
        content += _hist(tbe["phone_mean_tbe"].dropna().tolist(),
                         "Phone TBE distribution", "Mean TBE (s)", "#FF9800")
        content += "</div>\n"
        content += _subsection("Per-utterance TBE", _df_html(
            tbe,
            {"word_mean_tbe": "{:.4f}", "word_within_50ms": "{:.1%}",
             "phone_mean_tbe": "{:.4f}", "phone_within_50ms": "{:.1%}"},
        ))

    return _section("Forced alignment (MFA)", content)


def _build_pitch(out_dir: Path) -> str:
    p = out_dir / "pitch_f0_stats.csv"
    if not p.exists():
        return ""

    df = pd.read_csv(p)
    if df.empty:
        return _section("Pitch (CREPE F0)", "<p>No pitch data available.</p>")
    content = ""

    metrics = [
        ("f0_mean_hz",    "F0 mean (Hz)"),
        ("f0_range_st",   "F0 range (semitones)"),
        ("voiced_ratio",  "Voiced ratio"),
        ("vibrato_index", "Vibrato index"),
    ]

    if "technique" in df.columns:
        content += '<div class="flex-row">\n'
        for col, label in metrics:
            if col in df.columns:
                content += _boxplot(df, "technique", col,
                                    label + " by technique", label, TECH_COLORS)
        content += "</div>\n"

        # Summary table by technique
        agg = df.groupby("technique")[[c for c, _ in metrics if c in df.columns]].mean()
        fmt = {c: "{:.2f}" for c in agg.columns}
        content += _subsection("Mean pitch stats by technique",
                               _df_html(agg.reset_index(), fmt))

    if "group" in df.columns:
        content += '<div class="flex-row">\n'
        for col, label in metrics:
            if col in df.columns:
                content += _boxplot(df, "group", col,
                                    label + " by group", label, GROUP_COLORS)
        content += "</div>\n"

    # Per-utterance table
    show = [c for c in ["utt_id", "singer_id", "technique", "group",
                         "f0_mean_hz", "f0_std_hz", "f0_range_st",
                         "voiced_ratio", "vibrato_index"] if c in df.columns]
    fmt = {c: "{:.3f}" for c in show if c not in ("utt_id", "singer_id", "technique", "group")}
    content += _subsection("Per-utterance F0 stats", _df_html(df[show], fmt))

    return _section("Pitch extraction (CREPE)", content)


# ── CSS + JS + HTML shell ─────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
       background: #f5f7fa; color: #222; }
header { background: #1a237e; color: white; padding: 24px 40px; }
header h1 { font-size: 1.6rem; font-weight: 600; }
header p  { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
main { max-width: 1300px; margin: 0 auto; padding: 32px 24px; }
section { background: white; border-radius: 8px; box-shadow: 0 1px 4px #0001;
          margin-bottom: 32px; padding: 24px 28px; }
section h2 { font-size: 1.15rem; font-weight: 600; color: #1a237e;
             border-bottom: 2px solid #e3e8f0; padding-bottom: 8px; margin-bottom: 18px; }
.subsection { margin-top: 20px; }
.subsection h3 { font-size: 0.95rem; font-weight: 600; color: #444;
                 margin-bottom: 10px; }
.flex-row { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start;
            margin: 14px 0; }
.flex-row > * { flex: 1 1 auto; min-width: 280px; }
.metric-tiles { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
.tile { background: #f0f4ff; border-radius: 8px; padding: 14px 20px;
        text-align: center; min-width: 110px; }
.tile-val { display: block; font-size: 1.5rem; font-weight: 700; color: #1a237e; }
.tile-lbl { display: block; font-size: 0.72rem; color: #666; margin-top: 2px; }
table.data-table { border-collapse: collapse; font-size: 12px; width: 100%;
                   overflow-x: auto; display: block; }
table.data-table th { background: #1a237e; color: white; padding: 6px 10px;
                      text-align: left; cursor: pointer; white-space: nowrap; }
table.data-table th:hover { background: #283593; }
table.data-table td { padding: 5px 10px; border-bottom: 1px solid #eee;
                      white-space: nowrap; }
table.data-table tr:nth-child(even) { background: #f8f9ff; }
table.data-table tr:hover { background: #e8eaf6; }
p { margin: 8px 0; }
"""

_JS = """
document.querySelectorAll('table.data-table').forEach(table => {
  const ths = table.querySelectorAll('th');
  ths.forEach((th, col) => {
    th.dataset.dir = 'asc';
    th.addEventListener('click', () => {
      const dir = th.dataset.dir === 'asc' ? 1 : -1;
      th.dataset.dir = dir === 1 ? 'desc' : 'asc';
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        const av = a.cells[col]?.innerText.trim() ?? '';
        const bv = b.cells[col]?.innerText.trim() ?? '';
        const an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return dir * (an - bn);
        return dir * av.localeCompare(bv);
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
});
"""


def _html_shell(title: str, body: str, generated: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Generated {generated}</p>
</header>
<main>
{body}
</main>
<script>{_JS}</script>
</body>
</html>"""


# ── public entry point ────────────────────────────────────────────────────────

def make_report(out_dir: str | Path) -> Path:
    """Build a self-contained HTML report from the CSVs in ``out_dir``.

    Args:
        out_dir: The experiment results directory (contains the pipeline CSVs).

    Returns:
        Path to the written ``report.html``.
    """
    out_dir = Path(out_dir)
    experiment = out_dir.name
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    body = ""
    body += _build_inventory(out_dir)
    body += _build_asr(out_dir)
    body += _build_alignment(out_dir)
    body += _build_pitch(out_dir)

    html = _html_shell(f"Baseline report — {experiment}", body, generated)
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[report] written → {report_path}")
    return report_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m alt.report <results_dir>")
        sys.exit(1)
    make_report(sys.argv[1])
