"""
results.py — "Results" tab: browse the thesis numbers without re-running.

* Baselines: headline error (WER, or CER for zh/ja/ko) of every zero-shot
  model in every language, plus the per-technique breakdown as a chart.
* Fine-tuning ablation: the exp*_eval summaries (same song-disjoint test set).
* Calibration and pilot reports (markdown + figures), when generated.
"""

from __future__ import annotations

import re

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from app import state
from app.tabs.common import close

RESULTS = state.REPO / "results"
# Categorical slot 1 of the dataviz reference palette; bars are a single
# series, so one hue plus the title identifies them.
BAR_COLOR = "#2a78d6"


def baseline_table() -> pd.DataFrame:
    """One row per language × model with the headline metric."""
    rows = []
    for f in sorted(RESULTS.glob("baseline_*/asr_*_summary.csv")):
        s = pd.read_csv(f).iloc[0]
        metric = s["metric"] if isinstance(s.get("metric"), str) else "wer"
        rows.append({
            "language": f.parent.name.replace("baseline_", ""),
            "model": f.name.replace("asr_", "").replace("_summary.csv", ""),
            "metric": metric.upper(), "error": round(float(s[metric]), 3),
            "WER": round(float(s["wer"]), 3),
            "hallucination %": round(100 * float(s["hallucination_rate"]), 1),
            "n": int(s["n"])})
    return pd.DataFrame(rows)


def ablation_table() -> pd.DataFrame:
    """Fine-tuning experiments evaluated on the shared test manifest."""
    rows = []
    for f in sorted(RESULTS.glob("exp*_eval/asr_*_summary.csv")):
        s = pd.read_csv(f).iloc[0]
        rows.append({"experiment": f.parent.name.replace("_eval", ""),
                     "model": f.name.replace("asr_", "").replace("_summary.csv", ""),
                     "WER": round(float(s["wer"]), 3),
                     "hallucination %": round(100 * float(s["hallucination_rate"]), 1),
                     "n": int(s["n"])})
    return pd.DataFrame(rows)


def technique_chart(language: str, model: str):
    """Bar chart of the headline error per technique for one baseline run."""
    f = RESULTS / f"baseline_{language}" / f"asr_{model}_by_technique.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    metric = d["metric"].iloc[0] if "metric" in d and isinstance(d["metric"].iloc[0], str) else "wer"
    d = d.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(d["technique"], d[metric], color=BAR_COLOR, height=0.6)
    for y, v in enumerate(d[metric]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"{model} · {language}: {metric.upper()} by technique", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return close(fig)


def report_md(path) -> str:
    """Report markdown with image links stripped (figures shown in a gallery)."""
    if not path.exists():
        return f"*{path.relative_to(state.REPO)} not generated yet.*"
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", path.read_text(encoding="utf-8"))


def build(demo: gr.Blocks) -> None:
    """Create the tab's components and events inside the current Blocks."""
    with gr.Tab("Baselines"):
        base = baseline_table()
        gr.Dataframe(base, label="Zero-shot baselines (full corpus)")
        langs = sorted(base.language.unique()) if len(base) else []
        with gr.Row():
            lang = gr.Dropdown(langs, value="english" if "english" in langs else None,
                               label="Language")
            model = gr.Dropdown(sorted(base.model.unique()) if len(base) else [],
                                value="whisper_largev3", label="Model")
        chart = gr.Plot()
        gr.on([lang.change, model.change, demo.load], technique_chart,
              [lang, model], chart)
    with gr.Tab("Fine-tuning ablation"):
        gr.Markdown("English Whisper-large-v3 fine-tunes, evaluated on the test "
                    "manifest named in each configs/exp*_eval.yaml.")
        gr.Dataframe(ablation_table())
    with gr.Tab("Calibration"):
        gr.Markdown(report_md(RESULTS / "calibration" / "report.md"))
        gr.Gallery([str(p) for p in sorted((RESULTS / "calibration" / "figures").glob("*.png"))],
                   columns=2, height="auto", label="Figures")
    with gr.Tab("Pilot"):
        gr.Markdown(report_md(RESULTS / "pilot" / "report.md"))
