"""
explorer.py — "Dataset" tab: listen to and inspect real GTSinger takes.

For one phrase it shows the control take, the real technique take(s) of the
same phrase (two for Mixed_Voice_and_Falsetto) and the paired speech, each
with its spectrogram / F0 / technique-flag plot.
"""

from __future__ import annotations

import gradio as gr

from alt.augment.io import load_audio, load_meta
from alt.plots import plot_clip
from app import state
from app.tabs.common import clip_picker, close, init_picker

# Group slots shown side by side; a technique folder has one or two real groups.
_SLOTS = ["Control_Group", "tech_1", "tech_2", "Paired_Speech_Group"]


def _slot_groups(tech_folder: str) -> list[str | None]:
    """Map the four UI slots to concrete group folders for a technique folder."""
    from alt.augment.pairing import TECH_GROUPS
    tech = TECH_GROUPS.get(tech_folder, [])
    return ["Control_Group", tech[0] if tech else None,
            tech[1] if len(tech) > 1 else None, "Paired_Speech_Group"]


def build(demo: gr.Blocks) -> None:
    """Create the tab's components and events inside the current Blocks."""
    gr.Markdown("Pick a phrase: the **control** take and the **real technique** "
                "take are the same singer singing the same phrase — the pairs "
                "the augmentation is calibrated on.")
    picker = clip_picker()
    lyrics = gr.Markdown()
    show_plots = gr.Checkbox(True, label="Show spectrogram / F0 plots (slower)")

    # One row per group: player on the left, wide plot on the right.
    audios, plots = [], []
    for _ in _SLOTS:
        with gr.Row(equal_height=True):
            audios.append(gr.Audio(type="filepath", interactive=False, scale=1))
            plots.append(gr.Plot(show_label=False, scale=3))

    def render(lang, singer, tech, song, stem, with_plots):
        """Load every group take of the phrase into its slot."""
        out_audio, out_plot = [], []
        text = ""
        for group in _slot_groups(tech):
            path = state.take(lang, singer, tech, song, group, stem) if group else None
            if path is None:
                out_audio.append(gr.update(value=None, label=group or "—",
                                           visible=group is not None))
                out_plot.append(None)
                continue
            label = group.replace("_Group", "").replace("_", " ")
            out_audio.append(gr.update(value=str(path), label=label, visible=True))
            meta_path = path.with_suffix(".json")
            if not text:
                text = state.lyrics(meta_path)
            if with_plots:
                wav, sr = load_audio(path)
                meta = load_meta(meta_path) if meta_path.exists() else None
                out_plot.append(close(plot_clip(wav, sr, meta, title=label)))
            else:
                out_plot.append(None)
        return [f"**Lyrics:** {text}"] + out_audio + out_plot

    gr.on([picker.stem.change, show_plots.change], render,
          picker.inputs + [show_plots], [lyrics] + audios + plots)
    init_picker(demo, picker)
