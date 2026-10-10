"""
common.py — UI building blocks shared by several tabs.

``clip_picker`` builds the cascading language → singer → technique → song →
phrase dropdowns every tab uses to select a GTSinger phrase.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
import matplotlib.pyplot as plt

from app import state


@dataclass
class ClipPicker:
    """Handles of the cascading dropdowns (in selection order)."""
    language: gr.Dropdown
    singer: gr.Dropdown
    tech_folder: gr.Dropdown
    song: gr.Dropdown
    stem: gr.Dropdown
    split_badge: gr.Markdown

    @property
    def inputs(self) -> list:
        """The five dropdowns, as event inputs."""
        return [self.language, self.singer, self.tech_folder, self.song, self.stem]


def _dd(value, choices) -> dict:
    """Dropdown update that keeps ``value`` if still valid, else picks the first."""
    value = value if value in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def clip_picker(default_language: str = "english") -> ClipPicker:
    """Create the cascading phrase selector and wire its change events.

    Args:
        default_language: Language selected at start-up.

    Returns:
        The ``ClipPicker`` handles.
    """
    df = state.real()
    langs = state.options(df, "language")
    with gr.Row():
        language = gr.Dropdown(langs, value=default_language, label="Language")
        # Choices are filled at run time by the cascade; allow_custom_value
        # keeps Gradio from rejecting values it has not seen at build time
        # (e.g. direct API calls that skip the cascade events).
        singer = gr.Dropdown([], label="Singer", allow_custom_value=True)
        tech_folder = gr.Dropdown([], label="Technique folder", allow_custom_value=True)
        song = gr.Dropdown([], label="Song", allow_custom_value=True)
        stem = gr.Dropdown([], label="Phrase", allow_custom_value=True)
    split_badge = gr.Markdown()

    def _filter(lang, sng=None, tech=None, sg=None):
        """Index rows matching the selections made so far."""
        d = df[df["language"] == lang]
        if sng is not None:
            d = d[d["singer"] == sng]
        if tech is not None:
            d = d[d["tech_folder"] == tech]
        if sg is not None:
            d = d[d["song"] == sg]
        return d

    def on_language(lang, sng):
        return _dd(sng, state.options(_filter(lang), "singer"))

    def on_singer(lang, sng, tech):
        return _dd(tech, state.options(_filter(lang, sng), "tech_folder"))

    def on_tech(lang, sng, tech, sg):
        return _dd(sg, state.options(_filter(lang, sng, tech), "song"))

    def on_song(lang, sng, tech, sg, st):
        d = _filter(lang, sng, tech, sg)
        # A phrase exists if its control take exists (the pairing unit).
        stems = state.options(d[d["group"] == "Control_Group"], "stem") \
            or state.options(d, "stem")
        split = d["split"].iloc[0] if len(d) else "?"
        badge = ("🟥 **test song** — held out from every training set"
                 if split == "test" else "🟩 **train song**")
        return _dd(st, stems), badge

    language.change(on_language, [language, singer], singer)
    singer.change(on_singer, [language, singer, tech_folder], tech_folder)
    tech_folder.change(on_tech, [language, singer, tech_folder, song], song)
    song.change(on_song, [language, singer, tech_folder, song, stem],
                [stem, split_badge])

    picker = ClipPicker(language, singer, tech_folder, song, stem, split_badge)
    return picker


def init_picker(demo: gr.Blocks, picker: ClipPicker) -> None:
    """Populate the cascade once when the page loads."""
    demo.load(lambda lang: _dd(None, state.options(
        state.real()[state.real()["language"] == lang], "singer")),
        picker.language, picker.singer)


def close(fig: plt.Figure) -> plt.Figure:
    """Return a figure for Gradio and drop pyplot's reference to it.

    Gradio renders the figure object it receives; closing it in pyplot's
    registry prevents the figure manager from leaking memory across clicks.
    """
    plt.close(fig)
    return fig
