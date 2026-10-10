"""
aug_lab.py — "Augmentation lab" tab: synthesise a technique on a control take
and compare it with the real technique take of the same phrase.

Engines come from ``alt.augment.engines.ENGINES`` (legacy today; world_v2 and
psola are added there and appear here automatically). Parameters are edited as
JSON, pre-filled from the selected preset, so every engine's full parameter set
is reachable without a bespoke form per technique.
"""

from __future__ import annotations

import json

import gradio as gr
import numpy as np

from alt.augment.engines import ENGINES, augment
from alt.augment.io import load_audio, load_meta
from alt.augment.pairing import TECH_GROUPS
from alt.plots import plot_clip
from app import state
from app.tabs.common import clip_picker, close, init_picker


def _peak_normalise(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale down (never up) so playback does not clip; loudness otherwise kept."""
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav * (peak / m) if m > peak else wav


def build(demo: gr.Blocks) -> None:
    """Create the tab's components and events inside the current Blocks."""
    gr.Markdown("Choose a phrase, a target technique and an engine, then "
                "**Generate**. Compare the result with the **real** technique "
                "take of the same phrase (bottom). Prefer 🟩 train songs: 🟥 test "
                "songs are never augmented for training.")
    picker = clip_picker()
    engine_names = list(ENGINES)
    with gr.Row():
        group = gr.Dropdown([], label="Target technique group", allow_custom_value=True)
        engine = gr.Dropdown(engine_names, value=engine_names[0], label="Engine")
        preset = gr.Dropdown([], label="Preset / variant", allow_custom_value=True)
    engine_info = gr.Markdown()
    params = gr.Code(language="json", label="Parameters (editable)", lines=4)
    generate = gr.Button("Generate", variant="primary")
    status = gr.Markdown()

    with gr.Row(equal_height=True):
        ctrl_audio = gr.Audio(type="filepath", label="Control (input)",
                              interactive=False, scale=1)
        ctrl_plot = gr.Plot(show_label=False, scale=3)
    with gr.Row(equal_height=True):
        aug_audio = gr.Audio(label="Augmented", interactive=False, scale=1)
        aug_plot = gr.Plot(show_label=False, scale=3)
    with gr.Row(equal_height=True):
        real_audio = gr.Audio(type="filepath", label="Real technique take",
                              interactive=False, scale=1)
        real_plot = gr.Plot(show_label=False, scale=3)

    def on_tech(tech, eng):
        """Target groups = real groups of the folder that the engine supports."""
        supported = [g for g in TECH_GROUPS.get(tech, []) if g in ENGINES[eng].groups]
        unsupported = [g for g in TECH_GROUPS.get(tech, []) if g not in supported]
        note = f"*{ENGINES[eng].description}*"
        if unsupported:
            note += f"  \n⚠️ `{eng}` cannot synthesise: {', '.join(unsupported)}"
        return gr.update(choices=supported, value=supported[0] if supported else None), note

    def on_group(grp, eng, lang):
        """List the engine's presets for the chosen group."""
        if not grp:
            return gr.update(choices=[], value=None)
        presets = ENGINES[eng].presets(grp, lang)
        names = [f"v{k}" for k in range(len(presets))]
        return gr.update(choices=names, value=names[0] if names else None)

    def on_preset(grp, eng, lang, name):
        """Fill the JSON editor with the selected preset."""
        if not grp or not name:
            return ""
        presets = ENGINES[eng].presets(grp, lang)
        return json.dumps(presets[int(name[1:])], indent=1)

    def on_clip(lang, singer, tech, song, stem, grp):
        """Show the control and real takes of the selected phrase."""
        ctrl = state.take(lang, singer, tech, song, "Control_Group", stem)
        real = state.take(lang, singer, tech, song, grp, stem) if grp else None
        out = []
        for path, label in ((ctrl, "Control"), (real, f"Real {grp or ''}")):
            if path is None:
                out += [None, None]
                continue
            wav, sr = load_audio(path)
            meta_path = path.with_suffix(".json")
            meta = load_meta(meta_path) if meta_path.exists() else None
            out += [str(path), close(plot_clip(wav, sr, meta, title=label))]
        return out

    def on_generate(lang, singer, tech, song, stem, grp, eng, params_json):
        """Run the engine on the control take."""
        ctrl = state.take(lang, singer, tech, song, "Control_Group", stem)
        if ctrl is None or not grp:
            return None, None, "⚠️ Select a phrase that has a control take."
        try:
            p = json.loads(params_json or "{}")
        except json.JSONDecodeError as e:
            return None, None, f"⚠️ Invalid JSON: {e}"
        try:
            res = augment(eng, ctrl, grp, p, lang)
        except Exception as e:  # surface engine errors in the UI, not a traceback
            return None, None, f"⚠️ {type(e).__name__}: {e}"
        wav = _peak_normalise(res.wav)
        fig = close(plot_clip(wav, res.sr, res.meta,
                              title=f"{eng} → {grp} {json.dumps(p)}"))
        return (res.sr, wav.astype(np.float32)), fig, \
            f"✅ {eng} · {res.sr // 1000} kHz · {len(wav) / res.sr:.1f} s"

    picker.tech_folder.change(on_tech, [picker.tech_folder, engine], [group, engine_info])
    engine.change(on_tech, [picker.tech_folder, engine], [group, engine_info])
    group.change(on_group, [group, engine, picker.language], preset)
    preset.change(on_preset, [group, engine, picker.language, preset], params)
    gr.on([picker.stem.change, group.change], on_clip, picker.inputs + [group],
          [ctrl_audio, ctrl_plot, real_audio, real_plot])
    generate.click(on_generate, picker.inputs + [group, engine, params],
                   [aug_audio, aug_plot, status])
    init_picker(demo, picker)
