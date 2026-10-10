"""
asr.py — "ASR" tab: transcribe a dataset phrase, an upload or a recording with
any of the thesis models and score it against the reference lyrics.

Models: the zero-shot baselines of the registry (Whisper small / large-v2 /
large-v3, wav2vec2 — English only) and every fine-tuned Whisper checkpoint
found under results/finetune_whisper*/best_model. Only one model is kept in
GPU memory: switching model unloads the previous one (each large model is
~3 GB in fp16).
"""

from __future__ import annotations

import difflib
import html
import tempfile
import os
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from alt.asr import get_asr_model
from alt.metrics import score_pair
from alt.text import clean_text, primary_metric
from app import state
from app.tabs.common import clip_picker, init_picker

BASE_MODELS = ["whisper_largev3", "whisper_largev2", "whisper_small", "wav2vec2"]
ISO = {"english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
       "russian": "ru", "spanish": "es", "french": "fr", "german": "de",
       "italian": "it"}

_lock = threading.Lock()          # one transcription at a time (shared GPU)
_loaded: dict = {"key": None, "model": None}


def finetuned_models() -> dict[str, str]:
    """Display name → checkpoint dir of every fine-tuned Whisper on disk."""
    out = {}
    for best in sorted(state.REPO.glob("results/finetune_whisper*/best_model")):
        if (best / "config.json").exists():
            out[f"finetuned: {best.parent.name.replace('finetune_whisper', '') or '_pilot'}"] = str(best)
    return out


def _get_model(name: str, language: str):
    """Load (or reuse) a model; unloads the previous one to free VRAM."""
    key = (name, language)
    if _loaded["key"] == key:
        return _loaded["model"]
    if _loaded["model"] is not None:
        _loaded["model"].unload()
    ft = finetuned_models()
    if name in ft:
        m = get_asr_model("whisper_finetuned", device="cuda", batch_size=1,
                          language=ISO[language], model_id=ft[name])
    else:
        m = get_asr_model(name, device="cuda", batch_size=1, language=ISO[language])
    _loaded.update(key=key, model=m)
    return m


def word_diff(ref: str, hyp: str, language: str) -> str:
    """HTML diff of normalised reference vs hypothesis.

    Words for space-delimited languages, characters for zh/ja/ko. Deleted
    reference tokens are struck through (red), inserted ones green.
    """
    r, h = clean_text(ref, language), clean_text(hyp, language)
    char_level = primary_metric(language) == "cer"
    rt = list(r.replace(" ", "")) if char_level else r.split()
    ht = list(h.replace(" ", "")) if char_level else h.split()
    sep = "" if char_level else " "
    parts = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=rt, b=ht).get_opcodes():
        ref_seg = html.escape(sep.join(rt[a0:a1]))
        hyp_seg = html.escape(sep.join(ht[b0:b1]))
        if op == "equal":
            parts.append(ref_seg)
        if op in ("delete", "replace"):
            parts.append(f"<del style='color:#c0392b'>{ref_seg}</del>")
        if op in ("insert", "replace"):
            parts.append(f"<ins style='color:#1e8449'>{hyp_seg}</ins>")
    return f"<div style='font-size:1.05em;line-height:1.7'>{sep.join(parts) or '—'}</div>"


def build(demo: gr.Blocks) -> None:
    """Create the tab's components and events inside the current Blocks."""
    gr.Markdown("Transcribe a GTSinger phrase (pick below) **or** upload / "
                "record your own audio. Fine-tuned models appear automatically "
                "when their `best_model/` exists.")
    picker = clip_picker()
    group = gr.Dropdown(state.GROUP_ORDER, value="Control_Group",
                        label="Take (group) of the phrase")
    with gr.Row():
        own_audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                             label="…or your own audio (overrides the phrase)")
        reference = gr.Textbox(label="Reference lyrics (optional for own audio)", lines=2)
    model = gr.Dropdown(BASE_MODELS + list(finetuned_models()),
                        value="whisper_largev3", label="Model")
    run = gr.Button("Transcribe", variant="primary")
    player = gr.Audio(label="Input", interactive=False)
    hyp = gr.Textbox(label="Hypothesis", lines=2)
    scores = gr.Markdown()
    diff = gr.HTML()

    def on_run(lang, singer, tech, song, stem, grp, own, ref_text, model_name):
        if model_name == "wav2vec2" and lang != "english":
            return None, "", "⚠️ wav2vec2 is an English-only model.", ""
        if own:
            path, ref = own, ref_text or ""
        else:
            path = state.take(lang, singer, tech, song, grp, stem)
            if path is None:
                return None, "", f"⚠️ No {grp} take for this phrase.", ""
            path = str(path)
            ref = state.lyrics(Path(path).with_suffix(".json"))
        # Resample/convert through a temp WAV so any upload format works.
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav, sr)
        try:
            with _lock:
                text = _get_model(model_name, lang).transcribe([tmp.name])[0]
        finally:
            os.unlink(tmp.name)
        md = ""
        if ref:
            s = score_pair(ref, text, lang)
            metric = primary_metric(lang)
            md = (f"**{metric.upper()} = {s[metric]:.3f}**"
                  + (f" · WER = {s['wer']:.3f}" if metric != "wer" else "")
                  + f"  \n**Reference:** {html.escape(ref)}")
        return (sr, wav.astype(np.float32)), text, md, \
            (word_diff(ref, text, lang) if ref else "")

    run.click(on_run, picker.inputs + [group, own_audio, reference, model],
              [player, hyp, scores, diff])
    init_picker(demo, picker)
