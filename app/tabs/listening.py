"""
listening.py — "Listening test" tab: blind rating of pilot clips.

Each rater gets a seeded, shuffled session mixing real technique takes
(anchors), control takes and the augmented clips of every engine. Audio is
served as raw samples (Gradio writes a randomly named temp file), so neither
the file name nor the engine is visible. For each clip the rater gives:

* naturalness, 1 (clearly synthetic / robotic) … 5 (sounds like a real singer)
* which technique they hear (or "none")

Ratings are appended to results/listening/ratings.csv (file-locked, one row
per answer) and summarised per engine × technique at the bottom of the tab.
"""

from __future__ import annotations

import csv
import fcntl
import random
import zlib
from datetime import datetime, timezone

import gradio as gr
import numpy as np
import pandas as pd

from alt.augment.io import load_audio
from app import state

MANIFEST = state.REPO / "results" / "pilot" / "manifest.csv"
RATINGS = state.REPO / "results" / "listening" / "ratings.csv"
TECHNIQUES = ["vibrato", "breathy", "glissando", "pharyngeal", "falsetto",
              "mixed voice", "none (plain singing)"]
FIELDS = ["timestamp", "rater", "session_seed", "trial", "language", "group",
          "source", "variant", "clip", "naturalness", "technique_heard"]


# Technique groups of the pilot and their answer label in the rating form.
GROUP_LABELS = {"Vibrato_Group": "vibrato", "Breathy_Group": "breathy",
                "Glissando_Group": "glissando", "Pharyngeal_Group": "pharyngeal",
                "Falsetto_Group": "falsetto", "Mixed_Voice_Group": "mixed voice"}
# Techniques left out of a session by default (none: all six are rated).
DEFAULT_EXCLUDED: list[str] = []


def build_trials(rater: str, n: int = 40, languages: list[str] | None = None,
                 groups: list[str] | None = None) -> list[dict]:
    """Balanced, seeded trial list for one rater.

    Sources per phrase: the real technique take, the control take and the
    typical variant (v1) of every engine. Phrases are sampled so that each
    source type appears about equally often.

    Args:
        rater:     Rater ID (seeds the order, so a rater can resume).
        n:         Number of clips in the session.
        languages: Restrict to these languages (default: all in the pilot).
        groups:    Restrict to these technique groups (default: all).

    Returns:
        Trial dicts (clip path + hidden metadata).
    """
    man = pd.read_csv(MANIFEST)
    if languages:
        man = man[man.language.isin(languages)]
    if groups:
        man = man[man.group.isin(groups)]
    typical = man[man.variant == "v1"]
    items = []
    for _, r in typical.iterrows():
        items.append({"clip": r.out_wav, "source": r.engine, "variant": "v1",
                      "language": r.language, "group": r.group})
    for _, r in man.drop_duplicates("control_wav").iterrows():
        items.append({"clip": r.real_wav, "source": "real", "variant": "",
                      "language": r.language, "group": r.group})
        items.append({"clip": r.control_wav, "source": "control", "variant": "",
                      "language": r.language, "group": r.group})
    # crc32, not hash(): Python salts str hashes per process, which would give
    # a returning rater a different order.
    seed = zlib.crc32(rater.encode("utf-8")) if rater else 0
    rng = random.Random(seed)
    by_src: dict[str, list[dict]] = {}
    for it in items:
        by_src.setdefault(it["source"], []).append(it)
    for v in by_src.values():
        rng.shuffle(v)
    # Round-robin over sources → balanced session, then shuffle the order.
    trials, i = [], 0
    while len(trials) < n and any(by_src.values()):
        src = list(by_src)[i % len(by_src)]
        if by_src[src]:
            trials.append({**by_src[src].pop(), "session_seed": seed})
        i += 1
    rng.shuffle(trials)
    return trials


def append_rating(row: dict) -> None:
    """Append one answer to the ratings CSV (safe with concurrent raters)."""
    RATINGS.parent.mkdir(parents=True, exist_ok=True)
    new = not RATINGS.exists()
    with open(RATINGS, "a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
        fcntl.flock(fh, fcntl.LOCK_UN)


def summary() -> pd.DataFrame:
    """Mean naturalness and technique-recognition rate per source × group."""
    if not RATINGS.exists():
        return pd.DataFrame({"info": ["no ratings yet"]})
    r = pd.read_csv(RATINGS)
    target = r.group.str.replace("_Group", "").str.replace("_", " ").str.lower()
    r["recognised"] = r.technique_heard.str.lower() == target
    return (r.groupby(["source", "group"])
            .agg(n=("naturalness", "size"), naturalness=("naturalness", "mean"),
                 recognised=("recognised", "mean"))
            .round(2).reset_index())


def _audio(trial: dict):
    """Raw samples of a trial clip (hides the path from the rater)."""
    wav, sr = load_audio(state.REPO / trial["clip"])
    peak = float(np.max(np.abs(wav))) or 1.0
    return sr, (wav / max(peak, 1.0) * 0.95).astype(np.float32)


def build(demo: gr.Blocks) -> None:
    """Create the tab's components and events inside the current Blocks."""
    gr.Markdown(
        "Blind test. Enter a rater ID and press **Start**. For each clip rate "
        "how natural it sounds and which technique you hear. The same ID "
        "always gets the same clip order.\n\n*Pilot clips are needed: run "
        "`scripts/build_pilot.py` first.*")
    langs = (sorted(pd.read_csv(MANIFEST, usecols=["language"]).language.unique())
             if MANIFEST.exists() else [])
    with gr.Row():
        rater = gr.Textbox(label="Rater ID")
        n_trials = gr.Slider(10, 120, value=40, step=5, label="Clips in session")
        start = gr.Button("Start", variant="primary")
    with gr.Row():
        # Session filters: languages the rater understands, techniques they
        # can tell apart by ear.
        lang_sel = gr.CheckboxGroup(langs, value=langs, label="Languages")
        group_sel = gr.CheckboxGroup(
            [(lab, g) for g, lab in GROUP_LABELS.items()],
            value=[g for g in GROUP_LABELS if g not in DEFAULT_EXCLUDED],
            label="Techniques in the session")
    progress = gr.Markdown()
    audio = gr.Audio(label="Clip", interactive=False, autoplay=True)
    natural = gr.Radio([1, 2, 3, 4, 5], label="Naturalness (1 = robotic … 5 = real singer)")
    heard = gr.Radio(TECHNIQUES, label="Which technique do you hear?")
    submit = gr.Button("Submit & next")
    trials_state = gr.State([])
    idx_state = gr.State(0)
    with gr.Accordion("Results so far", open=False):
        table = gr.Dataframe(summary)
        refresh = gr.Button("Refresh")

    def on_start(r, n, ls, gs):
        keep = gr.update()
        if not r.strip():
            return [], 0, "⚠️ Enter a rater ID.", None, keep
        if not MANIFEST.exists():
            return [], 0, "⚠️ No pilot manifest found.", None, keep
        if not ls or not gs:
            return [], 0, "⚠️ Select at least one language and one technique.", None, keep
        trials = build_trials(r.strip(), int(n), languages=ls, groups=gs)
        if not trials:
            return [], 0, "⚠️ No clips for this selection.", None, keep
        # Answer options = the techniques in the session + "none".
        choices = [GROUP_LABELS[g] for g in GROUP_LABELS if g in gs] + [TECHNIQUES[-1]]
        return (trials, 0, f"Clip 1 / {len(trials)}", _audio(trials[0]),
                gr.update(choices=choices, value=None))

    def on_submit(r, trials, idx, nat, tech):
        if not trials or idx >= len(trials):
            return idx, "Session finished — thank you!", None, None, None
        if nat is None or tech is None:
            return idx, f"⚠️ Answer both questions (clip {idx + 1} / {len(trials)}).", \
                gr.update(), nat, tech
        t = trials[idx]
        append_rating({"timestamp": datetime.now(timezone.utc).isoformat(),
                       "rater": r.strip(), "session_seed": t["session_seed"],
                       "trial": idx, "language": t["language"], "group": t["group"],
                       "source": t["source"], "variant": t["variant"],
                       "clip": t["clip"], "naturalness": nat, "technique_heard": tech})
        idx += 1
        if idx >= len(trials):
            return idx, "✅ Session finished — thank you!", None, None, None
        return idx, f"Clip {idx + 1} / {len(trials)}", _audio(trials[idx]), None, None

    start.click(on_start, [rater, n_trials, lang_sel, group_sel],
                [trials_state, idx_state, progress, audio, heard])
    submit.click(on_submit, [rater, trials_state, idx_state, natural, heard],
                 [idx_state, progress, audio, natural, heard])
    refresh.click(summary, None, table)
