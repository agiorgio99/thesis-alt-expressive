"""
psola.py — PSOLA engine (Praat overlap-add) for the pitch-based techniques.

PSOLA shifts pitch by re-spacing the recording's own pitch periods instead of
re-generating the signal from a vocoder model, so timbre, breath noise and
micro-detail of the voice are preserved. It can only change pitch (and
duration), so it covers vibrato, glissando and the F0 component of falsetto;
voice-quality techniques stay with ``world_v2``.

The pitch gestures are exactly the v2 ones (``world_v2.vibrato_offsets`` /
``glissando_offsets`` with the same data-driven parameters and seeds), so the
A/B comparison isolates the *resynthesis method*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .engines import AugmentResult
from .features import F0_RANGE
from .io import load_audio, load_meta
from .labels import update_labels
from .pairing import GROUP_FLAG
from .world_core import FS_FRAMES, seed_for, splice, voice_of

GROUPS = ["Vibrato_Group", "Glissando_Group", "Falsetto_Group"]


def presets(group: str, language: str) -> list[dict]:
    """Same data-driven variants as v2 (falsetto: F0 part only).

    Args:
        group:    Target technique group.
        language: Lowercase language name.

    Returns:
        Parameter dicts consumed by ``run``.
    """
    from . import world_v2
    return _pitch_only(group, world_v2.presets(group, language))


def sample_presets(group: str, language: str, clip_key: str, n: int = 3,
                   tail_gamma: float = 1.0) -> list[dict]:
    """Same sampled variants as ``world_v2.sample_presets`` (falsetto: F0 part only).

    Args:
        group:      Target technique group.
        language:   Lowercase language name.
        clip_key:   Identifies the clip; seeds the draws (same draws as v2).
        n:          Number of variants.
        tail_gamma: 1 = follow the real distribution; < 1 = favour the tails.

    Returns:
        Parameter dicts consumed by ``run``.
    """
    from . import world_v2
    return _pitch_only(group, world_v2.sample_presets(group, language, clip_key, n,
                                                      tail_gamma))


def _pitch_only(group: str, presets_: list[dict]) -> list[dict]:
    """Keep only the pitch parameters PSOLA can realise (falsetto: F0 shift)."""
    if group != "Falsetto_Group":
        return presets_
    keep = ("f0_cents", "f0_cents_by_voice", "seed", "quantile")
    return [{k: p[k] for k in keep if k in p} for p in presets_]


def run(wav_path: str | Path, json_path: str | Path, group: str, params: dict,
        language: str) -> AugmentResult:
    """Augment a control take toward ``group`` with Praat PSOLA.

    Args:
        wav_path:  Control-take WAV (native rate kept).
        json_path: Its annotation.
        group:     "Vibrato_Group", "Glissando_Group" or "Falsetto_Group".
        params:    One of ``presets(group, language)``.
        language:  Lowercase language name.

    Returns:
        ``AugmentResult`` at the native sample rate.

    Raises:
        ValueError: For groups PSOLA cannot realise (voice quality).
    """
    import parselmouth
    from parselmouth.praat import call

    from . import world_v2
    from .world_core import harvest_f0

    if group not in GROUPS:
        raise ValueError(f"PSOLA only changes pitch; {group} needs world_v2")
    wav_path = Path(wav_path)
    wav, sr = load_audio(wav_path)
    meta = load_meta(json_path)
    spans, words = world_v2.target_spans(wav_path, group, language, meta)
    if not spans:
        return AugmentResult(wav, sr, meta, "psola", params)

    # Cents offset per 5 ms frame, from the same generators as v2.
    voice = voice_of(wav_path)
    params = world_v2.resolve_voice(params, voice)
    lo, hi = F0_RANGE.get(voice, F0_RANGE["unknown"])
    t, f0 = harvest_f0(wav, sr, lo, hi)
    rng = np.random.default_rng(seed_for(wav_path.stem, group, params.get("seed", 0)))
    if group == "Vibrato_Group":
        cents, _ = world_v2.vibrato_offsets(f0, spans, params, rng)
    elif group == "Glissando_Group":
        cents = world_v2.glissando_offsets(len(f0), spans, params, rng)
    else:  # falsetto: constant shift with 30 ms ramps inside the spans
        from .world_core import span_mask
        cents = span_mask(len(f0), spans) * params["f0_cents"]

    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    manip = call(snd, "To Manipulation", 0.01, lo, hi)
    tier = call(manip, "Extract pitch tier")
    n_pts = call(tier, "Get number of points")
    times = np.array([call(tier, "Get time from index", i + 1) for i in range(n_pts)])
    values = np.array([call(tier, "Get value at index", i + 1) for i in range(n_pts)])
    # Rebuild the tier with the offset applied at every original pitch point.
    frame_t = np.arange(len(cents)) / FS_FRAMES
    off = np.interp(times, frame_t, cents)
    new_tier = call("Create PitchTier", "shifted", snd.xmin, snd.xmax)
    for tt, v, o in zip(times, values, off):
        call(new_tier, "Add point", float(tt), float(v * 2 ** (o / 1200)))
    call([manip, new_tier], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    resyn = np.asarray(out.values[0], dtype=np.float64)
    spliced = splice(wav, resyn, sr, spans)
    return AugmentResult(spliced, sr, update_labels(meta, words, GROUP_FLAG[group]),
                         "psola", params)
