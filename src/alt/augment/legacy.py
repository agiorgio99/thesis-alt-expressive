"""
legacy.py — the original (v1) WORLD-vocoder technique augmentation.

Moved verbatim from ``scripts/build_augmented_dataset.py`` so that the existing
augmented dataset (data/GTSinger_Augmented/English) and every fine-tuning
result built on it stay bit-exactly reproducible. Do not "improve" anything in
this file — improvements belong to the ``world_v2`` / ``psola`` engines, which
are compared against this one.

Characteristics (documented because they explain the "robotic" sound and are
what v2 addresses):
* analysis/resynthesis at 16 kHz (GTSinger is 48 kHz) and of the whole clip;
* fixed light/mid/heavy parameter presets chosen on English;
* hard on/off boundaries at word/vowel edges, pure-sinusoid vibrato, score-line
  glissando replacing the sung F0, full-band aperiodicity changes, falsetto as
  a pitch shift.
"""

from __future__ import annotations

import numpy as np
import pyworld as pw

from .io import FRAME_MS, VOWELS_BY_LANGUAGE, midi_to_hz, t2f
from .labels import update_labels

# v1 works at 16 kHz (the ASR input rate), not at GTSinger's native 48 kHz.
SR = 16_000


def world_decompose(wav: np.ndarray):
    """WORLD analysis (Harvest F0, CheapTrick envelope, D4C aperiodicity) at 16 kHz.

    Args:
        wav: Mono float64 waveform at ``SR``.

    Returns:
        ``(f0, sp, ap)`` as returned by ``pyworld.wav2world``.
    """
    return pw.wav2world(wav, SR, frame_period=FRAME_MS)


def world_synthesize(f0, sp, ap) -> np.ndarray:
    """WORLD synthesis of a full clip at 16 kHz.

    Args:
        f0: F0 contour (Hz, 0 = unvoiced).
        sp: Spectral envelope.
        ap: Aperiodicity.

    Returns:
        The synthesised waveform.
    """
    return pw.synthesize(f0, sp, ap, SR, frame_period=FRAME_MS)


# ── Interpolation curves ──────────────────────────────────────────────────────

def cosine_curve(t: np.ndarray) -> np.ndarray:
    """Raised cosine: S-curve, zero velocity at both endpoints."""
    return (1.0 - np.cos(np.pi * t)) / 2.0


def ease_out_curve(t: np.ndarray) -> np.ndarray:
    """Quadratic ease-out: fast departure, slow landing."""
    return 1.0 - (1.0 - t) ** 2


CURVE_FNS = {"cosine": cosine_curve, "ease_out": ease_out_curve}

# ── Utterance-level augmentation functions ────────────────────────────────────

def augment_vibrato(f0, sp, ap, meta, params, vowels=None):
    """
    Sinusoidal F0 modulation on all vowels with note_dur >= 0.3 s.
    Params: rate_hz, extent_st, onset_ratio (fraction of vowel duration before
    vibrato starts — avoids attack wobble).
    """
    vowels = vowels if vowels is not None else VOWELS_BY_LANGUAGE["english"]
    rate_hz     = params["rate_hz"]
    extent_st   = params["extent_st"]
    onset_ratio = params.get("onset_ratio", 0.15)

    f0_out = f0.copy()
    affected = []

    for wi, w in enumerate(meta):
        if w["word"] == "<SP>":
            continue
        note_dur = float(w["note_dur"][0] if isinstance(w["note_dur"], list)
                         else w["note_dur"])
        if note_dur < 0.3:
            continue
        word_modified = False
        for ph, ps, pe in zip(w["ph"], w["ph_start"], w["ph_end"]):
            if ph.rstrip("012") not in vowels:
                continue
            v_dur = float(pe) - float(ps)
            if v_dur < 0.2:
                continue
            onset_s = float(ps) + v_dur * onset_ratio
            f_onset = t2f(onset_s)
            f_end   = min(t2f(float(pe)), len(f0_out))
            n_fr    = f_end - f_onset
            if n_fr < 5:
                continue
            t_loc = np.arange(n_fr) * (FRAME_MS / 1000.0)
            mod   = extent_st * np.sin(2.0 * np.pi * rate_hz * t_loc)
            for i, fi in enumerate(range(f_onset, f_end)):
                if fi < len(f0_out) and f0_out[fi] > 0:
                    f0_out[fi] *= 2.0 ** (mod[i] / 12.0)
            word_modified = True
        if word_modified:
            affected.append(wi)

    return world_synthesize(f0_out, sp, ap), update_labels(meta, affected, "vibrato")


def augment_breathy(f0, sp, ap, meta, params, vowels=None):
    """
    Push aperiodicity toward 1 on all non-SP words, reduce amplitude slightly.
    Params: alpha (0→0.5), amp (amplitude multiplier 0.84→0.92).
    ``vowels`` is accepted for a uniform augment_fn(...) calling convention
    across techniques but unused — this transform operates on whole words.
    """
    alpha = params["alpha"]
    amp   = params.get("amp", 1.0)

    ap_out   = ap.copy()
    affected = []

    for wi, w in enumerate(meta):
        if w["word"] == "<SP>":
            continue
        f_s = t2f(float(w["start_time"]))
        f_e = min(t2f(float(w["end_time"])), len(ap_out))
        ap_out[f_s:f_e] = alpha + (1.0 - alpha) * ap[f_s:f_e]
        affected.append(wi)

    wav_aug = world_synthesize(f0, sp, ap_out) * amp
    return wav_aug, update_labels(meta, affected, "breathy")


def augment_glissando(f0, sp, ap, meta, params, vowels=None):
    """
    Descending glissando on every vowel long enough (>= min_vowel_dur).
    Pitch starts at the note frequency and descends by delta_st semitones,
    following the shape given by `curve` (cosine S-curve or quadratic ease-out).
    Arithmetic is done in semitone space for perceptual uniformity.
    Params: delta_st, curve ('cosine' | 'ease_out'), min_vowel_dur.
    """
    vowels = vowels if vowels is not None else VOWELS_BY_LANGUAGE["english"]
    delta_st      = params["delta_st"]
    curve_fn      = CURVE_FNS[params.get("curve", "cosine")]
    min_vowel_dur = params.get("min_vowel_dur", 0.15)

    f0_out   = f0.copy()
    affected = []

    for wi, w in enumerate(meta):
        if w["word"] == "<SP>":
            continue
        notes = w.get("note", [])
        if not notes or notes[0] == 0:
            continue
        note_hz = midi_to_hz(int(notes[0]))
        word_modified = False
        for ph, ps, pe in zip(w["ph"], w["ph_start"], w["ph_end"]):
            if ph.rstrip("012") not in vowels:
                continue
            if float(pe) - float(ps) < min_vowel_dur:
                continue
            f_s = t2f(float(ps))
            f_e = min(t2f(float(pe)), len(f0_out))
            vi  = np.where(f0_out[f_s:f_e] > 0)[0]
            N   = len(vi)
            if N < 2:
                continue
            t_norm   = np.linspace(0.0, 1.0, N)
            alpha    = curve_fn(t_norm)           # [0 → 1]
            pitch_st = -delta_st * alpha           # [0 → -delta_st]
            f0_out[f_s + vi] = note_hz * (2.0 ** (pitch_st / 12.0))
            word_modified = True
        if word_modified:
            affected.append(wi)

    return world_synthesize(f0_out, sp, ap), update_labels(meta, affected, "glissando")


def _pharyngeal_filter(n_bins: int, gamma: float) -> np.ndarray:
    freqs = np.linspace(0, SR / 2, n_bins)
    boost = 1.0 + gamma * np.exp(-((freqs - 400) / 300) ** 2)
    attn  = np.ones_like(freqs)
    hi    = freqs > 3000
    attn[hi] = np.clip(1.0 - 0.35 * gamma * (freqs[hi] - 3000) / 3000, 0.55, 1.0)
    return (boost * attn).astype(np.float64)


def augment_pharyngeal(f0, sp, ap, meta, params, vowels=None):
    """
    Spectral envelope filter (Gaussian boost at 400 Hz, HF attenuation above 3 kHz)
    plus optional F0 depression and slight aperiodicity increase.
    Params: gamma (filter depth), press_st (F0 depression in st), ap_add.
    ``vowels`` is accepted for a uniform augment_fn(...) calling convention
    across techniques but unused — this transform operates on whole words.
    """
    gamma    = params["gamma"]
    press_st = params.get("press_st", 0.0)
    ap_add   = params.get("ap_add", 0.0)

    filt   = _pharyngeal_filter(sp.shape[1], gamma)
    f0_out = f0.copy()
    sp_out = sp.copy()
    ap_out = ap.copy()
    affected = []
    press_ratio = 2.0 ** (-press_st / 12.0)

    for wi, w in enumerate(meta):
        if w["word"] == "<SP>":
            continue
        f_s = t2f(float(w["start_time"]))
        f_e = min(t2f(float(w["end_time"])), len(f0_out))
        sp_out[f_s:f_e] = sp[f_s:f_e] * filt[None, :]
        if press_st > 0:
            voiced = f0_out[f_s:f_e] > 0
            f0_out[f_s:f_e][voiced] *= press_ratio
        if ap_add > 0:
            ap_out[f_s:f_e] = np.clip(ap[f_s:f_e] + ap_add, 0.0, 1.0)
        affected.append(wi)

    return world_synthesize(f0_out, sp_out, ap_out), update_labels(meta, affected, "pharyngeal")


def augment_falsetto(f0, sp, ap, meta, params, vowels=None):
    """
    F0 shift up by shift_st semitones (capped at 9 for WORLD stability) and
    aperiodicity increase (beta parameter pushes ap toward 1).
    Params: shift_st, beta.
    ``vowels`` is accepted for a uniform augment_fn(...) calling convention
    across techniques but unused — this transform operates on whole words.
    """
    shift_st = min(params["shift_st"], 9)
    beta     = params.get("beta", 0.15)

    ratio  = 2.0 ** (shift_st / 12.0)
    f0_out = f0.copy()
    ap_out = ap.copy()
    affected = []

    for wi, w in enumerate(meta):
        if w["word"] == "<SP>":
            continue
        f_s = t2f(float(w["start_time"]))
        f_e = min(t2f(float(w["end_time"])), len(f0_out))
        voiced = f0_out[f_s:f_e] > 0
        f0_out[f_s:f_e][voiced] *= ratio
        ap_out[f_s:f_e] = beta + (1.0 - beta) * ap[f_s:f_e]
        affected.append(wi)

    return world_synthesize(f0_out, sp, ap_out), update_labels(meta, affected, "falsetto")

# ── Technique configuration ───────────────────────────────────────────────────

TECHNIQUE_CONFIG = {
    "Vibrato": {
        "technique_key": "vibrato",
        "group_name":    "Vibrato_Group",
        "augment_fn":    augment_vibrato,
        "params": [
            dict(rate_hz=5.5, extent_st=1.0, onset_ratio=0.15),
            dict(rate_hz=6.5, extent_st=1.5, onset_ratio=0.15),
            dict(rate_hz=7.0, extent_st=2.0, onset_ratio=0.20),
        ],
    },
    "Breathy": {
        "technique_key": "breathy",
        "group_name":    "Breathy_Group",
        "augment_fn":    augment_breathy,
        "params": [
            dict(alpha=0.20, amp=0.92),
            dict(alpha=0.35, amp=0.88),
            dict(alpha=0.50, amp=0.84),
        ],
    },
    "Glissando": {
        "technique_key": "glissando",
        "group_name":    "Glissando_Group",
        "augment_fn":    augment_glissando,
        "params": [
            dict(delta_st=2, curve="cosine"),
            dict(delta_st=3, curve="cosine"),
            dict(delta_st=4, curve="cosine"),
            dict(delta_st=5, curve="ease_out"),
        ],
    },
    "Pharyngeal": {
        "technique_key": "pharyngeal",
        "group_name":    "Pharyngeal_Group",
        "augment_fn":    augment_pharyngeal,
        "params": [
            dict(gamma=0.40, press_st=0.5, ap_add=0.05),
            dict(gamma=0.70, press_st=1.0, ap_add=0.10),
            dict(gamma=1.00, press_st=2.0, ap_add=0.15),
        ],
    },
    "Mixed_Voice_and_Falsetto": {
        "technique_key": "falsetto",
        "group_name":    "Falsetto_Group",
        "augment_fn":    augment_falsetto,
        "params": [
            dict(shift_st=3, beta=0.10),
            dict(shift_st=5, beta=0.15),
            dict(shift_st=8, beta=0.20),
        ],
    },
}

