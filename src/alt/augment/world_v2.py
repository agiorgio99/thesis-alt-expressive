"""
world_v2.py — improved, data-driven WORLD augmentation engine.

Differences from v1 (``legacy.py``), each addressing a measured or audible
problem of v1:

* Parameters come from the calibration of the *same language*
  (``results/calibration/<lang>/params.json``: q25/q50/q75 of the real
  technique-vs-control effect), not from fixed English presets. The three
  variants are the three quartiles, ordered weak → strong.
* The technique is applied **where the singer actually used it**: the paired
  real take's per-phone flags (same phrase, same singer) select the spans on
  the control take; if no paired take exists, sung vowels are used.
* 48 kHz processing and local resynthesis + crossfade splice (``world_core``).
* Vibrato: added only up to the measured target extent (the control's own
  vibrato is taken into account), with delayed onset, rate/extent jitter and
  a small in-phase amplitude modulation, instead of a constant sinusoid.
* Glissando: an additive, sigmoid-shaped offset on the sung F0 that lands on
  the sung pitch — up or down with the measured probability — instead of
  replacing the F0 with a downward score line.
* Voice quality (breathy / pharyngeal / falsetto / mixed voice): band-wise
  aperiodicity change, measured spectral-envelope shape (ΔLTAS), explicit
  H1–H2 correction, level and F0 changes — all signed as measured, so e.g.
  pharyngeal *reduces* high-band aperiodicity as real singers do.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .engines import AugmentResult
from .features import AP_BANDS, match_control_span, phone_spans, vibrato_features
from .io import VOWELS_BY_LANGUAGE, is_vowel, load_audio, load_meta
from .labels import update_labels
from .pairing import GROUP_FLAG
from .world_core import (
    FS_FRAMES, analyse_clip, interp_curve_db, merge_spans, seed_for,
    smooth_noise, span_mask, splice, voice_of,
)

REPO = Path(__file__).resolve().parents[3]
CALIB_DIR = REPO / "results" / "calibration"

VQ_GROUPS = ["Breathy_Group", "Pharyngeal_Group", "Falsetto_Group", "Mixed_Voice_Group"]
GROUPS = ["Vibrato_Group", "Glissando_Group"] + VQ_GROUPS

# Variant k uses quantile QUANTS[k] of every measured effect, ordered so that
# the effect *magnitude* grows with k (weak → typical → strong).
QUANTS = ["q25", "q50", "q75"]
# Scale of the measured ΔLTAS shape per variant (the shape is a single median
# curve per language, so its strength is varied explicitly).
LTAS_SCALE = [0.6, 1.0, 1.4]
# Hard safety limits (dB / cents) against outlier calibration cells.
MAX_LEVEL_DB, MAX_SHAPE_DB, MAX_H1_DB = 6.0, 12.0, 12.0
AP_CENTRES = np.array([(lo + hi) / 2 for lo, hi in AP_BANDS], dtype=float)
AP_KEYS = [f"ap_{lo // 1000}k_{hi // 1000}k" for lo, hi in AP_BANDS]


# ── Calibration access ───────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def load_params(language: str) -> dict:
    """Load ``params.json`` of a language (written by calibration_report.py).

    Args:
        language: Lowercase language name.

    Returns:
        The parsed file.

    Raises:
        FileNotFoundError: Calibration not run yet for this language.
    """
    path = CALIB_DIR / language / "params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/measure_technique_features.py and "
            "scripts/calibration_report.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _q(spec: dict | None, k: int) -> float:
    """Quantile for variant ``k`` of one feature, magnitude-ordered.

    Sparse cells (``fallback_pooled`` present) use the pooled quantiles.
    Missing features return 0 (no change).
    """
    if not spec:
        return 0.0
    spec = spec.get("fallback_pooled") or spec
    order = QUANTS if spec["q50"] >= 0 else QUANTS[::-1]
    return float(spec[order[k]])


def presets(group: str, language: str) -> list[dict]:
    """Three data-driven variants (weak, typical, strong) for a group.

    Args:
        group:    Real technique group ("Vibrato_Group", ...).
        language: Lowercase language name.

    Returns:
        Parameter dicts consumed by ``run``.
    """
    g = load_params(language)["groups"].get(group, {})
    out = []
    for k in range(3):
        if group == "Vibrato_Group":
            p = {"extent_cents": _q(g.get("vib_extent_cents"), k),
                 # Rate and onset are gesture properties, not strengths: use
                 # the language's typical values, varied mildly by the jitter.
                 "rate_hz": _q(g.get("vib_rate_hz"), 1),
                 "onset_s": _q(g.get("vib_onset_s"), 1),
                 "am_db": 1.5, "jitter": 0.1}
        elif group == "Glissando_Group":
            p = {"glide_cents": _q(g.get("glide_abs_cents"), k),
                 "share_up": float(g.get("share_up", 0.5)), "steepness": 8.0}
        else:
            p = {"f0_cents": _q(g.get("f0_cents"), k),
                 # Per-voice F0 change (falsetto / mixed voice); ``run`` picks
                 # the entry of the clip's voice type, else uses f0_cents.
                 "f0_cents_by_voice": {v: _q(spec, k) for v, spec in
                                       g.get("f0_cents_by_voice", {}).items()},
                 "h1h2_db": _q(g.get("h1h2_db"), k),
                 "level_db": _q(g.get("level_db"), k),
                 "ap_delta": [_q(g.get(key), k) for key in AP_KEYS],
                 "ltas_scale": LTAS_SCALE[k],
                 # Calibration the ΔLTAS shape is read from in ``run``.
                 "targets": language}
        p["seed"] = k
        out.append({key: (round(v, 4) if isinstance(v, float) else v)
                    for key, v in p.items()})
    return out


# ── Sampled parameters (whole distribution, optional tail emphasis) ─────────
# The quartile presets above always produce the same three values per
# language, so the augmented data is much less diverse than the real data and
# never reaches the tails. ``sample_presets`` instead draws, for every clip,
# targets from the whole measured distribution (between the 2nd and 98th
# percentile, i.e. inside the human range).

U_RANGE = (0.02, 0.98)   # quantile range targets are drawn from


def strength_quantiles(n: int, rng: np.random.Generator, tail_gamma: float = 1.0) -> np.ndarray:
    """Stratified strength quantiles, ordered weak → strong.

    Variant ``k`` draws its quantile from the k-th of ``n`` equal strata, so a
    clip's variants stay ordered by strength while, over many clips, the
    draws are uniform in quantile space (= the real distribution is
    reproduced). ``tail_gamma < 1`` pushes draws toward both tails, to
    over-represent rare (under-represented) values.

    Args:
        n:          Number of variants.
        rng:        Seeded generator.
        tail_gamma: 1 = follow the real distribution; < 1 = favour the tails.

    Returns:
        ``n`` quantile levels inside ``U_RANGE``.
    """
    u = (np.arange(n) + rng.uniform(size=n)) / n
    # Symmetric warp around the median: |2u − 1|^γ (γ < 1 moves mass outward).
    c = 2 * u - 1
    v = 0.5 + 0.5 * np.sign(c) * np.abs(c) ** tail_gamma
    lo, hi = U_RANGE
    return lo + (hi - lo) * v


def quantile_value(spec: dict | None, u: float, by_magnitude: bool = True) -> float:
    """Value of one calibrated feature at quantile level ``u``.

    Interpolates the stored quantile grid (``qgrid``); older params files
    without a grid fall back to the quartiles (draws are then limited to the
    interquartile range).

    Args:
        spec:         Feature entry of params.json (may carry ``fallback_pooled``).
        u:            Quantile level in [0, 1].
        by_magnitude: Order so that larger ``u`` = larger |effect| (for
                      effects whose median is negative, ``u`` is mirrored).

    Returns:
        The interpolated value (0 when the feature is missing).
    """
    from .features import QGRID

    if not spec:
        return 0.0
    spec = spec.get("fallback_pooled") or spec
    if by_magnitude and spec["q50"] < 0:
        u = 1.0 - u
    if spec.get("qgrid"):
        return float(np.interp(u, QGRID, spec["qgrid"]))
    return float(np.interp(u, [0.25, 0.5, 0.75], [spec["q25"], spec["q50"], spec["q75"]]))


def sample_presets(group: str, language: str, clip_key: str, n: int = 3,
                   tail_gamma: float = 1.0) -> list[dict]:
    """Per-clip variants with targets sampled from the real distribution.

    Same parameter structure as ``presets`` (so ``run`` is unchanged), but:

    * the strength of variant k is drawn from the k-th stratum of the real
      distribution (``strength_quantiles``), deterministic per clip;
    * vibrato rate and onset are drawn independently from their full
      distributions (gesture variety), instead of the language median;
    * voice-quality features share one strength level per variant
      (comonotone), so a "strong breathy" variant is strong in every feature.

    Args:
        group:      Target technique group.
        language:   Lowercase language name whose calibration provides the
                    targets, or "pooled" (all languages together).
        clip_key:   Identifies the clip (e.g. control-take path); seeds the draws.
        n:          Number of variants.
        tail_gamma: See ``strength_quantiles``.

    Returns:
        Parameter dicts consumed by ``run`` (with ``quantile`` recorded).
    """
    g = load_params(language)["groups"].get(group, {})
    rng = np.random.default_rng(seed_for(clip_key, group, "sample", tail_gamma))
    out = []
    for k, u in enumerate(strength_quantiles(n, rng, tail_gamma)):
        if group == "Vibrato_Group":
            p = {"extent_cents": quantile_value(g.get("vib_extent_cents"), u),
                 "rate_hz": quantile_value(g.get("vib_rate_hz"), rng.uniform(*U_RANGE), False),
                 "onset_s": quantile_value(g.get("vib_onset_s"), rng.uniform(*U_RANGE), False),
                 "am_db": 1.5, "jitter": 0.1}
        elif group == "Glissando_Group":
            p = {"glide_cents": quantile_value(g.get("glide_abs_cents"), u),
                 "share_up": float(g.get("share_up", 0.5)), "steepness": 8.0}
        else:
            p = {"f0_cents": quantile_value(g.get("f0_cents"), u),
                 "f0_cents_by_voice": {v: quantile_value(spec, u) for v, spec in
                                       g.get("f0_cents_by_voice", {}).items()},
                 "h1h2_db": quantile_value(g.get("h1h2_db"), u),
                 "level_db": quantile_value(g.get("level_db"), u),
                 "ap_delta": [quantile_value(g.get(key), u) for key in AP_KEYS],
                 # Same mapping as the quartile presets (0.6 / 1.0 / 1.4 at
                 # q25 / q50 / q75), extended linearly to the tails.
                 "ltas_scale": float(0.6 + 1.6 * (u - 0.25)),
                 "targets": language}
        p["seed"] = k
        p["quantile"] = float(u)
        out.append({key: (round(v, 4) if isinstance(v, float) else v)
                    for key, v in p.items()})
    return out


# ── Where to apply ───────────────────────────────────────────────────────────

def target_spans(control_wav: Path, group: str, language: str,
                 ctrl_meta: list[dict]) -> tuple[list[tuple[float, float]], list[int]]:
    """Spans (control-take timing) where the technique should be applied.

    Uses the paired real take's flags when available; otherwise falls back to
    sung vowels (≥ 0.3 s for vibrato, ≥ 0.15 s otherwise).

    Args:
        control_wav: Control-take path.
        group:       Target technique group.
        language:    Lowercase language name.
        ctrl_meta:   Control-take annotation.

    Returns:
        ``(spans, affected_word_indices)``.
    """
    flag = GROUP_FLAG[group]
    vowels = VOWELS_BY_LANGUAGE[language]
    real_json = control_wav.parent.parent / group / (control_wav.stem + ".json")
    spans: list[tuple[float, float]] = []
    words: set[int] = set()
    if real_json.exists():
        for s in phone_spans(load_meta(real_json)):
            if not s[flag]:
                continue
            if group == "Vibrato_Group" and not is_vowel(s["ph"], vowels):
                continue
            c = match_control_span(s, ctrl_meta)
            if c is not None and c["end"] > c["start"]:
                spans.append((c["start"], c["end"]))
                words.add(s["word_idx"])
    if not spans:  # fallback: sung vowels
        min_dur = 0.3 if group == "Vibrato_Group" else 0.15
        for s in phone_spans(ctrl_meta):
            if is_vowel(s["ph"], vowels) and s["end"] - s["start"] >= min_dur:
                spans.append((s["start"], s["end"]))
                words.add(s["word_idx"])
    if group == "Vibrato_Group":
        spans = [sp for sp in spans if sp[1] - sp[0] >= 0.25]
    return merge_spans(spans), sorted(words)


# ── Transforms (operate in place on WORLD arrays) ────────────────────────────

def vibrato_offsets(f0: np.ndarray, spans: list[tuple[float, float]], p: dict,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame vibrato to add: pitch offset (cents) and loudness modulation (dB).

    Shared by the WORLD and PSOLA engines so both realise the same gesture.

    Args:
        f0:    Control-take F0 per frame (Hz, 0/NaN = unvoiced).
        spans: Target spans (s).
        p:     Vibrato parameters (see ``presets``).
        rng:   Seeded generator (jitter, start phase).

    Returns:
        ``(cents, am_db)`` arrays, zero outside the spans.
    """
    n_all = len(f0)
    cents, am = np.zeros(n_all), np.zeros(n_all)
    f0 = np.nan_to_num(f0, nan=0.0)
    for s, e in spans:
        a0 = max(0, int(round(s * FS_FRAMES)))
        a1 = min(n_all, int(round(e * FS_FRAMES)))
        n = a1 - a0
        if n < 10:
            continue
        # Existing (natural) vibrato of the control: add only what is missing
        # so the total matches the measured target extent.
        e0 = vibrato_features(f0[a0:a1])["vib_extent_cents"]
        e0 = 0.0 if not np.isfinite(e0) else e0
        add = np.sqrt(max(p["extent_cents"] ** 2 - e0 ** 2, 0.0))
        if add < 5.0:
            continue
        t = np.arange(n) / FS_FRAMES
        rate = p["rate_hz"] * (1 + 0.5 * p["jitter"] * smooth_noise(n, rng, 0.7))
        ext = add * (1 + p["jitter"] * smooth_noise(n, rng, 0.7))
        phase = 2 * np.pi * np.cumsum(rate) / FS_FRAMES + rng.uniform(0, 2 * np.pi)
        # Envelope: silent until the onset delay, rises over 150 ms, released
        # over the last 60 ms (keeps the span edges unmodulated).
        onset = min(max(p["onset_s"], 0.0), 0.4 * (e - s))
        env = np.clip((t - onset) / 0.15, 0, 1)
        env = 0.5 - 0.5 * np.cos(np.pi * env)
        rel = np.clip((t[-1] - t) / 0.06, 0, 1)
        env *= 0.5 - 0.5 * np.cos(np.pi * rel)
        osc = env * np.sin(phase)
        cents[a0:a1] = ext * osc
        # Small in-phase loudness modulation (real vibrato is not pitch-only).
        am[a0:a1] = p["am_db"] * osc
    return cents, am


def glissando_offsets(n_all: int, spans: list[tuple[float, float]], p: dict,
                      rng: np.random.Generator) -> np.ndarray:
    """Per-frame glide offset (cents): sigmoid into the sung pitch, up or down.

    An upward glide starts below the note and rises to it (scoop); a
    downward one starts above and falls to it. Either way the phrase lands in
    tune; the direction is drawn with the language's measured probability.

    Args:
        n_all: Number of frames of the clip.
        spans: Target spans (s).
        p:     Glissando parameters (see ``presets``).
        rng:   Seeded generator (direction draws).

    Returns:
        The offset array (cents), zero outside the spans.
    """
    cents = np.zeros(n_all)
    for s, e in spans:
        a0 = max(0, int(round(s * FS_FRAMES)))
        a1 = min(n_all, int(round(e * FS_FRAMES)))
        n = a1 - a0
        if n < 6:
            continue
        up = rng.uniform() < p["share_up"]
        size = p["glide_cents"] * (1 if up else -1)
        x = np.linspace(-1, 1, n) * p["steepness"] / 2
        sig = 1 / (1 + np.exp(-x))
        sig = (sig - sig[0]) / (sig[-1] - sig[0])
        off = -size * (1 - sig)
        # 30 ms fade-in of the offset so the span does not start with a jump.
        k = min(n, int(0.03 * FS_FRAMES))
        off[:k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / max(k, 1))
        cents[a0:a1] = off
    return cents


def _apply_pitch(a, cents: np.ndarray, am_db: np.ndarray | None = None) -> None:
    """Apply a cents offset (and optional loudness modulation) to WORLD arrays."""
    voiced = a.f0 > 0
    a.f0[voiced] *= 2 ** (cents[voiced] / 1200)
    if am_db is not None:
        a.sp *= (10 ** (am_db / 10))[:, None]


def _voice_quality(a, mask, p, curve_db) -> None:
    """Band aperiodicity, envelope shape, H1–H2, level and F0 changes."""
    freqs = a.freqs
    m = mask[:, None]
    # Envelope: measured ΔLTAS shape (level removed, bounded) + level.
    shape = np.zeros_like(freqs)
    if curve_db is not None:
        shape = np.clip(curve_db * p["ltas_scale"], -MAX_SHAPE_DB, MAX_SHAPE_DB)
    level = float(np.clip(p["level_db"], -MAX_LEVEL_DB, MAX_LEVEL_DB))
    gain_db = m * shape[None, :]
    # H1–H2: the shape already changes it by shape(f0) − shape(2 f0); add the
    # remainder as a smooth low-frequency shelf ending between f0 and 2 f0.
    f0 = np.where(a.f0 > 0, a.f0, np.nan)
    pred = np.interp(f0, freqs, shape) - np.interp(2 * f0, freqs, shape)
    extra = np.nan_to_num(np.clip(p["h1h2_db"] - pred, -MAX_H1_DB, MAX_H1_DB))
    edge_lo = np.nan_to_num(f0 * 1.2, nan=0.0)[:, None]
    edge_hi = np.nan_to_num(f0 * 1.8, nan=0.0)[:, None]
    shelf = np.clip((edge_hi - freqs[None, :]) / np.maximum(edge_hi - edge_lo, 1.0), 0, 1)
    gain_db = gain_db + m * extra[:, None] * shelf
    # Shape and H1 changes must not alter loudness by themselves: renormalise
    # each frame's envelope power, then apply the measured level change.
    before = a.sp.sum(axis=1)
    a.sp *= 10 ** (gain_db / 10)
    after = np.maximum(a.sp.sum(axis=1), 1e-20)
    a.sp *= ((before / after) * 10 ** (mask * level / 10))[:, None]
    # Aperiodicity: signed band deltas, interpolated across the spectrum.
    d_ap = np.interp(freqs, AP_CENTRES, np.asarray(p["ap_delta"], dtype=float))
    a.ap[:] = np.clip(a.ap + m * d_ap[None, :], 0.001, 0.999)
    # F0 shift (falsetto often sits higher than the control; ~0 for others).
    voiced = a.f0 > 0
    a.f0[voiced] *= 2 ** (mask[voiced] * p["f0_cents"] / 1200)


def resolve_voice(params: dict, voice: str) -> dict:
    """Use the voice-type specific F0 change when the preset provides one.

    Args:
        params: Engine parameters (may contain ``f0_cents_by_voice``).
        voice:  Voice type of the clip's singer.

    Returns:
        A copy with ``f0_cents`` set for this voice (unchanged otherwise).
    """
    by_voice = params.get("f0_cents_by_voice") or {}
    if voice in by_voice:
        params = {**params, "f0_cents": by_voice[voice]}
    return params


# ── Engine entry point ───────────────────────────────────────────────────────

def run(wav_path: str | Path, json_path: str | Path, group: str, params: dict,
        language: str) -> AugmentResult:
    """Augment a control take toward ``group`` with the v2 engine.

    Args:
        wav_path:  Control-take WAV (native rate is kept).
        json_path: Its annotation.
        group:     Target technique group.
        params:    One of ``presets(group, language)`` (or edited copy).
        language:  Lowercase language name.

    Returns:
        ``AugmentResult`` at the native sample rate.
    """
    import pyworld as pw

    wav_path = Path(wav_path)
    wav, sr = load_audio(wav_path)
    meta = load_meta(json_path)
    spans, words = target_spans(wav_path, group, language, meta)
    if not spans:
        return AugmentResult(wav, sr, meta, "world_v2", params)

    rng = np.random.default_rng(seed_for(wav_path.stem, group, params.get("seed", 0)))
    voice = voice_of(wav_path)
    params = resolve_voice(params, voice)
    a = analyse_clip(wav, sr, voice)
    if group == "Vibrato_Group":
        cents, am = vibrato_offsets(a.f0, spans, params, rng)
        _apply_pitch(a, cents, am)
    elif group == "Glissando_Group":
        _apply_pitch(a, glissando_offsets(len(a.f0), spans, params, rng))
    else:
        mask = span_mask(len(a.f0), spans)
        curve = None
        # Targets may come from another calibration than the clip's language
        # (e.g. "pooled"); vowels / spans above always use the clip's language.
        calib = load_params(params.get("targets", language))
        pts = calib["groups"].get(group, {}).get("delta_ltas_db")
        if pts:
            centres = np.asarray(calib["ltas_centres_hz"], dtype=float)
            vals = np.asarray(pts, dtype=float)
            # Keep only the spectral *shape*: level is applied separately.
            band = (centres >= 100) & (centres <= 8000) & np.isfinite(vals)
            vals = vals - np.nanmean(vals[band]) if band.any() else vals
            curve = interp_curve_db(a.freqs, centres, vals)
        _voice_quality(a, mask, params, curve)
    voc = pw.synthesize(a.f0, np.ascontiguousarray(a.sp), np.ascontiguousarray(a.ap),
                        sr, frame_period=1000.0 / FS_FRAMES)
    out = splice(wav, voc, sr, spans)
    return AugmentResult(out, sr, update_labels(meta, words, GROUP_FLAG[group]),
                         "world_v2", params)
