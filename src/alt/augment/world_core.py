"""
world_core.py — shared machinery of the improved (v2) WORLD engine.

Fixes the structural causes of the "robotic" v1 sound that are independent of
any single technique:

* **native rate** — analysis and synthesis at the file's own rate (48 kHz for
  GTSinger) instead of 16 kHz;
* **local resynthesis** — only the modified spans come from the vocoder; the
  rest of the clip is the original recording, bit-exact. The vocoded region is
  blended in with raised-cosine crossfades placed just outside the spans;
* **smooth control curves** — every parameter change is weighted by a frame
  mask with raised-cosine ramps, so nothing switches on/off in one frame.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from .features import F0_RANGE, Analysis, analyse
from .io import FRAME_MS
from .pairing import voice_type

FS_FRAMES = 1000.0 / FRAME_MS      # control-curve rate (frames per second)


def seed_for(*parts: object) -> int:
    """Deterministic RNG seed from any identifying values (stem, variant...).

    Args:
        *parts: Values identifying one generated clip.

    Returns:
        A 32-bit seed (stable across runs and machines, unlike ``hash``).
    """
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def voice_of(wav_path: str | Path) -> str:
    """Voice type from a GTSinger path ``.../<Singer>/<Tech>/<Song>/<Group>/x.wav``."""
    return voice_type(Path(wav_path).parents[3].name)


def analyse_clip(wav: np.ndarray, sr: int, voice: str) -> Analysis:
    """WORLD analysis at native rate (F0 from Harvest on a 16 kHz copy).

    Args:
        wav:   Mono float64 waveform.
        sr:    Sample rate.
        voice: Voice type (sets the F0 search range, see ``F0_RANGE``).

    Returns:
        The ``Analysis`` (t, f0, sp, ap at ``sr``).
    """
    return analyse(wav, sr, voice if voice in F0_RANGE else "unknown")


def span_mask(n_frames: int, spans: list[tuple[float, float]],
              ramp_s: float = 0.03) -> np.ndarray:
    """Frame weights in [0, 1]: 1 inside spans, raised-cosine ramps at edges.

    The ramps sit *inside* each span so the modification is fully off at the
    span boundaries (where the vocoded audio meets the original).

    Args:
        n_frames: Number of analysis frames.
        spans:    ``(start_s, end_s)`` intervals.
        ramp_s:   Ramp length in seconds.

    Returns:
        The mask (max over overlapping spans).
    """
    m = np.zeros(n_frames)
    r = max(1, int(round(ramp_s * FS_FRAMES)))
    for s, e in spans:
        a, b = int(round(s * FS_FRAMES)), int(round(e * FS_FRAMES))
        a, b = max(a, 0), min(b, n_frames)
        n = b - a
        if n <= 0:
            continue
        w = np.ones(n)
        k = min(r, n // 2)
        if k > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
            w[:k] = ramp
            w[n - k:] = ramp[::-1]
        m[a:b] = np.maximum(m[a:b], w)
    return m


def smooth_noise(n: int, rng: np.random.Generator, cutoff_hz: float = 1.0) -> np.ndarray:
    """Slowly varying zero-mean noise with unit std (for natural jitter).

    Args:
        n:         Number of frames.
        rng:       Random generator.
        cutoff_hz: Rough upper frequency of the fluctuations.

    Returns:
        The noise curve.
    """
    if n < 2:
        return np.zeros(n)
    win = max(3, int(FS_FRAMES / max(cutoff_hz, 0.1)))
    x = np.convolve(rng.standard_normal(n + win), np.hanning(win), "same")[:n]
    x -= x.mean()
    sd = x.std()
    return x / sd if sd > 0 else x


def interp_curve_db(freqs: np.ndarray, centres: np.ndarray,
                    values_db: np.ndarray) -> np.ndarray:
    """Interpolate a dB curve given on log-spaced centres onto envelope bins.

    Missing values (NaN) are ignored; outside the covered range the curve is
    held flat at its end values.

    Args:
        freqs:     Target bin frequencies (Hz).
        centres:   Frequencies (Hz) of ``values_db``.
        values_db: Curve values.

    Returns:
        The curve on ``freqs``.
    """
    ok = np.isfinite(values_db)
    if ok.sum() < 2:
        return np.zeros_like(freqs)
    lf = np.log(np.maximum(freqs, 1.0))
    return np.interp(lf, np.log(centres[ok]), values_db[ok])


def splice(original: np.ndarray, vocoded: np.ndarray, sr: int,
           spans: list[tuple[float, float]], pad_s: float = 0.04,
           xfade_s: float = 0.03) -> np.ndarray:
    """Insert vocoded regions into the original recording.

    The blend weight is 1 over each span extended by ``pad_s`` and falls to 0
    over ``xfade_s`` (raised cosine) beyond that, so the crossfade happens
    where the modification is already ramped off and both signals carry the
    same content. Outside the regions the original samples are untouched.

    Args:
        original: Original waveform.
        vocoded:  WORLD resynthesis with the modifications (same timeline).
        sr:       Sample rate.
        spans:    Modified ``(start_s, end_s)`` intervals.
        pad_s:    Margin kept fully vocoded around each span.
        xfade_s:  Crossfade length.

    Returns:
        The spliced waveform (length of ``original``).
    """
    n = len(original)
    voc = np.zeros(n)
    voc[:min(n, len(vocoded))] = vocoded[:n]
    w = np.zeros(n)
    xf = max(1, int(xfade_s * sr))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(xf) / xf)
    for s, e in spans:
        a = max(0, int((s - pad_s) * sr))
        b = min(n, int((e + pad_s) * sr))
        if b <= a:
            continue
        w[a:b] = 1.0
        lo = max(0, a - xf)
        w[lo:a] = np.maximum(w[lo:a], ramp[xf - (a - lo):])
        hi = min(n, b + xf)
        w[b:hi] = np.maximum(w[b:hi], ramp[::-1][:hi - b])
    return (1.0 - w) * original + w * voc


def merge_spans(spans: list[tuple[float, float]], gap_s: float = 0.02
                ) -> list[tuple[float, float]]:
    """Merge overlapping or nearly touching intervals.

    Args:
        spans: ``(start_s, end_s)`` intervals.
        gap_s: Intervals closer than this are joined.

    Returns:
        Sorted, merged intervals.
    """
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1] + gap_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def harvest_f0(wav: np.ndarray, sr: int, f0_floor: float, f0_ceil: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """F0 track only (no envelope/aperiodicity), Harvest on a 16 kHz copy.

    Args:
        wav:      Mono waveform.
        sr:       Its sample rate.
        f0_floor: Lowest F0 searched (Hz).
        f0_ceil:  Highest F0 searched (Hz).

    Returns:
        ``(times_s, f0_hz)`` on the ``FRAME_MS`` grid, 0 where unvoiced.
    """
    import librosa
    import pyworld as pw

    x16 = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16_000)
    f0, t = pw.harvest(np.ascontiguousarray(x16, dtype=np.float64), 16_000,
                       f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=FRAME_MS)
    return t, f0
