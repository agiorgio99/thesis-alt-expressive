"""
features.py — acoustic measurements of singing techniques on real recordings.

These estimators turn a phone-level span of a sung take into the quantities the
augmentation engines control, so that transform parameters can be *measured*
on the paired real data (technique take vs control take of the same phrase)
instead of chosen by ear:

* vibrato   — rate (Hz), extent (cents, peak), regularity, onset delay (s)
* glissando — glide size (cents, signed), duration, peak slope
* voice quality (breathy / falsetto / mixed voice / pharyngeal) — band
  aperiodicity, spectral tilt (dB/octave), H1–H2 (dB), level (dB), mean F0
  (cents re A4) and the long-term average spectrum (LTAS) on a fixed grid.

Analysis: F0 with WORLD Harvest on a 16 kHz copy (fast, sufficient for F0),
then CheapTrick envelope + D4C aperiodicity on the native-rate signal using the
same F0/time axis, so spectral measures cover the full band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import FRAME_MS, is_vowel

# Harvest F0 search ranges per voice type (Hz) — wide enough for falsetto and
# ornaments, narrow enough to reduce octave errors.
F0_RANGE: dict[str, tuple[float, float]] = {
    "soprano": (150.0, 1400.0),
    "alto": (110.0, 1100.0),
    "tenor": (80.0, 900.0),
    "bass": (55.0, 700.0),
    "unknown": (55.0, 1400.0),
}

# Aperiodicity bands (Hz). The upper bands carry aspiration noise (breathiness).
AP_BANDS: list[tuple[float, float]] = [(0, 1000), (1000, 3000), (3000, 5000),
                                       (5000, 8000), (8000, 12000)]

# Fixed log-frequency grid for LTAS vectors (band centres, Hz): 40 bands
# between 80 Hz and 16 kHz, comparable across clips and sample rates.
LTAS_CENTRES: np.ndarray = np.geomspace(80.0, 16000.0, 40)

# Quantile levels stored per feature in params.json ("qgrid"), so engines can
# sample targets from the whole real distribution. The ends (1st / 99th
# percentile) are taken as the limits of the human range.
QGRID: list[float] = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                      0.95, 0.98, 0.99]

# Vibrato search band (Hz) — sung vibrato lives at ~4.5–7.5 Hz.
VIB_BAND: tuple[float, float] = (3.0, 10.0)


@dataclass
class Analysis:
    """WORLD analysis of one take.

    Attributes:
        t:    Frame times (s).
        f0:   F0 per frame (Hz, 0 = unvoiced).
        sp:   Spectral envelope (power), frames × bins, at ``sr``.
        ap:   Aperiodicity (0..1), frames × bins, at ``sr``.
        sr:   Sample rate of ``sp``/``ap``/``wav``.
        wav:  The analysed waveform.
    """
    t: np.ndarray
    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    sr: int
    wav: np.ndarray

    def frames(self, start: float, end: float) -> slice:
        """Frame slice covering ``[start, end)`` seconds."""
        a = int(round(start * 1000.0 / FRAME_MS))
        b = int(round(end * 1000.0 / FRAME_MS))
        return slice(max(a, 0), min(b, len(self.f0)))

    @property
    def freqs(self) -> np.ndarray:
        """Centre frequency (Hz) of every envelope bin."""
        return np.linspace(0, self.sr / 2, self.sp.shape[1])


def analyse(wav: np.ndarray, sr: int, voice: str = "unknown") -> Analysis:
    """Run the WORLD analysis used by every measurement.

    Args:
        wav:   Mono float64 waveform (native rate, 48 kHz for GTSinger).
        sr:    Its sample rate.
        voice: Voice type ("soprano", "alto", "tenor", "bass") for the F0 range.

    Returns:
        The ``Analysis``.
    """
    import librosa
    import pyworld as pw

    lo, hi = F0_RANGE.get(voice, F0_RANGE["unknown"])
    x = np.ascontiguousarray(wav, dtype=np.float64)
    # F0 on a 16 kHz copy: Harvest cost scales with the rate, F0 does not need more.
    x16 = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16_000)
    f0, t = pw.harvest(np.ascontiguousarray(x16, dtype=np.float64), 16_000,
                       f0_floor=lo, f0_ceil=hi, frame_period=FRAME_MS)
    # Envelope / aperiodicity on the native-rate signal with the same F0 track.
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return Analysis(t, f0, sp, ap, sr, x)


# ── Span helpers ─────────────────────────────────────────────────────────────

def phone_spans(meta: list[dict]) -> list[dict]:
    """Flatten a word annotation into phone spans with their technique flags.

    Args:
        meta: GTSinger word list.

    Returns:
        One dict per phone: ``word_idx``, ``ph_idx``, ``word``, ``ph``,
        ``start``, ``end`` and the six technique flags as bools.
    """
    flags = ("vibrato", "breathy", "glissando", "pharyngeal", "falsetto", "mix")
    spans = []
    for wi, w in enumerate(meta):
        if str(w.get("word", "")).startswith("<"):
            continue
        for pi, (ph, ps, pe) in enumerate(zip(w["ph"], w["ph_start"], w["ph_end"])):
            span = {"word_idx": wi, "ph_idx": pi, "word": w["word"], "ph": ph,
                    "start": float(ps), "end": float(pe)}
            for f in flags:
                vals = w.get(f, [])
                span[f] = pi < len(vals) and str(vals[pi]) == "1"
            spans.append(span)
    return spans


def match_control_span(span: dict, ctrl_meta: list[dict]) -> dict | None:
    """Find the same phone in the control take (same word index + phone index).

    The two takes sing the same lyrics, so words/phones correspond by position;
    a mismatch in word text or phone label means the annotations diverge and
    the pair is skipped rather than guessed.

    Args:
        span:      A span from ``phone_spans`` of the technique take.
        ctrl_meta: Word annotation of the control take.

    Returns:
        The matching control span dict, or ``None``.
    """
    wi, pi = span["word_idx"], span["ph_idx"]
    if wi >= len(ctrl_meta):
        return None
    w = ctrl_meta[wi]
    if w.get("word") != span["word"] or pi >= len(w["ph"]) or w["ph"][pi] != span["ph"]:
        return None
    ctrl = {"start": float(w["ph_start"][pi]), "end": float(w["ph_end"][pi])}
    for f in ("vibrato", "breathy", "glissando", "pharyngeal", "falsetto", "mix"):
        vals = w.get(f, [])
        ctrl[f] = pi < len(vals) and str(vals[pi]) == "1"
    return ctrl


# ── Estimators ───────────────────────────────────────────────────────────────

def _cents(f0: np.ndarray, ref: float = 440.0) -> np.ndarray:
    """F0 (Hz) → cents re ``ref`` (NaN where unvoiced)."""
    out = np.full(f0.shape, np.nan)
    v = f0 > 0
    out[v] = 1200.0 * np.log2(f0[v] / ref)
    return out


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps (short unvoiced dropouts inside a vowel)."""
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x
    return np.interp(idx, idx[ok], x[ok])


def vibrato_features(f0: np.ndarray, frame_ms: float = FRAME_MS) -> dict:
    """Vibrato rate / extent / regularity / onset delay of one F0 segment.

    The contour (cents) is detrended with a quadratic fit (a vowel carries one
    note or a smooth transition, so this keeps the melody out of the
    residual), then band-passed to ``VIB_BAND`` in the frequency domain. The
    band's peak frequency is the rate, the band-passed signal's sinusoid
    amplitude (√2·std) the extent (peak deviation in cents, i.e. half the
    peak-to-peak swing), the share of in-band power within ±0.75 Hz of the
    peak the regularity, and the time its Hilbert envelope first reaches half
    of its maximum the onset delay.

    Args:
        f0:       F0 (Hz) of the span, 0 = unvoiced.
        frame_ms: Hop in ms.

    Returns:
        ``vib_rate_hz``, ``vib_extent_cents``, ``vib_regularity`` (0..1),
        ``vib_onset_s``; NaN when the span is too short / too unvoiced.
    """
    from scipy.signal import hilbert

    nan = {"vib_rate_hz": np.nan, "vib_extent_cents": np.nan,
           "vib_regularity": np.nan, "vib_onset_s": np.nan}
    fs = 1000.0 / frame_ms
    c = _cents(f0)
    if len(c) < 0.35 * fs or np.isfinite(c).mean() < 0.7:
        return nan
    c = _fill_gaps(c)
    x = np.arange(len(c))
    resid = c - np.polyval(np.polyfit(x, c, 2), x)
    # Band-pass by FFT masking (zero-phase), same length as the span.
    spec_full = np.fft.rfft(resid)
    freqs_full = np.fft.rfftfreq(len(resid), 1.0 / fs)
    mask = (freqs_full >= VIB_BAND[0]) & (freqs_full <= VIB_BAND[1])
    bp = np.fft.irfft(spec_full * mask, len(resid))
    # Rate from the zero-padded, Hann-windowed spectrum (fine frequency grid).
    n = 4096
    spec = np.abs(np.fft.rfft(resid * np.hanning(len(resid)), n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs >= VIB_BAND[0]) & (freqs <= VIB_BAND[1])
    if not band.any() or spec[band].sum() <= 0:
        return nan
    rate = float(freqs[band][np.argmax(spec[band])])
    near = band & (np.abs(freqs - rate) <= 0.75)
    regularity = float(spec[near].sum() / spec[band].sum())
    extent = float(np.sqrt(2.0) * np.std(bp))
    env = np.abs(hilbert(bp))
    win = max(3, int(round(fs / rate)))            # smooth over one cycle
    env = np.convolve(env, np.ones(win) / win, "same")
    onset = float(np.argmax(env >= 0.5 * env.max()) / fs) if env.max() > 0 else np.nan
    return {"vib_rate_hz": rate, "vib_extent_cents": extent,
            "vib_regularity": regularity, "vib_onset_s": onset}


def glide_features(f0: np.ndarray, frame_ms: float = FRAME_MS) -> dict:
    """Pitch glide over a span: signed size, duration and peak slope.

    Args:
        f0:       F0 (Hz) of the span.
        frame_ms: Hop in ms.

    Returns:
        ``glide_cents`` (end − start, median of the last/first 20 % of frames;
        positive = upward), ``glide_dur_s``, ``glide_max_slope_cents_s``.
    """
    fs = 1000.0 / frame_ms
    c = _cents(f0)
    nan = {"glide_cents": np.nan, "glide_dur_s": len(c) / fs,
           "glide_max_slope_cents_s": np.nan}
    if len(c) < 6 or np.isfinite(c).mean() < 0.6:
        return nan
    c = _fill_gaps(c)
    k = max(2, len(c) // 5)
    glide = float(np.median(c[-k:]) - np.median(c[:k]))
    # Slope on a 25 ms-smoothed contour so jitter does not dominate the maximum.
    sm = np.convolve(c, np.ones(5) / 5, "same")
    slope = float(np.max(np.abs(np.diff(sm[2:-2]))) * fs) if len(sm) > 6 else np.nan
    return {"glide_cents": glide, "glide_dur_s": len(c) / fs,
            "glide_max_slope_cents_s": slope}


def quality_features(a: Analysis, sl: slice) -> dict:
    """Voice-quality measures over the voiced frames of a span.

    Args:
        a:  Take analysis.
        sl: Frame slice of the span.

    Returns:
        ``ap_<lo>_<hi>`` band aperiodicity means, ``tilt_db_oct`` (LTAS slope
        100 Hz–8 kHz vs log2 f), ``h1h2_db`` (envelope-based H1*–H2*),
        ``level_db`` (RMS), ``f0_cents`` (median, re A4), ``voiced_ratio``,
        and ``ltas`` (dB on ``LTAS_CENTRES``, list).
    """
    f0 = a.f0[sl]
    voiced = f0 > 0
    out: dict = {"voiced_ratio": float(voiced.mean()) if len(f0) else 0.0}
    if voiced.sum() < 4:
        return out
    sp, ap = a.sp[sl][voiced], a.ap[sl][voiced]
    freqs = a.freqs
    for lo, hi in AP_BANDS:
        m = (freqs >= lo) & (freqs < min(hi, a.sr / 2))
        if m.any():
            out[f"ap_{lo // 1000}k_{hi // 1000}k"] = float(ap[:, m].mean())
    ltas_db = 10.0 * np.log10(sp.mean(axis=0) + 1e-20)
    fit = (freqs >= 100) & (freqs <= 8000)
    out["tilt_db_oct"] = float(np.polyfit(np.log2(freqs[fit]), ltas_db[fit], 1)[0])
    # H1*–H2*: envelope level at F0 vs at 2·F0, frame by frame.
    bins = freqs[1] - freqs[0]
    f0v = f0[voiced]
    i1 = np.clip(np.round(f0v / bins).astype(int), 0, sp.shape[1] - 1)
    i2 = np.clip(np.round(2 * f0v / bins).astype(int), 0, sp.shape[1] - 1)
    rows = np.arange(len(f0v))
    out["h1h2_db"] = float(np.median(10 * np.log10(sp[rows, i1] + 1e-20)
                                     - 10 * np.log10(sp[rows, i2] + 1e-20)))
    out["f0_cents"] = float(np.median(_cents(f0v)))
    hop = int(a.sr * FRAME_MS / 1000)
    seg = a.wav[sl.start * hop: sl.stop * hop]
    out["level_db"] = float(10 * np.log10(np.mean(seg ** 2) + 1e-20)) if len(seg) else np.nan
    ltas = np.interp(LTAS_CENTRES, freqs, ltas_db).round(2)
    # Bands above Nyquist do not exist in low-rate audio (e.g. 16 kHz legacy
    # output): mark them missing instead of repeating the last bin.
    ltas[LTAS_CENTRES > a.sr / 2] = np.nan
    out["ltas"] = ltas.tolist()
    return out


def span_features(a: Analysis, start: float, end: float) -> dict:
    """All measurements of one phone span.

    Args:
        a:     Take analysis.
        start: Span start (s).
        end:   Span end (s).

    Returns:
        Merged vibrato, glide and voice-quality features plus ``dur_s``.
    """
    sl = a.frames(start, end)
    f0 = a.f0[sl]
    feats = {"dur_s": end - start}
    feats.update(vibrato_features(f0))
    feats.update(glide_features(f0))
    feats.update(quality_features(a, sl))
    return feats


def technique_spans(meta: list[dict], flag: str, vowels: set[str]) -> list[dict]:
    """Spans to measure for a technique: flagged phones of the technique take.

    Vibrato and the voice-quality techniques are measured on flagged *vowels*
    (where they are realised); glissando on any flagged phone, since glides
    often run through voiced consonants and syllable boundaries.

    Args:
        meta:   Technique-take annotation.
        flag:   Technique flag ("vibrato", "glissando", "mix", ...).
        vowels: Language vowel inventory.

    Returns:
        The selected ``phone_spans`` entries.
    """
    spans = [s for s in phone_spans(meta) if s[flag]]
    if flag != "glissando":
        spans = [s for s in spans if is_vowel(s["ph"], vowels)]
    return [s for s in spans if s["end"] - s["start"] >= 0.12]
