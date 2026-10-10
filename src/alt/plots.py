"""
plots.py — audio visualisations shared by the Gradio app and the reports.

``plot_clip`` draws a log-frequency spectrogram with the sung F0 contour, the
score notes and the per-phone technique flags of a GTSinger annotation, which
is the view needed to judge by eye whether an augmentation looks like the real
technique (e.g. vibrato depth/rate, glide shape, breathiness in the high band).
"""

from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # headless backend: figures are rendered to images/UI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colours for the technique-flag strip (same hues as report.TECH_COLORS where
# the technique exists there).
FLAG_COLORS: dict[str, str] = {
    "vibrato": "#4C72B0",
    "breathy": "#55A868",
    "glissando": "#C44E52",
    "pharyngeal": "#8172B2",
    "falsetto": "#CCB974",
    "mix": "#64B5CD",
}


def f0_contour(wav: np.ndarray, sr: int, frame_ms: float = 5.0,
               f0_floor: float = 60.0, f0_ceil: float = 1200.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an F0 contour with WORLD Harvest (NaN where unvoiced).

    Harvest is the estimator the augmentation engines use, so the plotted
    contour is exactly what the transforms act upon.

    Args:
        wav:      Mono waveform.
        sr:       Sample rate.
        frame_ms: Hop in milliseconds.
        f0_floor: Lowest F0 searched (Hz).
        f0_ceil:  Highest F0 searched (Hz).

    Returns:
        ``(times_s, f0_hz)`` with NaN for unvoiced frames.
    """
    import pyworld as pw  # imported lazily — heavy C extension

    # Harvest cost grows with the sample rate; F0 needs no more than 16 kHz,
    # so high-rate audio (GTSinger is 48 kHz) is downsampled first (~3× faster).
    if sr > 16_000:
        import librosa
        wav = librosa.resample(np.asarray(wav, dtype=np.float32),
                               orig_sr=sr, target_sr=16_000)
        sr = 16_000
    x = np.ascontiguousarray(wav, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil,
                       frame_period=frame_ms)
    f0 = f0.astype(float)
    f0[f0 <= 0] = np.nan
    return t, f0


def plot_clip(wav: np.ndarray, sr: int, meta: list[dict] | None = None,
              title: str = "", fmax: float = 8000.0, show_f0: bool = True,
              f0: tuple[np.ndarray, np.ndarray] | None = None,
              ) -> plt.Figure:
    """Spectrogram + F0 + score notes + technique flags of one clip.

    Args:
        wav:     Mono waveform.
        sr:      Sample rate.
        meta:    Optional GTSinger word annotation; adds word labels, score
                 notes (dashed) and a per-phone technique-flag strip.
        title:   Figure title.
        fmax:    Upper frequency shown in the spectrogram (Hz).
        show_f0: Overlay the Harvest F0 contour.
        f0:      Pre-computed ``(times, f0_hz)`` to overlay instead of
                 estimating it (saves time when plotting the same clip twice).

    Returns:
        A matplotlib Figure (caller closes it / hands it to Gradio).
    """
    import librosa  # imported lazily — only needed for plotting

    # Three stacked panels sharing the time axis: spectrogram, F0 (log Hz, so
    # vibrato depth and glide size read as musical intervals), technique flags.
    fig, (ax, ax_f0, ax_flags) = plt.subplots(
        3, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [5, 3, 1.3]})
    duration = len(wav) / sr

    # Log-magnitude spectrogram (≈ 21 ms window) up to fmax.
    n_fft = 1024 if sr <= 16_000 else 2048
    spec = np.abs(librosa.stft(wav.astype(np.float32), n_fft=n_fft,
                               hop_length=n_fft // 4))
    db = librosa.amplitude_to_db(spec, ref=np.max)
    freqs = np.linspace(0, sr / 2, db.shape[0])
    keep = freqs <= fmax
    ax.imshow(db[keep], origin="lower", aspect="auto", cmap="magma",
              extent=[0, duration, 0, freqs[keep][-1]], vmin=-80, vmax=0)
    ax.set_ylabel("Hz")

    if show_f0:
        t, f = f0 if f0 is not None else f0_contour(wav, sr)
        ax_f0.plot(t, f, color="#1f77b4", lw=1.3, label="F0 (Harvest)")
        voiced = f[np.isfinite(f)]
        if voiced.size:
            # Frame the sung range with ~3 semitones of headroom on each side.
            ax_f0.set_ylim(np.nanmin(voiced) / 1.2, np.nanmax(voiced) * 1.2)
    ax_f0.set_yscale("log")
    # Plain Hz tick labels instead of 10^x notation on the log axis.
    ax_f0.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_f0.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_f0.set_ylabel("F0 (Hz)")
    ax_f0.grid(True, which="both", alpha=0.3)

    if meta:
        for w in meta:
            if w.get("word") in ("<SP>", "<AP>"):
                continue
            # Score notes as dashed lines at the note frequency.
            for n, ns, ne in zip(w.get("note", []), w.get("note_start", []),
                                 w.get("note_end", [])):
                if n:
                    hz = 440.0 * 2 ** ((n - 69) / 12)
                    ax_f0.hlines(hz, ns, ne, colors="grey", linestyles="--", lw=1.2)
            ax.text(float(w["start_time"]), fmax * 0.93, w["word"], color="white",
                    fontsize=8, va="top", clip_on=True)
            # Technique-flag strip: one row per technique, shaded where flagged.
            for row, (flag, color) in enumerate(FLAG_COLORS.items()):
                for on, ps, pe in zip(w.get(flag, []), w.get("ph_start", []),
                                      w.get("ph_end", [])):
                    if str(on) == "1":
                        ax_flags.barh(row, float(pe) - float(ps), left=float(ps),
                                      color=color, height=0.8)
        ax_flags.set_yticks(range(len(FLAG_COLORS)))
        ax_flags.set_yticklabels(list(FLAG_COLORS), fontsize=7)
        ax_flags.set_ylim(-0.6, len(FLAG_COLORS) - 0.4)
    else:
        ax_flags.set_yticks([])

    ax.set_ylim(0, fmax)
    ax.set_title(title, fontsize=10)
    if show_f0:
        ax_f0.plot([], [], color="grey", ls="--", label="score notes")
        ax_f0.legend(loc="upper right", fontsize=7)
    ax_flags.set_xlabel("time (s)")
    ax_flags.set_xlim(0, duration)
    fig.tight_layout()
    return fig
