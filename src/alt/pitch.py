"""
pitch.py — fundamental-frequency (F0) extraction with CREPE.

F0 contours are used in this thesis for the FFE (F0 Frame Error) metric and for
characterising expressive techniques (e.g. glissando widens within-phoneme F0
variance). CREPE is a CNN pitch tracker; its model capacity and frame step are
config-driven.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR, load_audio


@dataclass
class F0Contour:
    """A CREPE F0 estimate for one audio file.

    Attributes:
        time:       Frame timestamps in seconds.
        frequency:  Raw F0 estimate per frame, in Hz.
        confidence: CREPE voicing confidence per frame, in [0, 1].
        f0_voiced:  ``frequency`` with low-confidence frames set to NaN.
    """
    time: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray
    f0_voiced: np.ndarray


class CrepeExtractor:
    """Wrapper around the CREPE pitch tracker.

    Args:
        model_capacity: CREPE size — tiny | small | medium | large | full
                        (larger = more accurate, slower).
        step_ms:        F0 frame step in milliseconds.
        device:         "cuda" or "cpu" (CREPE/TensorFlow picks the GPU
                        automatically when one is visible).
        conf_threshold: Frames with confidence below this are marked unvoiced.
    """

    def __init__(self, model_capacity: str = "tiny", step_ms: int = 10,
                 device: str = "cpu", conf_threshold: float = 0.5) -> None:
        self.model_capacity = model_capacity
        self.step_ms = step_ms
        self.device = device
        self.conf_threshold = conf_threshold

    def extract(self, audio_path: str) -> F0Contour | None:
        """Run CREPE on one audio file and return its F0 contour.

        Args:
            audio_path: Path to the audio file (resampled to 16 kHz internally).

        Returns:
            An ``F0Contour``, or ``None`` if extraction failed.
        """
        try:
            import crepe
            audio, _ = load_audio(audio_path, sr=TARGET_SR)
            time, freq, conf, _ = crepe.predict(
                audio, TARGET_SR,
                model_capacity=self.model_capacity,
                step_size=self.step_ms,
                viterbi=True,
                verbose=0,
            )
            f0_voiced = freq.copy()
            f0_voiced[conf <= self.conf_threshold] = np.nan
            return F0Contour(time, freq, conf, f0_voiced)
        except Exception as exc:
            print(f"  [CREPE] failed on {audio_path}: {exc}")
            return None

    @staticmethod
    def summary_stats(contour: F0Contour) -> dict[str, float]:
        """Compute summary statistics from an F0 contour.

        Args:
            contour: An ``F0Contour`` produced by ``extract``.

        Returns:
            A dict with ``f0_mean_hz``, ``f0_std_hz``, ``f0_range_st``
            (semitone range), ``voiced_ratio`` and ``vibrato_index`` (mean of
            short-window F0 standard deviations). Empty if too few voiced
            frames are present.
        """
        valid = contour.f0_voiced[~np.isnan(contour.f0_voiced)]
        if len(valid) < 10:
            return {}
        semitones = 12 * np.log2(valid / 440.0 + 1e-8)
        win = 20
        win_stds = [np.std(valid[i:i + win])
                    for i in range(0, len(valid) - win, win // 2)]
        return {
            "f0_mean_hz": float(np.mean(valid)),
            "f0_std_hz": float(np.std(valid)),
            "f0_range_st": float(np.max(semitones) - np.min(semitones)),
            "voiced_ratio": float(len(valid) / len(contour.f0_voiced)),
            "vibrato_index": float(np.mean(win_stds)) if win_stds else 0.0,
        }
