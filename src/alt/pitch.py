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


def _stub_torchaudio() -> None:
    """Register a no-op torchaudio stub before torchcrepe is imported.

    torchcrepe imports torchaudio at module load time (for its own audio-loading
    helper), but we load audio ourselves with librosa and pass tensors directly
    to torchcrepe.predict(), so torchaudio is never actually called.
    Stubbing it avoids the broken .so linkage without losing any functionality.
    """
    import sys
    import types

    if "torchaudio" in sys.modules:
        return

    class _Stub(types.ModuleType):
        """Auto-creates child sub-modules on attribute access; callable no-op."""
        __file__ = "/dev/null"   # prevents _Stub leaking into os.stat/__file__ checks
        __path__: list = []
        def __getattr__(self, name: str) -> "_Stub":
            child = _Stub(f"{self.__name__}.{name}")
            sys.modules[child.__name__] = child
            setattr(self, name, child)
            return child
        def __call__(self, *args, **kwargs):
            return None

    root = _Stub("torchaudio")
    sys.modules["torchaudio"] = root


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
            import torch
            _stub_torchaudio()
            import torchcrepe
            audio, _ = load_audio(audio_path, sr=TARGET_SR)
            audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            hop = int(TARGET_SR * self.step_ms / 1000)
            pitch, periodicity = torchcrepe.predict(
                audio_t, TARGET_SR,
                hop_length=hop,
                fmin=32.7, fmax=1975.5,
                model=self.model_capacity,
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
                batch_size=512,
                device=self.device,
                pad=True,
            )
            freq = pitch.squeeze(0).cpu().numpy()
            conf = periodicity.squeeze(0).cpu().numpy()
            time = np.arange(len(freq)) * self.step_ms / 1000.0
            f0_voiced = freq.copy()
            f0_voiced[conf <= self.conf_threshold] = np.nan
            return F0Contour(time, freq, conf, f0_voiced)
        except Exception as exc:
            print(f"  [pitch] failed on {audio_path}: {exc}")
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
