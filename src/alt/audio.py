"""
audio.py — audio I/O and format conversion helpers.

ASR models and forced aligners (MFA/SOFA) all expect 16 kHz mono WAV input.
This module centralises that conversion plus simple loading so no other module
has to know the resampling details.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# Target sample rate required by Whisper / wav2vec2 / MFA / SOFA / CREPE.
TARGET_SR: int = 16_000


def convert_to_16k(src: str | Path, dst: str | Path, overwrite: bool = False) -> bool:
    """Convert any audio file to 16 kHz mono 16-bit WAV via ffmpeg.

    Args:
        src:       Path to the source audio file (any format ffmpeg reads).
        dst:       Path of the WAV file to write.
        overwrite: If False and ``dst`` already exists, skip and return True.

    Returns:
        True on success (or skip), False if ffmpeg returned a non-zero code.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists() and not overwrite:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src), "-ar", str(TARGET_SR), "-ac", "1",
         "-acodec", "pcm_s16le", str(dst)],
        capture_output=True,
    )
    return result.returncode == 0


def load_audio(path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 waveform at a given sample rate.

    Args:
        path: Path to the audio file.
        sr:   Target sample rate; the signal is resampled if it differs.

    Returns:
        A tuple ``(waveform, sample_rate)`` where ``waveform`` is a 1-D
        float32 numpy array in the range [-1, 1].
    """
    import librosa                    # imported lazily — heavy import
    y, file_sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), file_sr


def duration_seconds(path: str | Path) -> float:
    """Return the duration of an audio file in seconds.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds, or ``float('nan')`` if the file cannot be read.
    """
    import librosa
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return float("nan")
