"""
io.py — shared I/O and annotation helpers for the augmentation engines.

Everything that is not specific to one augmentation engine lives here: audio +
GTSinger-JSON loading, frame/time conversion, MIDI helpers and the per-language
vowel inventories used to find the sung (vowel) part of each word.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

# WORLD analysis hop used by every engine (ms). 5 ms is pyworld's default and
# the value the legacy (v1) augmentation was built with.
FRAME_MS: float = 5.0

# Per-language vowel-phone sets, used by vowel-timed transforms (vibrato,
# glissando) to find the sung portion of each word. English uses ARPAbet
# (stress digits stripped before lookup — see ``is_vowel``). Every other
# language uses GTSinger's IPA-based phone alphabet, matched as complete
# strings (diacritics/length marks included, e.g. "aː", "ɛ̃"). Built from a
# 300-file per-language sample of the real ``ph`` annotations in
# data/GTSinger/<Language>; vowel-match rates land at 36.7–47.2 % for every
# language (English 36.8 %), a consistent range.
VOWELS_BY_LANGUAGE: dict[str, set[str]] = {
    "english": {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                "IH", "IY", "OW", "OY", "UH", "UW"},
    "chinese": {  # pinyin finals (rimes) — the vowel-bearing part of the syllable
        "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian",
        "iang", "iao", "ie", "in", "ing", "iong", "iou", "o", "ong", "ou", "u", "ua",
        "uai", "uan", "uang", "uei", "uen", "uo", "v", "van", "ve", "vn",
    },
    "french": {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ɑ̃", "ɔ", "ɔ̃", "ə", "ɛ", "ɛ̃"},
    "german": {"a", "aj", "aw", "aː", "eː", "iː", "oː", "uː", "yː", "øː", "ə", "ɛ",
               "ɪ", "ɔ", "ɔʏ", "ɐ", "œ", "ʊ", "ʏ"},
    "italian": {"a", "e", "i", "o", "u", "ɔ", "ɛ"},
    "japanese": {"a", "aː", "e", "eː", "i", "iː", "i̥", "o", "oː", "ɨ", "ɨː", "ɨ̥", "ɯ", "ɯ̥"},
    "korean": {"e", "eː", "i", "iː", "o", "oː", "u", "uː", "ɐ", "ɛ", "ɛː", "ɨ", "ɨː", "ʌ", "ʌː"},
    "russian": {"a", "e", "i", "o", "u", "æ", "ɐ", "ə", "ɛ", "ɨ", "ɪ", "ɵ", "ʉ", "ʊ"},
    "spanish": {"a", "e", "i", "o", "u"},
}

# GTSinger's numeric codes for the comma-separated per-word ``tech`` field.
TECH_CODE: dict[str, str] = {
    "mix": "1", "falsetto": "2", "breathy": "3",
    "pharyngeal": "4", "glissando": "5", "vibrato": "6",
}


def t2f(t_sec: float, frame_ms: float = FRAME_MS) -> int:
    """Convert a time in seconds to the nearest analysis-frame index.

    Args:
        t_sec:    Time in seconds.
        frame_ms: Analysis hop in milliseconds.

    Returns:
        The rounded frame index.
    """
    return int(round(t_sec * 1000.0 / frame_ms))


def midi_to_hz(midi: float) -> float:
    """Convert a MIDI note number to frequency in Hz (A4 = 69 = 440 Hz).

    Args:
        midi: MIDI note number.

    Returns:
        Frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def is_vowel(ph: str, vowels: set[str]) -> bool:
    """Return whether a GTSinger phone label is a vowel for a language.

    ARPAbet stress digits ("AH0", "IY1") are stripped before the lookup; IPA
    labels have none, so this is a no-op for the other languages.

    Args:
        ph:     Phone label from the annotation ``ph`` list.
        vowels: The language's vowel set (see ``VOWELS_BY_LANGUAGE``).

    Returns:
        True if the phone is a vowel.
    """
    return ph.rstrip("012") in vowels


def load_meta(json_path: str | Path) -> list[dict]:
    """Load a GTSinger word-level JSON annotation.

    Args:
        json_path: Path to the ``.json`` file next to a WAV.

    Returns:
        The list of word dicts (``word``, ``start_time``, ``ph``, ``note`` ...).
    """
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_audio(wav_path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float64, optionally resampled.

    Args:
        wav_path: Audio file path.
        sr:       Target sample rate, or ``None`` to keep the native rate
                  (48 kHz for GTSinger).

    Returns:
        ``(wav, sample_rate)``.
    """
    wav, native_sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float64)
    if sr is not None and native_sr != sr:
        import librosa  # imported lazily — only needed when resampling
        wav = librosa.resample(wav.astype(np.float32), orig_sr=native_sr,
                               target_sr=sr).astype(np.float64)
        return wav, sr
    return wav, native_sr


def load_utterance(wav_path: str | Path, json_path: str | Path,
                   sr: int | None = 16_000) -> tuple[np.ndarray, list[dict]]:
    """Load a GTSinger utterance (audio + word annotation).

    Kept with the legacy signature (audio resampled to 16 kHz by default) so
    the v1 engine is reproduced exactly.

    Args:
        wav_path:  Audio file path.
        json_path: Matching ``.json`` annotation.
        sr:        Target sample rate (``None`` keeps the native rate).

    Returns:
        ``(wav, meta)``.
    """
    wav, _ = load_audio(wav_path, sr)
    return wav, load_meta(json_path)
