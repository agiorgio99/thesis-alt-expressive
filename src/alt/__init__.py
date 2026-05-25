"""
alt — Automatic Lyric Transcription pipeline for expressive singing.

A modular, config-driven baseline pipeline. Each subtask (dataset loading, ASR,
forced alignment, pitch tracking) is an interchangeable component selected by a
string key in the experiment config — see ``configs/baseline.yaml``.

Subpackages / modules
---------------------
* config     — typed experiment configuration (YAML + dataclasses).
* text       — shared text normalisation / phoneme conversion.
* audio      — audio I/O and 16 kHz conversion.
* dataset    — dataset adapters (GTSinger, VocalSet, ...) + registry.
* asr        — ASR model wrappers (Whisper, wav2vec2, FireRedASR) + registry.
* alignment  — forced-alignment wrappers (MFA, SOFA) + registry.
* pitch      — CREPE F0 extraction.
* metrics    — WER / PER / hallucination / TBE computation.
* pipeline   — orchestrator that runs the selected components end to end.
"""

__version__ = "0.1.0"
