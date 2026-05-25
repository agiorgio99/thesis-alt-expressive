# thesis-alt-expressive

Automatic Lyric Transcription (ALT) for expressive singing — baseline pipeline.

This repository reproduces the **Phase 1 baseline** (Whisper / wav2vec2 /
FireRedASR transcription, MFA / SOFA alignment, CREPE pitch tracking on the
GTSinger English subset) as a **modular, config-driven pipeline**. Later phases
(data augmentation, fine-tuning) extend it without rewriting the core.

## Design idea

Every subtask is an interchangeable component selected by a string in the
config. To swap a model you change one line of YAML — no code edits:

| Subtask    | Config key            | Built-in choices                                            |
|------------|-----------------------|-------------------------------------------------------------|
| Dataset    | `data.name`           | `gtsinger`, `vocalset` *(stub)*                             |
| ASR        | `asr.model_name`      | `whisper_small`, `whisper_largev2`, `whisper_largev3`, `wav2vec2`, `fireredasr` |
| Alignment  | `alignment.aligner`   | `mfa`, `sofa`                                               |
| Pitch      | `pitch.model_capacity`| CREPE `tiny` … `full`                                       |

Each subtask also has its own `device` field, so ASR can run on GPU while
alignment runs on CPU in the same experiment.

## Layout

```
thesis-alt-expressive/
├── configs/        YAML experiment configs (baseline.yaml provided)
├── data/           datasets + scratch (gitignored — download separately)
├── notebooks/      exploratory Jupyter notebooks
├── results/        per-experiment metric CSVs (gitignored)
├── scripts/
│   └── run_pipeline.py     CLI entry point
└── src/alt/        the pipeline package
    ├── config.py           typed config (YAML + dataclasses + CLI overrides)
    ├── text.py             shared text normalisation / phonemes
    ├── audio.py            audio I/O + 16 kHz conversion
    ├── dataset.py          dataset adapters + registry
    ├── asr.py              ASR model wrappers + registry
    ├── alignment.py        forced-alignment wrappers + registry
    ├── pitch.py            CREPE F0 extraction
    ├── metrics.py          WER / PER / TBE / FFE
    └── pipeline.py         orchestrator
```

## Install

```bash
pip install -r requirements.txt
```

External tools (not on PyPI): `ffmpeg`, MFA (`conda install -c conda-forge
montreal-forced-aligner`), and — only if used — SOFA and FireRedASR repos.
See `requirements.txt` for the exact links.

## Run

```bash
# Full baseline
python scripts/run_pipeline.py --config configs/baseline.yaml

# Quick smoke test: 20 utterances, Whisper small, CPU
python scripts/run_pipeline.py --config configs/baseline.yaml \
    --set data.limit=20 asr.model_name=whisper_small asr.device=cpu

# Only the ASR stage
python scripts/run_pipeline.py --config configs/baseline.yaml --stage asr
```

Results are written to `results/<experiment_name>/` as CSV files
(`inventory.csv`, `asr_<model>.csv`, per-technique / per-singer breakdowns,
`alignment_<aligner>_tbe.csv`, `pitch_f0_stats.csv`).

## Extending

* **New dataset** — subclass `DatasetAdapter`, decorate with
  `@register_dataset("name")`, set `data.name`. (`VocalSetAdapter` is a stub
  showing the pattern.)
* **New ASR model** — subclass `ASRModel`, register with `@register_asr`.
* **New aligner** — subclass `Aligner`, register with `@register_aligner`.

## Status

Phase 1 (baseline) scaffold. Augmentation (PDAugment, F0 perturbation) and
Whisper fine-tuning (Phase 2–3) are not yet implemented — `metrics.f0_frame_error`
is already provided for the FFE analysis those phases need.
