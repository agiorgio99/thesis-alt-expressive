# ExpressiveALT: Technique-Specific Data Augmentation and Whisper Fine-Tuning for Automatic Lyrics Transcription on Expressive Singing

Automatic Lyric Transcription (ALT) for expressive singing. A modular,
config-driven pipeline covering all three phases of the thesis:

1. **Baseline** — Whisper / wav2vec2 / FireRedASR transcription, MFA forced
   alignment, CREPE pitch tracking on the GTSinger English subset.
2. **Data augmentation** — WORLD-vocoder-based, technique-specific synthesis
   (vibrato, breathy, glissando, pharyngeal, mixed/falsetto) that turns neutral
   `Control_Group` recordings into labelled technique-mimicking samples.
3. **Fine-tuning** — Whisper large-v3 fine-tuned on the augmented data, with a
   controlled 5-experiment comparative study isolating the quality vs.
   quantity contribution of augmented vs. real training data.

**Central finding:** WORLD-augmented technique samples are a viable,
data-collection-free substitute for real recordings when fine-tuning Whisper
for expressive singing ALT, provided the training set reaches sufficient size
(overall WER 15.8% → 11.3%, a ~28% relative reduction, on a shared held-out
test set).

## Thesis summary

### Baseline (Phase 1) — ASR on original GTSinger technique groups (n=2,705)

| Model | WER | PER | Hallucination rate |
|---|---|---|---|
| **Whisper large-v3** | **18.75%** | **12.14%** | **2.70%** |
| Whisper large-v2 | 22.68% | 15.43% | 3.80% |
| Whisper small | 30.21% | 20.35% | 4.10% |
| wav2vec2-large-960h | 52.65% | 31.56% | 11.30% |

MFA forced alignment (word level): mean TBE 267.6 ms, 58.7% within 50 ms.
CREPE F0 extraction succeeded on ~3,725 utterances.

### Augmentation (Phase 2)

WORLD-vocoder decomposition (f0 / spectral envelope / aperiodicity) driven by
per-technique acoustic models, applied to the 1,254 `Control_Group` WAVs to
produce **3,987 augmented samples** across 5 techniques
(`data/GTSinger_Augmented/English/`). Ground-truth GTSinger JSON annotations
are deep-copied and re-labelled with the synthesised technique.

### Fine-tuning (Phase 3) — controlled 5-experiment comparison

Shared held-out test set (n=428, seed=42, stratified by technique). All
fine-tuned conditions: `--freeze-encoder --gradient-checkpointing --optim
adamw_8bit`, 3 epochs, lr=1e-5.

| Exp | Training data | WER | PER | Halluc |
|---|---|---|---|---|
| C — vanilla Whisper | none | 15.8% | 9.5% | 0.9% |
| B — orig_only | 20% original | 15.8% | 9.5% | 0.9% |
| 2 — aug_only | all augmented (small) | 28.9% | 22.7% | 0.7% |
| 1 — mixed | all augmented + 20% original | 11.5% | 7.1% | 0.5% |
| **A — aug_matched** | all augmented, size-matched | **11.3%** | **6.9%** | 0.5% |

Key findings:

- Fine-tuning on a small amount of real data alone (Exp B) has **zero
  effect** — the sample size is below the threshold needed to move the
  decoder.
- Augmented data is a **valid substitute for real data**, given sufficient
  quantity: Exp A (11.3%) ≈ Exp 1 (11.5%), a 0.14 pp gap within noise.
- The large Exp 2 → Exp 1/A jump is a **data quantity effect, not quality** —
  augmented samples are not inferior, there simply weren't enough of them in
  Exp 2.

A pilot fine-tune (all augmented data only, 5 epochs, `results/finetune_whisper/`)
achieved 15.0% WER vs. an 18.75% baseline on the full original test set,
motivating the controlled study above.

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
├── configs/        YAML experiment configs (baseline + augmented + per-experiment eval)
├── data/           datasets — GTSinger + GTSinger_Augmented (gitignored — download, see below)
├── results/        per-experiment metric CSVs + HTML reports (tracked);
│                   finetune_whisper*/ checkpoints (gitignored — download, see below)
├── scripts/
│   ├── run_pipeline.py             CLI entry point (ASR / alignment / pitch)
│   ├── build_augmented_dataset.py  builds GTSinger_Augmented from Control_Group WAVs
│   └── finetune_whisper.py         fine-tunes Whisper large-v3 (mixed / aug_only / aug_matched / orig_only)
├── src/alt/        the pipeline package
│   ├── config.py           typed config (YAML + dataclasses + CLI overrides)
│   ├── text.py             shared text normalisation / phonemes
│   ├── audio.py            audio I/O + 16 kHz conversion
│   ├── dataset.py          dataset adapters + registry + manifest filtering
│   ├── asr.py              ASR model wrappers + registry (Whisper / wav2vec2 / FireRedASR)
│   ├── alignment.py        forced-alignment wrappers + registry
│   ├── pitch.py            CREPE F0 extraction
│   ├── metrics.py          WER / PER / TBE / FFE
│   ├── report.py           HTML report generator
│   └── pipeline.py         orchestrator
└──environment.yml
```

## Install

```bash
conda env create -f environment.yml
conda activate thesis-alt
```

`environment.yml` covers the full pipeline: baseline ASR/alignment/pitch,
WORLD augmentation, and Whisper fine-tuning (incl. `bitsandbytes` for the
`adamw_8bit` optimizer used in the Phase 3 experiments).

External tools not installable via conda/pip:
- **MFA** — keep it in its own env: `conda create -n aligner -c conda-forge montreal-forced-aligner`
- **FireRedASR** — clone [FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR) and point `asr.extra.repo_dir` / `asr.extra.model_dir` at it (only needed if you select `asr.model_name: fireredasr`)
- **SOFA** — only needed if `alignment.aligner: sofa` is selected

## Data & fine-tuned model weights

`data/` and the `results/finetune_whisper*/` checkpoint directories are
gitignored (audio corpus + model weights are too large for git). Download
them from this Google Drive folder:

**https://drive.google.com/drive/folders/1_Yq2Dpr6zHRgGMcrGrO7SGSJMavCvIgX?usp=sharing**

The folder mirrors this repo's `data/` and `results/` directories exactly.
To use it:

1. Download the `data/` and `results/` folders from the Drive link.
2. Copy/merge them into the root of your local clone (i.e. so you end up with
   `thesis-alt-expressive/data/...` and `thesis-alt-expressive/results/...`,
   merging with the `results/` content already tracked in git rather than
   overwriting it).

What's inside:

| Path (after merging) | Contents |
|---|---|
| `data/GTSinger/English/` | Original GTSinger English corpus (3 singers, 5 techniques) |
| `data/GTSinger_Augmented/English/` | 3,987 WORLD-augmented technique samples (Phase 2 output) |
| `results/finetune_whisper/best_model/` | Pilot fine-tune (all augmented data, 5 epochs) |
| `results/finetune_whisper_mixed/best_model/` | Exp 1 — all augmented + 20% original |
| `results/finetune_whisper_aug_only/best_model/` | Exp 2 — all augmented data only |
| `results/finetune_whisper_aug_matched/best_model/` | Exp A — all augmented, size-matched to Exp 1 |
| `results/finetune_whisper_orig_only/best_model/` | Exp B — 20% original data only |

All CSV metrics and HTML reports (`results/baseline_english/`,
`results/augmented_eval/`, `results/exp*_eval/`, etc.) are already committed
to this repo and need no download.

## Run

```bash
# Full baseline
python scripts/run_pipeline.py --config configs/baseline.yaml

# Quick smoke test: 20 utterances, Whisper small, CPU
python scripts/run_pipeline.py --config configs/baseline.yaml \
    --set data.limit=20 asr.model_name=whisper_small asr.device=cpu

# Only the ASR stage
python scripts/run_pipeline.py --config configs/baseline.yaml --stage asr

# Evaluate a fine-tuned checkpoint (after downloading results/, see above)
python scripts/run_pipeline.py --config configs/finetuned_eval.yaml

# Rebuild the augmented dataset from the original Control_Group WAVs
python scripts/build_augmented_dataset.py \
    --src data/GTSinger/English --dst data/GTSinger_Augmented/English

# Fine-tune Whisper large-v3 (e.g. the aug_matched experiment)
python scripts/finetune_whisper.py --mode aug_matched \
    --aug-src data/GTSinger_Augmented/English --orig-src data/GTSinger/English \
    --freeze-encoder --gradient-checkpointing --optim adamw_8bit
```

Results are written to `results/<experiment_name>/` as CSV files
(`inventory.csv`, `asr_<model>.csv`, per-technique / per-singer breakdowns,
`alignment_<aligner>_tbe.csv`, `pitch_f0_stats.csv`) plus an HTML report.

## Extending

* **New dataset** — subclass `DatasetAdapter`, decorate with
  `@register_dataset("name")`, set `data.name`. (`VocalSetAdapter` is a stub
  showing the pattern.)
* **New ASR model** — subclass `ASRModel`, register with `@register_asr`.
* **New aligner** — subclass `Aligner`, register with `@register_aligner`.
