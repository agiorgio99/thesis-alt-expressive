# Reproducing ExpressiveALT

Step-by-step guide to rebuild every result in this repository from scratch:
baseline ASR evaluation on GTSinger (9 languages), WORLD-vocoder technique
augmentation, the Whisper fine-tuning ablation, and the CLAP embedding analysis.

All commands are run from the repository root.

---

## 0. Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu, kernel 6.x) |
| GPU | NVIDIA, ≥ 16 GB VRAM for Whisper-large-v3 fine-tuning (tested: RTX 5080). Baseline inference fits in less. |
| Disk | ~47 GB extracted GTSinger (9 languages) + ~30 GB of zips during extraction (deletable afterwards) + ~10 GB augmented audio + ~3 GB per fine-tuned model |
| Software | Conda (Miniforge recommended), CUDA 12.8 driver for Blackwell GPUs |

## 1. Environment

```bash
conda env create -f environment.yml
conda activate thesis-alt
```

Optional — only needed for forced alignment (English baseline, `alignment.enabled: true`):

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner
```

## 2. Data: GTSinger

1. Download the per-language GTSinger archives (`English.zip`, `Chinese.zip`,
   `French.zip`, `German.zip`, `Italian.zip`, `Japanese.zip`, `Korean.zip`,
   `Russian.zip`, `Spanish.zip`) from the official GTSinger release
   (NeurIPS 2024 Datasets & Benchmarks, see the paper / Hugging Face page).
2. Extract them into the layout the code expects:

   ```bash
   python scripts/extract_gtsinger_zips.py --zip-dir ~/Downloads --dry-run   # preview
   python scripts/extract_gtsinger_zips.py --zip-dir ~/Downloads
   ```

   Result: `data/GTSinger/<Language>/<Singer>/<Technique>/<Song>/<Group>/*.{wav,json,TextGrid}`.
   The script decodes zip filenames as UTF-8 (GTSinger's archives do not set the
   UTF-8 flag, so a plain `unzip` mangles Chinese/Japanese/Korean/Russian song
   folder names).

3. Sanity check — utterance counts per language should be:

   | English | Chinese | French | German | Italian | Japanese | Korean | Russian | Spanish |
   |---|---|---|---|---|---|---|---|---|
   | 3934 | 10188 | 3675 | 4855 | 2647 | 2832 | 3327 | 2419 | 4113 |

## 3. Fixed, song-disjoint test splits

Every experiment of a language is evaluated on the same held-out test set.
GTSinger repeats songs across technique folders and singers, so the split is
made over **whole songs** (~30 % of original technique utterances, balanced per
technique); no lyrics of a test song ever appear in training or validation.

```bash
for l in en zh fr de it ja ko ru es; do
    python scripts/finetune_whisper.py --split-only --language $l
done
```

Writes `results/splits/test_manifest_<lang>.json` (audio paths) and
`test_manifest_<lang>.songs.json` (held-out songs + statistics). The split is
deterministic for a given `--seed` (default 42); the committed manifests are the
ones used in the thesis.

## 4. Baseline ASR evaluation

```bash
python scripts/run_pipeline.py --config configs/baseline.yaml            # English (+ MFA alignment, CREPE pitch)
python scripts/run_pipeline.py --config configs/baseline_chinese.yaml    # likewise for
# baseline_{french,german,italian,japanese,korean,russian,spanish}.yaml
```

- Output: `results/baseline_<lang>/asr_<model>.csv` (per utterance),
  `asr_<model>_summary.csv`, `_by_technique.csv`, `_by_singer_id.csv`, `report.html`.
- Metric: **WER** for space-delimited languages, **CER** for Chinese, Japanese
  and Korean (GTSinger annotates them per character/syllable, so WER is not
  meaningful there). Chinese hypotheses are mapped to Simplified characters
  (OpenCC) before scoring.
- Quick smoke test: `--set data.limit=16`.
- HTML reports are not committed (7–30 MB each); rebuild one with
  `--stage report`.

## 5. Technique augmentation (WORLD vocoder)

For every song with a `Control_Group`, each control clip is re-synthesised with
the technique of its folder (vibrato, breathy, glissando, pharyngeal, falsetto)
over a fixed parameter grid (3–4 variants per clip, identical for every
language):

```bash
python scripts/build_augmented_dataset.py --language english --dry-run
python scripts/build_augmented_dataset.py --language english \
    --src data/GTSinger/English --dst data/GTSinger_Augmented/English --workers 8
```

Repeat with `--language <name>` and the matching `--src/--dst` for the other
languages. Vowel positions (where vibrato/glissando are applied) use a
per-language phone inventory (`VOWELS_BY_LANGUAGE`).

### 5b. Data-driven augmentation (v2) — calibration and pilot

The v1 presets above were tuned by ear on English. The v2 pipeline measures
what each technique really changes, per language, on GTSinger's paired takes
(technique vs control take of the same phrase), then compares three engines:
`legacy` (v1), `world_v2` (48 kHz, local resynthesis, calibrated parameters)
and `psola` (Praat overlap-add, pitch techniques only).

```bash
# 1. Measure real technique effects (train songs only; CPU, ~1.5 h for 9 languages)
python scripts/measure_technique_features.py --language all --workers 24 --legacy-sample 30
# 2. Report (do the v1 presets match real singers?) + per-language params.json
python scripts/calibration_report.py          # → results/calibration/report.md
# 3. Pilot: N control takes per language × technique, every engine, 3 variants + passthrough
python scripts/build_pilot.py --n-clips 8 --workers 20
# 4. Compare engines: objective re-measurement, CLAP direction/magnitude/artefact, Whisper Δerror
python scripts/evaluate_pilot.py               # → results/pilot/report.md
# 5. Engine outputs on the pilot pairs, measured on the calibration spans (~45 min)
python scripts/measure_engine_outputs.py --workers 24
# 6. Distributions per technique / language (outliers, diversity, tails,
#    register, single-singer dependence)  → results/distributions/report.md
python scripts/distribution_analysis.py
```

Full augmentation with the data-driven engines (float WAV, native 48 kHz,
`augment_manifest.csv` with every variant's parameters):

```bash
python scripts/build_augmented_dataset.py --language english --engine auto \
    --param-mode sampled --targets language --exclude-test-songs --workers 24
# → data/GTSinger_aug_auto/English   (fine-tune with --aug-src pointing there)
```

`--engine auto` = PSOLA for vibrato / glissando, world_v2 for the rest;
`--param-mode quartile|sampled|tails` (q25/q50/q75 presets, per-clip draws from
the whole real distribution, or draws favouring its tails); `--targets pooled`
uses all languages together instead of the clip's language.

Code: `src/alt/augment/` (`features.py` estimators, `world_v2.py`, `psola.py`,
`legacy.py` = v1 kept bit-exact, `engines.py` registry).

## 5c. Showcase / listening-test app

```bash
python -m app.state --rebuild      # index data/ (≈1 s)
python -m app.main                 # http://127.0.0.1:7860
```

Tabs: *Dataset* (listen to control / technique / speech takes with
spectrogram, F0, score and technique flags), *Augmentation lab* (run any
engine with editable parameters next to the real take), *ASR* (zero-shot or
fine-tuned Whisper on a phrase, an upload or the microphone, with WER/CER and
a word diff), *Listening test* (blind ratings → `results/listening/ratings.csv`),
*Results* (baselines, ablation, calibration, pilot). Audio is read from
`data/` at runtime and never bundled.

## Tests

```bash
python -m pytest tests/ -q
```

## 6. Fine-tuning ablation (Whisper-large-v3)

Four training regimes, all evaluated on the same test manifest:

| `--mode` | Training data |
|---|---|
| `mixed` | all augmented + 20 % original technique clips |
| `aug_only` | all augmented only |
| `aug_matched` | augmented, size-matched to `mixed` |
| `orig_only` | 20 % original only |

```bash
python scripts/finetune_whisper.py --mode mixed --language en --dry-run   # print split/data stats
python scripts/finetune_whisper.py --mode mixed --language en \
    --optim adamw_8bit --gradient-checkpointing
```

Models go to `results/finetune_whisper_<mode>[_<lang>]/best_model/` (not
committed). Checkpoint selection uses WER (CER for zh/ja/ko). Evaluate on the
test set with the matching config, e.g.:

```bash
python scripts/run_pipeline.py --config configs/exp1_mixed_eval.yaml --stage asr
python scripts/run_pipeline.py --config configs/exp_c_baseline_eval.yaml --stage asr   # zero-shot reference
```

## 7. CLAP analysis (English)

```bash
python scripts/clap_embed.py --roots data/GTSinger/English data/GTSinger_Augmented/English
python scripts/clap_world_passthrough.py          # WORLD "null augmentation" reference
python scripts/clap_analysis.py --emb-dir results/clap
```

## What is (not) in the repository

| Committed | Not committed (regenerate with the steps above) |
|---|---|
| code, configs, test-split manifests, per-utterance and summary CSVs, CLAP embeddings | GTSinger audio (`data/`), augmented audio, fine-tuned weights, HTML reports, run logs |
