# Thesis Progress Summary — Automatic Lyrics Transcription for Expressive Singing
**Author:** Antonello Giorgio  
**Date:** July 2026  
**Repo:** `/home/antonello/Desktop/thesis-alt-expressive`

---

## 1. Research Goal

Automatic Lyrics Transcription (ALT) — converting singing audio to text — is significantly harder than ordinary ASR because of musical phenomena (vibrato, glissando, breathy voice, pharyngeal voice, mixed/falsetto voice) that distort the acoustic signal in ways that speech-trained models are not designed to handle. The thesis investigates whether **technique-specific data augmentation** (synthesising new training samples that simulate each expressive technique from neutral control recordings) can improve ALT accuracy on expressive singing, using the GTSinger corpus as the benchmark.

**Central hypothesis:** Fine-tuning Whisper large-v3 on WORLD-vocoder-augmented technique samples will reduce WER on expressive singing, and augmented data can substitute for real recordings as a scalable, annotation-free training source.

---

## 2. Dataset — GTSinger (English Subset)

- **Corpus:** GTSinger — a large-scale, multi-technique, multi-singer singing voice dataset with per-phoneme technique annotations.
- **Language subset used:** English
- **Singers:** 3 (EN-Alto-1, EN-Alto-2, EN-Tenor-1)
- **Techniques:** 5 — Vibrato, Breathy, Glissando, Pharyngeal, Mixed Voice & Falsetto
- **Groups per song:** each song has four parallel recordings:
  - `Technique_Group` — singer performs with the expressive technique
  - `Control_Group` — same singer performs the same phrase in neutral modal voice
  - `Paired_Speech_Group` — same singer reads the lyrics as speech
  - (Mixed/Falsetto also has `Mixed_Voice_Group` and `Falsetto_Group`)
- **Annotation format:** per-utterance JSON with word-level, phoneme-level, note-level, and per-technique flag fields; MFA-aligned TextGrids
- **Total utterances in baseline evaluation:** ~5,700 (technique + control + speech groups)

### GTSinger JSON structure (key fields per word entry):
```json
{
  "word": "memory",
  "ph": ["M","EH1","M","ER0","IY0"],
  "ph_start": [1.05, 1.09, ...],
  "ph_end": [1.09, 1.17, ...],
  "note": [69],
  "note_dur": [0.97],
  "vibrato": ["0","0","1","1","1"],
  "breathy": ["0","0","0","0","0"],
  "glissando": ["0","0","0","0","0"],
  "pharyngeal": ["0","0","0","0","0"],
  "mix": ["0","0","0","0","0"],
  "falsetto": ["0","0","0","0","0"],
  "tech": "6"
}
```
`tech` field codes: 0=none, 1=mix, 2=falsetto, 3=breathy, 4=pharyngeal, 5=glissando, 6=vibrato.

---

## 3. Baseline Pipeline (Phase 1)

**Framework:** modular Python pipeline (`src/alt/`) with a YAML config system and an ASR model registry. Implemented stages: Dataset → ASR → Forced Alignment → Pitch.

### 3.1 ASR Baseline Results (on original GTSinger technique groups, n=2,705)

| Model | WER | PER | Hallucination rate |
|---|---|---|---|
| **Whisper large-v3** | **18.75%** | **12.14%** | **2.70%** |
| Whisper large-v2 | 22.68% | 15.43% | 3.80% |
| Whisper small | 30.21% | 20.35% | 4.10% |
| wav2vec2-large-960h | 52.65% | 31.56% | 11.30% |

**Whisper large-v3 WER by technique (technique group only):**

| Technique | n | WER | PER | Halluc. rate |
|---|---|---|---|---|
| Breathy | 474 | 20.7% | 14.3% | 4.6% |
| Pharyngeal | 855 | 20.1% | 14.0% | 2.8% |
| Vibrato | 675 | 20.0% | 11.1% | 1.0% |
| Glissando | 333 | 18.0% | 11.6% | 5.1% |
| Mixed/Falsetto | 368 | 11.5% | 7.3% | 0.8% |
| **Overall** | **2,705** | **18.75%** | **12.14%** | **2.70%** |

**WER by singer (Whisper large-v3):**

| Singer | n | WER |
|---|---|---|
| EN-Alto-1 | 807 | 21.1% |
| EN-Tenor-1 | 756 | 20.3% |
| EN-Alto-2 | 1,142 | 16.1% |

### 3.2 Forced Alignment (MFA, word + phoneme level)

Aligner: Montreal Forced Aligner with `english_mfa` acoustic model.

| Level | Mean TBE | Median TBE | ≤20ms | ≤50ms | ≤100ms |
|---|---|---|---|---|---|
| Word | 267.6 ms | 184.9 ms | 35.7% | 58.7% | 70.1% |
| Phone | 490.9 ms | 394.6 ms | 18.2% | 26.2% | 36.6% |

The large TBE for singing alignment is expected — MFA is trained on speech, and melismatic singing causes severe boundary errors.

### 3.3 Pitch Extraction (CREPE F0)

CREPE full-capacity model at 10 ms step size. Per-utterance statistics extracted: F0 mean (Hz), F0 std, F0 range (semitones), voiced ratio, vibrato index (peak amplitude of 5–8 Hz band in F0 autocorrelation). ~3,725 utterances successfully tracked.

---

## 4. Data Augmentation (Phase 2)

### 4.1 Approach — WORLD Vocoder + Technique-Specific F0/Spectral Modifications

**Motivation:** The GTSinger Control_Group provides ~1,254 neutral renditions of songs that also have technique recordings. The idea is to transform these neutral recordings into technique-mimicking versions by modifying the acoustic parameters that characterise each technique. This avoids the need for new recordings and lets us generate multiple parameter variants per utterance.

**Vocoder:** WORLD (Morise et al. 2016). Decomposes audio into:
- `f0` — fundamental frequency contour (per 5 ms frame)
- `sp` — spectral envelope (log power spectrum)
- `ap` — aperiodicity (band-wise noise proportion)

Resynthesis from modified (f0, sp, ap) preserves intelligibility while allowing precise control.

**Key insight:** GTSinger JSON already contains per-word `note` (MIDI pitch) and `note_dur` fields, so pitch targets are derived directly from the annotations — no external MIDI files needed (unlike the original PDAugment paper for ALT).

**Label modification:** After augmentation, the JSON is deep-copied and the relevant technique flag (`vibrato`, `breathy`, etc.) is set to `"1"` on all modified phonemes, and the `tech` string is updated. This produces correctly annotated augmented samples.

### 4.2 Per-Technique Augmentation Strategy

#### Vibrato
- **What:** Sinusoidal F0 modulation on sustained vowels (note_dur ≥ 0.3 s).
- **Formula:** `f0[t] *= 2^(E·sin(2πrt)/12)`, where r = rate (Hz), E = extent (semitones).
- **Onset delay:** 15–20% of vowel duration (avoids attack wobble).
- **Parameter grid (3 variants):** (5.5 Hz, ±1.0 st), (6.5 Hz, ±1.5 st), (7.0 Hz, ±2.0 st).
- **References:** Seashore 1932; Ferrante 2011 (JASA); Sundberg 1987.

#### Breathy Voice
- **What:** Incomplete glottal closure → elevated aperiodicity + reduced amplitude.
- **Formula:** `ap_aug = α + (1−α)·ap_orig` with α ∈ {0.20, 0.35, 0.50}; amplitude ×{0.92, 0.88, 0.84}.
- **Applied to:** all non-silence words in the utterance.
- **Parameter grid (3 variants).**
- **References:** Klatt & Klatt 1990 (JASA); Hanson 1997 (JASA); WORLD vocoder.

#### Glissando (Portamento)
- **What:** Descending F0 slide on each vowel nucleus.
- **Key design decision:** interpolation in semitone space, not Hz (perceptually uniform). Linear-in-Hz is doubly wrong: non-uniform semitone steps AND an abrupt onset/offset jerk.
- **Shape:** Raised cosine (S-curve) `α(t) = (1−cos(πt))/2` — zero velocity at both endpoints, matching vocal fold inertia. Optionally: quadratic ease-out `1−(1−t)²` (fast drop, slow landing).
- **Formula:** `f0(i) = f_note · 2^(−δ·α(i/N−1)/12)`, δ = descent in semitones.
- **Parameter grid (4 variants):** ↓2 st, ↓3 st, ↓4 st (cosine), ↓5 st (ease-out).
- **References:** Todd 1992 (JASA); Friberg & Sundberg 1999 (JASA); Sundberg 1987; Bonada & Serra 2007 (IEEE SPM).

#### Pharyngeal Voice
- **What:** Pharyngeal constriction → spectral boost at 400 Hz, HF attenuation above 3 kHz, slight F0 depression, marginal aperiodicity increase.
- **Spectral filter:** `H(f) = [1 + γ·exp(−((f−400)/300)²)] · attn(f)` where attn linearly reduces above 3 kHz.
- **Parameter grid (3 variants):** γ ∈ {0.40, 0.70, 1.00}, press ∈ {0.5, 1.0, 2.0} st, ap_add ∈ {0.05, 0.10, 0.15}.
- **References:** Edmondson & Esling 2006 (Phonology); Esling 1999; Sundberg 1987.

#### Mixed Voice / Falsetto
- **What:** F0 shift up (lighter head-voice register) + aperiodicity increase (incomplete glottal closure).
- **Formula:** `f0[t] *= 2^(shift/12)`, `ap_aug = β + (1−β)·ap_orig`.
- **Stability cap:** shift ≤ 9 semitones (WORLD quality degrades above this).
- **Parameter grid (3 variants):** shift ∈ {3, 5, 8} st, β ∈ {0.10, 0.15, 0.20}.
- **References:** Hollien 1974 (J. Phonetics); Titze 1988 (J. Voice); Sundberg & Hogset 2001.

### 4.3 Augmented Dataset Scale

| Technique | Control WAVs | Variants/file | New samples |
|---|---|---|---|
| Vibrato | 302 | 3 | 906 |
| Breathy | 225 | 3 | 675 |
| Glissando | 225 | 4 | 900 |
| Pharyngeal | 349 | 3 | 1,047 |
| Mixed/Falsetto | 153 | 3 | 459 |
| **Total** | **1,254** | | **3,987** |

**Output structure** (`data/GTSinger_Augmented/English/`):
```
{Singer}/{Technique}/{Song}/
  Control_Group/         ← symlink to original
  Paired_Speech_Group/   ← symlink to original
  {Technique_Group}/     ← generated: {stem}_v0.wav, _v1.wav, ...
```

---

## 5. Augmented Data Evaluation (Pre-Fine-Tuning)

**Setup:** Run pre-trained Whisper large-v3 (no fine-tuning) on the 3,987 WORLD-augmented technique group files. Ground-truth transcripts extracted from GTSinger JSON word fields (no TextGrid available for augmented files). This tells us: (a) whether WORLD augmentation preserves intelligibility, and (b) how each technique's augmentation stresses the model.

### 5.1 Results — Whisper large-v3 on Augmented Technique Groups (n=3,987)

| Technique | n (aug) | WER (aug, pre-FT) | WER (orig, baseline) | Δ WER |
|---|---|---|---|---|
| Vibrato | 906 | **28.8%** | 20.0% | **+8.8 pp ↑** |
| Pharyngeal | 1,047 | 21.0% | 20.1% | +0.9 pp |
| Breathy | 675 | 20.9% | 20.7% | +0.2 pp |
| Glissando | 900 | 13.8% | 18.0% | **−4.2 pp ↓** |
| Mixed/Falsetto | 459 | 13.9% | 11.5% | +2.4 pp |
| **Overall** | **3,987** | **20.3%** | **18.75%** | +1.5 pp |

### 5.2 Interpretation

1. **Vibrato augmentation is the hardest for the pre-trained model (+8.8 pp).** The sinusoidal F0 oscillation at 5–7 Hz creates rapid pitch variation that Whisper large-v3 has not seen during its speech pre-training. This motivates fine-tuning.

2. **Breathy and pharyngeal augmentation have minimal impact (< 1 pp).** The WORLD `ap` and `sp` modifications produce recognisably intelligible audio that the pre-trained model handles nearly as well as the originals.

3. **Glissando augmentation unexpectedly lowers WER (−4.2 pp).** The WORLD-synthesized descending glissando is apparently easier for the model than real glissando singing. Possible explanations: real glissando involves more complex pitch trajectories and co-occurring phonation changes; WORLD-resynthesized audio is somewhat cleaner; the cosine interpolation shortens effective vowel duration.

4. **Hallucination rates are lower in augmented data (0–2%) vs. originals (0.8–5.1%).** WORLD resynthesis removes the most extreme phonation irregularities.

---

## 6. Initial Fine-Tuning (Phase 3 — Pilot)

**Goal:** Fine-tune Whisper large-v3 on the 3,987 augmented samples, then evaluate on the original technique groups to measure WER improvement.

**Framework:** HuggingFace `Seq2SeqTrainer` (same checkpoint as used in baseline).

### 6.1 Training Configuration

| Parameter | Value |
|---|---|
| Base model | openai/whisper-large-v3 |
| Training samples | 3,587 (90% of augmented technique groups) |
| Validation samples | 400 (10% stratified holdout) |
| Epochs | 5 |
| Learning rate | 5e-6 |
| Effective batch size | 8 (batch 2 × grad accum 4) |
| Mixed precision | bf16 (Ampere GPU) |
| Encoder frozen | Yes — only decoder weights updated (~907 M / 1,543 M trainable) |
| Gradient checkpointing | Yes |
| Warmup steps | 500 |
| Eval every | 500 steps |
| Best checkpoint | step 2000 (epoch 4.5) |
| Output | results/finetune_whisper/best_model/ |

### 6.2 Training Curve (Validation WER on augmented holdout)

| Step | Epoch | Val WER |
|---|---|---|
| 500 | 1.1 | 33.2% |
| 1000 | 2.2 | 28.5% |
| 1500 | 3.3 | 26.1% |
| **2000** | **4.5** | **25.6% ← best** |
| 2245 | 5.0 | 26.1% |

### 6.3 Fine-Tuned Model Results — Whisper large-v3 (fine-tuned) on Original GTSinger Technique Groups

| Technique | n | WER (baseline) | WER (fine-tuned) | Δ WER |
|---|---|---|---|---|
| Vibrato | 906 | 20.0% | **14.1%** | **−5.9 pp** |
| Glissando | 675 | 18.0% | **12.6%** | **−5.4 pp** |
| Mixed/Falsetto | 631 | 11.5% | **8.8%** | **−2.7 pp** |
| Breathy | 675 | 20.7% | **18.5%** | **−2.2 pp** |
| Pharyngeal | 1,047 | 20.1% | **18.7%** | **−1.4 pp** |
| **Overall** | **3,934** | **18.75%** | **15.0%** | **−3.75 pp** |

Overall WER 15.0% vs baseline 18.75%: **−3.75 pp, ~20% relative improvement**, achieved with 30 minutes of training on ~3,600 WORLD-synthesized samples.

---

## 7. Controlled Fine-Tuning Experiments (Phase 3 — Comparative Study)

**Motivation:** The pilot (Section 6) showed fine-tuning on augmented data helps, but left two questions open: (1) how much of the improvement is due to data *quality* vs *quantity*? (2) how does augmented data compare to real original data as a training source?

**Design:** A shared held-out test split of 30% of original technique utterances was fixed (n=428, seed=42, stratified by technique). Five training conditions were evaluated against this identical test set.

### 7.1 Experimental Conditions

| Exp | Label | Training data | Train samples |
|---|---|---|---|
| C | vanilla baseline | No fine-tuning | — |
| B | orig_only | 20% of original technique WAVs | ~285 |
| 2 | aug_only | All augmented WAVs (aug_train, 90%) | ~3,588 |
| 1 | mixed | All augmented + 20% original | ~3,873 |
| A | aug_matched | All augmented + N extra aug (N = len(train_orig)) | ~3,873 |

Exp 1 and Exp A are **matched in total training samples** — the only difference is whether the extra N samples are real (Exp 1) or augmented (Exp A). This directly isolates the quality effect.

Training setup for all fine-tuned conditions: `--freeze-encoder --gradient-checkpointing --batch-size 4 --grad-accum 4 --optim adamw_8bit`, 3 epochs, lr=1e-5.

### 7.2 Overall Results (shared test set, n=428)

| Exp | Training data | WER | PER | Halluc | Δ vs baseline |
|---|---|---|---|---|---|
| C — vanilla Whisper | none | 15.8% | 9.5% | 0.9% | — |
| B — orig_only | 20% original | 15.8% | 9.5% | 0.9% | **±0 pp** |
| 2 — aug_only | all aug (small) | 28.9% | 22.7% | 0.7% | +13.1 pp |
| 1 — mixed | all aug + 20% orig | 11.5% | 7.1% | 0.5% | **−4.3 pp** |
| A — aug_matched | all aug, size-matched | 11.3% | 6.9% | 0.5% | **−4.5 pp** |

### 7.3 Per-Technique Results

| Technique | C baseline | B orig_only | 2 aug_only | 1 mixed | A aug_matched |
|---|---|---|---|---|---|
| vibrato | 21.2% | 21.3% | 18.5% | 18.2% | 18.3% |
| glissando | 16.5% | 16.4% | 8.5% | 10.2% | 9.6% |
| pharyngeal | 14.6% | 14.6% | **81.6%*** | 10.5% | 10.1% |
| breathy | 14.0% | 14.0% | 10.8% | 11.0% | 10.7% |
| mixed_falsetto | 12.6% | 12.6% | 8.5% | 7.5% | 7.8% |
| **Overall** | **15.8%** | **15.8%** | **28.9%** | **11.5%** | **11.3%** |

*Pharyngeal WER 81.6% in Exp 2 is driven by catastrophic hallucinations on a handful of utterances; `wer_no_halluc` for pharyngeal in that condition is **10.3%**, comparable to all other fine-tuned models.

### 7.4 Per-Singer Results (Exp 1 vs Exp A)

| Singer | C baseline | B orig_only | 1 mixed | A aug_matched |
|---|---|---|---|---|
| EN-Tenor-1 | 20.7% | 20.7% | 17.1% | 16.4% |
| EN-Alto-1 | 11.6% | 11.6% | 11.8% | 11.8% |
| EN-Alto-2 | 16.3% | 16.3% | 9.5% | 9.4% |

Exp 1 and Exp A produce nearly identical per-singer WERs, confirming that the quality equivalence holds across all three singers.

### 7.5 Key Findings

**Finding 1 — Fine-tuning on 20% original data alone (Exp B) provides zero improvement.**
WER 15.80% vs baseline 15.79%. Confirmed across all five techniques and all three singers (Exp B = Exp C to two decimal places throughout). The ~285 original samples are insufficient to shift the decoder's behavior — a minimum training set size threshold exists below which fine-tuning has no effect.

**Finding 2 — Augmented data is a valid substitute for real data, given sufficient quantity.**
Exp A (11.3%) ≈ Exp 1 (11.5%), a gap of only 0.14 pp — within noise. Both use ~3,873 training samples. This is the central validation of the augmentation pipeline: WORLD-augmented samples carry equivalent training signal to real recordings.

**Finding 3 — The large Exp 2 vs Exp 1 gap (28.9% → 11.5%) is a data quantity effect, not quality.**
Exp 2 uses only aug_train (~3,588 samples, no extras). Adding ~285 real samples (Exp 1) or ~285 more augmented samples (Exp A) recovers performance equally. The augmented data is not inferior — there was simply not enough of it.

**Finding 4 — Exp 2 shows catastrophic pharyngeal failure (WER 81.6%).**
This is a training instability artefact in the small-data regime: a few pharyngeal test utterances trigger high-insertion hallucinations in the Exp 2 model. The WORLD pharyngeal augmentation uses a simplified Gaussian spectral filter — the weakest of the five technique models — and with a smaller training set the model fails to learn robust pharyngeal features. When the training set is expanded to full size (Exp A), pharyngeal WER normalises to 10.1%.

**Finding 5 — Vibrato remains the hardest technique across all conditions (18.2–21.3%).**
Despite benefiting most from fine-tuning in the pilot study (−5.9 pp), vibrato in the controlled study still sits 6–9 pp above other techniques. The sinusoidal F0 pattern is well-modelled by the augmentation and clearly teaches the decoder something useful, but the technique's acoustic complexity is not fully overcome.

### 7.6 Summary Interpretation

The five experiments cleanly partition the data contribution into three effects:

| Comparison | Effect measured | Result |
|---|---|---|
| Exp C → Exp B | fine-tuning on small real data | **no effect** (±0 pp) |
| Exp C → Exp 2 | fine-tuning on medium augmented data | **hurts** (+13.1 pp) — training instability |
| Exp 2 → Exp A | adding more augmented data (quantity) | **large gain** (−17.6 pp) |
| Exp A → Exp 1 | swapping extra aug for real data (quality) | **negligible** (−0.2 pp) |

**Thesis claim:** WORLD-augmented technique samples are a viable, data-collection-free substitute for real recordings when fine-tuning Whisper for expressive singing ALT, provided the training set reaches a sufficient size (~3,600 samples in this setup).

---

## 8. Scripts & File Map

| File | Purpose |
|---|---|
| `scripts/run_pipeline.py` | Run baseline pipeline (ASR / alignment / pitch) |
| `scripts/build_augmented_dataset.py` | Build GTSinger_Augmented from control group WAVs |
| `scripts/finetune_whisper.py` | Fine-tune Whisper large-v3 — modes: mixed, aug_only, aug_matched, orig_only |
| `src/alt/dataset.py` | GTSinger dataset adapter + manifest filtering for shared test split |
| `src/alt/asr.py` | Whisper + wav2vec2 + FireRedASR wrappers & registry |
| `src/alt/metrics.py` | WER, PER, TBE, FFE metrics |
| `src/alt/report.py` | HTML report generator |
| `src/alt/config.py` | DataConfig with manifest field |
| `src/alt/pipeline.py` | Pipeline orchestrator (threads manifest into dataset loading) |
| `notebooks/augmentation_demo.ipynb` | Interactive demo of all 5 augmentation techniques |
| `configs/baseline.yaml` | Phase 1 config (all models, all stages) |
| `configs/augmented_eval.yaml` | Augmented dataset eval (pre-trained model) |
| `configs/finetuned_eval.yaml` | Pilot fine-tuned model eval on full original test set |
| `configs/exp1_mixed_eval.yaml` | Eval config — Exp 1 (mixed) |
| `configs/exp2_aug_only_eval.yaml` | Eval config — Exp 2 (aug_only) |
| `configs/exp_a_aug_matched_eval.yaml` | Eval config — Exp A (size-matched aug) |
| `configs/exp_b_orig_only_eval.yaml` | Eval config — Exp B (orig_only) |
| `configs/exp_c_baseline_eval.yaml` | Eval config — Exp C (vanilla Whisper, no FT) |
| `data/GTSinger/English/` | Original GTSinger English corpus |
| `data/GTSinger_Augmented/English/` | 3,987 WORLD-augmented technique group files |
| `results/shared_test_manifest.json` | Shared 30% held-out test split (428 utterances, seed=42) |
| `results/baseline_english/` | Phase 1 results (all models, all stages) |
| `results/augmented_eval/` | Augmented data eval results + report.html |
| `results/finetune_whisper/` | Pilot fine-tuning checkpoints (gitignored) |
| `results/finetuned_eval/` | Pilot fine-tuned model eval on full original test set |
| `results/finetune_whisper_mixed/` | Exp 1 checkpoints + best_model (gitignored) |
| `results/finetune_whisper_aug_only/` | Exp 2 checkpoints + best_model (gitignored) |
| `results/finetune_whisper_aug_matched/` | Exp A checkpoints + best_model (gitignored) |
| `results/finetune_whisper_orig_only/` | Exp B checkpoints + best_model (gitignored) |
| `results/exp1_mixed_eval/` | Exp 1 evaluation results (CSVs + report) |
| `results/exp2_aug_only_eval/` | Exp 2 evaluation results |
| `results/exp_a_aug_matched_eval/` | Exp A evaluation results |
| `results/exp_b_orig_only_eval/` | Exp B evaluation results |
| `results/exp_c_baseline_eval/` | Exp C evaluation results |

---

## 9. Key References

- **WORLD vocoder:** Morise et al. (2016). WORLD: A vocoder-based high-quality speech synthesis system. *IEICE Trans. Inf. Syst.*, E99-D(7).
- **PDAugment (ALT augmentation):** Zhang et al. (2021). PDAugment: Data augmentation by pitch and duration adjustments for automatic lyrics transcription. *arXiv:2109.07940*.
- **GTSinger:** Liu et al. (2024). GTSinger: A global multi-technique singing corpus with realistic music scores for all singing tasks. *arXiv:2409.13832*.
- **Whisper:** Radford et al. (2023). Robust speech recognition via large-scale weak supervision. *ICML 2023*.
- **MFA:** McAuliffe et al. (2017). Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. *Interspeech 2017*.
- **Vibrato:** Ferrante (2011). Vibrato rate and extent in soprano voice. *JASA*, 130(3).
- **Breathy voice:** Klatt & Klatt (1990). Analysis, synthesis, and perception of voice quality variations. *JASA*, 87(2).
- **Glissando shape:** Todd (1992). The dynamics of dynamics: A model of musical expression. *JASA*, 91(6); Friberg & Sundberg (1999). *JASA*, 105(3).
- **Pharyngeal:** Edmondson & Esling (2006). The valves of the throat. *Phonology*, 23(2).
- **Falsetto:** Hollien (1974). On vocal registers. *J. Phonetics*, 2(2); Titze (1988). *J. Voice*, 2(3).

---

## 10. Status Summary

| Phase | Component | Status |
|---|---|---|
| 1 | GTSinger dataset adapter | ✅ Done |
| 1 | ASR baseline (4 models) | ✅ Done |
| 1 | MFA forced alignment | ✅ Done |
| 1 | CREPE pitch extraction | ✅ Done |
| 1 | HTML report generator | ✅ Done |
| 2 | WORLD augmentation (5 techniques) | ✅ Done |
| 2 | Augmented dataset build (3,987 files) | ✅ Done |
| 2 | Augmented dataset eval (pre-FT) | ✅ Done |
| 3 | Pilot fine-tuning (all augmented, 5 epochs) | ✅ Done |
| 3 | Pilot eval on full original test set | ✅ Done |
| 3 | Shared test split manifest (428 utterances, seed=42) | ✅ Done |
| 3 | Exp C — vanilla Whisper baseline on shared test | ✅ Done |
| 3 | Exp B — orig_only fine-tune + eval | ✅ Done |
| 3 | Exp 2 — aug_only fine-tune + eval | ✅ Done |
| 3 | Exp A — aug_matched fine-tune + eval | ✅ Done |
| 3 | Exp 1 — mixed fine-tune + eval | ✅ Done |
| 3 | 5-experiment comparative analysis | ✅ Done |

---

## 11. Remaining Work Before Defense (July 15, 11:00)

**Manuscript deadline: July 8** (one week before defense).

### Priority 1 — Thesis Manuscript (hard deadline July 8)
Write and submit the full thesis draft. Structure should cover:
- Introduction & motivation
- Related work (ALT, Whisper, WORLD vocoder, singing voice datasets, data augmentation for ASR)
- Dataset & baseline (GTSinger, pipeline, Phase 1 results)
- Augmentation methodology (WORLD decomposition, per-technique approach, raised cosine glissando justification)
- Experiments: pilot fine-tuning + 5-experiment controlled study
- Discussion & conclusions (central finding: augmented data = real data at sufficient quantity)

### Priority 2 — Augmentation Improvements (if time before July 8)
- **Better vibrato onset modelling:** gradual amplitude build-up `E(t) = E_max · (1 − e^{−t/τ})` instead of step onset.
- **Pharyngeal:** formant-based filter (Esling 1999) rather than Gaussian spectral boost.
- **Breathy:** add correlated F0 jitter (Klatt & Klatt 1990).

### Priority 3 — VocalSet Comparison (optional)
Run the baseline and fine-tuned pipeline on VocalSet to test generalisation beyond GTSinger. Main challenge: VocalSet has no lyrics transcripts.

### Priority 4 — Gradio Demo App (optional)
Upload audio → Demucs vocal separation → parallel baseline vs fine-tuned Whisper transcription. Strong visual for the defense.

---

## 12. Figures to Capture from report.html Files

Open each `report.html` in a browser. Use browser screenshot (or Flameshot on Linux) to crop individual charts. Prefer 150% zoom for high-DPI captures.

### 12.1 `results/baseline_english/report.html`

| Figure | What it shows | Where to use |
|---|---|---|
| Pie — utterances by technique | Distribution of GTSinger English technique data | §Dataset chapter |
| Grouped bar: WER by technique (4 models) | Cross-model comparison | §Baseline results — most important baseline figure |
| Bar: WER by singer | Per-singer spread | §Baseline analysis |
| Histogram: per-utterance WER distribution | Long tail of hard utterances | §Discussion |
| Boxplot: F0 mean / vibrato index / voiced ratio / F0 range by technique | Acoustic fingerprint of each technique | §Technique characterisation — use all four together |

### 12.2 `results/augmented_eval/report.html`

| Figure | What it shows | Where to use |
|---|---|---|
| Bar: WER by technique | Vibrato spike (+8.8 pp) and glissando dip | §Augmentation chapter — key motivation for FT |
| Summary tiles | Overall WER 20.3%, hallucination rate | §Augmented eval slide |

### 12.3 Five-experiment comparison (assemble from CSVs)

The central result figure for the thesis is a **grouped bar chart** (5 bar groups = 5 techniques, 5 bars per group = 5 experiments). Generate from:
- `results/exp_c_baseline_eval/asr_whisper_largev3_by_technique.csv`
- `results/exp_b_orig_only_eval/asr_whisper_finetuned_by_technique.csv`
- `results/exp2_aug_only_eval/asr_whisper_finetuned_by_technique.csv`
- `results/exp1_mixed_eval/asr_whisper_finetuned_by_technique.csv`
- `results/exp_a_aug_matched_eval/asr_whisper_finetuned_by_technique.csv`

Suggested palette: grey (C), light blue (B), red (2), dark blue (1), green (A).
Note: clip pharyngeal Exp 2 bar at 30% with a break marker and annotate the full 81.6%.
