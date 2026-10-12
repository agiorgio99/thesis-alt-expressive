# Literature notes (meeting list C: Ariana, vibrato literature)

Notes on papers read for the thesis, with what each one means for this
project. Numbers are quoted from the papers; "→ us" lines link them to the
code and results here.

---

## 1. Shuyunying, Pereira, Zhang — *Probing transposition in audio embeddings: human key change vs. algorithmic pitch-shift* (ISMIR LBD 2026, submitted)

Ariana Pereira (UPF) is the second author (contact via Rafael, list C item 12).

**Setup.** One vocalist, 5 melodies (~5 s, sustained "ah"), original key (O)
and human re-performances (H) at ±2, ±4, ±7 semitones, 2 takes each.
Each original also transposed with Rubber Band (phase vocoder) without
(M_rub) and with (M_rubF) formant preservation; 190 recordings.
Embeddings: MULE (contrastive log-mel), CLAP (audio–text, HTSAT), MERT-330M
(last hidden layer, mean-pooled). Distance = angle, arccos(cos(x, y)).
Machine–human gap Δ = [d(O, M) − d(O, H)] / d(O1, O2), normalised by the
same-melody same-key retake distance. Statistics on the 5 melody-level means
(the two takes are not independent).

**Findings.**
- Without formant preservation, machine transpositions lie farther from the
  original than human ones (Δ: MULE 0.65*, CLAP 0.21 n.s., MERT 0.61**).
- Formant preservation shrinks the gap (MULE −0.39, CLAP 0.00, MERT 0.08):
  statistically indistinguishable from human under CLAP and MERT.
- But M_rubF does not coincide with H: d(H, M_rubF) ≈ d(O, H). Same distance
  from the original ≠ same representation.
- The gap grows with interval size without formant preservation (largest at
  ±7 st) and is flat with it; up/down asymmetries exist only for the machine
  shifts and disappear with formant preservation.
- CLAP is the least sensitive model: it does not separate all conditions.
- Embeddings carry key information (distance grows with interval for H too)
  and lean strongly on timbre / spectral envelope.

**→ us.**
- Our pilot evaluation of the engines uses only CLAP (scripts/clap_*.py,
  results/pilot). This paper finds CLAP the *least* discriminative model →
  add MERT (330M, last layer, mean pool) as a second embedding. Cheap, and
  makes the "v2 is closer to real than v1" claim much stronger.
- Their Δ normalisation transfers directly: d(control, augmented) vs
  d(control, real technique), normalised by a within-singer baseline
  (GTSinger has no retakes; candidates: distance between the Falsetto and
  Mixed Voice takes of the same phrase, or between control takes of
  neighbouring phrases). Also report d(augmented, real technique), the
  analogue of their H–M_rubF result.
- Our pitch engines (WORLD, PSOLA) keep the spectral envelope like their
  formant-preserving condition, and world_v2 additionally applies the
  measured median ΔLTAS for falsetto / mixed voice → in principle closer to
  a human register change than either Rubber Band condition. Worth testing.
- Large shifts are the risky ones: our falsetto F0 jump is ~330–420 cents
  median and up to an octave for basses, i.e. beyond their ±7 st. Falsetto is
  also the only technique that is significantly harder for Whisper
  (results/paper/report.md, 6/8 languages), so it matters most.
- Statistical design matches ours (melody-level means ↔ our song bootstrap).

**Questions for Ariana.** Which formant analysis / pitch-shifting tools
(Praat, LPC, WORLD?) and settings; do the human re-performances show formant
shifts with pitch (how much of the envelope should move with F0?); would
their Δ protocol accept technique-vs-control instead of key change; plans to
extend to more singers (GTSinger has 20, with paired control takes).

---

## 2. Diaz & Rothman — *Acoustical comparison between samples of good and poor vibrato in singers* (J. Voice 17(2), 2003)

**Setup.** 8 professional singers (5 m, 3 f), 5 good + 5 poor vibrato samples
each (unanimous judgement of 4 judges), samples > 1.5 s, constant pitch and
vowel. Time-varying rate and extent from LPC on 170 ms windows with a 12 ms
hop (≈ 80 values/s). Four periodicity measures, each as % of the mean:
rate SD, rate max−min, extent SD, extent max−min.

**Findings (group means, good vs poor).**

| measure (% of mean)        | good | poor | p     |
|----------------------------|-----:|-----:|------:|
| rate variability (SD)      | 4.8  | 5.9  | 0.23  |
| rate total deviation       | 21.9 | 28.3 | 0.13  |
| extent variability (SD)    | 6.6  | 10.3 | 0.005 |
| extent total deviation     | 48.8 | 75.2 | 0.007 |

Good vibrato = more periodic; the quality cue is mainly the stability of the
**extent**, not of the rate (singers control rate better than extent).

**→ us.**
- Our vibrato generator (world_v2.vibrato_offsets, shared with PSOLA) adds
  slow jitter with `jitter = 0.1`: extent SD ≈ 10 % (= their *poor* mean) and
  rate SD ≈ 5 % (between good and poor). Suggest extent jitter ≈ 0.065 and
  rate jitter ≈ 0.045 (their "good" values), or better: measure the same
  quantities on GTSinger and sample them like the other targets.
- Measurement: add time-varying rate/extent (170 ms windows, 12 ms hop on
  our F0 track) to features.py → rate-SD % and extent-SD % per span,
  comparable with this paper. Our F0 track already has a finer time
  resolution than their 80 values/s, so LPC itself is not needed.

---

## 3. Anand, Wingate, Smith, Shrivastav — *Acoustic parameters critical for an appropriate vibrato* (J. Voice 26(6), 2012)

**Setup.** 8 voice majors (bass, tenor, alto, soprano), /a/ at low, middle,
high pitch; 1 s segments resynthesised with STRAIGHT with vibrato rate
2–10 Hz (step 2) × depth 2–26 Hz (step 6) → 600 stimuli; 9 listeners
(4 expert, 5 naive), yes/no "appropriate for current-day opera", each
stimulus 3 times at 70 and 90 dB SPL.

**Findings.**
- Rate: 6 Hz most appropriate; drops above and below; 2 Hz always rejected;
  < 4 Hz is "wobble". At high mean f0 (> ~25 st above 106 Hz) 4–8 Hz is
  acceptable; at low f0 only a narrow band around 6 Hz.
- Depth (peak deviation in Hz): 2 and 8 Hz preferred; appropriateness falls
  above 8 Hz, most steeply for bass/tenor. Higher f0 tolerates wider depth.
- Interaction: wide depth is acceptable only at ~6 Hz; slow + wide = worst.
- Loudness (70 vs 90 dB) no effect. Naive vs expert listeners no difference
  (t = −1.87, p = 0.06). Inter-listener r = 0.69, intra-listener ICC 0.66.
- Literature ranges quoted: mean rates 4–7 Hz; typical extent ±1 semitone
  (~6 % of f0); Ekholm et al. 5.1–6.8 Hz, 3.5–8 % of f0; Howes et al.
  6.28–7.59 Hz, 0.41–1.58 st.
- Caveat: judgements of opera "appropriateness" on isolated vowels, not
  naturalness in pop/folk singing.

**→ us.**
- GTSinger real vibrato (calibration spans, all languages, n = 3339): rate
  median 5.5 Hz (IQR 4.7–6.2); 14.6 % below 4 Hz, but the 5 % quantile is
  3.03 Hz = the bottom of our search band (VIB_BAND = 3–10 Hz), so most of
  these are failed estimates (as FINDINGS.md already notes). Extent median
  54 cents peak, 95 % = 160 cents; 19 % above 100 cents (±1 st).
- Guard rails for sampled vibrato: rate within 4–8 Hz (no "wobble", no
  trill-like speed), and treat rate estimates at the band edge as missing
  when building the target distribution. Wide extents only near 5–7 Hz.
  These are perceptual limits for *good* vibrato; our targets come from real
  GTSinger singing, which is the right reference for ASR realism, so use
  them as bounds, not as targets.
- Vibrato vs trill boundary (meeting item 13): cited source Hakes, Shipp &
  Doherty 1987, "Acoustic properties of straight tone, vibrato, trill and
  trillo", J. Voice 1:148–156 — still to read.
- Listening test: naive listeners judge vibrato like experts → non-expert
  raters are defensible; a yes/no "appropriate / natural vibrato" item with
  repeated presentations gives a reliability measure (ICC) cheaply.
