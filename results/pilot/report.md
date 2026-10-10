# Augmentation pilot — engine comparison

Per language × technique group, N control takes (train songs) were augmented by each engine (3 variants weak→strong + a passthrough). See scripts/evaluate_pilot.py for the metric definitions.

## Engines overall (share of cells passing; mean metrics)

| engine   |   pass_objective |   pass_clap |   pass_asr |   obj_mean_iqr_dist |   direction_score |   magnitude_ratio |   artefact_ratio |
|:---------|-----------------:|------------:|-----------:|--------------------:|------------------:|------------------:|-----------------:|
| legacy   |            0.941 |       0.644 |      0.667 |               0.682 |             0.178 |             1.133 |            2.555 |
| psola    |            1     |       0.21  |      0.79  |               0.456 |             0.46  |             0.37  |            0.092 |
| world_v2 |            1     |       0.309 |      0.728 |               0.392 |             0.461 |             0.417 |            0.309 |

## By technique

|                                   |   pass_objective |   pass_clap |   pass_asr |
|:----------------------------------|-----------------:|------------:|-----------:|
| ('Breathy_Group', 'legacy')       |              1   |        0.85 |       0.78 |
| ('Breathy_Group', 'world_v2')     |              1   |        0.26 |       0.78 |
| ('Falsetto_Group', 'legacy')      |              1   |        0.93 |       0.81 |
| ('Falsetto_Group', 'psola')       |              1   |        0.19 |       0.85 |
| ('Falsetto_Group', 'world_v2')    |              1   |        0.52 |       0.7  |
| ('Glissando_Group', 'legacy')     |              1   |        0.26 |       0.7  |
| ('Glissando_Group', 'psola')      |              1   |        0.15 |       0.89 |
| ('Glissando_Group', 'world_v2')   |              1   |        0.11 |       0.89 |
| ('Mixed_Voice_Group', 'world_v2') |              1   |        0.3  |       0.74 |
| ('Pharyngeal_Group', 'legacy')    |              1   |        0.33 |       0.74 |
| ('Pharyngeal_Group', 'world_v2')  |              1   |        0.33 |       0.7  |
| ('Vibrato_Group', 'legacy')       |              0.7 |        0.85 |       0.3  |
| ('Vibrato_Group', 'psola')        |              1   |        0.3  |       0.63 |
| ('Vibrato_Group', 'world_v2')     |              1   |        0.33 |       0.56 |

## By language

|                          |   pass_objective |   pass_clap |   pass_asr |
|:-------------------------|-----------------:|------------:|-----------:|
| ('chinese', 'legacy')    |             1    |        0.67 |       0.67 |
| ('chinese', 'psola')     |             1    |        0.22 |       1    |
| ('chinese', 'world_v2')  |             1    |        0.5  |       0.94 |
| ('english', 'legacy')    |             0.93 |        0.73 |       0.8  |
| ('english', 'psola')     |             1    |        0.22 |       1    |
| ('english', 'world_v2')  |             1    |        0.28 |       0.67 |
| ('french', 'legacy')     |             0.87 |        0.8  |       0.73 |
| ('french', 'psola')      |             1    |        0.22 |       0.67 |
| ('french', 'world_v2')   |             1    |        0.28 |       0.72 |
| ('german', 'legacy')     |             1    |        0.67 |       0.13 |
| ('german', 'psola')      |             1    |        0.22 |       0.78 |
| ('german', 'world_v2')   |             1    |        0.44 |       0.5  |
| ('italian', 'legacy')    |             1    |        0.67 |       0.67 |
| ('italian', 'psola')     |             1    |        0.22 |       0.33 |
| ('italian', 'world_v2')  |             1    |        0.39 |       0.61 |
| ('japanese', 'legacy')   |             1    |        0.53 |       0.73 |
| ('japanese', 'psola')    |             1    |        0.22 |       0.67 |
| ('japanese', 'world_v2') |             1    |        0.22 |       0.67 |
| ('korean', 'legacy')     |             0.87 |        0.67 |       0.67 |
| ('korean', 'psola')      |             1    |        0    |       1    |
| ('korean', 'world_v2')   |             1    |        0    |       0.83 |
| ('russian', 'legacy')    |             0.93 |        0.6  |       0.8  |
| ('russian', 'psola')     |             1    |        0.22 |       0.89 |
| ('russian', 'world_v2')  |             1    |        0.28 |       0.83 |
| ('spanish', 'legacy')    |             0.87 |        0.47 |       0.8  |
| ('spanish', 'psola')     |             1    |        0.33 |       0.78 |
| ('spanish', 'world_v2')  |             1    |        0.39 |       0.78 |

## Objective: distance from the real median (IQR units, typical variant v1, median over languages; ≤ 1 = realistic)

|                                                   |   legacy |   psola |   world_v2 |
|:--------------------------------------------------|---------:|--------:|-----------:|
| ('Breathy_Group', 'Δ H1–H2 (dB)')                 |     0.6  |  nan    |       0.04 |
| ('Breathy_Group', 'Δ aperiodicity 1–3 kHz')       |     0.45 |  nan    |       0.77 |
| ('Breathy_Group', 'Δ aperiodicity 5–8 kHz')       |     0.4  |  nan    |       0.65 |
| ('Breathy_Group', 'Δ level (dB)')                 |     0.21 |  nan    |       0.12 |
| ('Breathy_Group', 'Δ spectral tilt (dB/oct)')     |     0.17 |  nan    |       0.13 |
| ('Falsetto_Group', 'Δ F0 (cents)')                |     0.24 |    0.34 |       0.35 |
| ('Falsetto_Group', 'Δ H1–H2 (dB)')                |     0.27 |    0.4  |       0.07 |
| ('Falsetto_Group', 'Δ aperiodicity 5–8 kHz')      |     0.57 |    0.83 |       0.54 |
| ('Falsetto_Group', 'Δ spectral tilt (dB/oct)')    |     0.61 |    0.33 |       0.14 |
| ('Glissando_Group', 'glide duration (s)')         |     0.37 |    0.37 |       0.37 |
| ('Glissando_Group', 'glide size (cents, + = up)') |     0.69 |    0.09 |       0.14 |
| ('Mixed_Voice_Group', 'Δ F0 (cents)')             |   nan    |  nan    |       0.06 |
| ('Mixed_Voice_Group', 'Δ H1–H2 (dB)')             |   nan    |  nan    |       0.06 |
| ('Mixed_Voice_Group', 'Δ aperiodicity 5–8 kHz')   |   nan    |  nan    |       0.54 |
| ('Mixed_Voice_Group', 'Δ spectral tilt (dB/oct)') |   nan    |  nan    |       0.12 |
| ('Pharyngeal_Group', 'Δ F0 (cents)')              |     1.46 |  nan    |       0    |
| ('Pharyngeal_Group', 'Δ H1–H2 (dB)')              |     0.37 |  nan    |       0.04 |
| ('Pharyngeal_Group', 'Δ aperiodicity 5–8 kHz')    |     0.41 |  nan    |       0.21 |
| ('Pharyngeal_Group', 'Δ level (dB)')              |     0.38 |  nan    |       0.3  |
| ('Pharyngeal_Group', 'Δ spectral tilt (dB/oct)')  |     0.43 |  nan    |       0.1  |
| ('Vibrato_Group', 'vibrato extent (cents, peak)') |     2.48 |    0.19 |       0.15 |
| ('Vibrato_Group', 'vibrato onset delay (s)')      |     0.25 |    0.22 |       0.19 |
| ('Vibrato_Group', 'vibrato rate (Hz)')            |     0.55 |    0.09 |       0.09 |
