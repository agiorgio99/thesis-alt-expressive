# CLAP embedding analysis — augmented vs original

- Model: `laion/larger_clap_music_and_speech`
- Utterances embedded: 9018 (real 3777, augmented 3987, control 1254, WORLD passthrough 1254)
- Embedding dim: 512

## How to read this

- **auc_real_tech_vs_control** — validity check, section 0. Real technique takes vs their real control takes: same singer, same phrase, same session, technique the only systematic difference. **Near 0.5 would mean CLAP is blind to technique and nothing below is interpretable; high values license the rest.** A high AUC alongside a high `cos_real_control` is not a contradiction: the technique displacement is small in norm but consistent in direction.
- **direction_vs_resynth / magnitude_vs_resynth** — section 4b. The same geometry measured from the WORLD passthrough centroid rather than the raw control, i.e. what the *transformation* contributed once resynthesis is taken as given. These are the numbers to quote if the passthrough population is present.
- **auc_debiased** — separability after projecting out the single vocoder direction. The drop from `auc_raw` is the share of the real-vs-augmented separability owed to resynthesis rather than to the technique models. **Values well below 0.5 do not mean the populations are inversely separable**: with group-disjoint folds they mean the residual separating direction flips sign from fold to fold, i.e. nothing stable is left. Read them as 0.5-equivalent. Note also that when the axis is estimated from `aug - control` rather than from the passthrough, it removes some genuine technique signal along with the artefact, so these are a lower bound on what survives.
- **direction_score** — cosine between the displacement augmentation applies to a control clip and the displacement that separates real technique recordings from control. **1.0 = the augmentation pushes audio exactly the way the real technique does; 0 = orthogonal; negative = the wrong way.**
- **magnitude_ratio** — how far it pushes, relative to the real gap. <1 under-shoots (too subtle), >1 over-shoots (caricature).
- **fd_ratio_aug_over_control** — Fréchet(real, aug) / Fréchet(real, control). **<1 means augmentation moved the audio closer to real technique than doing nothing.**
- **auc_mean** — real-vs-augmented classifier. 0.5 = indistinguishable, 1.0 = the augmentation leaves an obvious domain signature.
- **aug->real accuracy** — a technique classifier trained only on synthetic audio, tested on real audio: the CLAP-space analogue of the Phase-3 fine-tuning result.
- **mmd_p_*** — label-permutation p-value for the MMD. A raw MMD^2 is not interpretable on its own (the unbiased estimator can go slightly negative when the populations overlap); read the p-value instead. Note that MMD is low-powered at these sample sizes in 512 dimensions — treat `auc_mean` as the more sensitive test and MMD as a distribution-free confirmation.

## 0. Validity check — can CLAP see technique on REAL audio?

```
     technique    n  auc_real_tech_vs_control  auc_std split_by
       breathy  450                    0.9459   0.0214     song
     glissando  450                    0.9388   0.0090     song
mixed_falsetto  317                    0.9959   0.0050     song
    pharyngeal  698                    0.9011   0.0478     song
       vibrato  604                    0.9738   0.0112     song
           ALL 2519                    0.8423   0.0330     song
```

## 1–2. Centroid geometry and displacement direction

```
     technique  n_real  n_aug  n_control  cos_real_aug  cos_real_control  cos_aug_control  direction_score  magnitude_ratio  spread_real  spread_aug
       breathy     225    675        225        0.6945            0.9777           0.7042           0.0776           3.6411       0.1159      0.1248
     glissando     225    900        225        0.6614            0.9878           0.6835          -0.0792           5.0934       0.1207      0.1486
mixed_falsetto     164    459        153        0.7144            0.9400           0.6604           0.3995           2.3791       0.0734      0.0865
    pharyngeal     349   1047        349        0.6910            0.9847           0.7184          -0.0926           4.2937       0.1482      0.1573
       vibrato     302    906        302        0.7876            0.9656           0.7422           0.4238           2.7367       0.1264      0.1157
```

## 3. Distribution distances (Fréchet / MMD)

```
     technique  fd_real_aug  fd_real_control  fd_ratio_aug_over_control  mmd_real_aug  mmd_p_real_aug  mmd_real_control  mmd_p_real_control
       breathy       0.5907           0.0538                    10.9699        0.5331           0.005            0.0572               0.005
     glissando       0.6477           0.0316                    20.5011        0.5206           0.005            0.0284               0.005
mixed_falsetto       0.5477           0.1221                     4.4845        0.6794           0.005            0.2095               0.005
    pharyngeal       0.6026           0.0448                    13.4476        0.4326           0.005            0.0317               0.005
       vibrato       0.4371           0.0729                     5.9981        0.4113           0.005            0.0760               0.005
```

## 4. Real vs augmented separability (singer-disjoint AUC)

```
     technique    n  auc_mean  auc_std
       breathy  900       1.0      0.0
     glissando 1125       1.0      0.0
mixed_falsetto  623       NaN      NaN
    pharyngeal 1396       1.0      0.0
       vibrato 1208       1.0      0.0
           ALL 5252       1.0      0.0
```

## 4b. Geometry with the resynthesis artefact removed

```
     technique  direction_vs_resynth  magnitude_vs_resynth  fd_real_resynth  cos_real_resynth  direction_debiased  magnitude_debiased
       breathy                0.1416                1.5107           0.5556            0.7361              0.1067              1.6424
     glissando               -0.0359                2.1717           0.5731            0.7231              0.0615              2.3336
mixed_falsetto                0.4578                0.7787           0.6293            0.6961              0.3993              1.0408
    pharyngeal                0.0620                0.9320           0.5471            0.7362              0.0439              1.0391
       vibrato                0.6652                1.4665           0.5346            0.7464              0.6505              1.4684
```

## 4c. Separability before/after removing the vocoder axis

```
     technique    n  auc_raw  auc_debiased
       breathy  900      1.0        0.9792
     glissando 1125      1.0        0.8816
mixed_falsetto  623      NaN           NaN
    pharyngeal 1396      1.0        0.2997
       vibrato 1208      1.0        0.8811
           ALL 5252      1.0        0.2582
```

## 5. Cross-domain technique probe

```
direction  n_train  n_test  accuracy  balanced_accuracy  chance
real->aug     1265    3987    0.5360             0.5818     0.2
aug->real     3987    1265    0.3628             0.4095     0.2
```

## 6. kNN retrieval purity against real audio

```
     technique  k  n_aug_queries  purity_aug_query  purity_real_query
       breathy 10            675            0.3415             0.6422
     glissando 10            900            0.2197             0.4889
mixed_falsetto 10            459            0.5183             0.8561
    pharyngeal 10           1047            0.2919             0.7195
       vibrato 10            906            0.6935             0.7629
```

## 7. Zero-shot text prompting

```
     technique origin    n  zeroshot_accuracy  mean_sim_correct_prompt  chance
       breathy   real  225             0.0000                   0.1699  0.1667
       breathy    aug  675             0.0000                   0.0757  0.1667
     glissando   real  225             0.7378                   0.3480  0.1667
     glissando    aug  900             0.6578                   0.2866  0.1667
mixed_falsetto   real  164             0.0000                   0.2003  0.1667
mixed_falsetto    aug  459             0.0000                   0.1143  0.1667
    pharyngeal   real  349             0.0000                   0.1262  0.1667
    pharyngeal    aug 1047             0.0000                   0.1933  0.1667
       vibrato   real  302             0.0000                   0.2816  0.1667
       vibrato    aug  906             0.0055                   0.3011  0.1667
```

## Paired analysis (augmented clip vs its source control)

```
     technique    n  cos_to_source  shift_norm  shift_alignment
       breathy  675         0.6479      0.8319           0.0662
     glissando  900         0.6184      0.8665          -0.0615
mixed_falsetto  459         0.6306      0.8550           0.3485
    pharyngeal 1047         0.6578      0.8188          -0.0778
       vibrato  906         0.6873      0.7825           0.3368
```

## CLAP distance vs measured WER

```
     technique  fd_real_aug  fd_real_control  fd_ratio_aug_over_control  mmd_real_aug  mmd_p_real_aug  mmd_real_control  mmd_p_real_control  wer_real  per_real  wer_aug  per_aug  wer_gap_aug_minus_real
       breathy       0.5907           0.0538                    10.9699        0.5331           0.005            0.0572               0.005    0.2075    0.1434   0.2089   0.1426                  0.0014
     glissando       0.6477           0.0316                    20.5011        0.5206           0.005            0.0284               0.005    0.1796    0.1157   0.1376   0.0859                 -0.0420
mixed_falsetto       0.5477           0.1221                     4.4845        0.6794           0.005            0.2095               0.005    0.1150    0.0730   0.1391   0.0839                  0.0241
    pharyngeal       0.6026           0.0448                    13.4476        0.4326           0.005            0.0317               0.005    0.2005    0.1402   0.2096   0.1380                  0.0091
       vibrato       0.4371           0.0729                     5.9981        0.4113           0.005            0.0760               0.005    0.2004    0.1114   0.2879   0.1561                  0.0875
```

## Figures

- `geometry_bars.png`
- `probe_confusion_aug_to_real.png`
- `probe_confusion_real_to_aug.png`
- `projection_tsne.png`