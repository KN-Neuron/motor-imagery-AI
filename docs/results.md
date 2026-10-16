# Results at a glance

Every number in one place: what data, which split, which settings. Numbers are from the full from-scratch run (`results/final/summary_all.txt`, 2026-10-10); the section in the last column points to the discussion in [`docs/lab_notebook.md`](lab_notebook.md), whose earlier tables differ by up to 0.4 pp. Raw tables and figures from the full run are written by `scripts/run_all.sh` to `results/final/` (`summary_all.txt`, `REPORT.md`, `figures/`).

## Common setup

* **Data:** PhysioNet EEGMMIDB, 160 Hz. *Imagery (MI)* = runs R04, R08, R12 (left vs right fist). *Executed movement (MM)* = runs R03, R07, R11. N = 106 subjects after the standard exclusions (105 in the first benchmark, which also drops subject 089).
* **Split (N-LNSO):** by subject, 5 outer folds of ~21 test subjects; each subject is a test subject exactly once, scored by a model that never saw it. ~15% of the training subjects select the best epoch. 3 seeds, averaged per subject.
* **Metric:** accuracy on unseen subjects, mean over subjects, **[95% bootstrap CI over subjects]**. Chance for left vs right = 50%.
* **Normalization:** per subject, label-free. `zscore` = per subject and channel z-score, `EA` = Euclidean alignment, `EMS` = exponential moving standardization, `none`.
* **Models:** `EEGNet`, `Shallow` (ShallowConvNet), `Deep`, `CSP+LDA`, `TS+LR` (Tangent Space + logistic regression), `EEGNet-Transformer`.

## 1. The old ~84%: it reproduces, and it is mostly eyes (MI, N = 106)

| Experiment | Channels | Band (Hz) | Window (s) | Model + norm | Accuracy | Section |
|---|---|---|---|---|---|---|
| old setting reproduced | 64 | 0.5-45 | 0-4 | EEGNet zscore | **83.5 [81.5, 85.5]** | 3.4 |
| old setting, narrower band | 64 | 7-30 | 0-4 | EEGNet zscore | 81.8 [79.7, 84.0] | 3.4 |
| band 4-40 | 64 | 4-40 | 0-4 | EEGNet zscore | 77.2 | 3.4 |
| without the first 0.5 s | 64 | 7-30 | 0.5-4 | EEGNet zscore | 65.1 [62.6, 67.7] | 3.4 |
| motor channels only | 21 motor | 7-30 | 0-4 | EEGNet zscore | 68.3 [66.0, 70.8] | 3.4 |
| wideband, motor channels only | 21 motor | 0.5-45 | 0-4 | EEGNet zscore | 81.0 [78.6, 83.2] | 3.5 |
| wideband, **frontal channels only** | 10 frontal (Fp, AF, F7, F8) | 0.5-45 | 0-4 | EEGNet zscore | **81.5 [79.3, 83.7]** | 3.5 |
| wideband, occipital channels only | 9 occipital | 0.5-45 | 0-4 | EEGNet zscore | 69.1 [67.1, 71.0] | 3.5 |
| **F7 and F8 only, below 4 Hz** | F7, F8 | 0.5-4 | 0-4 | EEGNet zscore | **81.3 [79.1, 83.4]** | 3.6 |
| F7-F8 difference, 8 averages | F7, F8 | 0.5-4 | 0-4 | LDA | 77.4 [75.1, 79.7] | 3.6 |
| motor channels after eye regression | 21 motor, 6 frontal regressed out | 0.5-45 | 0-4 | EEGNet zscore | 69.7 [67.6, 71.8] | 3.6 |

Reading: a pair of frontal channels below 4 Hz, with no motor cortex at all, gives 81%. The eye and gaze signal explains most of the old 84%; removing it per subject by regression drops motor-channel accuracy from ~81% to ~69%. Evidence is circumstantial: the dataset has no EOG channel and the target appears on the left or right of the screen.

## 2. Honest motor imagery after eye regression (MI, N = 106)

21 motor channels, six frontal references (Fp1, Fp2, AF7, AF8, F7, F8) regressed out per subject **before** normalization.

| Model + norm | 7-30 Hz, 0-4 s | 7-30 Hz, 0.5-4 s (**main**) | 0.5-45 Hz, 0.5-4 s |
|---|---|---|---|
| EEGNet zscore | 68.8 [66.4, 71.3] | **66.6 [64.2, 69.1]** | 67.0 [64.9, 69.2] |
| EEGNet max-norm zscore | 68.5 | 66.5 | 66.5 |
| Shallow zscore | 68.0 | 64.1 | 62.9 |
| CSP+LDA EA | 67.2 | 66.2 | 57.9 |
| TS+LR EA | 65.9 | 64.5 | 59.4 |
| EEGNet none | 64.9 | 62.9 | 62.1 |

Residual below 4 Hz (EEGNet, 21 motor channels, 0.5-4 Hz, 0-4 s): 80.8% without regression, **62.1% [60.4, 63.8]** with regression. Whether the remainder is un-removed eye signal or slow motor potentials cannot be decided on this data (section 3.7). The configuration was chosen after looking at results on the same folds, so these numbers are exploratory.

## 3. Executed movement vs imagery (N = 106, same settings)

| Setting | Imagery | Movement |
|---|---|---|
| old setting, 64 ch, 0.5-45 Hz, EEGNet zscore | 83.5 | 85.0 [83.1, 86.8] |
| F7/F8 below 4 Hz, EEGNet zscore | 81.3 | 78.8 [76.2, 81.2] |
| F7-F8, LDA | 77.4 | 74.0 [71.3, 76.7] |
| motor channels, eye regression, 7-30 Hz, 0.5-4 s: EEGNet zscore | 66.6 | 64.7 [62.7, 66.8] |
| same, CSP+LDA EA | 66.2 | 69.4 [66.7, 72.0] |
| same, **TS+LR EA** | 64.5 | **69.8 [67.0, 72.4]** |

Movement is better than imagery only for covariance methods (about +3 to +5 pp, ~70%). Eyes dominate movement too. Paired tests imagery vs movement were not run.

## 4. Hyperparameter tuning does not help (MI, nested inside the training subjects)

| Model | Fixed hyperparameters | After tuning |
|---|---|---|
| EEGNet zscore (40 of 288 grid combinations) | 66.6 [64.2, 69.1] | 66.4 [63.8, 68.9] |
| CSP+LDA EA (n_components) | 66.2 | 66.4 [63.5, 69.4] |
| TS+LR EA (C) | 64.5 | 65.7 [62.8, 68.7] |
| Shallow zscore (8 candidates) | 64.1 | 64.4 [62.0, 66.7] |
| EEGNet-Transformer zscore (8 candidates) | n/a | 63.9 [61.5, 66.3] |

Tuning changes results by -0.2 to +1.2 pp, inside the CIs. The transformer is the weakest model.

## 5. Replication of a published 5-class CNN-GRU ([Bouchane et al. 2025](https://www.mdpi.com/1424-8220/25/5/1399))

Reimplemented from the paper's description, 3 seeds. 5 classes = left fist, right fist, both fists, both feet, baseline (B = rest, 50% of windows). Balanced accuracy is the right metric (chance = 20% for 5 classes). Left vs right uses plain accuracy (chance 50%).

| Experiment | Subjects | Task | Split | Accuracy | **Balanced acc.** |
|---|---|---|---|---|---|
| paper7_random | 7 | 5-class | random instances (6 channel pairs of one window can be on both sides) | 73.6 | **80.2** |
| paper7_trial | 7 | 5-class | all pairs of a window together, subjects shared | 23.0 | **20.9** |
| paper7_subject | 7 | 5-class | by subject | 22.5 | **20.7** |
| all_random | 103 | 5-class | random instances | 39.2 | 39.2 |
| all_trial | 103 | 5-class | trial-grouped | 32.2 | 25.6 |
| all_subject | 103 | 5-class | by subject | 30.9 | 24.8 |
| all_random_lr | 103 | left vs right | random instances | 69.8 | 69.8 |
| all_subject_lr | 103 | left vs right | by subject | **51.2 [50.4, 52.1]** | 51.3 |

A high score appears only when windows leak across the split. Without leakage the 7-subject setup is at chance, and left vs right on 103 subjects is at chance (majority class 50.4%). The paper does not state which split its main tables use; this shows that such numbers occur only with a leaking split, not that its split leaked.

## 6. The 4-way pipeline (`train.py`, original result)

| Setting | Split | Result |
|---|---|---|
| binary left vs right (`binary_all` run of 2026-03-20), 5 seeds | by subject: TRAIN 60% / VAL 20% / DEV 10% / HOLDOUT 10% | Dev 87.2 ± 4.6, **Holdout 84.3 ± 2.8** (accuracy ×100) |

Holdout is ~10 subjects, so a single value is noisy. The old setting reproduces under N-LNSO at 83.5% (section 1). Dev is higher than Holdout in this run, consistent with the Dev/CV score taking the best accuracy over epochs on the evaluated fold (an optimistic choice; the holdout is not affected).

## 7. Other checks

| Experiment | Result | Section |
|---|---|---|
| Within-subject CSP+LDA (4-40 Hz, 0.5-2.5 s, 105 subjects) | 57.7 | 3.2 |
| Within-subject TS+LR (same) | 54.7 | 3.2 |
| Cross-subject TS+LR N-LNSO (same) | 63.6 | 3.2 |
| Main benchmark (4-40 Hz, 0.5-2.5 s, 64 ch, N = 105): EEGNet zscore | **71.4 [69.4, 73.5]** | 3.3 |
| same: EEGNet max-norm EA / CSP+LDA EA / EEGNet EMS | 67.8 / 67.5 / 67.3 | 3.3 |
| same: Shallow EA / EEGNet EA / EEGNet none / TS+LR EA / CSP+LDA none | 65.1 / 65.0 / 63.9 / 63.6 / 57.6 | 3.3 |
| same: Deep EA | 50.5 (does not learn) | 3.3 |

The 71.4% of the first benchmark includes the frontal channels and the first 0.5 s, so it also contains the eye signal. Rerunning everything from scratch (`scripts/run_all.sh`) reproduced these numbers within 0.4 pp.
