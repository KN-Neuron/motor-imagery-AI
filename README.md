# Motor imagery (left vs right hand) on EEGMMIDB: cross-subject evaluation without leakage

Author: Grzegorz Szczepanek. Developed within **KN Neuron**, Neuroinformatics Student Research Group, Wrocław University of Technology. Contributors: see the git history.

Left/right-hand motor imagery classification on the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64 channels, 160 Hz). The project started from a pipeline reporting 84% and turned into a study of what that number actually measured, under a strict subject-level protocol.

## Results at a glance

Accuracy on unseen subjects, left vs right hand, chance 50%. Brackets: 95% bootstrap CI over subjects.

| # | Experiment | Data | Split | Result |
|---|---|---|---|---|
| 1 | Original 4-way pipeline (`train.py`) | MI, binary left/right | by subject 60/20/10/10, holdout ~10 subjects, 5 seeds | **84.3 ± 2.8** (holdout) |
| 2 | Same old setting, rerun | MI, 64 ch, 0.5-45 Hz, 0-4 s | N-LNSO, 106 subjects | **83.5 [81.5, 85.5]** |
| 3 | Eye signal only (F7 and F8, below 4 Hz) | MI, 2 frontal channels | N-LNSO, 106 subjects | 81.3 [79.1, 83.4] |
| 4 | Motor imagery after eye regression | MI, 21 motor ch, 7-30 Hz, 0.5-4 s | N-LNSO, 106 subjects | **66.6 [64.2, 69.1]** |
| 5 | Executed movement, same settings, Tangent Space + LR | MM, 21 motor ch, 7-30 Hz, 0.5-4 s | N-LNSO, 106 subjects | 69.8 [67.0, 72.4] |
| 6 | Replicated published setup, left vs right | MI, 103 subjects | random instances / by subject | 69.8 / **51.2** [50.4, 52.1] |

Rows 2 to 6 are the same protocol and the same subjects, so they are comparable. Row 1 used a single small holdout.

## Main findings

All numbers are accuracy on unseen subjects (N = 106), mean over subjects with a 95% bootstrap CI over subjects. Chance is 50%.

| Question | Result |
|---|---|
| Does the original ~84% reproduce? | Yes, 83.5% (64 channels, 0.5 to 45 Hz). |
| Where does it come from? | Mostly eye and gaze signal, not motor imagery. Frontal channels F7/F8 below 4 Hz alone give 81.3% (EEGNet). An LDA on F7 and F8 alone gives 77.4%. The cue appears on the left or right of the screen and the dataset has no EOG channel. This evidence is circumstantial. |
| Honest motor imagery accuracy? | **66.6% [64.2, 69.1]** (EEGNet, 21 motor channels, frontal channels regressed out per subject, 7 to 30 Hz, 0.5 to 4 s). 68.8% if the first 0.5 s after the cue is included. |
| Does executed movement do better? | Only with covariance methods: 69.8% [67.0, 72.4] (Tangent Space + LR) vs 64.5% for imagery. EEGNet does not improve (64.7%). |
| Does hyperparameter tuning help? | No. Nested tuning gives 66.4% vs 66.6% for EEGNet (40 of 288 grid combinations) and changes the other models by -0.2 to +1.2 pp. |
| Does a spatial EEGNet-Transformer help? | No, it is the weakest model: 63.9% [61.5, 66.3]. |
| Do very high published numbers hold up? | We replicated a CNN-GRU 5-class setup ([Bouchane et al. 2025, Sensors 25(5):1399](https://www.mdpi.com/1424-8220/25/5/1399)) from its description. With a random instance split we obtain 80.2% balanced accuracy. When the six channel pairs of the same window stay on one side of the split it drops to 20.9% (chance is 20%), and with a subject-level split to 20.7%. Left/right on 103 subjects: 69.8% with a random split, 51.2% [50.4, 52.1] by subject. |

Every number with its data, channels, band, window, split and model: [`docs/results.md`](docs/results.md).

Caveats: the MI configuration was chosen after looking at results on the same folds, so the 66.6% is exploratory; the replication follows the paper's description, not its code, and the paper does not state its main split.
Full history, every table and every mistake made along the way: [`docs/lab_notebook.md`](docs/lab_notebook.md) (Polish).

## Evaluation protocol (N-LNSO)

* Splits are by subject, before any processing. 5 outer folds of ~21 test subjects; each subject is a test subject exactly once, scored by a model that never saw it.
* Inside the training subjects, ~15% are held out to select the best epoch; the test fold is used once.
* Normalization is per subject and label-free (z-score, EMS or Euclidean alignment). Eye-channel regression is per subject and runs before normalization.
* 3 seeds per model; the subject is the independent unit for all CIs (bootstrap over subjects, Wilcoxon with Holm correction for paired comparisons).
* Hyperparameter tuning, when used, is nested in the training subjects.

```
Split subject IDs into 5 groups (~21 subjects each), before any processing. Repeat for fold 1..5:

  fold 1:  [ TEST ][  ...  ][  ...  ][  ...  ][  ...  ]
  fold 2:  [  ...  ][ TEST ][  ...  ][  ...  ][  ...  ]
  ...                                         (every subject is in TEST exactly once)

  one fold:  4 groups (~85 subjects) -> TRAIN (~72) + VAL (~13, random subjects, 15%)
             per-subject preprocessing, no labels (EOG regression, then z-score / EA / EMS)
                 |                    |                      |
          fit the network      pick the best epoch     TEST: one evaluation by a model
          (3 seeds)            on VAL                  that never saw these subjects

  tuned variant: inside the 85 training subjects, 3 subject groups; each candidate is trained on 2
                 groups and scored on the 3rd; the best is refit on all 85, then TEST once.

  result: accuracy per subject (mean of 3 seeds) -> mean over 106 subjects + bootstrap 95% CI over subjects
```

## Reproduce

```bash
poetry install                       # or: pip install -r requirements.txt
make test                            # 102 tests, ~3 min, no data needed
make all                             # everything from scratch (GPU, several hours): results/final/
```

`scripts/run_all.sh` runs every experiment (phases: `mi mm legacy main replication tuned`), saves training curves, logs, environment and figures, and writes `results/final/REPORT.md`. Select phases with `PHASES="replication"`, point to local EDF files with `DATA_DIR=/path`. Finished models are reused, so an interrupted run can simply be restarted. Without `DATA_DIR`, data is downloaded with kagglehub.

## The 4-way pipeline (`train.py`)

The first version of the project, designed before the study above, and methodologically sound: subjects are split **before any preprocessing** into TRAIN 60% / VAL 20% / DEV 10% / HOLDOUT 10%. VAL selects the checkpoint (early stopping), DEV compares models and preprocessing, grid searches use GroupKFold on TRAIN+VAL subjects only, and the HOLDOUT is evaluated once at the end. There is no leakage between subjects. Its holdout result (84.3%, mean of 5 seeds) is the number that started the study, and it reproduced (83.5%) under the N-LNSO protocol. It was the right design for the question as it was then posed; what the study found is a property of the data (eye and gaze signal), not a flaw of the pipeline.

N-LNSO keeps the same principles and changes one thing: instead of a single holdout of ~10 subjects (noisy), five rotating test folds of ~21 subjects, so every subject is tested once and the CIs are over all 106. A 4-way run with a larger holdout (e.g. 20 subjects) is a reasonable alternative when a single final number is wanted.

```
4-way subject split:
  - TRAIN   (60%)  gradient descent, backprop
  - VAL     (20%)  early stopping, checkpoint selection
  - DEV     (10%)  model / preprocessing comparison across stages
  - HOLDOUT (10%)  touched ONCE at the very end, reported result

  ┌─────────────────────────────────────────────────────────┐
  │  Split subject IDs (before ANY preprocessing)           │
  │  ┌───────┐ ┌─────┐ ┌─────┐ ┌─────────┐                  │
  │  │ TRAIN │ │ VAL │ │ DEV │ │ HOLDOUT │                  │
  │  └───┬───┘ └──┬──┘ └──┬──┘ └────┬────┘                  │
  │      │        │       │         │                       │
  │  Preprocess  Preprocess  Preprocess  Preprocess         │
  │  (separate)  (separate)  (separate)  (separate)         │
  │      │        │       │         │                       │
  │      ▼        ▼       │         │ (locked away)         │
  │   train()  checkpoint │         │                       │
  │      │     selection  │         │                       │
  │      │        │       │         │                       │
  │  Stages 1-6   │       ▼         │                       │
  │  grid search  │   evaluate      │                       │
  │  CV on train+ │   best model    │                       │
  │  val subjects │   on DEV        │                       │
  │      │        │       │         │                       │
  │  Stage 7:     │       │         ▼                       │
  │  final retrain│       │    ONE evaluation               │
  │  with best    │       │    (reported result)            │
  │  config       │       │                                 │
  └─────────────────────────────────────────────────────────┘

  Stages 2-6 (CV, grid searches): GroupKFold on TRAIN+VAL only
  Stage 1 (single run):           train on TRAIN, checkpoint on VAL, eval on DEV
  Stage 7 (final retrain):        train on TRAIN, checkpoint on VAL, eval on DEV
  Stage 8 (holdout):              ONE evaluation on HOLDOUT, the reported result
```

Stages: single EEGNet run, EEGNet and ShallowConvNet grids, CSP + classical ML grid (LDA, SVM, RF, KNN, GB, LR, MLP), preprocessing grid (band, window, baseline), joint preprocessing x model grid, final retrain with the best configuration, one holdout evaluation. Grid searches write a CSV checkpoint and resume after a crash or `Ctrl+C`; the final retrain saves full learning curves.

```bash
python train.py                                                  # default config (configs/default.yaml)
python train.py --config configs/full_binary_all_channels.yaml   # custom config
python train_multiple_splits.py --config configs/full_binary_all_channels.yaml --runs 5   # repeated splits
```

Example of the final output (5 seeds, run of 2026-03-20). The best configuration differs between seeds, the Dev score is higher than the holdout score:

```
 Seed  |  Dev Acc   | Holdout Acc  | Best Model
  42   |   0.9167   |    0.8326    | EEGNet(f1=8,d=2,do=0.25,lr=0.001)
  43   |   0.8170   |    0.8968    | EEGNet(f1=8,d=2,do=0.5,lr=0.0005)
  44   |   0.8727   |    0.8197    | EEGNet(f1=8,d=2,do=0.25,lr=0.001)
  45   |   0.8250   |    0.8374    | EEGNet(f1=16,d=2,do=0.5,lr=0.001)
  46   |   0.9286   |    0.8286    | EEGNet(f1=8,d=2,do=0.5,lr=0.001)
MEAN Dev Acc:     0.8720 ± 0.0457
MEAN Holdout Acc: 0.8430 ± 0.0275
```

Any field can be overridden with a partial YAML (`data.n_subjects`, `training.epochs`, ...). Usage from a notebook, the data flow, module mapping and a guide to adding a model, a preprocessing step or a stage: [`docs/legacy_pipeline.md`](docs/legacy_pipeline.md).

Known issues of this pipeline (details in [`docs/code_review_map.md`](docs/code_review_map.md)): the grid-search loops had an indentation bug introduced during the March refactoring, fixed and covered by tests; the CV score in the grids is the best accuracy over epochs on the evaluated fold, so Dev accuracy is optimistic (the holdout is not affected); it was not re-run end to end after the refactoring.

## Repository layout

```
src/eval/         N-LNSO, models behind one interface, tuning, statistics, simulations
src/data/         loading, preprocessing, per-subject normalization
src/models/       EEGNet, ShallowConvNet, EEGNet-Transformer
src/replication/  replication of the CNN-GRU 5-class setup
src/ba/           BrainAccess module (not validated on real data yet)
scripts/          run_all.sh, benchmarks, figures, reports
configs/          one YAML per experiment
tests/            102 tests, incl. end-to-end on synthetic EDF files
docs/             lab notebook, research plan, protocol, related work
train.py ...      the 4-way pipeline (see above and docs/legacy_pipeline.md)
notebooks/        exploratory notebooks (see notebooks/README.md)
```

## Documentation

| file | content |
|---|---|
| [`docs/results.md`](docs/results.md) | all results in tables: data, split, settings, accuracy with CI |
| [`docs/lab_notebook.md`](docs/lab_notebook.md) | chronological log: all runs, results, conclusions, mistakes (Polish) |
| [`docs/code_review_map.md`](docs/code_review_map.md) | what each module does, suggested review order, known problems |
| [`docs/legacy_pipeline.md`](docs/legacy_pipeline.md) | the earlier 4-way pipeline (`train.py`) and how to extend it |
| [`docs/protocol_eksperymentu.md`](docs/protocol_eksperymentu.md), [`docs/research_plan.md`](docs/research_plan.md), [`docs/related_work.md`](docs/related_work.md) | protocol, plan, literature (some citations marked "to verify") |
| [`docs/TODO.md`](docs/TODO.md) | team task list |

## Limitations

* EEGMMIDB has no EOG channel and no fixation instruction, so the eye explanation of the old 84% cannot be separated from slow motor potentials below 4 Hz on this data. A dedicated protocol (central cue, fixation, recorded EOG) is the planned next step.
* Own BrainAccess recordings are too short (one subject, minutes) for any conclusion; the BrainAccess code is prepared for larger data and untested on real recordings.
* `train.py` was not re-run end to end after the refactoring; the numbers in this README come from `src/eval/` (N-LNSO), apart from its original 84.3% holdout result.

## License

MIT, see [`LICENSE`](LICENSE). Copyright (c) 2026 KN Neuron, Grzegorz Szczepanek.
