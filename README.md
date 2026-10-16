# Motor imagery (left vs right hand) on EEGMMIDB: cross-subject evaluation without leakage

Developed by **KN Neuron**, Neuroinformatics Student Research Group, Wrocław University of Technology.

Left/right-hand motor imagery classification on the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64 channels, 160 Hz). The project started from a pipeline reporting 84% and turned into a study of what that number actually measured, under a strict subject-level protocol.

## Main findings

All numbers are accuracy on unseen subjects (N = 106), mean over subjects with a 95% bootstrap CI over subjects. Chance is 50%.

| Question | Result |
|---|---|
| Does the original ~84% reproduce? | Yes, 83.5% (64 channels, 0.5 to 45 Hz). |
| Where does it come from? | Mostly eye and gaze signal, not motor imagery. Frontal channels F7/F8 below 4 Hz alone give 81.3% (EEGNet). An LDA on F7 and F8 alone gives 77.4%. The cue appears on the left or right of the screen and the dataset has no EOG channel. This evidence is circumstantial. |
| Honest motor imagery accuracy? | **66.7% [64.2, 69.2]** (EEGNet, 21 motor channels, frontal channels regressed out per subject, 7 to 30 Hz, 0.5 to 4 s). 68.8% if the first 0.5 s after the cue is included. |
| Does executed movement do better? | Only with covariance methods: 69.8% [67.0, 72.4] (Tangent Space + LR) vs 64.5% for imagery. EEGNet does not improve (64.7%). |
| Does hyperparameter tuning help? | No. Nested tuning gives 66.6% vs 66.7% for EEGNet (40 of 288 grid combinations) and changes the other models by -0.1 to +1.2 pp. |
| Does a spatial EEGNet-Transformer help? | No, it is the weakest model: 63.9% [61.5, 66.4]. |
| Do very high published numbers hold up? | We replicated a CNN-GRU 5-class setup ([Bouchane et al. 2025, Sensors 25(5):1399](https://www.mdpi.com/1424-8220/25/5/1399)) from its description. With a random instance split we obtain 80.2% balanced accuracy. When the six channel pairs of the same window stay on one side of the split it drops to 20.9% (chance is 20%), and with a subject-level split to 20.7%. Left/right on 103 subjects: 69.8% with a random split, 51.2% [50.4, 52.1] by subject. |

Caveats: the MI configuration was chosen after looking at results on the same folds, so the 66.7% is exploratory; the replication follows the paper's description, not its code, and the paper does not state its main split.
Full history, every table and every mistake made along the way: [`docs/lab_notebook.md`](docs/lab_notebook.md) (Polish).

## Evaluation protocol (N-LNSO)

* Splits are by subject, before any processing. 5 outer folds of ~21 test subjects; each subject is a test subject exactly once, scored by a model that never saw it.
* Inside the training subjects, ~15% are held out to select the best epoch; the test fold is used once.
* Normalization is per subject and label-free (z-score, EMS or Euclidean alignment). Eye-channel regression is per subject and runs before normalization.
* 3 seeds per model; the subject is the independent unit for all CIs (bootstrap over subjects, Wilcoxon with Holm correction for paired comparisons).
* Hyperparameter tuning, when used, is nested in the training subjects.

## Reproduce

```bash
poetry install                       # or: pip install -r requirements.txt
make test                            # 102 tests, ~3 min, no data needed
make all                             # everything from scratch (GPU, several hours): results/final/
```

`scripts/run_all.sh` runs every experiment (phases: `mi mm legacy main replication tuned`), saves training curves, logs, environment and figures, and writes `results/final/REPORT.md`. Select phases with `PHASES="replication"`, point to local EDF files with `DATA_DIR=/path`. Finished models are reused, so an interrupted run can simply be restarted. Without `DATA_DIR`, data is downloaded with kagglehub.

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
train.py ...      the earlier 4-way pipeline (see docs/legacy_pipeline.md)
notebooks/        exploratory notebooks (see notebooks/README.md)
```

## Documentation

| file | content |
|---|---|
| [`docs/lab_notebook.md`](docs/lab_notebook.md) | chronological log: all runs, results, conclusions, mistakes (Polish) |
| [`docs/code_review_map.md`](docs/code_review_map.md) | what each module does, suggested review order, known problems |
| [`docs/legacy_pipeline.md`](docs/legacy_pipeline.md) | the earlier 4-way pipeline (`train.py`) and how to extend it |
| [`docs/protocol_eksperymentu.md`](docs/protocol_eksperymentu.md), [`docs/research_plan.md`](docs/research_plan.md), [`docs/related_work.md`](docs/related_work.md) | protocol, plan, literature (some citations marked "to verify") |
| [`docs/TODO.md`](docs/TODO.md) | team task list |

## Limitations

* EEGMMIDB has no EOG channel and no fixation instruction, so the eye explanation of the old 84% cannot be separated from slow motor potentials below 4 Hz on this data. A dedicated protocol (central cue, fixation, recorded EOG) is the planned next step.
* Own BrainAccess recordings are too short (one subject, minutes) for any conclusion; the BrainAccess code is prepared for larger data and untested on real recordings.
* `train.py` was not re-run end to end after the refactoring; its results are not used in the study.

## License

MIT (see `pyproject.toml`).
