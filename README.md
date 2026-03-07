# MindStride — EEG Motor Imagery Classification

Modular Python pipeline for classifying left-hand / right-hand / rest motor imagery from the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64-channel EEG, 160 Hz).

Developed by **KN Neuron** — Neuroinformatics Student Research Group, Wrocław University of Technology.

## Project Structure

```
motor-imagery-AI/
├── configs/
│   └── default.yaml                # all hyperparameters in one place
├── src/
│   ├── __init__.py
│   ├── config.py                   # YAML loading & merging
│   ├── engine.py                   # train_step, eval_step, train loop, CV
│   ├── utils.py                    # plotting, seeding, model save/load
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # download dataset, read EDF files
│   │   ├── preprocessing.py        # epoching, filtering, normalization
│   │   ├── dataset.py              # PyTorch EEGDataset
│   │   └── splitter.py             # subject-based split, DataLoaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── eegnet.py               # EEGNet architecture
│   └── pipelines/
│       ├── __init__.py
│       ├── csp_ml.py               # CSP + classical ML grid search
│       ├── two_stage.py            # mu-wave gating + binary EEGNet
│       ├── grid_search.py          # EEGNet / preprocessing / joint grids
│       └── fbcsp.py                # FBCSP stub (TODO)
├── notebooks/
│   └── experiments.ipynb           # interactive notebook importing modules
├── tests/
│   └── ...                         # pytest suite
├── train.py                        # CLI entry point
├── pyproject.toml                  # Poetry config
├── requirements.txt
└── requirements-dev.txt
```

## Quick Start

```bash
# Install with Poetry
poetry install

# Or with pip
pip install -r requirements.txt

# Train with default config
python train.py

# Train with custom config
python train.py --config configs/my_experiment.yaml
```

## Usage from Notebook

```python
from src.config import load_config
from src.data import download_dataset, load_raw_subjects, epoch_subjects, subject_split, make_dataloaders
from src.models import EEGNet
from src.engine import train, cross_validate_subjects
from src.pipelines import run_csp_ml_grid, run_preprocessing_grid, run_joint_grid
from src.utils import set_seeds, get_device, plot_training_curves, plot_confusion_matrix

cfg = load_config()
set_seeds(cfg["seed"])
device = get_device()

subjects_data = download_dataset()
raw_data = load_raw_subjects(subjects_data, cache_dir="cache")
X, y, subjects, _ = epoch_subjects(raw_data, event_id={"left_hand": 2, "right_hand": 3},
                                    channels=cfg["channels"]["motor_channels"],
                                    label_offset=2)
split = subject_split(X, y, subjects)
loaders = make_dataloaders(split)
model = EEGNet(chans=21, classes=2, time_points=X.shape[2]).to(device)
```

## Configuration

All hyperparameters live in `configs/default.yaml`. Override any field with a partial YAML:

```yaml
# configs/quick_debug.yaml
data:
  n_subjects: 10
  cache_dir: cache
training:
  epochs: 5
```

## Notebook → Module Mapping

| Notebook Section | Module |
|---|---|
| §1–4: Data loading, EDA | `src/data/loader.py` |
| §4: Preprocessing & epoching | `src/data/preprocessing.py` |
| §5–6: Split & DataLoaders | `src/data/splitter.py`, `src/data/dataset.py` |
| §7: EEGNet | `src/models/eegnet.py` |
| §8: Training engine | `src/engine.py` |
| §9–12: Single run, CV, grid search | `src/engine.py`, `src/pipelines/grid_search.py` |
| §13–14: CSP + classical ML | `src/pipelines/csp_ml.py` |
| §15: Motor cortex channels | Same modules, different config |
| §16: Two-stage mu-wave gating | `src/pipelines/two_stage.py` |
| §17: Binary L/R pipeline | Same modules, `task.mode: binary` |
| §18: Preprocessing grid search | `src/pipelines/grid_search.py` |
| §19: Joint preprocessing × model search | `src/pipelines/grid_search.py` |
| §20: FBCSP (stub) | `src/pipelines/fbcsp.py` |

## License

MIT
