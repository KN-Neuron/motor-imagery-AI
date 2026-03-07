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

## Instrukcja jak dodawać nowe eksperymenty

### 1. Nowy model (np. EEGConformer, ShallowConvNet)

Stwórz plik `src/models/shallow_convnet.py` z klasą dziedziczącą po `nn.Module`:

```python
class ShallowConvNet(nn.Module):
    def __init__(self, chans, classes, time_points, ...):
        ...
    def forward(self, x):
        ...
```

Dodaj import w `src/models/__init__.py`:

```python
from src.models.shallow_convnet import ShallowConvNet
```

Potem w `train.py` w STAGE 1 (lub nowym STAGE) po prostu zamień `EEGNet(...)` na `ShallowConvNet(...)`. Reszta pipeline'u (loss, optimizer, `train()`, `predict_with_model()`) działa identycznie bo bierze `nn.Module`.

### 2. Nowy klasyczny ML model (np. XGBoost)

Wystarczy dodać wpis w `src/pipelines/csp_ml.py` w funkcji `get_ml_models()`:

```python
"XGBoost": {
    "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
    "params": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.01, 0.1],
        "csp__n_components": csp_components,
    },
},
```

I dodaj `"XGBoost"` do listy w YAML `csp_ml.models`. Pipeline CSP → Scaler → Classifier + GridSearchCV ogarnie resztę automatycznie.

### 3. Nowy preprocessing (np. ICA, inny filtr)

Edytuj `src/data/preprocessing.py` — dodaj krok w `epoch_subjects()`, albo stwórz nową funkcję np. `epoch_subjects_with_ica()`. Potem w `train.py` wywołaj ją zamiast `epoch_subjects()`.

Albo jeśli chcesz to włączyć do grid searcha — dodaj parametr w `preprocessing_grid` w YAML i obsłuż go w `epoch_with_params()`.

### 4. Nowy stage w train.py

Schemat jest zawsze taki sam — wklej blok między istniejące STAGE'e:

```python
# ════════════════════════════════════════════════════════
# STAGE X: Moja nowa rzecz
# ════════════════════════════════════════════════════════
if run_cfg.get("my_new_stage", False):
    print("\n" + "=" * 60)
    print("  STAGE X: Moja nowa rzecz")
    print("=" * 60)

    # ... twoja logika ...

    logger.log_stage("my_new_stage", {
        "accuracy": wynik,
        "cokolwiek": inne_metryki,
    })
```

I w YAML dodaj:

```yaml
run:
  my_new_stage: true
```

Logger zapisze wynik do JSON natychmiast.

### 5. Nowy config bez zmian w kodzie

Najczęstszy case — chcesz sprawdzić inne hiperparametry. Zero zmian w Pythonie, robisz nowy YAML:

```yaml
# configs/experiment_wide_band.yaml
preprocessing:
  bandpass: [4.0, 40.0]
training:
  epochs: 150
  lr: 0.0005
eegnet:
  f1: 16
  d: 4
  dropout_rate: 0.25
run:
  single_run: true
  cross_validation: true
  eegnet_grid_search: false    # skip — testujesz ręczne params
  csp_ml_grid: false
  preprocessing_grid: false
  joint_grid: false
  final_retrain: false
```

```bash
poetry run python train.py --config configs/experiment_wide_band.yaml
```

Wynik ląduje w osobnym JSON z timestampem, nie nadpisuje poprzednich.

### Podsumowanie flow'u

```
YAML config → train.py czyta run: → odpala włączone STAGE'e
                                   → każdy STAGE używa modułów z src/
                                   → każdy STAGE loguje do JSON przez ResultsLogger
```

Moduły są od siebie niezależne — `engine.py` nie wie nic o `csp_ml.py`, `EEGNet` nie wie nic o preprocessingu. `train.py` to jedyne miejsce które je klei razem.

## License

MIT
