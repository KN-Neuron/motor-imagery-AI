#  EEG Motor Imagery Classification

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

# Run repeated splits for stable evaluation (e.g., 5 runs with different seeds)
python train_multiple_splits.py --config configs/full_binary_all_channels.yaml --runs 5
```

### Safety Features & Checkpointing

The grid search pipelines (EEGNet, ShallowConvNet, Preprocessing, and Joint) will now automatically log their progress dynamically to CSV files in the `outputs/` directory.

- **Crash Recovery:** If the execution stops (due to out-of-memory errors, system crash, or pressing `Ctrl+C`), the partial grid search results are preserved.
- **Resuming:** Running the script again will automatically read the `.csv` checkpoint from the previous stages (if pointing to the exact same directory depending on how output timestamping is configured, or manually moving the checkpoint) and skip the already completed combinations, saving hours of computation.
- **Learning Curves:** During the `Final Retrain` stage (Stage 7), full epoch histories (loss, accuracy validation) are saved directly in `outputs/final_retrain_curves.png` and recorded in the JSON logger.

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

## Data flow

```

4-way subject split:
  - TRAIN  (60%) — gradient descent, backprop
  - VAL    (20%) — early stopping, checkpoint selection
  - DEV    (10%) — model/preprocessing comparison across stages
  - HOLDOUT(10%) — touched ONCE at the very end, reported in paper

Data flow:
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
  Stage 1 (single run): train on TRAIN, checkpoint on VAL, eval on DEV
  Stage 7 (final retrain): train on TRAIN, checkpoint on VAL, eval on DEV
  Stage 8 (holdout): ONE evaluation on HOLDOUT — this goes in the paper

```

Usage:
```
    python train.py --config configs/full_binary_all_channels.yaml
```

## Notebook → Module Mapping

| Notebook Section                        | Module                                          |
| --------------------------------------- | ----------------------------------------------- |
| §1–4: Data loading, EDA                 | `src/data/loader.py`                            |
| §4: Preprocessing & epoching            | `src/data/preprocessing.py`                     |
| §5–6: Split & DataLoaders               | `src/data/splitter.py`, `src/data/dataset.py`   |
| §7: EEGNet                              | `src/models/eegnet.py`                          |
| §8: Training engine                     | `src/engine.py`                                 |
| §9–12: Single run, CV, grid search      | `src/engine.py`, `src/pipelines/grid_search.py` |
| §13–14: CSP + classical ML              | `src/pipelines/csp_ml.py`                       |
| §15: Motor cortex channels              | Same modules, different config                  |
| §16: Two-stage mu-wave gating           | `src/pipelines/two_stage.py`                    |
| §17: Binary L/R pipeline                | Same modules, `task.mode: binary`               |
| §18: Preprocessing grid search          | `src/pipelines/grid_search.py`                  |
| §19: Joint preprocessing × model search | `src/pipelines/grid_search.py`                  |
| §20: FBCSP (stub)                       | `src/pipelines/fbcsp.py`                        |


## Example training output
```
============================================================
  ALL DONE — results saved to outputs/20260320_141237_binary_all.json
============================================================
```

> Dawna tabela wyników (84,3% na holdoucie) została usunięta: była niespójna z `summary_metrics.csv` (74,9 do 77,9%) i TODO (80%), opierała się na jednym holdoucie i wyborze po epokach. Jedyne źródło prawdy: [`docs/results.md`](docs/results.md), generowane automatycznie (`make reproduce`).

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
  eegnet_grid_search: false # skip — testujesz ręczne params
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

## Zmiany do przeglądu (dodane od 2026-10-06)

Stan sprzed zmian: branch `pre-rework` (commit `9a2dc2f`). Pełna lista: `git diff pre-rework..development`.
Przebieg prac, wszystkie wyniki i popełnione błędy: **`docs/lab_notebook.md`**.

### Mapa nowego kodu

| plik | co robi |
|---|---|
| `src/eval/nlnso.py` | N-LNSO: podział osób na foldy zewnętrzne + walidacja z puli treningowej; zwraca wyniki per osoba i per próba |
| `src/eval/data.py` | `load_raw` (dane + wykluczenia z logiem), `build_epochs` (epoki + cache), `regress_out` (regresja EOG per osoba) |
| `src/eval/pipelines.py` | modele w jednym interfejsie `fit/predict_proba`: EEGNet (± max-norm), Shallow, Deep, CSP+LDA, TS+LR, `heog_lda`, `eegnet_transformer` (`src/models/eegnet_transformer.py`) |
| `src/eval/stats.py` | bootstrap CI po osobach, próg dwumianowy, Wilcoxon, Holm, liczebność próby |
| `src/eval/resume.py` | zapis skończonych par (model, normalizacja) i wznawianie |
| `src/eval/tuning.py` | strojenie hiperparametrów w wewnętrznej pętli N-LNSO: k-fold po osobach treningowych, wybór, refit, test raz |
| `src/data/normalization.py` | normalizacje (zscore, EMS, EA), `PreprocMeta`, `check_compatibility` |
| `src/data/subjects.py` | presety wykluczeń osób + filtr z logowaniem |
| `src/data/preprocessing.py`, `loader.py`, `loading.py` | zmiany: parametr `normalization`, EMS na ciągłym sygnale, `find_edf_files`, wykluczenia z configu |
| `src/eval/degradation.py`, `study.py`, `montages.py`, `smr.py` | symulacje pseudo-BrainAccess, few-shot, krzywa "ile prób", predyktor SMR |
| `src/eval/sanity.py` | within-subject CV (sanity check) |
| `src/ba/` | moduł BrainAccess (wczytywanie sesji, foldy blokowe, checkpointy z metadanymi, BIDS) |
| `scripts/run_benchmark.py` | główny bieg: siatka (model, normalizacja) z configu → `per_subject.csv`, `per_trial.csv`, `run_meta.json` |
| `scripts/run_legacy_check.sh` | jedna komenda do wszystkich wariantów `configs/legacy_*.yaml` + tabela |
| `scripts/run_tuned.py`, `run_night.sh` | bieg nocny: `configs/mi_tuned.yaml` → `results/mi_tuned` (+ `selection.csv`) |
| `scripts/run_mi_check.sh` | benchmark MI po regresji EOG + kontrola niskich częstotliwości |
| `scripts/summarize_results.py` | tabela średnich z CI dla katalogów wyników |
| `scripts/run_simulations.py`, `make_report.py`, `sanity_check.py`, `power_analysis.py`, `train_pretrained.py` | symulacje, raport `docs/results.md`, sanity check, moc, model do BA |
| `configs/benchmark.yaml` | główny benchmark; `configs/legacy_*.yaml` = warianty diagnostyczne (pasmo, okno, kanały, oczy); `configs/mi_*.yaml` = benchmark MI po regresji EOG (`mi_reg_mu_beta_nocue` = kandydat na główny) |
| `tests/` | ok. 70 testów, w tym end-to-end na syntetycznych EDF (`test_scripts_smoke.py`, ok. 90 s) |
| `docs/` | `lab_notebook.md`, `research_plan.md`, `related_work.md`, `protocol_eksperymentu.md` |
| `Makefile` | `make reproduce`, `make test` |

Zmienione stare pliki: `train.py` (przekazuje `normalization`, zapisuje metadane), `src/models/eegnet.py`
(max-norm jako opcja `use_max_norm`), `src/config.py`, `configs/*.yaml` (`normalize` → `normalization`).
Usunięte: skrypty `*_ba_data.py` itp. (zastąpione przez `src/ba/`), 4 testy importujące nieistniejące moduły.
Z innych branchy (nie moje): `src/data/augmentation.py`, notebooki EDA/Hjorth, `docs/archive/`.

### Proponowana kolejność przeglądu (od największego ryzyka)

1. `src/eval/nlnso.py` + `tests/test_eval.py`: czy osoby treningowe, walidacyjne i testowe są rozłączne, czy test jest użyty raz.
2. `src/data/normalization.py`: czy normalizacje są per osoba i bez etykiet.
3. `src/eval/data.py`: wykluczenia, cache, kolejność kanałów, `regress_out`.
4. `src/data/preprocessing.py`: filtr i epoki (diff względem `pre-rework`).
5. `src/eval/pipelines.py`: `TorchPipeline.fit` (wybór epoki na walidacji), `HeogLDA`.
6. `src/eval/tuning.py` + `tests/test_tuning.py`: czy wybór hiperparametrów nie widzi osób testowych.
7. `src/eval/stats.py`, `scripts/run_benchmark.py`, `src/eval/resume.py`.
8. Symulacje (`degradation.py`, `study.py`): znane problemy niżej.
9. `src/ba/`: nieuruchamiane na prawdziwych danych.

### Znane problemy (nienaprawione)

- `train.py` po zmianach tylko skompilowany, nie uruchomiony; w starych configach `normalize: true` zamieniłem na `normalization: "none"`, co nie odpowiada przebiegowi z 84%.
- Kontrola zgodności preprocessingu nie sprawdza nazwy normalizacji.
- Few-shot k=40 niemożliwe przy ok. 22 próbach na klasę (NaN w raporcie).
- Deep ConvNet przy oknie 2 s nie uczy się (zmniejszone jądra).
- Listy kanałów MIDI16/MAXI32 są założeniem; część cytowań w `docs/related_work.md` "do weryfikacji"; `poetry.lock` nieodświeżony.
- Wnioski o ruchach oczu: patrz `docs/lab_notebook.md`, sekcja 4.

### Uruchamianie

```bash
make test                                         # testy
make reproduce DATA_DIR=/sciezka/do/physionet     # benchmark + symulacje + docs/results.md (bez DATA_DIR: kagglehub)
nohup bash scripts/run_legacy_check.sh > legacy_check.log 2>&1 &   # warianty diagnostyczne + results/legacy_summary.txt
nohup bash scripts/run_night.sh > night.log 2>&1 &         # strojenie w N-LNSO + results/night_summary.txt
```

# TODO


| What | Status |
|------|--------|
| PhysioNet MI data loading (R04, R08, R12) | ✅ |
| EDA on single subject (PSD, raw signal) | ✅ |
| Preprocessing: bandpass 7-30 Hz, epoching 0-4s | ✅ |
| Normalizacja: jawny parametr `preprocessing.normalization` (none, zscore_subject_channel, exp_moving_standardization, euclidean_alignment); wcześniej z-score był zakomentowany | ✅ (wybór domyślnej po ablacji) |
| Class balancing (downsampling to smallest class) | ✅ |
| Weighted CrossEntropyLoss everywhere | ✅ |
| Subject-based split (70/15/15, no leakage) | ✅ |
| EEGNet — 64 channels, 3 classes | ✅ |
| EEGNet — 21 motor cortex channels, 3 classes | ✅ |
| EEGNet — 21ch binary L/R (no rest) | ✅ |
| Subject-based K-Fold CV (GroupKFold) | ✅ |
| EEGNet hyperparameter grid search (lr, dropout, f1, d) | ✅ |
| Final model retrain with best params | ✅ |
| CSP One-vs-Rest | ✅ |
| CSP Pairwise (MultiClassCSP) | ✅ |
| CSP binary (native, for L/R) | ✅ |
| Grid search over 7 classical ML models (LDA, SVM, RF, KNN, GB, LR, MLP) | ✅ |
| class_weight='balanced' in ML models | ✅ |
| Two-stage pipeline: mu-wave gating + binary L/R | ✅ |
| Comparison of all approaches (bar charts, confusion matrices) | ✅ |
| Full pipeline repeated on both 64ch and 21ch | ✅ |
| Preprocessing grid search (tmin/tmax/bandpass/baseline) | ✅ |
| Joint grid search — preprocessing × all models (EEGNet + 7 ML) | ✅ |
| Dawny "best combo 80%" jest niezweryfikowany (patrz `docs/results.md`) | ⚠️ |
| FBCSP (Filter Bank CSP) | ❌ |
| Data augmentation (sliding window, noise, warping) | ❌ |
| Ensemble (voting/stacking of best models) | ❌ |
| Feature extraction: Hjorth parameters, kurtosis, variance | ❌ |
| Feature extraction: band powers, spectral entropy, mu/beta ratio | ❌ |
| Feature extraction: wavelets/STFT → 2D for CNN | ❌ |
| Feature extraction: connectivity (PLV, coherence) | ❌ |
| Feature fusion + selection (mutual information, RFE) | ❌ |
| Subject-adaptive bandpass | ❌ |
| Riemannian geometry (pyriemann) | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Attention-based EEGNet | ❌ |
| Transfer learning (pretrain → fine-tune per subject) | ❌ |
| Sliding window inference | ❌ |
| EEGConformer / ATCNet (braindecode) | ❌ |
| GAN — synthetic EEG epoch augmentation (WGAN-GP / conditional GAN) | ❌ |
| Variational Autoencoder — latent space + classification on embeddings | ❌ |
| Contrastive encoder (SimCLR/BYOL-style) — self-supervised pre-training | ❌ |
| Autoencoder denoising — pre-train encoder → fine-tune classifier | ❌ |
| ICA artifact removal (eye blinks, muscle artifacts) + comparison w/wo | ❌ |
| Per-trial normalization (z-score per epoch instead of per subject) | ❌ |
| Per-sample normalization (z-score per timepoint across channels) | ❌ |
| Euclidean Alignment (covariance matrix centering per subject) | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Min-max normalization comparison | ❌ |
| Robust scaling (median/IQR — resistant to EEG artifacts) | ❌ |
| Normalization strategy grid search (z-score vs min-max vs robust vs EA vs per-sample) | ❌ |
| Full joint grid search on 64 channels (preprocessing × all models) | ❌ |
| 64ch vs 21ch comparison (best combos head-to-head, same preprocessing) | ❌ |
| Generative models approach | ❌ |
| Statistical analysis (p-values between approaches) | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Blink trigger (real-time app) | ❌ |
| Comparison between channels. Check only MI channels vs all channels vs only non MI channels | ❌ |
| Retrain EEGNet na 16ch subset (BrainAccess MIDI channels) z PhysioNet | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Channel mapping PhysioNet→MIDI + sferyczna interpolacja brakujących | ❌ |
| Try to find BCI-Illiterate people and do a full result analysis | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Resample 250Hz + fine-tune na własnych danych z BrainAccess MIDI | ❌ |

## License

MIT
