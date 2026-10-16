# Mapa kodu do przeglądu


Stan sprzed przebudowy (6.10.2026): lokalny branch `pre-rework` (nie wypchnięty; hash zmienił się po przepisaniu historii 11.10). Zakres zmian: `git diff pre-rework..development`.
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
| `scripts/run_all.sh` | **wszystko od zera** do `results/final`: biegi, krzywe uczenia (`history.csv`), logi, środowisko, `figures/`, `REPORT.md` |
| `scripts/make_figures.py` | wykresy i `REPORT.md` dla całego drzewa wyników |
| `src/replication/`, `scripts/run_replication_bouchane.py` | replikacja Bouchane et al. 2025 (3 sposoby podziału, kilka seedów) |
| `scripts/summarize_results.py` | tabela średnich z CI dla katalogów wyników |
| `scripts/run_simulations.py`, `make_report.py`, `sanity_check.py`, `power_analysis.py`, `train_pretrained.py` | symulacje, raport `docs/results.md`, sanity check, moc, model do BA |
| `configs/benchmark.yaml` | główny benchmark; `configs/legacy_*.yaml` = warianty diagnostyczne (pasmo, okno, kanały, oczy); `configs/mi_*.yaml` = benchmark MI po regresji EOG (`mi_reg_mu_beta_nocue` = kandydat na główny) |
| `tests/` | 102 testy, w tym end-to-end na syntetycznych EDF (`test_scripts_smoke.py`, ok. 90 s) |
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

### Znane problemy (stan na 2026-10-16)

- `train.py`: skompilowany i sprawdzony częściowo testami (siatki, etap 7 CSP+ML), ale nie uruchomiony end-to-end po zmianach.
  W starych configach `normalize: true` zamieniono na `normalization: "none"`, co nie odpowiada przebiegowi z 84%.
- CV w `train.py` i w siatkach bierze maksimum dokładności po epokach na ocenianym foldzie (optymistyczne Dev Acc);
  wymaga decyzji, jak wybierać epokę.
- Kontrola zgodności preprocessingu nie sprawdza nazwy normalizacji.
- Few-shot k=40 niemożliwe przy ok. 22 próbach na klasę (NaN w raporcie); k=0 jest transduktywne (normalizacja liczona
  na próbach testowych bez etykiet).
- Deep ConvNet przy oknie 2 s nie uczy się; Shallow może być zaniżony (brak regularyzacji).
- Listy kanałów MIDI16/MAXI32 są założeniem; część cytowań w `docs/related_work.md` "do weryfikacji"; `poetry.lock` nieodświeżony.
- Wybór konfiguracji (okno, pasmo, kanały, regresja EOG) po obejrzeniu wyników N-LNSO: liczby MI są eksploracyjne.
- Notebooki poza `notebooks/archive_leakage/`: patrz `notebooks/README.md`.
- Wnioski o ruchach oczu: `docs/lab_notebook.md`, sekcja 4.

### Uruchamianie

```bash
make test                                         # testy
make reproduce DATA_DIR=/sciezka/do/physionet     # benchmark + symulacje + docs/results.md (bez DATA_DIR: kagglehub)
nohup bash scripts/run_legacy_check.sh > legacy_check.log 2>&1 &   # warianty diagnostyczne + results/legacy_summary.txt
nohup bash scripts/run_night.sh > night.log 2>&1 &         # strojenie w N-LNSO + results/night_summary.txt
nohup bash scripts/run_all.sh > run_all.log 2>&1 &          # wszystko od zera + wykresy + results/final/REPORT.md
```
