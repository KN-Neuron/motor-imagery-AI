# Notebooki

Notebooki eksploracyjne z początku projektu (EDA, pierwsze modele, augmentacja, cechy Hjortha). **Wyniki z nich nie są podstawą żadnego wniosku w pracy.** Wiarygodna ewaluacja (podział po osobach, N-LNSO) jest w `src/eval/` i `scripts/`, a wyniki w `docs/lab_notebook.md`.

| plik / katalog | status |
|---|---|
| `archive_leakage/` | losowy podział epok bez grupowania po osobach; tylko historia (patrz README w katalogu) |
| `brain_board_EDA*.ipynb`, `brain_board_EDA_JS.ipynb` | eksploracja danych. Komórka 36 w `brain_board_EDA_JS.ipynb` dopasowuje CSP na całych danych przed CV bez grupowania po osobach (wyciek; do weryfikacji) |
| `brain_board_augmentation.ipynb` | augmentacja (`src/data/augmentation.py`). Komórka 19 używa `StratifiedKFold(shuffle=True)` na epokach wielu osób (ta sama osoba w treningu i teście; do weryfikacji), więc porównanie z/bez augmentacji jest na przeciekającym podziale |
| `brain_board_ensemble_subject_based.ipynb` | ensemble z podziałem po osobach |
| `hjorth_parameters.ipynb` | cechy Hjortha, CV w obrębie osoby (within-subject) |
| `experiments*.ipynb`, `left-right-and-rest*.ipynb`, `mindstride_fixed_v2.ipynb` | wcześniejsze wersje pipelinu (Copy1 = kopie robocze) |
| `models/`, `results/` | pliki wyjściowe notebooków |

Uwagi o komórkach 36 i 19 pochodzą z przeglądu kodu w drugiej sesji i nie były przeze mnie uruchamiane ponownie.
