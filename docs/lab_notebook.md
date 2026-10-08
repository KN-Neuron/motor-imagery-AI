# Dziennik eksperymentów (od 2026-10-06)

Wszystkie wyniki: PhysioNet EEGMMIDB, runy R04/R08/R12 (wyobrażanie lewej/prawej pięści, T1/T2), 160 Hz.
Wartości to średnia dokładność po osobach w %, w nawiasie 95% CI z bootstrapu po osobach.
Protokół N-LNSO (każda osoba testowana raz, wybór checkpointu na walidacji z puli treningowej),
5 foldów zewnętrznych, 3 seedy uśrednione per osoba, 50 epok, bez strojenia hiperparametrów.
Próg istotności per osoba przy 45 próbach: 64,4% (dwumianowy, p<0,05).

## 1. Stan wyjściowy

- Branch `pre-rework` (lokalnie, commit `9a2dc2f`) = stan sprzed zmian.
- `train.py`: podział osób TRAIN/VAL/DEV/HOLDOUT (60/20/10/10), holdout użyty raz. Protokół poprawny, bez wycieku.
- Wynik z 2026-03-20 (`outputs/20260320_141237_binary_all.json`), 5 seedów = 5 losowych podziałów osób:
  holdout 83,3 / 89,7 / 82,0 / 83,7 / 82,9, **średnio 84,3 ± 2,8**; dev 87,2 ± 4,6.
  Config: pasmo z siatki {0-49, 0,5-45 Hz}, okno 0-4 s, `normalize: true`, EEGNet f1=16, d=2,
  temp_kernel=64, dropout 0,25, 100 epok, patience 15, wd 1e-3, wszystkie osoby.
- Kod z 2026-03-20 nie jest w historii git (`train.py` i `preprocessing.py` pojawiają się w `788acd8`, 2026-03-21).
  W repo normalizacja w `epoch_subjects` była zakomentowana.
- Notebooki z prawdziwym wyciekiem (podział po próbach) przeniesione do `notebooks/archive_leakage/`.

## 2. Przebudowa (commity d5ade4a ... c4e479c)

N-LNSO, statystyka po osobach, jawna opcja `normalization`, `PreprocMeta` + kontrola zgodności,
konfigurowalne wykluczenia osób, baseline'y, symulacje pseudo-BrainAccess, few-shot, moduł `src/ba/`,
eksport BIDS, `make reproduce`, zapis częściowy i wznawianie benchmarku.

## 3. Uruchomienia i wyniki (chronologicznie)

### 3.1 Smoke run, 15 osób (stary kod z błędem wykluczeń)
Wszystko na poziomie losowym: csp_lda|EA 55,3; ts_lr|EA 54,5; csp_lda|none 53,8; EEGNet 48,5-51,7;
shallow 50,5; deep 49,1. Przyczyna: ok. 10 osób w treningu. Przy okazji znaleziony błąd: ścieżka
kagglehub po cichu wykluczała 038/082/089/104 wbrew configowi (naprawione, `d4e813f`).

### 3.2 Sanity check, 105 osób (`scripts/sanity_check.py`)
| ustawienia | within-subject CSP+LDA | within-subject TS+LR | cross-subject N-LNSO TS+LR |
|---|---|---|---|
| 4-40 Hz, 0,5-2,5 s | 57,7 | 54,7 | 63,6 |
| 8-30 Hz, 0,5-3,5 s | 59,0 | 57,3 | 64,7 |

Wniosek: pipeline działa; trening cross-subject lepszy niż na jednej osobie (ok. 45 prób).

### 3.3 Główny benchmark (`configs/benchmark.yaml`: 4-40 Hz, 0,5-2,5 s, 64 kan., wykluczenie 089, N=105)
| pipeline | wynik |
|---|---|
| eegnet + zscore_subject_channel | **71,5 [69,4; 73,5]** |
| eegnet_maxnorm + EA | 67,8 [66,0; 69,7] |
| csp_lda + EA | 67,5 [65,0; 69,9] |
| eegnet + EMS | 67,3 [65,5; 69,2] |
| shallow + EA | 65,1 [62,9; 67,2] |
| eegnet + EA | 65,0 [63,3; 66,6] |
| eegnet + none | 64,0 [62,5; 65,5] |
| ts_lr + EA | 63,6 [61,3; 65,9] |
| csp_lda + none | 57,6 [56,0; 59,3] |
| deep + EA | 50,5 (nie nauczył się; jądra zmniejszone przy 321 próbkach) |

Względem eegnet+EA (Wilcoxon, Holm): zscore +6,5 pp (p<0,0001), max-norm +2,9 pp (p=0,0001),
EMS +2,4 pp (p=0,026), csp_lda+EA +2,5 pp (p=0,032). SD różnic per osoba 8,6 pp
(26 osób na wykrycie 5 pp, 149 na 2 pp).

Symulacje pseudo-BrainAccess (tylko EEGNet, dane PhysioNet sztucznie psute; listy kanałów MIDI16/MAXI32 założone):
- szum + dryf: spadek 13-21 pp (do poziomu losowego) dla każdej normalizacji;
- model EA karmiony zscore: spadek 9,5 pp, **guard tego nie wykrywa** (do naprawy);
- dla EA podzbiory kanałów trenowane od nowa są lepsze od 64 kanałów (niewyjaśnione).

Few-shot: k=10/20 bez zysku; **k=40 = NaN, błąd projektu** (ok. 22 próby na klasę). Do naprawy.

### 3.4 Replikacja starych ustawień w N-LNSO (wszystkie osoby, N=106)
| wariant | pasmo | okno | kanały | eegnet none | eegnet zscore | csp_lda none |
|---|---|---|---|---|---|---|
| legacy_replica | 7-30 | 0-4 | 64 | 68,6 | 81,8 [79,7; 84,0] | 60,7 |
| legacy_window_only | 4-40 | 0-4 | 64 | 68,8 | 77,2 | 58,3 |
| legacy_no_cue | 7-30 | 0,5-4 | 64 | 61,1 | 65,0 | 59,9 |
| legacy_motor21 | 7-30 | 0-4 | 21 ruchowe | 62,8 | 68,4 | 59,1 |
| legacy_wideband | 0,5-45 | 0-4 | 64 | 82,6 | **83,5 [81,5; 85,5]** | 53,5 |

**Stary wynik 84% się odtwarza** (83,5% na 106 osobach). Różnica względem głównego benchmarku wynika z pasma i okna, nie z ewaluacji.

### 3.5 Lokalizacja sygnału w szerokim paśmie (0,5-45 Hz, N=106)
| wariant | okno | kanały | eegnet none | eegnet zscore |
|---|---|---|---|---|
| legacy_wb_no_cue | 0,5-4 | 64 | 78,6 | 78,5 [76,0; 80,9] |
| legacy_wb_mi_window | 0,5-2,5 | 64 | 77,2 | 77,1 [74,4; 79,8] |
| legacy_wb_motor21 | 0-4 | 21 ruchowe | 80,3 | 81,0 [78,6; 83,2] |
| **legacy_wb_frontal** | 0-4 | 10 czołowych (Fp/AF/F7/F8) | 81,0 | **81,5 [79,3; 83,7]** |
| legacy_wb_occipital | 0-4 | 9 potylicznych | 68,2 | 69,1 [67,1; 71,0] |

### 3.6 Testy ruchów oczu (N=106, okno 0-4 s)
| wariant | pasmo | kanały | wynik |
|---|---|---|---|
| legacy_eye_heog, EEGNet zscore | 0,5-4 | tylko F7, F8 | **81,3 [79,1; 83,4]** |
| legacy_eye_heog, LDA na 8 średnich F7-F8 | 0,5-4 | F7-F8 | **77,4 [75,1; 79,7]** (none: 77,2) |
| legacy_eye_motor_regressed, EEGNet zscore | 0,5-45 | 21 ruchowych po regresji 6 czołowych | **69,3 [67,3; 71,2]** (none: 66,3) |
| legacy_eye_frontal_mu_beta, EEGNet zscore | 7-30 | 10 czołowych | 77,5 [75,3; 79,7] (none: 66,5) |

### 3.7 Benchmark MI po regresji EOG i kontrola rezydualna (2026-10-08, commit f79cef6, N=106)
21 kanałów ruchowych, regresja 6 czołowych (Fp1, Fp2, AF7, AF8, F7, F8) per osoba przed normalizacją.

| model | mu_beta 7-30, 0-4 s | mu_beta_nocue 7-30, 0,5-4 s | wb_nocue 0,5-45, 0,5-4 s |
|---|---|---|---|
| eegnet zscore | **68,8 [66,3; 71,3]** | **66,7 [64,2; 69,2]** | 67,0 [64,9; 69,2] |
| eegnet_maxnorm zscore | 68,5 | 66,5 | 66,5 |
| shallow zscore | 68,0 | 64,1 | 62,9 |
| csp_lda EA | 67,2 | 66,2 | 57,9 |
| ts_lr EA | 65,9 | 64,5 | 59,4 |
| eegnet none | 64,9 | 62,8 | 62,0 |

Kontrola, EEGNet, 21 kanałów ruchowych, 0,5-4 Hz, 0-4 s:

| wariant | zscore | none |
|---|---|---|
| bez regresji (`mi_ctrl_lowfreq_raw`) | 80,8 [78,5; 83,0] | 80,2 |
| po regresji (`mi_ctrl_lowfreq_reg`) | 62,1 [60,4; 63,8] | 62,2 |

Obserwacje:
- Regresja zabiera większość sygnału poniżej 4 Hz na kanałach ruchowych (80,8 → 62,1), ale nie do 50%.
  Reszta to albo niedoregresowane oczy (regresja liniowa, 6 referencji), albo wolne potencjały ruchowe;
  tych dwóch nie da się tu rozdzielić.
- W 7-30 Hz regresja prawie nic nie zmienia: motor21 bez regresji 68,4 (3.4), po regresji 68,8.
  Kanały ruchowe w mu/beta raczej nie niosą sygnału ocznego (w przeciwieństwie do czołowych, 77,5%).
- Usunięcie 0-0,5 s kosztuje ok. 2 pp (68,8 → 66,7), nie 17 pp jak na 64 kanałach.
- Szerokie pasmo po regresji nie pomaga EEGNet (67,0 vs 66,7), a szkodzi CSP/TS.
- Modele mieszczą się w ok. 64-69%; różnice w obrębie CI, nie testowane parami.

### 3.8 Bieg nocny: strojenie hiperparametrów w N-LNSO (2026-10-09, `configs/mi_tuned.yaml`, N=106)
Dane jak `mi_reg_mu_beta_nocue`; wybór na 3 grupach osób treningowych w każdym foldzie zewnętrznym.

| model | stałe hiperparametry (3.7) | po strojeniu |
|---|---|---|
| EEGNet zscore (40 z 288) | 66,7 [64,2; 69,2] | 66,6 [64,1; 69,1] |
| CSP+LDA EA (n_components) | 66,2 | 66,4 [63,5; 69,4] |
| TS+LR EA (C) | 64,5 | 65,7 [62,8; 68,7] |
| Shallow zscore (8) | 64,1 | 64,4 [62,0; 66,7] |
| SpatialEEGNetTransformer zscore (8) | (brak) | 63,9 [61,5; 66,4] |

Obserwacje:
- Strojenie nic nie daje (zmiany od -0,1 do +1,2 pp, w obrębie CI).
- Wybory EEGNet są niestabilne między foldami (np. f1 4 albo 16, lr 0,001 albo 0,005, jądro 32 albo 128)
  przy bardzo podobnych wynikach wewnętrznych (0,645-0,687): powierzchnia wyników jest płaska, a różnice
  między foldami wynikają z osób, nie z hiperparametrów.
- Transformer jest najsłabszy, a jego wyniki wewnętrzne mają największy rozrzut (0,567-0,649),
  co wskazuje na niestabilny trening przy ok. 3300 próbach treningowych. Różnicy względem EEGNet nie testowałem parami.
- Wniosek: górna granica ok. 66-67% wynika z danych (MI L/R między osobami na EEGMMIDB po usunięciu oczu),
  nie z wyboru modelu ani hiperparametrów.

## 4. Wnioski robocze (stan na 2026-10-07, po testach oczu)

1. Stary pipeline był poprawny metodologicznie; 84% to nie wyciek ani przypadek.
2. **Dwa kanały przy oczach (F7, F8) poniżej 4 Hz dają 81%, prawie tyle co 64 kanały (83,5%)**;
   jedna cecha "F7 minus F8" uśredniona w czasie z LDA daje 77%. Kanały ruchowe po regresji kanałów
   czołowych spadają z 81 do 69%. Najprostsze wyjaśnienie: model w dużej mierze rozpoznaje kierunek
   spojrzenia (cel w EEGMMIDB pojawia się z lewej/prawej strony ekranu), nie wyobrażenie ruchu.
   Bardzo silna poszlaka, ale nie pomiar wprost: EEGMMIDB nie ma kanałów EOG.
3. W paśmie 7-30 Hz usunięcie pierwszych 0,5 s kosztuje 17 pp, w szerokim tylko 5 pp
   (spójne z utrzymanym spojrzeniem przez całą próbę).
4. Kanały czołowe niosą informację także w 7-30 Hz (77,5% z zscore), więc samo odcięcie niskich
   częstotliwości NIE usuwa oczu. Główny benchmark (4-40 Hz, 64 kanały, 71,5%) też może być
   częściowo "oczny". Najlepsze obecne oszacowanie MI: kanały ruchowe po regresji EOG,
   ok. 66-69% (może być zaniżone, jeśli regresja zabiera sygnał mózgowy, albo zawyżone, jeśli
   regresja liniowa nie usuwa oczu w całości).
5. Dla BrainAccess: bodziec w środku ekranu i fiksacja; inaczej model może działać na oczach.
6. (2026-10-08, po 3.7) Uczciwe oszacowanie MI L/R na EEGMMIDB w N-LNSO: **ok. 67% (EEGNet zscore,
   kanały ruchowe, 7-30 Hz, 0,5-4 s, po regresji EOG: 66,7 [64,2; 69,2])**. Kandydat na główny
   benchmark: `mi_reg_mu_beta_nocue` (najbardziej zachowawczy: bez cue, bez niskich częstotliwości,
   bez kanałów czołowych). Różnica 83,5 → 66,7 to górne oszacowanie udziału oczu i bodźca.
7. Wnioski z regresji EOG (3.6, 3.7):
   - Ograniczenie do kanałów ruchowych NIE usuwa oczu: potencjały oczne rozchodzą się po głowie
     (motor21 w 0,5-45 Hz: 81%, w 0,5-4 Hz: 80,8%, prawie tyle co 64 kanały).
   - Regresja per osoba, bez etykiet, na 6 kanałach przyocznych (Fp1, Fp2, AF7, AF8, F7, F8 jako
     zastępcze EOG) usuwa większość tego przecieku: 0,5-4 Hz na kanałach ruchowych 80,8 → 62,1%.
   - W 7-30 Hz regresja nie zmienia wyniku (68,4 → 68,8%): tam ochronę dają pasmo i dobór kanałów,
     regresja jest zabezpieczeniem i nie zabiera sygnału MI.
   - Ograniczenia: tylko liniowa; kanały czołowe zawierają też aktywność mózgu (może być częściowo
     usunięta); brak prawdziwego EOG w EEGMMIDB; resztkowe 62% w 0,5-4 Hz nierozstrzygnięte
     (oczy albo wolne potencjały ruchowe).
   - Dla publikacji: każdy wynik L/R na EEGMMIDB bez kontroli oczu jest podejrzany; kontrola =
     regresja + test samego F7-F8 (77% z LDA na jednej cesze).

### Plan dalej (uzgodnić po przeglądzie kodu)
1. Nowy główny benchmark MI: kanały ruchowe + regresja EOG; porównać pasmo 7-30 vs 0,5-45 Hz i okno,
   na wszystkich modelach (ok. 1-2 h). Przygotowane: `mi_reg_mu_beta`, `mi_reg_mu_beta_nocue`, `mi_reg_wb_nocue`.
2. Kontrola rezydualna: kanały ruchowe w 0,5-4 Hz bez i po regresji (`mi_ctrl_lowfreq_raw`, `mi_ctrl_lowfreq_reg`).
   Uruchomienie obu punktów: `nohup bash scripts/run_mi_check.sh > mi_check.log 2>&1 &` → `results/mi_summary.txt`.
3. Dopiero potem bieg nocny: grid search hiperparametrów (jak w starym `train.py`) w wewnętrznej
   pętli N-LNSO, tylko na osobach walidacyjnych, na ustawieniach, które przejdą punkty 1-2.
4. BrainAccess: bodziec w środku ekranu + fiksacja; regresja EOG albo bez kanałów czołowych.
5. Potencjalny wkład publikacyjny: ilościowy udział ruchów oczu w wynikach L/R MI na EEGMMIDB
   (2 kanały F7/F8 < 4 Hz ≈ 81% vs 64 kanały 83,5%).

Stan: użytkownik przegląda i waliduje cały dodany kod przed dalszymi biegami
(mapa zmian i kolejność przeglądu: README, sekcja "Zmiany do przeglądu").

### Plan: ruch wykonywany (2026-10-09)
Te same kontrole na przebiegach z ruchem (R03/R07/R11, lewa/prawa pięść), osobne configi `configs/mm_*.yaml`
i katalogi `results/mm_*` (wyniki MI nienaruszone): stare ustawienie 64 kan. 0,5-45 Hz, F7/F8 < 4 Hz,
benchmark po regresji (7-30 Hz i 0,5-45 Hz, 0,5-4 s), kontrola 0,5-4 Hz bez/po regresji.
Uruchomienie: `nohup bash scripts/run_movement_check.sh > movement_check.log 2>&1 &` → `results/movement_summary.txt`.

### Literatura: analiza krytyczna Bouchane et al. 2025 (2026-10-09)
Bouchane M., Guo W., Yang S. *Hybrid CNN-GRU Models for Improved EEG Motor Imagery Classification.*
Sensors 25(5):1399, 2025. https://www.mdpi.com/1424-8220/25/5/1399 (pełny tekst: PMC11902626).
Twierdzą: 98,9-99,7% na EEGMMIDB, 5 klas (LF, RF, obie pięści, stopy, baseline), 2 kanały na wejściu, CNN-GRU.

Problemy znalezione w tekście (cytaty dosłowne):
1. Mała próba, ta sama osoba w treningu i teście: "the model was trained and tested on seven subjects,
   achieving a macro average accuracy of 98.88%"; per osoba "10-fold cross-validation" w obrębie osoby.
   Dataset opisany jako 103 osoby, wyniki na 7. U nas: 106 osób, test zawsze na nowych osobach.
2. Brak opisu podziału trening/test dla głównych tabel (3-4); "each experimental run is divided into
   4 s segments": przy losowym podziale segmentów sąsiednie fragmenty tej samej próby trafiają do obu zbiorów.
3. Ta sama próba jako kilka przykładów: "Each SMA is formed by the data related to each channel couple
   combination and is considered independent from the other couples as a separate input pattern."
   Jedna próba = 6 przykładów z sąsiednich, silnie skorelowanych par kanałów (FC1-FC2, C1-C2, C3-C4...);
   para w treningu, sąsiednia para tej samej próby w teście.
4. Klasy pochodzą z różnych przebiegów (LF/RF z R04/R08/R12, obie pięści/stopy z R06/R10/R14, baseline
   osobno): przy podziale segmentów model może rozpoznawać nagranie zamiast zadania.
5. SMOTE deklarowany tylko na treningu ("The validation and test datasets remain unchanged"),
   nieweryfikowalne bez opisu podziału. Brak kodu (Data Availability: tylko link do PhysioNet).
6. Oczy: ICA bez szczegółów, ale pasmo 8-30 Hz i kanały ruchowe (u nas analogiczny motor21 7-30 Hz: 68%),
   więc oczy raczej nie tłumaczą 99%; główne wyjaśnienie to sposób podziału.

Status: analiza tekstu, NIE obalenie empiryczne. Do zrobienia, żeby był to wniosek publikacyjny:
odtworzyć ich ustawienie (pary kanałów jako osobne przykłady, losowy podział segmentów, 7 osób)
i to samo z podziałem po osobach (N-LNSO), na tych samych danych. Oczekiwanie: wysoki wynik przy
podziale segmentów, spadek do poziomu ok. 50-67% (2 klasy) / znacznie niżej dla 5 klas przy podziale po osobach.
Do sekcji "related work": przykład zawyżonych wyników na EEGMMIDB przy podziale nieopisanym / nie po osobach;
nasz wkład = ile zostaje przy podziale po osobach (ok. 67%) i ile dają oczy (83,5 → 67).

## 5. Moje błędy w trakcie (do pamiętania)

- Twierdziłem, że `BAD_SUBJECTS` nie istnieje (istniało).
- Nazwałem 84% wyciekiem, potem "szczęśliwym holdoutem"; oba twierdzenia błędne.
- W starych configach zamieniłem `normalize: true` na `normalization: "none"` jako "zgodne ze starym
  zachowaniem"; dla przebiegu z 84% nieprawdziwe. Do poprawy.
- Okno 0,5-2,5 s i pasmo 4-40 Hz w benchmarku wybrałem a priori; komentarz o koszcie 2 s (~1,3 pp) się nie potwierdził.
- Wniosek "przewaga siedzi w pierwszych 0,5 s" był prawdziwy tylko dla 7-30 Hz.

- Twierdziłem, że informacja z czoła siedzi tylko w niskich częstotliwościach; test 7-30 Hz temu przeczy.
- `build_epochs` podawał nazwy kanałów w kolejności z pliku EDF, a dane były w kolejności z configu
  (MNE `pick`). Na klasyfikację bez wpływu (dotyczyło tylko metadanych przy podzbiorach kanałów);
  naprawione przed regresją EOG, która wymaga poprawnych nazw.

- Klucz cache epok nie zawierał przebiegów (tylko id osób i ustawienia): ruch i wyobrażenie z tymi samymi
  ustawieniami trafiłyby w ten sam plik. Wcześniejszych wyników nie dotyczy (wszystkie na R04/R08/R12);
  naprawione przed biegiem ruchu (odcisk nagrania w kluczu, test).
- Regresja EOG była wykonywana PO normalizacji; przy EA to błąd (EA miesza kanały). Od commitu
  z `scripts/run_mi_check.sh` regresja idzie na surowych epokach, normalizacja po niej. Wynik
  `legacy_eye_motor_regressed` (69,3%) policzony jeszcze starą kolejnością (zscore przed regresją;
  dla zscore różnica powinna być niewielka, nie sprawdzone).

## 6. Otwarte

- Benchmark MI policzony (3.7); użytkownik zatwierdził `mi_reg_mu_beta_nocue` jako główny (2026-10-08).
- Bieg nocny przygotowany (`scripts/run_night.sh`, `configs/mi_tuned.yaml`, `src/eval/tuning.py`):
  te same dane co `mi_reg_mu_beta_nocue`; w każdym foldzie zewnętrznym osoby treningowe dzielone na 3 grupy,
  każdy kandydat uczony na 2 grupach (checkpoint na osobach walidacyjnych foldu), oceniany na trzeciej;
  najlepszy (średnia dokładność per osoba) uczony na całym treningu, 3 seedy, test raz.
  Kandydaci: EEGNet 40 losowych z 288 kombinacji siatki ze starego `train.py` (+ domyślny),
  Shallow 8, CSP n_components 5, TS+LR C 4. Wybór na ok. 25 osobach na grupę: spodziewany zysk mały,
  szum wyboru porównywalny z różnicami między kandydatami.
- Dodana sieć autora `SpatialEEGNetTransformer` (`src/models/eegnet_transformer.py`, w rejestrze jako
  `eegnet_transformer`, w biegu nocnym 8 kandydatów: lr, dropout, num_layers). Poprawka względem szkicu:
  liczba tokenów z próbnego przejścia (oryginał wysypywał się np. dla T=639; dla naszego T=561 działał).
- Rezydualne 62% w 0,5-4 Hz po regresji: oczy czy wolne potencjały ruchowe? Nierozstrzygnięte.
- Commity d4e813f..0423ad0 mają linię Co-Authored-By; użytkownik nie chce jej nigdy. Przepisanie historii wymaga force pusha (decyzja użytkownika).
- Poprawki: guard sprawdzający nazwę normalizacji; usunąć k=40; `normalize` w starych configach.
- Push do `development` na GitHubie zwraca 500; commity idą przez branch `legacy-check`.
- Niezweryfikowane: listy kanałów BrainAccess, część cytowań "do weryfikacji", `poetry.lock`,
  stary `train.py` po zmianach (tylko kompilacja).
