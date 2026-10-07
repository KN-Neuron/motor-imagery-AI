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

## 4. Wnioski robocze (stan na 2026-10-07)

1. Stary pipeline był poprawny metodologicznie; 84% to nie wyciek ani przypadek.
2. **Same kanały czołowe dają tyle co pełny model**, a cały zysk przychodzi z pasma poniżej ok. 7 Hz;
   CSP (tylko moc) w szerokim paśmie spada do losowego. Najprostsze wyjaśnienie: model rozpoznaje
   kierunek spojrzenia (cel w EEGMMIDB pojawia się z lewej/prawej strony ekranu), nie wyobrażenie ruchu.
   **Hipoteza, nie dowód**: EEGMMIDB nie ma kanałów EOG.
3. W paśmie 7-30 Hz usunięcie pierwszych 0,5 s kosztuje 17 pp, w szerokim tylko 5 pp
   (spójne z utrzymanym spojrzeniem przez całą próbę).
4. Do czasu wykluczenia oczu jako liczbę MI raportujemy wariant bez najniższych częstotliwości
   (np. 71,5% przy 4-40 Hz), a i tam udział oczu nie jest wykluczony.
5. Dla BrainAccess: bodziec w środku ekranu i fiksacja; inaczej model może działać na oczach.

## 5. Moje błędy w trakcie (do pamiętania)

- Twierdziłem, że `BAD_SUBJECTS` nie istnieje (istniało).
- Nazwałem 84% wyciekiem, potem "szczęśliwym holdoutem"; oba twierdzenia błędne.
- W starych configach zamieniłem `normalize: true` na `normalization: "none"` jako "zgodne ze starym
  zachowaniem"; dla przebiegu z 84% nieprawdziwe. Do poprawy.
- Okno 0,5-2,5 s i pasmo 4-40 Hz w benchmarku wybrałem a priori; komentarz o koszcie 2 s (~1,3 pp) się nie potwierdził.
- Wniosek "przewaga siedzi w pierwszych 0,5 s" był prawdziwy tylko dla 7-30 Hz.

## 6. Otwarte

- Testy rozstrzygające oczy: bipolarny F7-F8 w 0,5-4 Hz z LDA; kanały ruchowe po regresji kanałów
  czołowych; kanały czołowe w 7-30 Hz.
- Poprawki: guard sprawdzający nazwę normalizacji; usunąć k=40; `normalize` w starych configach.
- Push do `development` na GitHubie zwraca 500; commity idą przez branch `legacy-check`.
- Niezweryfikowane: listy kanałów BrainAccess, część cytowań "do weryfikacji", `poetry.lock`,
  stary `train.py` po zmianach (tylko kompilacja).
