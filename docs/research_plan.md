# Plan badawczy (do zamrożenia przed zbieraniem danych)

Status: **szkic**. Kryteria sukcesu i porażki poniżej mają być zamrożone (data, hash commita, najlepiej rejestracja np. OSF) **przed** pierwszą sesją nagrań. Wszystkie liczby oznaczone "ZAŁOŻENIE" mają być zastąpione wartościami zmierzonymi na PhysioNet (`make reproduce`, potem `python scripts/power_analysis.py --per-subject results/main/per_subject.csv --ref "eegnet|euclidean_alignment"`). Dane BrainAccess dostępne obecnie (kilkanaście minut od jednej osoby, zadanie wykonania ruchu) nie służą do żadnych wniosków ani strojenia.

## Zasady metodologiczne (obowiązują wszystkie pytania)

1. Twierdzenia cross-subject tylko przy podziale po osobach (N-LNSO, Del Pup i in. 2025). Żadnego losowego podziału epok.
2. Wybór checkpointu i hiperparametrów wyłącznie na walidacji z danych treningowych. Zbiór testowy użyty raz na końcu.
3. Normalizacja i macierze alignmentu liczone bez etykiet i tylko na danych danej osoby lub sesji.
4. Jednostką niezależną jest osoba. Przedziały ufności: bootstrap po osobach. Seedy uśredniane wewnątrz osoby.
5. Pomiar jest wyłącznie na wyobrażeniu ruchu (MI). Zadanie "hand clench" to wykonanie ruchu (ME) i nie jest walidacją MI.

## Pytania badawcze

### RQ1 (główne): transfer PhysioNet na BrainAccess z alignmentem i kalibracją few-shot

- **H1a:** model wytrenowany na PhysioNet na wspólnym podzbiorze kanałów, z Euclidean Alignment i adaptacją na k próbach kalibracyjnych, osiąga na MI z BrainAccess wyższą dokładność per osoba niż ten sam model bez adaptacji (k = 0).
- **H1b:** przy k = 20 prób na klasę taki model jest lepszy od EEGNet trenowanego od zera na tych samych k próbach (kalibracja: 0, 10, 20, 40 na klasę).
- **Miary:** dokładność per osoba na sesji testowej (blok MI), mediana i średnia z CI (bootstrap po osobach), odsetek osób powyżej progu dwumianowego dla faktycznej liczby prób testowych (Combrisson i Jerbi 2015), odsetek osób >= 70%.
- **Testy:** Wilcoxon sparowany po osobach (te same osoby w obu warunkach), korekta Holma po porównaniach (k, metoda). Efekt: różnica median i CI.
- **Kryterium sukcesu (do zamrożenia):** H1b potwierdzona, jeśli różnica średnich >= 5 pp, p (Holm) < 0,05 i CI różnicy nie obejmuje 0, przy N >= 12 osób, oraz mediana dokładności MI przy k = 20 > próg istotności per osoba. **Porażka:** CI różnicy obejmuje 0 lub mediana <= próg. Taki wynik jest publikowalny jako wynik negatywny z oceną BCI inefficiency.
- **Symulacja wstępna (PhysioNet):** `scripts/run_simulations.py` (degradacje pseudo-BrainAccess, kalibracja few-shot) testuje metodę przed zebraniem danych. Wynik symulacji nie zastępuje pomiaru.

### RQ2 (ablacja): normalizacja, pasmo, okno

- **H2:** niezgodność skali, pasma lub kanałów między treningiem a testem obniża dokładność silniej niż różnica metody klasyfikacji, a alignment i wspólny preprocessing przywracają większość spadku. Hipoteza z raportu (niezgodność skali i pasma sama sprowadza model do poziomu losowego) jest sprawdzana, nie zakładana.
- **Plan:** czynniki: normalizacja (none, zscore_subject_channel, exp_moving_standardization, euclidean_alignment), pasmo (4 do 40, 8 do 30, 0,5 do 45), okno (1, 2, 4 s), kanały (64, 32, 16, kanały zerowe jako negatywna kontrola). PhysioNet na PhysioNet w N-LNSO oraz degradacje pseudo-BrainAccess; potem ten sam układ na BrainAccess.
- **Miary i testy:** spadek dokładności względem warunku zgodnego, sparowany po osobach, CI bootstrap po osobach; Wilcoxon z korektą Holma.
- **Sukces:** co najmniej jedna degradacja daje spadek > 5 pp z CI wykluczającym 0, a alignment odzyskuje > połowę spadku. **Porażka:** brak spadków istotnych (wtedy poziom losowy na BrainAccess ma inne przyczyny niż niezgodność preprocessingu).

### RQ3 (pomocnicze): predyktor SMR Blankertza

- **H3:** predyktor SMR liczony z 2 minut spoczynku z otwartymi oczami koreluje z dokładnością MI per osoba (na PhysioNet: run R01, na BrainAccess: blok spoczynkowy). Implementacja `src/eval/smr.py` jest przybliżeniem (Blankertz i in. 2010, r = 0,53 przy N = 80 wg raportu, liczba do weryfikacji).
- **Miara i test:** korelacja Spearmana, CI bootstrap po osobach. Wyniki z PhysioNet traktowane jako replikacja kierunkowa, z BrainAccess jako eksploracja.
- **Sukces:** rho > 0 z CI wykluczającym 0 na PhysioNet. Na BrainAccess przy N rzędu 12 do 16 test ma niską moc (patrz niżej), więc wynik pozostaje eksploracyjny.

## Analiza mocy

Narzędzie: `scripts/power_analysis.py`. Test sparowany (t, przybliżenie do Wilcoxona), alfa 0,05 dwustronnie, moc 0,8, rozkład nieskończenie-centralny t (`n_subjects_paired`).

Liczby przy ZAŁOŻENIU SD per osoba oraz SD różnic sparowanych = 10 pp (do zastąpienia zmierzonymi):

| scenariusz | wynik |
|---|---|
| próg istotności per osoba, 45 prób (p<0,05) | 64,4% |
| próg istotności per osoba, 100 prób | 59,0% |
| próg istotności per osoba, 200 prób | 56,5% |
| osób do wykazania średniej 60% > 50% | 10 |
| osób do wykrycia różnicy 5 pp między warunkami | 34 |
| osób do wykrycia różnicy 8 pp między warunkami | 15 |
| osób do wykrycia różnicy 2 pp między warunkami | 199 |

Wnioski: (1) próg per osoba dla 45 prób (64,4%) potwierdza wyliczenie z raportu (około 64%). Raport pisze "100 prób na klasę daje próg około 56%": to 200 prób łącznie (56,5%), czyli zgodne. (2) Przy N = 12 do 16 osób test potwierdzający ma moc tylko dla efektów >= około 8 pp (przy SD różnic 10 pp). Różnicę 5 pp trzeba traktować jako eksploracyjną, chyba że zmierzone SD różnic okaże się mniejsze (np. 6 pp daje około 15 osób dla 5 pp). (3) Wartość SD zależy od danych: po uruchomieniu na serwerze skrypt przeliczy tabelę na faktycznych wynikach N-LNSO (SD per osoba i SD różnic).

## Minimalny plan próby

- **Osoby:** 12 do 16 (RQ1 potwierdzająco dla efektów >= 8 pp; RQ2 i RQ3 eksploracyjnie). Pilotaż 3 do 4 osób służy wyłącznie debugowaniu pipeline'u i nie daje wniosków.
- **Sesje:** 2 sesje w różne dni (odstęp >= 24 h), co pozwala ocenić transfer cross-session i podział po sesjach.
- **Próby:** po 80 do 100 prób MI na klasę na sesję (160 do 200 łącznie, próg per osoba 56 do 59%). Blok ME jako kontrola (np. 40 prób na klasę). Spoczynek 2 min z otwartymi oczami. Szczegóły: `docs/protocol_eksperymentu.md`.
- **Zbiór:** dodatkowo zbiór kalibracyjny/testowy wydzielany po blokach, nigdy po losowych epokach.

## Kontrola artefaktów

Blok ME wywołuje EMG i artefakty ruchowe, na które suche elektrody są podatne (wg raportu, Sultana i in. 2025, do weryfikacji). Dobry wynik na ME nie jest dowodem na rytmy mu i beta. Wymagane: rejestracja EMG lub akcelerometru w bloku ME (`artifact_control` w `src/ba/pipeline.py` zarezerwowane), brak prób z ruchem w blokach MI (kontrola EMG w MI), raport odrzuconych prób.

## Wymagania etyczne i dane

- **Zgoda komisji ds. etyki** przed rekrutacją. Nie wiadomo (do ustalenia z opiekunem koła), czy PWr ma własną komisję ds. etyki badań z udziałem ludzi, czy trzeba złożyć wniosek do komisji bioetycznej innej uczelni. Numer zgody będzie wymagany przez czasopisma.
- **Świadoma zgoda** pisemna: cel, procedura, czas, brak korzyści medycznych, prawo wycofania bez konsekwencji, zakres udostępniania danych. Osobna zgoda na udostępnienie zanonimizowanych danych w repozytorium otwartym.
- **RODO:** EEG to dane osobowe (jeśli możliwa identyfikacja). Pseudonimizacja (kod `p001`), klucz łączący kod z osobą przechowywany osobno i poza zbiorem, minimalizacja metadanych (bez dat urodzenia, bez pełnych dat nagrania), podstawa prawna i okres przechowywania opisane we wniosku, administrator danych wskazany, procedura usunięcia na żądanie. Opis do weryfikacji przez inspektora ochrony danych uczelni.
- **Format i udostępnianie:** BIDS-EEG (`src/ba/bids.py`, Pernet i in. 2019), anonimizacja nagłówków EDF (brak imienia i daty urodzenia), licencja danych (np. CC0 lub CC-BY) wybrana po konsultacji z komisją.
- **Bezpieczeństwo:** badanie nieinwazyjne, kryteria wykluczenia (np. padaczka) w formularzu zgody; do uzgodnienia z komisją.

## Reprodukowalność

Stałe seedy (`seed`, `eval.seeds` w `configs/benchmark.yaml`), konfiguracja i hash commita w nagłówku każdego raportu, środowisko: `pyproject.toml` + `poetry.lock` (przed publikacją wykonać `poetry lock` po dodaniu zależności `pyriemann`, `mne-bids`, `edfio`, `tabulate`) oraz `make env`. `make reproduce DATA_DIR=...` odtwarza tabele i wykresy z `docs/results.md`.
