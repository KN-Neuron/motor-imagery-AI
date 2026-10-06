# Protokół eksperymentu (opis do wniosku do komisji)

Wersja robocza. Parametry z oznaczeniem "do ustalenia" wymagają decyzji zespołu i komisji.

## Cel i uczestnicy

Zebranie nagrań EEG z systemu BrainAccess podczas wyobrażania ruchu dłoni (MI) i wykonywania ruchu (ME, kontrola) w celu oceny transferu modeli trenowanych na publicznym zbiorze PhysioNet EEGMMIDB. Liczba uczestników: 12 do 16 osób dorosłych, zdrowych, praworęcznych (kryteria do ustalenia). Każda osoba bierze udział w 2 sesjach w różne dni (odstęp co najmniej 24 godziny).

## Sprzęt

BrainAccess (wariant MIDI 16 kanałów lub MAXI 32 kanały, suche elektrody), częstotliwość próbkowania zgodna z urządzeniem i zapisana w pliku, rejestracja EMG przedramienia lub akcelerometru dłoni w blokach ME (do ustalenia). Rozmieszczenie elektrod i nazwy kanałów zapisane w metadanych BIDS (`channels.tsv`, `electrodes.tsv`).

## Przebieg sesji (około 60 do 75 minut łącznie z założeniem czepka)

1. Informacja, zgoda, ankieta (wiek, ręczność, doświadczenie z BCI; bez danych identyfikujących). Założenie sprzętu, kontrola impedancji lub jakości sygnału.
2. **Spoczynek z otwartymi oczami: 2 minuty** (fiksacja na krzyżu). Dla predyktora SMR (RQ3). Opcjonalnie 1 minuta z zamkniętymi oczami.
3. **Bloki MI (wyobrażanie ruchu zaciskania lewej/prawej dłoni):** 5 bloków po 20 prób (10 na klasę, losowa kolejność), łącznie 100 prób MI na sesję, 50 na klasę (do ustalenia, docelowo do 80 do 100 na klasę na sesję wg `docs/research_plan.md`). Instrukcja kinestetycznego wyobrażania, bez ruchu i napinania mięśni.
4. **Bloki ME (wykonanie zaciskania dłoni, kontrola):** 2 bloki po 20 prób (10 na klasę), łącznie 40 prób, **zawsze po blokach MI** (aby uniknąć wpływu wykonania na wyobrażanie), osobno oznaczone w pliku zdarzeń jako ME.
5. Przerwy: 1 do 2 minuty między blokami, 5 minut w połowie sesji.

## Struktura jednej próby

Krzyż fiksacyjny 2 s, bodziec (strzałka w lewo lub w prawo) 1 s, zadanie 4 s, odpoczynek 3 do 5 s (losowy jitter, aby uniknąć synchronizacji z rytmami). Całkowity czas próby około 10 do 12 s, czyli blok 20 prób trwa około 4 minut. Dla wsparcia porównania z PhysioNet okno analizy to 2 s w obrębie zadania (np. 0,5 do 2,5 s po bodźcu), zgodne z konfiguracją treningu.

## Zdarzenia i etykiety

`events.tsv`: `onset`, `duration`, `trial_type`: `LEFT_HAND_IMAGERY`, `RIGHT_HAND_IMAGERY` (MI), `LEFT_HAND_CLENCH`, `RIGHT_HAND_CLENCH` (ME). Pole `trial_type` jednoznacznie rozróżnia MI i ME; kod (`TASK_KIND` w `src/ba/pipeline.py`) odrzuca mieszanie. Bloki numerowane (podział po blokach i sesjach, nigdy po losowych epokach).

## Kontrola jakości i artefakty

Odrzucanie prób po amplitudzie (próg do ustalenia), notowanie ruchów oczu i mięśni, brak ICA jako wymogu (patrz `docs/research_plan.md`). Prób z ruchem w blokach MI nie zaliczać do analizy; raportować liczbę odrzuceń.

## Dane i ochrona prywatności

Pseudonimizacja (kod `pNNN`), klucz osobno i szyfrowany, eksport do BIDS-EEG (`src.ba.bids.export_bids`), usunięcie z nagłówka EDF imienia i daty urodzenia, daty nagrania uogólnione. Czas przechowywania i administrator danych: do ustalenia z uczelnią. Udostępnienie otwarte tylko za osobną zgodą uczestnika.

## Ryzyka dla uczestnika

Minimalne: nieinwazyjny zapis, możliwy dyskomfort suchych elektrod i zmęczenie. Kryteria wykluczenia i procedura przerwania badania: do uzgodnienia z komisją.
