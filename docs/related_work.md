# Related work: cytowania

Status weryfikacji: **zweryfikowane** = autor, rok, czasopismo i DOI potwierdzone w Crossref (api.crossref.org) lub arXiv API w dniu 2026-10-06 (sprawdzono zgodność tytułu, pierwszego autora, czasopisma, roku, tomu i stron). **do weryfikacji** = z raportu `docs/literature_review.md` lub z pamięci, nie potwierdzone niezależnie. Zawartość merytoryczna (liczby wyników, treść wniosków) pochodzi z raportu i **nie została sprawdzona** w pełnych tekstach, chyba że zaznaczono inaczej. Przed użyciem w artykule sprawdź liczby w oryginale.

## Zweryfikowane (metadane)

| Praca | Czasopismo, rok | DOI | Użycie w projekcie |
|---|---|---|---|
| Dose, Møller, Iversen, Puthusserypady. An end-to-end deep learning approach to MI-EEG signal classification for BCIs | Expert Systems with Applications 114:532-542, 2018 | 10.1016/j.eswa.2018.08.031 | wynik referencyjny na PhysioNet (80,38% wg raportu, liczba do weryfikacji) |
| Xu i in. Cross-Dataset Variability Problem in EEG Decoding With Deep Learning | Front Hum Neurosci 14, 2020 | 10.3389/fnhum.2020.00103 | transfer cross-dataset, alignment |
| Brookshire i in. Data leakage in deep learning studies of translational EEG | Front Neurosci 18, 2024 | 10.3389/fnins.2024.1373515 | podział po osobach |
| Combrisson, Jerbi. Exceeding chance level by chance: the caveat of theoretical chance levels in brain signal classification | J Neurosci Methods 250:126-136, 2015 | 10.1016/j.jneumeth.2015.01.010 | próg istotności z rozkładu dwumianowego (test `test_binomial_threshold_matches_combrisson`) |
| Schirrmeister i in. Deep learning with convolutional neural networks for EEG decoding and visualization | Hum Brain Mapp 38:5391-5420, 2017 | 10.1002/hbm.23730 | Shallow/Deep ConvNet, exponential moving standardization |
| He, Wu. Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach | IEEE Trans Biomed Eng 67(2):399-410, 2020 | 10.1109/TBME.2019.2913914 | Euclidean Alignment |
| Junqueira i in. A systematic evaluation of Euclidean alignment with deep learning for EEG decoding | J Neural Eng 21(3):036038, 2024 | 10.1088/1741-2552/ad4f18 | EA z sieciami głębokimi |
| Lawhern i in. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces | J Neural Eng 15(5):056013, 2018 | 10.1088/1741-2552/aace8c | EEGNet i max-norm |
| Blankertz i in. Neurophysiological predictor of SMR-based BCI performance | NeuroImage 51:1303-1309, 2010 | 10.1016/j.neuroimage.2010.03.022 | predyktor SMR (RQ3) |
| Sannelli i in. A large scale screening study with a SMR-based BCI: Categorization of BCI users and differences in their SMR activity | PLOS ONE 14(1):e0207351, 2019 | 10.1371/journal.pone.0207351 | częstość BCI inefficiency |
| Lee i in. EEG dataset and OpenBMI toolbox for three BCI paradigms | GigaScience 8(5), 2019 | 10.1093/gigascience/giz002 | BCI illiteracy |
| Pernet i in. EEG-BIDS, an extension to the brain imaging data structure for electroencephalography | Scientific Data 6:103, 2019 | 10.1038/s41597-019-0104-8 | format `src/ba/bids.py` |
| Schalk i in. BCI2000: A General-Purpose Brain-Computer Interface System | IEEE Trans Biomed Eng 51:1034-1043, 2004 | 10.1109/TBME.2004.827072 | twórcy zbioru EEGMMIDB (cytowanie samego zbioru razem z Goldberger i in. 2000, do weryfikacji) |
| Thompson. Critiquing the Concept of BCI Illiteracy | Sci Eng Ethics 25:1217-1233, 2019 (Crossref: rok online 2018) | 10.1007/s11948-018-0061-1 | terminologia |
| Popescu i in. Single Trial Classification of Motor Imagination Using 6 Dry EEG Electrodes | PLoS ONE 2:e637, 2007 | 10.1371/journal.pone.0000637 | suche elektrody |
| Jayaram, Barachant. MOABB: trustworthy algorithm benchmarking for BCIs | J Neural Eng 15:066011, 2018 | 10.1088/1741-2552/aadea0 | benchmark |
| Lotte i in. A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update | J Neural Eng 15:031005, 2018 | 10.1088/1741-2552/aab2f2 | przegląd |
| Roy i in. Deep learning-based electroencephalography analysis: a systematic review | J Neural Eng 16:051001, 2019 | 10.1088/1741-2552/ab260c | przegląd |
| Craik, He, Contreras-Vidal. Deep learning for electroencephalogram (EEG) classification tasks: a review | J Neural Eng 16:031001, 2019 | 10.1088/1741-2552/ab0ab5 | przegląd |
| Zanini i in. Transfer Learning: A Riemannian Geometry Framework With Applications to Brain-Computer Interfaces | IEEE Trans Biomed Eng 65(5):1107-1116, 2018 | 10.1109/TBME.2017.2742541 | re-centering Riemannowski |
| Rodrigues, Congedo, Jutten. Riemannian Procrustes Analysis: Transfer Learning for Brain-Computer Interfaces | IEEE Trans Biomed Eng 66:2390-2401, 2019 | 10.1109/TBME.2018.2889705 | RPA |
| Zhang i in. Predicting Inter-session Performance of SMR-Based Brain-Computer Interface Using Resting-State Spectral Entropy (tytuł wg Crossref: "Predicting Inter-session Performance of SMR-Based Brain-Computer Interface...") | Brain Topography 28:680-690, 2015 | 10.1007/s10548-015-0429-3 | alternatywny predyktor spoczynkowy (zgodność z tezą "entropia widmowa C3 r = 0,65" z raportu do weryfikacji) |
| Shuqfa i in. Decoding Multi-Class Motor Imagery and Motor Execution Tasks Using Riemannian Geometry Algorithms on Large EEG Datasets | Sensors 23(11):5051, 2023 | 10.3390/s23115051 | ME vs MI na PhysioNet (raport powołuje się także na "Shuqfa i in. 2024" w sprawie osoby 106: nie sprawdzono) |

## Zweryfikowane tylko w arXiv (tytuł, autor, rok; wersja czasopismowa do weryfikacji)

| Praca | arXiv | Uwagi |
|---|---|---|
| Wang X. i in. An Accurate EEGNet-based Motor-Imagery Brain-Computer Interface for Low-Power Edge Computing, 2020 | 2004.00077 | raport podaje IEEE MeMeA 2020 (venue do weryfikacji); wyniki 82,43% / 84,32% z raportu, nie sprawdzone |
| Del Pup i in. The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject classification, 2025 | 2505.13021 | źródło N-LNSO; wersja w Computers in Biology and Medicine do weryfikacji |
| Chevallier i in. The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark, 2024 | 2404.15319 | |
| Köllőd i in. Deep comparisons of Neural Networks from the EEGNet family, 2023 | 2302.08797 | raport podaje Electronics 12:2743; DOI `10.3390/electronics12132743` z próby NIE istnieje w Crossref, więc **DOI do weryfikacji**. Lista wykluczeń 88, 89, 92, 100 stąd (do weryfikacji w tekście) |
| Wang J. i in. CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding, 2024 (ICLR 2025 wg raportu) | 2412.07236 | |
| Wu D. i in. Revisiting Euclidean Alignment for Transfer Learning in EEG-Based Brain-Computer Interfaces, 2025 | 2502.09203 | liczba "59,71% do 79,79%" z raportu, nie sprawdzona |

## Do weryfikacji (nie potwierdzone)

- Gwon, Ahn 2024, NeuroImage (transfer ME do MI): brak DOI, wyszukiwanie Crossref przerwane limitem zapytań.
- Allison, Neuper 2010, "Could anyone use a BCI?", Springer: brak DOI.
- Sultana i in. 2025, J Neural Eng 22(6):066045, doi:10.1088/1741-2552/ae2e8a (z raportu; wyszukiwanie po tytule nie znalazło).
- Frontiers 2025, doi:10.3389/fnins.2025.1689647: Crossref wskazuje na pracę **Zheng i in., "Motor imagery EEG classification via wavelet-packet synthetic augmentation..."**. Teza raportu, że ta praca wyklucza osoby 38, 88, 89, 92, 100, 104, **nie została sprawdzona**.
- Frontiers 2025 TCPL (doi:10.3389/fnins.2025.1689286), Sensors 2026 (doi:10.3390/s26113310), Biosensors 2026 16(9):467, arXiv:2607.22778, arXiv:2512.08959, arXiv:2507.07622, arXiv:2006.00622, Roots i in. Sensors 2023 23(18):7908, Sensors 2021 doi:10.3390/s21196672, Front Neurosci 2022 PMC9124859: wszystko z raportu, nic nie sprawdzone. Identyfikatory z dat po 2026 mogą być błędne.
- Altaheri 2023 (ATCNet), Song 2023 (EEG Conformer), LaBraM, EEGPT, BIOT: DOI do weryfikacji.
- Goldberger i in. 2000 (PhysioNet), cytowanie zbioru EEGMMIDB: wymagane przez PhysioNet, DOI do weryfikacji.
