# Zadania zespołu



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
| Data augmentation (sliding window, noise, warping) | 🔧 notebook `brain_board_augmentation.ipynb`, poza N-LNSO |
| Ensemble (voting/stacking of best models) | 🔧 notebook `brain_board_ensemble_subject_based.ipynb`, poza N-LNSO |
| Feature extraction: Hjorth parameters, kurtosis, variance | 🔧 notebook `hjorth_parameters.ipynb` (CV w obrębie osoby), poza N-LNSO |
| Feature extraction: band powers, spectral entropy, mu/beta ratio | ❌ |
| Feature extraction: wavelets/STFT → 2D for CNN | ❌ |
| Feature extraction: connectivity (PLV, coherence) | ❌ |
| Feature fusion + selection (mutual information, RFE) | ❌ |
| Subject-adaptive bandpass | ❌ |
| Riemannian geometry (pyriemann) | ✅ `ts_lr` (Tangent Space + LR) w `src/eval`, wyniki w `docs/lab_notebook.md` |
| Attention-based EEGNet | ✅ `SpatialEEGNetTransformer` (`src/models/eegnet_transformer.py`): 63,9%, najsłabszy z modeli |
| Transfer learning (pretrain → fine-tune per subject) | ❌ |
| Sliding window inference | ❌ |
| EEGConformer / ATCNet (braindecode) | ❌ |
| GAN — synthetic EEG epoch augmentation (WGAN-GP / conditional GAN) | ❌ |
| Variational Autoencoder — latent space + classification on embeddings | ❌ |
| Contrastive encoder (SimCLR/BYOL-style) — self-supervised pre-training | ❌ |
| Autoencoder denoising — pre-train encoder → fine-tune classifier | ❌ |
| ICA artifact removal (eye blinks, muscle artifacts) + comparison w/wo | ❌ zamiast tego regresja kanałów czołowych per osoba (notatka 3.7) |
| Per-trial normalization (z-score per epoch instead of per subject) | ❌ |
| Per-sample normalization (z-score per timepoint across channels) | ❌ |
| Euclidean Alignment (covariance matrix centering per subject) | ✅ `src/data/normalization.py`, używane w wynikach |
| Min-max normalization comparison | ❌ |
| Robust scaling (median/IQR — resistant to EEG artifacts) | ❌ |
| Normalization strategy grid search (z-score vs min-max vs robust vs EA vs per-sample) | 🔧 none, z-score, EMS, EA gotowe; min-max, robust i per-sample brak |
| Full joint grid search on 64 channels (preprocessing × all models) | ❌ |
| 64ch vs 21ch comparison (best combos head-to-head, same preprocessing) | ✅ notatka 3.3 i 3.4 |
| Generative models approach | ❌ |
| Statistical analysis (p-values between approaches) | ✅ bootstrap CI po osobach, Wilcoxon z korektą Holma (`src/eval/stats.py`, `scripts/make_figures.py`) |
| Blink trigger (real-time app) | ❌ |
| Comparison between channels. Check only MI channels vs all channels vs only non MI channels | ✅ ruchowe vs czołowe vs potyliczne vs wszystkie (`configs/legacy_*.yaml`, notatka 3.5) |
| Retrain EEGNet na 16ch subset (BrainAccess MIDI channels) z PhysioNet | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Channel mapping PhysioNet→MIDI + sferyczna interpolacja brakujących | ❌ |
| Try to find BCI-Illiterate people and do a full result analysis | 🔧 zaimplementowane w `src/eval`, wyniki po uruchomieniu na serwerze |
| Resample 250Hz + fine-tune na własnych danych z BrainAccess MIDI | ❌ |

Stan na 2026-10-16. Pozycje oznaczone ✅ z wynikami: `docs/lab_notebook.md`.
