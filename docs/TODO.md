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
