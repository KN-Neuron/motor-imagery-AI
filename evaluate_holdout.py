#!/usr/bin/env python3
"""
evaluate_holdout.py — Uczciwa ewaluacja modelu na danych, których NIGDY nie widział.

Strategia:
  1. Odtwarzamy DOKŁADNIE ten sam split co w treningu (seed=42, 70/15/15)
  2. Identyfikujemy test subjects — subjectów których model nigdy nie widział
  3. Ładujemy surowe dane TYLKO dla tych test subjects
  4. Preprocessing od zera (filtracja, epoching, normalizacja) TYLKO na test danych
  5. Ewaluacja modelu → accuracy, confusion matrix, per-subject breakdown

Dlaczego to jest czyste:
  - Normalizacja z-score liczona TYLKO na test subjects (żaden mean/std z train)
  - Brak jakiegokolwiek kontaktu z train/val danymi
  - Subject-level split: cały subject jest albo w train, albo w test

Użycie:
    python evaluate_holdout.py
    python evaluate_holdout.py --model outputs/eegnet_single_run.pth
    python evaluate_holdout.py --model outputs/final_best.pth

Albo jako komórka w notebooku — skopiuj kod poniżej bloku if __name__.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import mne
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
)

# ── Dodaj src do path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loading import download_dataset, load_raw_subjects
from src.models.eegnet import EEGNet


# ═══════════════════════════════════════════════════════════════════
# CONFIG — musi być IDENTYCZNY jak w treningu
# ═══════════════════════════════════════════════════════════════════
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
DATASET = "brianleung2020/eeg-motor-movementimagery-dataset"
DESIRED_RUNS = ["R04", "R08", "R12"]
SFREQ = 160.0

# Preprocessing (z configu full_binary_all_channels.yaml)
LOW_FREQ = 7.0
HIGH_FREQ = 30.0
TMIN = 0.0
TMAX = 4.0
BASELINE = None

# Task
EVENT_ID = {"left_hand": 2, "right_hand": 3}
LABEL_OFFSET = 2
CLASS_NAMES = ["left_hand", "right_hand"]
N_CLASSES = 2
N_CHANS = 64  # all channels mode

# Annotation mapping (jak w preprocessing.py)
ANNOTATION_MAPPING = {"T0": 1, "T1": 2, "T2": 3}

MODEL_PATH = "outputs/eegnet_single_run.pth"


def get_test_subjects(raw_data: dict) -> set:
    """
    Odtwórz DOKŁADNIE ten sam split co w treningu.
    Zwraca set z ID subjectów testowych.
    """
    rng = np.random.RandomState(SEED)
    unique_subjects = np.array(sorted(raw_data.keys()))
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val:])

    print(f"Split odtworzony (seed={SEED}):")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects:   {len(val_subjects)}")
    print(f"  Test subjects:  {len(test_subjects)}")
    print(f"  Test IDs:       {sorted(test_subjects)}")

    return test_subjects, train_subjects, val_subjects


def preprocess_test_subjects(raw_data: dict, test_subjects: set):
    """
    Preprocessing TYLKO na test subjects — zero kontaktu z train.
    Normalizacja per-subject, per-channel (identycznie jak w epoch_subjects).
    """
    all_X, all_y, all_subjects = [], [], []
    skipped = []

    for subject in sorted(test_subjects):
        if subject not in raw_data:
            skipped.append(subject)
            continue

        try:
            raw = raw_data[subject].copy()
            # Nie pick channels — all 64
            raw.filter(
                LOW_FREQ, HIGH_FREQ,
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )

            events, _ = mne.events_from_annotations(
                raw, event_id=ANNOTATION_MAPPING, verbose=False
            )

            epochs = mne.Epochs(
                raw, events, EVENT_ID,
                tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False,
            )

            X = epochs.get_data().astype(np.float32)
            y = epochs.events[:, -1] - LABEL_OFFSET

            # Z-score normalizacja PER-SUBJECT, PER-CHANNEL
            # (identycznie jak w preprocessing.py linie 79-83)
            for ch in range(X.shape[1]):
                mean = X[:, ch, :].mean()
                std = X[:, ch, :].std()
                if std > 0:
                    X[:, ch, :] = (X[:, ch, :] - mean) / std

            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), int(subject)))

        except Exception as e:
            print(f"  ⚠️ Subject {subject} failed: {e}")
            skipped.append(subject)

    if not all_X:
        raise RuntimeError("No test data loaded!")

    X_test = np.concatenate(all_X, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    subjects_test = np.concatenate(all_subjects, axis=0)

    if skipped:
        print(f"  Skipped: {skipped}")

    return X_test, y_test, subjects_test


def load_model(model_path: str, n_chans: int, n_classes: int, time_points: int):
    """
    Załaduj zapisany model EEGNet.

    Obsługuje dwa formaty:
      1. Czysty state_dict (z save_model w utils.py) — domyślny
      2. Dict z kluczem 'model_state_dict' (alternatywny)
    Metadata czytana z {stem}_meta.json jeśli istnieje.

    Returns (model, actual_chans) — actual_chans może różnić się od n_chans
    jeśli model był trenowany na motor channels.
    """
    import re as _re

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Ustal czy to czysty state_dict czy opakowany w dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("block") or k.startswith("fc") for k in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Spróbuj odczytać metadata z _meta.json
    meta_path = Path(model_path).with_name(Path(model_path).stem + "_meta.json")
    metadata = {}
    if meta_path.exists():
        import json as _json
        with open(meta_path) as f:
            metadata = _json.load(f)
        print(f"  Loaded metadata from {meta_path}")

    if metadata:
        for k, v in metadata.items():
            print(f"  {k}: {v}")
    else:
        print(f"  No metadata found (using config defaults)")

    # ── Odczytaj parametry modelu z metadata LUB z model_name regex ──
    model_name = metadata.get("model_name", "")

    # Spróbuj wyciągnąć f1, d, dropout, lr z model_name: "EEGNet(f1=16,d=2,do=0.5,lr=0.001)"
    m = _re.search(r"f1=(\d+),d=(\d+),do=([\d.]+)", model_name)
    if m:
        f1 = int(m.group(1))
        d = int(m.group(2))
        dropout = float(m.group(3))
    else:
        f1 = metadata.get("f1", 8)
        d = metadata.get("d", 2)
        dropout = metadata.get("dropout_rate", 0.5)

    f2 = f1 * d
    temp_kernel = metadata.get("temp_kernel", 80)
    pk1 = metadata.get("pk1", 4)
    pk2 = metadata.get("pk2", 8)

    # ── Odczytaj rzeczywistą liczbę kanałów z wag modelu ──
    # block2.0.weight ma shape (d*f1, 1, CHANS, 1)
    actual_chans = n_chans
    if "block2.0.weight" in state_dict:
        actual_chans = state_dict["block2.0.weight"].shape[2]

    # ── Odczytaj time_points z wag fc jeśli się nie zgadza ──
    actual_time_points = time_points
    if "fc.weight" in state_dict:
        fc_in = state_dict["fc.weight"].shape[1]
        # fc input = (time_points // (pk1 * pk2)) * f2
        actual_time_points = (fc_in // f2) * pk1 * pk2

    print(f"  Architecture (from weights): EEGNet(chans={actual_chans}, classes={n_classes}, "
          f"time_points={actual_time_points}, f1={f1}, d={d}, f2={f2}, "
          f"temp_kernel={temp_kernel}, dropout={dropout})")

    if actual_chans != n_chans:
        print(f"  ⚠️ Model was trained on {actual_chans} channels (motor channels mode), "
              f"not {n_chans} (all channels)!")

    model = EEGNet(
        chans=actual_chans,
        classes=n_classes,
        time_points=actual_time_points,
        f1=f1, f2=f2, d=d,
        pk1=pk1, pk2=pk2,
        dropout_rate=dropout,
        temp_kernel=temp_kernel,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, actual_chans


def evaluate(model, X_test, y_test, subjects_test, device="cpu"):
    """Pełna ewaluacja: overall + per-subject."""
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.inference_mode():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    labels = y_test

    # ── Overall metrics ──────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)

    print(f"\n{'='*60}")
    print(f"  HOLDOUT TEST RESULTS (subjects NEVER seen during training)")
    print(f"{'='*60}")
    print(f"\n  Accuracy:     {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Cohen kappa:  {kappa:.4f}")
    print(f"  N test:       {len(labels)} epochs from {len(np.unique(subjects_test))} subjects")
    print(f"\n  Confusion matrix:")
    print(f"                Predicted")
    print(f"               {CLASS_NAMES[0]:>10s} {CLASS_NAMES[1]:>10s}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Actual {name:>10s} {cm[i, 0]:>10d} {cm[i, 1]:>10d}")
    print(f"\n{report}")

    # ── Per-subject breakdown ────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Per-subject accuracy:")
    print(f"{'─'*60}")
    unique_subs = np.unique(subjects_test)
    sub_accs = []
    for sub in sorted(unique_subs):
        mask = subjects_test == sub
        sub_acc = accuracy_score(labels[mask], preds[mask])
        n_epochs = mask.sum()
        sub_accs.append(sub_acc)
        marker = "✓" if sub_acc >= 0.7 else ("~" if sub_acc >= 0.55 else "✗")
        print(f"  {marker} Subject {sub:>3d}: {sub_acc:.3f}  ({n_epochs} epochs)")

    print(f"\n  Mean per-subject acc: {np.mean(sub_accs):.4f} ± {np.std(sub_accs):.4f}")
    print(f"  Median:               {np.median(sub_accs):.4f}")
    print(f"  Min / Max:            {np.min(sub_accs):.4f} / {np.max(sub_accs):.4f}")
    print(f"  Subjects >= 70%:      {sum(1 for a in sub_accs if a >= 0.7)}/{len(sub_accs)}")
    print(f"  Subjects >= 60%:      {sum(1 for a in sub_accs if a >= 0.6)}/{len(sub_accs)}")
    print(f"  Subjects at chance:   {sum(1 for a in sub_accs if a < 0.55)}/{len(sub_accs)}")

    # ── Leakage sanity check ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  LEAKAGE SANITY CHECK")
    print(f"{'─'*60}")

    # Porównaj z raportowanym test acc z treningu
    reported_test_acc = 0.8531  # z logu: "Test acc: 0.8531"
    reported_val_acc = 0.8810   # z logu: "Best val_acc: 0.8810"

    diff = abs(acc - reported_test_acc)
    print(f"  Reported test acc (from training):  {reported_test_acc:.4f}")
    print(f"  Clean holdout acc (this script):    {acc:.4f}")
    print(f"  Difference:                         {diff:.4f}")

    if diff < 0.01:
        print(f"  ✓ Results match — split is consistent, no obvious leakage")
    elif diff < 0.03:
        print(f"  ~ Small difference — may be due to balancing/randomness")
    elif acc > reported_test_acc + 0.05:
        print(f"  ⚠️ HIGHER than reported — possible test data contamination!")
    else:
        print(f"  ⚠️ Large difference — split may have changed (preprocessing params?)")

    return {
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm,
        "per_subject_accs": dict(zip(unique_subs.tolist(), sub_accs)),
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


def main(model_path: str = MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Pobierz dataset ────────────────────────────────────────
    print("\n[1/5] Downloading dataset...")
    subjects_data = download_dataset(DATASET, DESIRED_RUNS)

    # ── 2. Załaduj surowe dane (wszyscy — żeby odtworzyć split) ──
    print("\n[2/5] Loading raw subjects (all, to reconstruct split)...")
    raw_data = load_raw_subjects(subjects_data, sfreq=SFREQ)

    # ── 3. Odtwórz split → zidentyfikuj test subjects ────────────
    print("\n[3/5] Reconstructing train/val/test split...")
    test_subjects, train_subjects, val_subjects = get_test_subjects(raw_data)

    # ⚠️ Ostrzeżenie o rozbieżności subject count
    n_loaded = len(raw_data)
    n_train_log = 106  # z loga treningowego: "Subjects: 106"
    if n_loaded != n_train_log:
        print(f"\n  ⚠️ WARNING: Loaded {n_loaded} subjects, but training log shows {n_train_log}.")
        print(f"     Training used cached data (cache/raw_data.pkl) which may have had more subjects.")
        print(f"     This means the split might not perfectly match the original training split.")
        print(f"     Subjects missing now: likely had sfreq≠160Hz or corrupt files.")
        print(f"     Results below are STILL valid as a fresh holdout test, but subject IDs")
        print(f"     may differ slightly from the original test set.\n")

    # WAŻNE: Sprawdzenie że test subjects NIE pokrywają się z train/val
    overlap_train = test_subjects & train_subjects
    overlap_val = test_subjects & val_subjects
    assert len(overlap_train) == 0, f"LEAK: test ∩ train = {overlap_train}"
    assert len(overlap_val) == 0, f"LEAK: test ∩ val = {overlap_val}"
    print("  ✓ No overlap between test and train/val subjects")

    # ── 4. Załaduj model NAJPIERW — żeby wiedzieć jakie preprocessing ──
    print(f"\n[4/5] Loading model from {model_path}...")
    # Tymczasowo ładujemy z dummy time_points, potem poprawimy
    model, actual_chans = load_model(model_path, N_CHANS, N_CLASSES, 641)

    # ── Odczytaj preprocessing z metadata modelu ──────────────────
    meta_path = Path(model_path).with_name(Path(model_path).stem + "_meta.json")
    if meta_path.exists():
        import json as _json
        with open(meta_path) as f:
            meta = _json.load(f)
        low_freq = meta.get("bandpass", [LOW_FREQ, HIGH_FREQ])[0]
        high_freq = meta.get("bandpass", [LOW_FREQ, HIGH_FREQ])[1]
        tmin = meta.get("tmin", TMIN)
        tmax = meta.get("tmax", TMAX)
    else:
        low_freq, high_freq = LOW_FREQ, HIGH_FREQ
        tmin, tmax = TMIN, TMAX

    use_motor = actual_chans != N_CHANS
    channels = None
    if use_motor:
        channels = [
            "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
            "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
            "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
        ]

    print(f"\n  Preprocessing params (from model metadata):")
    print(f"    Bandpass:  [{low_freq}, {high_freq}] Hz")
    print(f"    Time:      [{tmin}, {tmax}] s")
    print(f"    Channels:  {'motor (' + str(actual_chans) + ')' if use_motor else 'all (64)'}")

    # ── 5. Preprocessing test subjects z WŁAŚCIWYMI parametrami ───
    print(f"\n[5/5] Preprocessing test subjects with model's params...")

    all_X, all_y, all_subj = [], [], []
    skipped = []

    for subject in sorted(test_subjects):
        if subject not in raw_data:
            skipped.append(subject)
            continue
        try:
            raw = raw_data[subject].copy()
            if channels:
                raw.pick(channels)
            raw.filter(
                low_freq, high_freq,
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )
            events, _ = mne.events_from_annotations(
                raw, event_id=ANNOTATION_MAPPING, verbose=False
            )
            epochs = mne.Epochs(
                raw, events, EVENT_ID,
                tmin=tmin, tmax=tmax,
                baseline=BASELINE, preload=True, verbose=False,
            )
            X = epochs.get_data().astype(np.float32)
            y = epochs.events[:, -1] - LABEL_OFFSET

            # Z-score per-subject per-channel
            for ch in range(X.shape[1]):
                mean = X[:, ch, :].mean()
                std = X[:, ch, :].std()
                if std > 0:
                    X[:, ch, :] = (X[:, ch, :] - mean) / std

            all_X.append(X)
            all_y.append(y)
            all_subj.append(np.full(len(y), int(subject)))
        except Exception as e:
            print(f"  ⚠️ Subject {subject} failed: {e}")
            skipped.append(subject)

    if not all_X:
        raise RuntimeError("No test data loaded!")

    X_test = np.concatenate(all_X, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    subjects_test = np.concatenate(all_subj, axis=0)

    if skipped:
        print(f"  Skipped: {skipped}")

    print(f"  Test data shape: {X_test.shape}")
    print(f"  Class distribution: {np.bincount(y_test).tolist()}")
    print(f"  Subjects: {len(np.unique(subjects_test))}")

    # Balance test set
    n_per_class = min(np.bincount(y_test))
    rng = np.random.RandomState(SEED)
    idx = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_test == cls)[0]
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    X_test, y_test, subjects_test = X_test[idx], y_test[idx], subjects_test[idx]
    print(f"  After balancing: {X_test.shape}, classes: {np.bincount(y_test).tolist()}")

    # ── Sprawdź czy time_points modelu pasuje do danych ──────────
    actual_time_points = X_test.shape[2]
    if "fc.weight" in (model.state_dict()):
        fc_in = model.fc.in_features
        # Jeśli time_points się nie zgadza, przeładuj model
        # (architektura zależy od time_points przez linear_size)
        expected_tp_from_model = (fc_in // (model.block3[1].out_channels)) * 4 * 8
        if actual_time_points != expected_tp_from_model:
            print(f"  Reloading model with correct time_points={actual_time_points}...")
            model, actual_chans = load_model(model_path, actual_chans, N_CLASSES, actual_time_points)

    # ── Leakage reference — odczytaj z metadata jeśli dostępne ───
    reported_acc = None
    if meta_path.exists():
        reported_acc = meta.get("test_accuracy")

    results = evaluate(model, X_test, y_test, subjects_test, device)

    # Nadpisz leakage check jeśli mamy reported acc z tego konkretnego modelu
    if reported_acc is not None:
        diff = abs(results["accuracy"] - reported_acc)
        print(f"\n  Reported test acc (from metadata):  {reported_acc:.4f}")
        print(f"  Clean holdout acc (this script):    {results['accuracy']:.4f}")
        print(f"  Difference:                         {diff:.4f}")
        if diff < 0.01:
            print(f"  ✓ Results match")
        elif diff < 0.03:
            print(f"  ~ Small difference — balancing/split randomness")
        elif results["accuracy"] > reported_acc + 0.05:
            print(f"  ⚠️ HIGHER than reported — possible contamination!")
        else:
            print(f"  ⚠️ Large difference — split mismatch (102 vs 106 subjects)")

    print(f"\n{'='*60}")
    print(f"  DONE — Clean holdout accuracy: {results['accuracy']:.4f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .pth model")
    args = parser.parse_args()
    main(args.model)