#!/usr/bin/env python3
"""
clean_train_eval.py — Trenuj i ewaluuj EEGNet z zerowym wyciekiem danych.

Gwarancje:
  1. Split na SUBJECT level: 90% train+val, 10% test — ZANIM cokolwiek zostanie przetworzone
  2. Preprocessing (filtracja, epoching, normalizacja) odbywa się OSOBNO
     na train+val i na test — żaden mean/std nie przecieka
  3. Model NIGDY nie widzi test data — nawet nie dostaje test_dataloader
  4. Ewaluacja na teście to OSOBNA funkcja, wywoływana po zamknięciu treningu
  5. Parametry preprocessingu i modelu zdefiniowane W JEDNYM MIEJSCU na górze

Architektura: EEGNet (najlepszy potwierdzony model — 85.3% na holdout)
Dataset: PhysioNet Motor Imagery (binary: left_hand vs right_hand)

Użycie:
    python clean_train_eval.py
    python clean_train_eval.py --seed 123        # inny seed
    python clean_train_eval.py --epochs 150      # więcej epok
    python clean_train_eval.py --test-ratio 0.15 # 85/15 split
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import EEGDataset
from src.data.loading import download_dataset, load_raw_subjects
from src.models.eegnet import EEGNet


# ═══════════════════════════════════════════════════════════════════
# JEDYNY BLOK KONFIGURACJI — wszystko zdefiniowane tu
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # Data
    "dataset": "brianleung2020/eeg-motor-movementimagery-dataset",
    "desired_runs": ["R04", "R08", "R12"],
    "sfreq": 160.0,

    # Preprocessing
    "low_freq": 7.0,
    "high_freq": 30.0,
    "tmin": 0.0,
    "tmax": 4.0,
    "baseline": None,
    "channels": None,       # None = all 64

    # Task
    "event_id": {"left_hand": 2, "right_hand": 3},
    "label_offset": 2,
    "class_names": ["left_hand", "right_hand"],
    "n_classes": 2,

    # EEGNet architecture (potwierdzone jako najlepsze na holdout: 85.3%)
    "f1": 8,
    "d": 2,
    "temp_kernel": 80,
    "pk1": 4,
    "pk2": 8,
    "dropout_rate": 0.5,

    # Training
    "epochs": 100,
    "lr": 0.001,
    "batch_size": 64,
    "val_fraction_of_train": 0.15,  # 15% train subjects → val

    # Split
    "test_ratio": 0.10,    # 10% subjects held out for test
    "seed": 42,

    # MNE annotation mapping
    "annotation_mapping": {"T0": 1, "T1": 2, "T2": 3},
}


# ═══════════════════════════════════════════════════════════════════
# KROK 1: Podział subjectów — ZANIM dotkniesz danych
# ═══════════════════════════════════════════════════════════════════

def split_subjects(
    all_subject_ids: list[str],
    test_ratio: float,
    val_fraction_of_train: float,
    seed: int,
) -> dict[str, list[str]]:
    """
    Podziel ID subjectów na train / val / test.

    Test subjects są wyciągane PIERWSZE i odłożone na bok.
    Potem train subjects dzielone na train i val.

    Zwraca dict z kluczami 'train', 'val', 'test' — listy subject IDs.
    """
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(all_subject_ids))
    rng.shuffle(ids)

    n_test = max(1, int(len(ids) * test_ratio))
    test_ids = ids[:n_test].tolist()
    trainval_ids = ids[n_test:].tolist()

    # Podziel trainval na train i val
    rng2 = np.random.RandomState(seed + 1)  # inny seed żeby nie korelować
    trainval_arr = np.array(trainval_ids)
    rng2.shuffle(trainval_arr)

    n_val = max(1, int(len(trainval_arr) * val_fraction_of_train))
    val_ids = trainval_arr[:n_val].tolist()
    train_ids = trainval_arr[n_val:].tolist()

    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ═══════════════════════════════════════════════════════════════════
# KROK 2: Preprocessing — OSOBNO per split
# ═══════════════════════════════════════════════════════════════════

def preprocess_subjects(
    raw_data: dict[str, mne.io.Raw],
    subject_ids: list[str],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Przetwórz surowe dane dla KONKRETNYCH subjectów.

    Każdy subject przetwarzany niezależnie:
      1. Pick channels (jeśli motor)
      2. Bandpass filter
      3. Epoching z event_id
      4. Z-score normalizacja PER-SUBJECT PER-CHANNEL
         (statystyki liczone TYLKO na danych tego subjecta)

    Zwraca (X, y, subjects) — niezbalansowane.
    """
    all_X, all_y, all_subj = [], [], []
    skipped = []

    for sid in sorted(subject_ids):
        if sid not in raw_data:
            skipped.append(sid)
            continue
        try:
            raw = raw_data[sid].copy()

            if cfg["channels"] is not None:
                raw.pick(cfg["channels"])

            raw.filter(
                cfg["low_freq"], cfg["high_freq"],
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )

            events, _ = mne.events_from_annotations(
                raw, event_id=cfg["annotation_mapping"], verbose=False
            )

            epochs = mne.Epochs(
                raw, events, cfg["event_id"],
                tmin=cfg["tmin"], tmax=cfg["tmax"],
                baseline=cfg["baseline"],
                preload=True, verbose=False,
            )

            X = epochs.get_data().astype(np.float32)
            y = (epochs.events[:, -1] - cfg["label_offset"]).astype(np.int64)

            # Normalizacja z-score — per-subject, per-channel
            # Statystyki TYLKO z tego subjecta, TYLKO z tego splitu
            for ch in range(X.shape[1]):
                ch_data = X[:, ch, :]
                mean = ch_data.mean()
                std = ch_data.std()
                if std > 0:
                    X[:, ch, :] = (ch_data - mean) / std

            all_X.append(X)
            all_y.append(y)
            all_subj.append(np.full(len(y), int(sid)))

        except Exception as e:
            print(f"  ⚠️ Subject {sid}: {e}")
            skipped.append(sid)

    if not all_X:
        raise RuntimeError(f"No data loaded for subjects: {subject_ids}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subj = np.concatenate(all_subj, axis=0)

    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")

    return X, y, subj


def balance_classes(
    X: np.ndarray, y: np.ndarray, subj: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample do mniejszej klasy."""
    rng = np.random.RandomState(seed)
    counts = np.bincount(y)
    min_count = min(counts)

    idx = []
    for cls in range(len(counts)):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)

    return X[idx], y[idx], subj[idx]


# ═══════════════════════════════════════════════════════════════════
# KROK 3: Trening — BEZ dostępu do test data
# ═══════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epochs: int,
) -> tuple[nn.Module, dict]:
    """
    Trenuj model. Best checkpoint na val_acc.

    UWAGA: ta funkcja NIE przyjmuje test_loader.
    Test jest OSOBNY krok, po zamknięciu treningu.
    """
    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), desc="Training"):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_loss = total_loss / total
        train_acc = correct / total

        if scheduler is not None:
            scheduler.step()

        # ── Val ──
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        # ── Checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train: {train_loss:.4f} / {train_acc:.4f} | "
                  f"val: {val_loss:.4f} / {val_acc:.4f}")

    # Przywróć best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best val_acc: {best_val_acc:.4f}")

    history["best_val_acc"] = best_val_acc
    return model, history


# ═══════════════════════════════════════════════════════════════════
# KROK 4: Ewaluacja — OSOBNA funkcja, OSOBNE dane
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subjects_test: np.ndarray,
    class_names: list[str],
    device: str,
) -> dict:
    """Ewaluacja modelu na danych testowych. Model jest w trybie eval, grady wyłączone."""
    model.eval()
    model.to(device)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.inference_mode():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()

    labels = y_test
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, digits=4)

    # ── Print ──
    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS (subjects NEVER seen during training)")
    print(f"{'='*60}")
    print(f"\n  Accuracy:     {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Cohen kappa:  {kappa:.4f}")
    print(f"  N epochs:     {len(labels)} from {len(np.unique(subjects_test))} subjects")
    print(f"\n  Confusion matrix:")
    print(f"               {'  '.join(f'{n:>10s}' for n in class_names)}")
    for i, name in enumerate(class_names):
        row = '  '.join(f'{cm[i, j]:>10d}' for j in range(len(class_names)))
        print(f"  {name:>10s}  {row}")
    print(f"\n{report}")

    # ── Per-subject ──
    print(f"{'─'*60}")
    print(f"  Per-subject accuracy:")
    print(f"{'─'*60}")
    sub_accs = []
    for sub in sorted(np.unique(subjects_test)):
        mask = subjects_test == sub
        sub_acc = accuracy_score(labels[mask], preds[mask])
        n = mask.sum()
        sub_accs.append(sub_acc)
        marker = "✓" if sub_acc >= 0.7 else ("~" if sub_acc >= 0.55 else "✗")
        print(f"  {marker} Subject {sub:>3d}: {sub_acc:.3f}  ({n} epochs)")

    print(f"\n  Mean per-subject: {np.mean(sub_accs):.4f} ± {np.std(sub_accs):.4f}")
    print(f"  Median:           {np.median(sub_accs):.4f}")
    print(f"  Min / Max:        {np.min(sub_accs):.4f} / {np.max(sub_accs):.4f}")

    return {
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "per_subject_accs": {int(s): float(a) for s, a in zip(sorted(np.unique(subjects_test)), sub_accs)},
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN — orchestracja kroków w poprawnej kolejności
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clean EEGNet train + eval")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--test-ratio", type=float, default=CONFIG["test_ratio"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = {**CONFIG, "seed": args.seed, "epochs": args.epochs,
           "test_ratio": args.test_ratio, "lr": args.lr}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {cfg['seed']}")
    print(f"Split: {100 - cfg['test_ratio']*100:.0f}% train+val / {cfg['test_ratio']*100:.0f}% test")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ══════════════════════════════════════════════════════════════
    # KROK 1: Pobierz dane, zrób split na SUBJECT IDs
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  KROK 1: Ładowanie danych i subject-level split")
    print(f"{'='*60}")

    subjects_data = download_dataset(cfg["dataset"], cfg["desired_runs"])
    raw_data = load_raw_subjects(subjects_data, sfreq=cfg["sfreq"])

    all_subject_ids = sorted(raw_data.keys())
    print(f"  Dostępnych subjectów: {len(all_subject_ids)}")

    split = split_subjects(
        all_subject_ids,
        test_ratio=cfg["test_ratio"],
        val_fraction_of_train=cfg["val_fraction_of_train"],
        seed=cfg["seed"],
    )

    # ── WERYFIKACJA: zero overlap ──
    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])

    assert len(train_set & val_set) == 0,   f"LEAK: train ∩ val = {train_set & val_set}"
    assert len(train_set & test_set) == 0,  f"LEAK: train ∩ test = {train_set & test_set}"
    assert len(val_set & test_set) == 0,    f"LEAK: val ∩ test = {val_set & test_set}"
    assert train_set | val_set | test_set == set(all_subject_ids), "Missing subjects!"

    print(f"  Train subjects: {len(split['train'])}")
    print(f"  Val subjects:   {len(split['val'])}")
    print(f"  Test subjects:  {len(split['test'])}")
    print(f"  Test IDs:       {sorted(split['test'])}")
    print(f"  ✓ Zero overlap between all splits")

    # ══════════════════════════════════════════════════════════════
    # KROK 2: Preprocessing — OSOBNO per split
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  KROK 2: Preprocessing (OSOBNO per split)")
    print(f"{'='*60}")

    print("\n  [train]")
    X_train, y_train, subj_train = preprocess_subjects(raw_data, split["train"], cfg)
    print(f"    Shape: {X_train.shape}, classes: {np.bincount(y_train).tolist()}")

    print("  [val]")
    X_val, y_val, subj_val = preprocess_subjects(raw_data, split["val"], cfg)
    print(f"    Shape: {X_val.shape}, classes: {np.bincount(y_val).tolist()}")

    print("  [test]")
    X_test, y_test, subj_test = preprocess_subjects(raw_data, split["test"], cfg)
    print(f"    Shape: {X_test.shape}, classes: {np.bincount(y_test).tolist()}")

    # ── Balansowanie per-split ──
    print("\n  Balancing per split...")
    X_train, y_train, subj_train = balance_classes(X_train, y_train, subj_train, cfg["seed"])
    X_val, y_val, subj_val = balance_classes(X_val, y_val, subj_val, cfg["seed"])
    X_test, y_test, subj_test = balance_classes(X_test, y_test, subj_test, cfg["seed"])

    print(f"    Train: {X_train.shape}, classes: {np.bincount(y_train).tolist()}")
    print(f"    Val:   {X_val.shape}, classes: {np.bincount(y_val).tolist()}")
    print(f"    Test:  {X_test.shape}, classes: {np.bincount(y_test).tolist()}")

    # ── Sanity: shapes match ──
    n_chans = X_train.shape[1]
    time_points = X_train.shape[2]
    assert X_val.shape[1] == n_chans and X_test.shape[1] == n_chans
    assert X_val.shape[2] == time_points and X_test.shape[2] == time_points
    print(f"    Channels: {n_chans}, Time points: {time_points}")

    # ══════════════════════════════════════════════════════════════
    # KROK 3: Trening (TYLKO train + val)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  KROK 3: Trening EEGNet (test data NIEDOSTĘPNE)")
    print(f"{'='*60}")

    train_loader = DataLoader(
        EEGDataset(X_train, y_train), batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        EEGDataset(X_val, y_val), batch_size=cfg["batch_size"], shuffle=False
    )
    # UWAGA: NIE tworzymy test_loader tutaj. Test jest w kroku 4.

    f2 = cfg["f1"] * cfg["d"]
    model = EEGNet(
        chans=n_chans, classes=cfg["n_classes"], time_points=time_points,
        f1=cfg["f1"], f2=f2, d=cfg["d"],
        temp_kernel=cfg["temp_kernel"],
        pk1=cfg["pk1"], pk2=cfg["pk2"],
        dropout_rate=cfg["dropout_rate"],
    ).to(device)

    print(f"  Model: EEGNet(f1={cfg['f1']}, d={cfg['d']}, f2={f2}, "
          f"temp_kernel={cfg['temp_kernel']}, dropout={cfg['dropout_rate']})")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted loss (na train labels)
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    model, history = train_model(
        model, train_loader, val_loader,
        loss_fn, optimizer, scheduler,
        device, cfg["epochs"],
    )

    # ══════════════════════════════════════════════════════════════
    # KROK 4: Ewaluacja na TEST (pierwszy i jedyny kontakt z test)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  KROK 4: Ewaluacja na TEST (pierwszy kontakt z tymi danymi)")
    print(f"{'='*60}")

    results = evaluate_on_test(
        model, X_test, y_test, subj_test, cfg["class_names"], device
    )

    # ══════════════════════════════════════════════════════════════
    # KROK 5: Zapis
    # ══════════════════════════════════════════════════════════════
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model
    model_path = out_dir / f"clean_eegnet_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)

    # Metadata
    meta = {
        "timestamp": timestamp,
        "config": {k: v for k, v in cfg.items()
                   if k not in ("annotation_mapping", "event_id")},
        "split": {
            "train_subjects": sorted(split["train"]),
            "val_subjects": sorted(split["val"]),
            "test_subjects": sorted(split["test"]),
            "n_train_epochs": int(len(y_train)),
            "n_val_epochs": int(len(y_val)),
            "n_test_epochs": int(len(y_test)),
        },
        "training": {
            "best_val_acc": float(history["best_val_acc"]),
            "final_train_acc": float(history["train_acc"][-1]),
            "final_val_acc": float(history["val_acc"][-1]),
        },
        "test_results": {
            "accuracy": results["accuracy"],
            "kappa": results["kappa"],
            "per_subject_accs": results["per_subject_accs"],
        },
        "architecture": {
            "chans": n_chans,
            "time_points": time_points,
            "f1": cfg["f1"],
            "d": cfg["d"],
            "f2": f2,
            "temp_kernel": cfg["temp_kernel"],
            "dropout_rate": cfg["dropout_rate"],
        },
    }

    meta_path = out_dir / f"clean_eegnet_{timestamp}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  Model saved: {model_path}")
    print(f"  Metadata:    {meta_path}")

    print(f"\n{'='*60}")
    print(f"  GOTOWE — Test accuracy: {results['accuracy']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()