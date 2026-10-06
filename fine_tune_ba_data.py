import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
import numpy as np
import os
import pandas as pd
from src.models.eegnet import EEGNet
from sklearn.metrics import accuracy_score, classification_report

def get_physionet_channels():
    return [
        'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
        'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.',
        'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..',
        'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.',
        'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
    ]

def map_channels(data, original_names, target_names):
    target_data = np.zeros((len(target_names), data.shape[1]))
    norm_original = {name.replace('.', '').upper(): i for i, name in enumerate(original_names)}
    for i, target_name in enumerate(target_names):
        norm_target = target_name.replace('.', '').upper()
        if norm_target in norm_original:
            target_data[i, :] = data[norm_original[norm_target], :]
    return target_data

def get_data_from_folder(folder_path, bp_low=0.5, bp_high=45.0, tmin=1.0, tmax=4.0):
    edf_path = os.path.join(folder_path, "session.edf")
    events_path = os.path.join(folder_path, "events.tsv")
    
    if not os.path.exists(edf_path) or not os.path.exists(events_path):
        return None, None
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(160.0)
    raw.filter(bp_low, bp_high, fir_design='firwin', verbose=False)
    
    physionet_ch = get_physionet_channels()
    mapped_data = map_channels(raw.get_data(), raw.ch_names, physionet_ch)
    
    info = mne.create_info(physionet_ch, sfreq=160.0, ch_types='eeg')
    raw_64 = mne.io.RawArray(mapped_data, info)
    
    events_df = pd.read_csv(events_path, sep='\t')
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    mne_events = []
    for _, row in events_df.iterrows():
        if row['trial_type'] in event_id_mapping:
            sample = int(row['onset'] * 160.0)
            mne_events.append([sample, 0, event_id_mapping[row['trial_type']]])
            
    if not mne_events:
        return None, None
        
    mne_events = np.array(mne_events)
    epochs = mne.Epochs(raw_64, mne_events, event_id=event_id_mapping, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    target_len = 641
    if X.shape[2] > target_len:
        X = X[:, :, :target_len]
    elif X.shape[2] < target_len:
        pad = np.zeros((X.shape[0], X.shape[1], target_len - X.shape[2]))
        X = np.concatenate([X, pad], axis=2)
        
    # Normalization
    for ch in range(X.shape[1]):
        mean = X[:, ch, :].mean()
        std = X[:, ch, :].std()
        if std > 0:
            X[:, ch, :] = (X[:, ch, :] - mean) / std
            
    return X, y

def main():
    device = torch.device("cpu") # Force CPU
    print(f"Using device: {device}")
    
    ba_data_root = "cache/BA_DATA"
    folders = sorted([f for f in os.listdir(ba_data_root) if os.path.isdir(os.path.join(ba_data_root, f)) and not f.startswith('.')])
    
    all_X = []
    all_y = []
    
    print("Loading and preprocessing all data...")
    for folder in folders:
        X, y = get_data_from_folder(os.path.join(ba_data_root, folder))
        if X is not None:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        print("No data found!")
        return

    # Combine all sessions
    X_total = np.concatenate(all_X, axis=0)
    y_total = np.concatenate(all_y, axis=0)
    
    # Shuffle
    indices = np.arange(len(X_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_total = X_total[indices]
    y_total = y_total[indices]
    
    # 70/30 Split
    split_idx = int(0.7 * len(X_total))
    X_train, X_test = X_total[:split_idx], X_total[split_idx:]
    y_train, y_test = y_total[:split_idx], y_total[split_idx:]
    
    print(f"Total samples: {len(X_total)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Convert to Tensors
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # Load model
    model = EEGNet(chans=64, classes=2, time_points=641, f1=16, f2=32, d=2, dropout_rate=0.5)
    model.load_state_dict(torch.load("outputs/final_best.pth", map_location=device))
    model.to(device)
    
    # Fine-tuning strategy: Only tune the final layer first
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting fine-tuning (FC layer only)...")
    epochs = 50
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Evaluation
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
        
        acc = accuracy_score(all_true, all_preds)
        if acc > best_acc:
            best_acc = acc
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Test Acc: {acc:.4f}")

    print(f"\nBest Test Acc during FC tuning: {best_acc:.4f}")

    # Second phase: Unfreeze everything with very low LR
    print("\nStarting fine-tuning (All layers, low LR)...")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    for epoch in range(20):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
        acc = accuracy_score(all_true, all_preds)
        if acc > best_acc:
            best_acc = acc
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+51}/{epochs+20}, Test Acc: {acc:.4f}")

    print(f"\nFinal Best Test Acc: {best_acc:.4f}")
    print("\nFinal Evaluation on Test Split (30%):")
    print(classification_report(all_true, all_preds, target_names=['LEFT', 'RIGHT']))
    
    # Save the fine-tuned model
    save_path = "outputs/ba_data_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuned model saved to {save_path}")

if __name__ == "__main__":
    main()
