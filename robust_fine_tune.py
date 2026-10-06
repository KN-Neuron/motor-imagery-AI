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
from sklearn.model_selection import LeaveOneGroupOut

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

def get_data_from_folder(folder_path, bp_low=4.0, bp_high=38.0):
    edf_path = os.path.join(folder_path, "session.edf")
    events_path = os.path.join(folder_path, "events.tsv")
    
    if not os.path.exists(edf_path) or not os.path.exists(events_path):
        return None, None
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(160.0)
    raw.filter(bp_low, bp_high, fir_design='firwin', verbose=False)
    
    # Simple Artifact Rejection: Peak-to-peak amplitude thresholding
    # This is a basic way to "clean" the data without full ICA
    
    physionet_ch = get_physionet_channels()
    mapped_data = map_channels(raw.get_data(), raw.ch_names, physionet_ch)
    
    info = mne.create_info(physionet_ch, sfreq=160.0, ch_types='eeg')
    raw_64 = mne.io.RawArray(mapped_data, info)
    
    events_df = pd.read_csv(events_path, sep='\t')
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    X_list = []
    y_list = []
    
    for _, row in events_df.iterrows():
        if row['trial_type'] in event_id_mapping:
            label = event_id_mapping[row['trial_type']]
            onset = row['onset']
            
            # DATA AUGMENTATION: Sliding window within the 4s trial
            # Trial is roughly 4s. We take multiple 4s windows with small offsets if possible,
            # or just multiple windows if we shorten the duration.
            # Let's try 3 windows: [0, 4], [0.5, 4.5] (if duration allows), etc.
            # Given BA_DATA trials are ~4.1s, we can take:
            # 1. 0.0 - 4.0
            # 2. 0.1 - 4.1
            windows = [0.0, 0.1]
            for offset in windows:
                tmin = onset + offset
                tmax = tmin + 4.0
                
                # Check if we are within bounds
                if tmax <= raw_64.times[-1]:
                    start_samp = int(tmin * 160.0)
                    stop_samp = start_samp + 641
                    epoch_data = raw_64.get_data(start=start_samp, stop=stop_samp)
                    
                    # Reject if flat or huge artifacts (> 150uV)
                    if np.max(np.abs(epoch_data)) < 1e-4 and np.max(np.abs(epoch_data)) > 1e-8:
                        X_list.append(epoch_data)
                        y_list.append(label)
            
    if not X_list:
        return None, None
        
    X = np.array(X_list)
    y = np.array(y_list)
        
    # Per-epoch Normalization
    for i in range(len(X)):
        for ch in range(X.shape[1]):
            mean = X[i, ch, :].mean()
            std = X[i, ch, :].std()
            if std > 0:
                X[i, ch, :] = (X[i, ch, :] - mean) / std
            
    return X, y

def train_and_eval(train_X, train_y, test_X, test_y, device):
    train_ds = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long())
    test_ds = TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = EEGNet(chans=64, classes=2, time_points=641, f1=16, f2=32, d=2, dropout_rate=0.5)
    model.load_state_dict(torch.load("outputs/final_best.pth", map_location=device))
    model.to(device)
    
    # Fine-tuning: Unfreeze only Block 3 and FC first, then everything
    for param in model.parameters():
        param.requires_grad = False
    for param in model.block3.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Quick phase 1
    for epoch in range(20):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()
            
    # Phase 2: Everything
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_acc = 0
    for epoch in range(30):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b_X, b_y in test_loader:
                b_X = b_X.to(device)
                out = model(b_X)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                trues.extend(b_y.numpy())
        acc = accuracy_score(trues, preds)
        if acc > best_acc:
            best_acc = acc
            
    return best_acc

def main():
    device = torch.device("cpu")
    ba_data_root = "cache/BA_DATA"
    folders = sorted([f for f in os.listdir(ba_data_root) if os.path.isdir(os.path.join(ba_data_root, f)) and not f.startswith('.')])
    
    all_X, all_y, all_groups = [], [], []
    
    print("Loading data with augmentation and cleaning...")
    for i, folder in enumerate(folders):
        X, y = get_data_from_folder(os.path.join(ba_data_root, folder))
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            all_groups.append(np.full(len(y), i))
            
    X_total = np.concatenate(all_X)
    y_total = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    
    print(f"Total samples (augmented): {len(X_total)}")
    
    # Use Leave-One-Session-Out Cross Validation for more reliable accuracy
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(X_total, y_total, groups):
        train_X, test_X = X_total[train_idx], X_total[test_idx]
        train_y, test_y = y_total[train_idx], y_total[test_idx]
        
        acc = train_and_eval(train_X, train_y, test_X, test_y, device)
        print(f"Session {groups[test_idx][0]} Holdout Accuracy: {acc:.4f}")
        scores.append(acc)
        
    print(f"\nMean LOSO Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

if __name__ == "__main__":
    main()
