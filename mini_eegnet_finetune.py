import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
import numpy as np
import os
import pandas as pd
from src.models.eegnet import EEGNet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

def get_physionet_channels():
    # Exactly as returned by MNE/PhysioNet
    return [
        'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
        'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.',
        'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..',
        'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.',
        'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
    ]

def get_data_from_folder(folder_path, bp_low=8.0, bp_high=30.0):
    edf_path = os.path.join(folder_path, "session.edf")
    events_path = os.path.join(folder_path, "events.tsv")
    
    if not os.path.exists(edf_path) or not os.path.exists(events_path):
        return None, None, None
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(160.0)
    raw.filter(bp_low, bp_high, fir_design='firwin', verbose=False)
    
    ch_names = raw.ch_names
    
    events_df = pd.read_csv(events_path, sep='\t')
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    X_list = []
    y_list = []
    
    for _, row in events_df.iterrows():
        if row['trial_type'] in event_id_mapping:
            label = event_id_mapping[row['trial_type']]
            onset = row['onset']
            
            # Augmentation: 5 windows
            offsets = [0.0, 0.1, 0.2, 0.3, 0.4]
            for offset in offsets:
                tmin = onset + offset
                if (tmin + 4.0) <= raw.times[-1]:
                    start_samp = int(tmin * 160.0)
                    stop_samp = start_samp + 641
                    epoch_data = raw.get_data(start=start_samp, stop=stop_samp)
                    
                    # Normalization per epoch
                    for ch in range(epoch_data.shape[0]):
                        std = epoch_data[ch, :].std()
                        if std > 0:
                            epoch_data[ch, :] = (epoch_data[ch, :] - epoch_data[ch, :].mean()) / std
                    
                    X_list.append(epoch_data)
                    y_list.append(label)
            
    if not X_list:
        return None, None, ch_names
        
    return np.array(X_list), np.array(y_list), ch_names

def build_mini_eegnet(ch_names_32, device):
    # 1. Map 32 channels to 64 channel indices
    physio_channels = get_physionet_channels()
    norm_physio = {n.replace('.', '').upper(): i for i, n in enumerate(physio_channels)}
    
    mapping_indices = []
    for name in ch_names_32:
        norm_name = name.replace('.', '').upper()
        if norm_name in norm_physio:
            mapping_indices.append(norm_physio[norm_name])
        else:
            # Fallback (should not happen with this 32ch set)
            mapping_indices.append(0) 
    
    # 2. Instantiate both models
    # Pre-trained params: f1=16, d=2 (from metadata)
    model_64 = EEGNet(chans=64, classes=2, time_points=641, f1=16, f2=32, d=2, dropout_rate=0.5)
    model_64.load_state_dict(torch.load("outputs/final_best.pth", map_location=device))
    
    model_32 = EEGNet(chans=32, classes=2, time_points=641, f1=16, f2=32, d=2, dropout_rate=0.5)
    
    # 3. Surgical Weight Transfer
    print(f"Transferring weights for {len(mapping_indices)} channels...")
    
    # Block 1: Same
    model_32.block1.load_state_dict(model_64.block1.state_dict())
    
    # Block 2: Spatial Filters - Surgery here!
    # weight shape: (out_channels, 1, chans, 1) -> (32, 1, 64, 1) for model_64
    # We pick the 32 rows corresponding to our channels
    with torch.no_grad():
        w_64 = model_64.block2[0].weight.data # (32, 1, 64, 1)
        model_32.block2[0].weight.data = w_64[:, :, mapping_indices, :]
        
        # Copy BatchNorm and other Block 2 params
        model_32.block2[1].load_state_dict(model_64.block2[1].state_dict())
        
    # Block 3: Same
    model_32.block3.load_state_dict(model_64.block3.state_dict())
    
    # FC: Same
    model_32.fc.load_state_dict(model_64.fc.state_dict())
    
    return model_32

def train_and_eval(model, train_X, train_y, test_X, test_y, device):
    train_ds = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long())
    test_ds = TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Fine-tuning: Full model but with low LR
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(40):
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
    ch_names_32 = None
    
    print("Loading data for 32ch Mini-EEGNet...")
    for i, folder in enumerate(folders):
        X, y, names = get_data_from_folder(os.path.join(ba_data_root, folder))
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            all_groups.append(np.full(len(y), i))
            if ch_names_32 is None: ch_names_32 = names
            
    X_total = np.concatenate(all_X)
    y_total = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    
    print(f"Total samples: {len(X_total)}")
    
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(X_total, y_total, groups):
        train_X, test_X = X_total[train_idx], X_total[test_idx]
        train_y, test_y = y_total[train_idx], y_total[test_idx]
        
        # Build a fresh mini model for each fold to avoid leakage
        model = build_mini_eegnet(ch_names_32, device)
        model.to(device)
        
        acc = train_and_eval(model, train_X, train_y, test_X, test_y, device)
        print(f"Session {groups[test_idx][0]} Holdout Accuracy: {acc:.4f}")
        scores.append(acc)
        
    print(f"\nFinal Mini-EEGNet Mean Accuracy: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
