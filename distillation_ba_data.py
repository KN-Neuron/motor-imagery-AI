import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return [
        'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
        'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.',
        'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..',
        'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.',
        'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
    ]

def map_channels_to_64(data, original_names):
    target_names = get_physionet_channels()
    target_data = np.zeros((len(target_names), data.shape[1]))
    norm_original = {name.replace('.', '').upper(): i for i, name in enumerate(original_names)}
    for i, target_name in enumerate(target_names):
        norm_target = target_name.replace('.', '').upper()
        if norm_target in norm_original:
            target_data[i, :] = data[norm_original[norm_target], :]
    return target_data

def get_data_for_distillation(folder_path, bp_low=8.0, bp_high=30.0):
    edf_path = os.path.join(folder_path, "session.edf")
    events_path = os.path.join(folder_path, "events.tsv")
    if not os.path.exists(edf_path) or not os.path.exists(events_path):
        return None, None, None, None
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(160.0)
    raw.filter(bp_low, bp_high, fir_design='firwin', verbose=False)
    
    events_df = pd.read_csv(events_path, sep='\t')
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    X32_list, X64_list, y_list = [], [], []
    
    for _, row in events_df.iterrows():
        if row['trial_type'] in event_id_mapping:
            label = event_id_mapping[row['trial_type']]
            onset = row['onset']
            offsets = [0.0, 0.1, 0.2, 0.3, 0.4]
            for offset in offsets:
                tmin = onset + offset
                if (tmin + 4.0) <= raw.times[-1]:
                    start_samp = int(tmin * 160.0)
                    stop_samp = start_samp + 641
                    data_32 = raw.get_data(start=start_samp, stop=stop_samp)
                    data_64 = map_channels_to_64(data_32, raw.ch_names)
                    
                    # Norm
                    for ch in range(data_32.shape[0]):
                        s = data_32[ch, :].std()
                        if s > 0: data_32[ch, :] = (data_32[ch, :] - data_32[ch, :].mean()) / s
                    for ch in range(data_64.shape[0]):
                        s = data_64[ch, :].std()
                        if s > 0: data_64[ch, :] = (data_64[ch, :] - data_64[ch, :].mean()) / s
                        
                    X32_list.append(data_32)
                    X64_list.append(data_64)
                    y_list.append(label)
                    
    return np.array(X32_list), np.array(X64_list), np.array(y_list), raw.ch_names

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # Soft loss (KL Divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    # Hard loss (Cross Entropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss

def train_student(student, teacher, train_loader, test_loader, device, epochs=50):
    optimizer = optim.Adam(student.parameters(), lr=0.0001)
    teacher.eval()
    
    best_acc = 0
    for epoch in range(epochs):
        student.train()
        for b32, b64, blabels in train_loader:
            b32, b64, blabels = b32.to(device), b64.to(device), blabels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(b64)
            student_logits = student(b32)
            loss = distillation_loss(student_logits, teacher_logits, blabels)
            loss.backward()
            optimizer.step()
            
        student.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b32, b64, blabels in test_loader:
                out = student(b32.to(device))
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                trues.extend(blabels.numpy())
        acc = accuracy_score(trues, preds)
        if acc > best_acc: best_acc = acc
    return best_acc

def main():
    device = torch.device("cpu")
    ba_data_root = "cache/BA_DATA"
    folders = sorted([f for f in os.listdir(ba_data_root) if os.path.isdir(os.path.join(ba_data_root, f)) and not f.startswith('.')])
    
    all_X32, all_X64, all_y, all_groups = [], [], [], []
    for i, folder in enumerate(folders):
        X32, X64, y, _ = get_data_for_distillation(os.path.join(ba_data_root, folder))
        if X32 is not None:
            all_X32.append(X32); all_X64.append(X64); all_y.append(y)
            all_groups.append(np.full(len(y), i))
            
    X32_total = np.concatenate(all_X32); X64_total = np.concatenate(all_X64)
    y_total = np.concatenate(all_y); groups = np.concatenate(all_groups)
    
    # Teacher (Pre-trained 64ch)
    teacher = EEGNet(chans=64, classes=2, time_points=641, f1=16, f2=32, d=2)
    teacher.load_state_dict(torch.load("outputs/final_best.pth", map_location=device))
    teacher.to(device)
    
    logo = LeaveOneGroupOut()
    scores = []
    for train_idx, test_idx in logo.split(X32_total, y_total, groups):
        # Student (Fresh 32ch)
        student = EEGNet(chans=32, classes=2, time_points=641, f1=16, f2=32, d=2)
        student.to(device)
        
        train_ds = TensorDataset(torch.from_numpy(X32_total[train_idx]).float(), 
                                 torch.from_numpy(X64_total[train_idx]).float(), 
                                 torch.from_numpy(y_total[train_idx]).long())
        test_ds = TensorDataset(torch.from_numpy(X32_total[test_idx]).float(), 
                                torch.from_numpy(X64_total[test_idx]).float(), 
                                torch.from_numpy(y_total[test_idx]).long())
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        
        acc = train_student(student, teacher, train_loader, test_loader, device)
        print(f"Fold Accuracy: {acc:.4f}")
        scores.append(acc)
    
    print(f"\nFinal Distillation Mean Accuracy: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
