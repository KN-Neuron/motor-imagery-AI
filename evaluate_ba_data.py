import torch
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
    """
    Map original channels to target 64 channels.
    """
    target_data = np.zeros((len(target_names), data.shape[1]))
    
    # Create normalized mapping (strip dots and uppercase)
    norm_original = {name.replace('.', '').upper(): i for i, name in enumerate(original_names)}
    
    found = 0
    found_names = []
    for i, target_name in enumerate(target_names):
        norm_target = target_name.replace('.', '').upper()
        if norm_target in norm_original:
            target_data[i, :] = data[norm_original[norm_target], :]
            found += 1
            found_names.append(target_name)
            
    print(f"  Mapped {found}/{len(target_names)} channels: {', '.join(found_names[:5])}...")
    missing_in_original = [name for name in original_names if name.replace('.', '').upper() not in [tn.replace('.', '').upper() for tn in target_names]]
    if missing_in_original:
        print(f"  Channels in original not in target: {', '.join(missing_in_original)}")
    return target_data

def evaluate_folder(folder_path, model, device, bp_low=0.5, bp_high=45.0):
    edf_path = os.path.join(folder_path, "session.edf")
    events_path = os.path.join(folder_path, "events.tsv")
    
    if not os.path.exists(edf_path) or not os.path.exists(events_path):
        print(f"  Missing files in {folder_path}")
        return None, None
    
    # 1. Load data
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # 2. Preprocess
    raw.resample(160.0)
    raw.filter(bp_low, bp_high, fir_design='firwin', verbose=False)
    
    # 3. Channel Mapping
    physionet_ch = get_physionet_channels()
    data = raw.get_data()
    mapped_data = map_channels(data, raw.ch_names, physionet_ch)
    
    # Create a new raw object with 64 channels
    info = mne.create_info(physionet_ch, sfreq=160.0, ch_types='eeg')
    raw_64 = mne.io.RawArray(mapped_data, info)
    
    # 4. Epoching
    events_df = pd.read_csv(events_path, sep='\t')
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    mne_events = []
    for _, row in events_df.iterrows():
        if row['trial_type'] in event_id_mapping:
            sample = int(row['onset'] * 160.0)
            mne_events.append([sample, 0, event_id_mapping[row['trial_type']]])
            
    if not mne_events:
        print(f"  No relevant events in {folder_path}")
        return None, None
        
    mne_events = np.array(mne_events)
    epochs = mne.Epochs(raw_64, mne_events, event_id=event_id_mapping, tmin=0.0, tmax=4.0, baseline=None, preload=True, verbose=False)
    
    X = epochs.get_data() # (N, 64, T)
    y = epochs.events[:, -1]
    
    # Ensure T=641
    if X.shape[2] > 641:
        X = X[:, :, :641]
    elif X.shape[2] < 641:
        pad = np.zeros((X.shape[0], X.shape[1], 641 - X.shape[2]))
        X = np.concatenate([X, pad], axis=2)
        
    # 5. Normalization (per-subject channel-wise z-score)
    for ch in range(X.shape[1]):
        mean = X[:, ch, :].mean()
        std = X[:, ch, :].std()
        if std > 0:
            X[:, ch, :] = (X[:, ch, :] - mean) / std
            
    print(f"  Data range after norm: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}, std={X.std():.2f}")
            
    # 6. Inference
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    return y, preds

def main():
    # Force CPU because of CUDA compatibility issues in this environment
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = EEGNet(chans=64, classes=2, time_points=641, f1=16, f2=32, d=2, dropout_rate=0.5)
    model.load_state_dict(torch.load("outputs/final_best.pth", map_location=device))
    model.to(device)
    model.eval()
    
    ba_data_root = "cache/BA_DATA"
    folders = [f for f in os.listdir(ba_data_root) if os.path.isdir(os.path.join(ba_data_root, f)) and not f.startswith('.')]
    
    # Parameter adjustment: try a few bandpass filters and time windows
    bp_configs = [
        (0.5, 45.0), 
        (7.0, 30.0)
    ]
    time_windows = [
        (0.0, 4.0),
        (0.5, 3.5),
        (1.0, 4.0)
    ]
    
    for low, high in bp_configs:
        for tmin, tmax in time_windows:
            print(f"\n--- Bandpass: {low}-{high} Hz, Window: {tmin}-{tmax}s ---")
            all_y = []
            all_preds = []
            
            for folder in sorted(folders):
                # We need to modify evaluate_folder to accept tmin, tmax
                y, preds = evaluate_folder_ext(os.path.join(ba_data_root, folder), model, device, low, high, tmin, tmax)
                if y is not None:
                    all_y.extend(y)
                    all_preds.extend(preds)
            
            if all_y:
                total_acc = accuracy_score(all_y, all_preds)
                print(f"OVERALL ACCURACY: {total_acc:.4f}")

def evaluate_folder_ext(folder_path, model, device, bp_low, bp_high, tmin, tmax):
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
    
    # Target length is 641 (4s at 160Hz)
    target_len = 641
    if X.shape[2] > target_len:
        X = X[:, :, :target_len]
    elif X.shape[2] < target_len:
        pad = np.zeros((X.shape[0], X.shape[1], target_len - X.shape[2]))
        X = np.concatenate([X, pad], axis=2)
        
    for ch in range(X.shape[1]):
        mean = X[:, ch, :].mean()
        std = X[:, ch, :].std()
        if std > 0:
            X[:, ch, :] = (X[:, ch, :] - mean) / std
            
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    return y, preds

if __name__ == "__main__":
    main()
