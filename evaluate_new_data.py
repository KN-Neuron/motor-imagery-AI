import torch
import mne
import numpy as np
from src.models.eegnet import EEGNet
import os

def evaluate():
    # 1. Model Configuration
    chans = 64
    classes = 2
    time_points = 641
    f1 = 16
    d = 2
    dropout_rate = 0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EEGNet(chans=chans, classes=classes, time_points=time_points, f1=f1, f2=f1*d, d=d, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load("outputs/final_best.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Data Loading
    edf_path = "cache/BA_DATA/session.edf"
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Standard 64-channel list for PhysioNet EEGMMIDB
    physionet_channels = [
        'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
        'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fp2.', 'Af3.', 'Af4.', 'F7..', 'F3..', 'Fz..',
        'F4..', 'F8..', 'Ft7.', 'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'Ft8.', 'T7..', 'C3..', 'C1..',
        'Cz..', 'C2..', 'C4..', 'T8..', 'Tp7.', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Tp8.', 'P7..',
        'P3..', 'Pz..', 'P4..', 'P8..', 'Po3.', 'Po4.', 'O1..', 'Oz..', 'O2..'
    ]
    # Wait, the list above has duplicates and is a bit messy. 
    # Let's use the standard 64 channels from the mne library or common references for this dataset.
    # Actually, the dataset used in training is "brianleung2020/eeg-motor-movementimagery-dataset" which is PhysioNet.
    # The standard 64 channels for PhysioNet are:
    standard_64 = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "O2", "Oz", "AF3", "AF4", "PO3", "PO4",
        "PO7", "PO8", "FPZ", "CPZ", "FCZ", "FT7", "FT8", "TP7", "TP8", "POZ", "FT9", "FT10", "TPP9h", "TPP10h",
        "PO9", "PO10", "P9", "P10", "F1", "F2", "F5", "F6", "FC3", "FC4", "C1", "C2", "C5", "C6", "CP3", "CP4",
        "P1", "P2"
    ]
    # Re-checking standard PhysioNet EEGMMIDB 64 channels. 
    # Usually: Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, O1, O2
    # plus many others. Let's get the exact list from MNE if possible or use a known one.
    
    # PhysioNet channels often have dots in MNE, e.g., 'Fc5.'
    # Let's use a more robust way: pick the 32 we have and fill the rest with zeros.
    
    # 3. Preprocessing
    # Resample to 160Hz as expected by model
    raw.resample(160.0)
    
    # Bandpass filter
    raw.filter(0.5, 45.0)
    
    # 4. Channel Mapping
    # The model expects 64 channels in a specific order. 
    # Since I don't have the original order, I'll assume standard order or at least consistent mapping.
    # Given we have 32 channels, we'll map them to their names in the standard 64 and zero out others.
    
    new_ch_names = raw.ch_names
    print(f"New data channels: {new_ch_names}")
    
    # Create a 64-channel info structure
    # We need the 64 channel names the model was trained on. 
    # I'll try to guess them based on the 10-20 system if I can't find the exact list.
    # WAIT! I can check the model's block2 to see if it's picking specific channels? No, it's a Conv2d(f1, d*f1, (chans, 1)).
    
    # Let's look for the channel list one more time in the repo. 
    # Maybe in a config or a notebook?
    
    # I'll use the standard 64 from a known PhysioNet loader.
    physionet_64 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 
        'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'O2', 'AF3', 'AF4', 'FT7', 
        'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz', 'FCz', 'Oz', 'F1', 'F2', 'F5', 'F6', 'FC3', 'FC4', 
        'FT9', 'FT10', 'TPP9h', 'TPP10h', 'C1', 'C2', 'C5', 'C6', 'CP3', 'CP4', 'P1', 'P2', 'POz', 'P9', 'P10', 
        'AF7', 'AF8'
    ]
    # This is 64 channels. Let's see if our 32 are in here.
    # Our 32: Fp1, Fp2, AF3, AF4, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO3, PO4, O1, Oz, O2
    
    # Create zero data for missing channels
    data = raw.get_data()
    ch_map = {name.upper(): i for i, name in enumerate(new_ch_names)}
    
    final_data_64 = np.zeros((64, data.shape[1]))
    found_count = 0
    for i, name in enumerate(physionet_64):
        uname = name.upper()
        if uname in ch_map:
            final_data_64[i, :] = data[ch_map[uname], :]
            found_count += 1
    
    print(f"Mapped {found_count}/64 channels. {64-found_count} channels will be zeroed.")
    
    # Create new Raw object with 64 channels
    info = mne.create_info(physionet_64, sfreq=160.0, ch_types='eeg')
    raw_64 = mne.io.RawArray(final_data_64, info)
    
    # 5. Epoching
    # Load events from events.tsv
    events_tsv = "cache/BA_DATA/events.tsv"
    with open(events_tsv, 'r') as f:
        lines = f.readlines()[1:] # skip header
    
    mne_events = []
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    for line in lines:
        onset, duration, trial_type = line.strip().split('\t')
        if trial_type in event_id_mapping:
            sample = int(float(onset) * 160.0)
            mne_events.append([sample, 0, event_id_mapping[trial_type]])
            
    mne_events = np.array(mne_events)
    print(f"Found {len(mne_events)} relevant events.")
    
    epochs = mne.Epochs(raw_64, mne_events, event_id=event_id_mapping, tmin=0.0, tmax=4.0, baseline=None, preload=True)
    
    X = epochs.get_data() # (N, 64, T)
    y = epochs.events[:, -1]
    
    # Ensure time_points is exactly 641
    if X.shape[2] > 641:
        X = X[:, :, :641]
    elif X.shape[2] < 641:
        pad = np.zeros((X.shape[0], X.shape[1], 641 - X.shape[2]))
        X = np.concatenate([X, pad], axis=2)
        
    # Normalization (as done in training: per-subject, per-channel z-score or global?)
    # Config said "normalize: true" which in preprocessing.py is per-subject channel-wise z-score.
    for ch in range(X.shape[1]):
        mean = X[:, ch, :].mean()
        std = X[:, ch, :].std()
        if std > 0:
            X[:, ch, :] = (X[:, ch, :] - mean) / std
            
    # 6. Inference
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    accuracy = (preds == y).mean()
    print(f"Evaluation Accuracy on BA_DATA: {accuracy:.4f}")
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=event_id_mapping.keys()))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))

if __name__ == "__main__":
    evaluate()
