import mne
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from mne.decoding import CSP
from sklearn.metrics import accuracy_score

def get_data_from_folder(folder_path, bp_low=8.0, bp_high=30.0):
    edf_path = os.path.join(folder_path, "session.edf")
    events_path = os.path.join(folder_path, "events.tsv")
    
    if not os.path.exists(edf_path) or not os.path.exists(events_path):
        return None, None
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(bp_low, bp_high, fir_design='firwin', verbose=False)
    
    # Use only available 32 channels - no mapping to 64 to avoid adding zeros
    # This is better for CSP
    
    events_df = pd.read_csv(events_path, sep='\t')
    event_id_mapping = {'LEFT_HAND_CLENCH': 0, 'RIGHT_HAND_CLENCH': 1}
    
    mne_events = []
    for _, row in events_df.iterrows():
        if row['trial_type'] in event_id_mapping:
            sample = int(row['onset'] * raw.info['sfreq'])
            mne_events.append([sample, 0, event_id_mapping[row['trial_type']]])
            
    if not mne_events:
        return None, None
        
    mne_events = np.array(mne_events)
    # 0.5 to 3.5s window is usually best for CSP features
    epochs = mne.Epochs(raw, mne_events, event_id=event_id_mapping, tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
            
    return X, y

def main():
    ba_data_root = "cache/BA_DATA"
    folders = sorted([f for f in os.listdir(ba_data_root) if os.path.isdir(os.path.join(ba_data_root, f)) and not f.startswith('.')])
    
    all_X, all_y, all_groups = [], [], []
    
    print("Loading data for CSP+LDA (32 channels)...")
    for i, folder in enumerate(folders):
        X, y = get_data_from_folder(os.path.join(ba_data_root, folder))
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            all_groups.append(np.full(len(y), i))
            
    X_total = np.concatenate(all_X)
    y_total = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    
    print(f"Total samples: {len(X_total)}")
    
    # CSP + LDA Pipeline
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(X_total, y_total, groups):
        train_X, test_X = X_total[train_idx], X_total[test_idx]
        train_y, test_y = y_total[train_idx], y_total[test_idx]
        
        clf.fit(train_X, train_y)
        acc = accuracy_score(test_y, clf.predict(test_X))
        print(f"Session {groups[test_idx][0]} Holdout Accuracy: {acc:.4f}")
        scores.append(acc)
        
    print(f"\nFinal CSP+LDA Mean Accuracy: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
