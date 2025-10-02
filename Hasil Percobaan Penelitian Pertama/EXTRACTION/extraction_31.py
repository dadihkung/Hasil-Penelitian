import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def extract_features_windowed(df, sfreq, window_sec=1, 
                              use_stat=True, use_band=False, use_hjorth=False, use_cca=False, 
                              target_freqs=None):
    window_size = int(sfreq * window_sec)
    features = []

    for i in range(0, len(df) - window_size + 1, window_size):
        win = df.iloc[i:i+window_size]
        row = {
            'subject': win['subject'].iloc[0],
            'loop': win['loop'].iloc[0],
            'start_time': win['time'].iloc[0],
            'end_time': win['time'].iloc[-1],
        }

        for ch in ['EXG0', 'EXG1']:
            sig = win[ch].values

            if use_stat:
                row[f'{ch}_mean'] = np.mean(sig)
                row[f'{ch}_std'] = np.std(sig)
                row[f'{ch}_var'] = np.var(sig)
                row[f'{ch}_skew'] = skew(sig)
                row[f'{ch}_kurtosis'] = kurtosis(sig)

            if use_cca:
                # Placeholder untuk CCA – isi sesuai kebutuhan
                # Contoh: row[f'{ch}_cca_score'] = <hasil perhitungan CCA>
                row[f'{ch}_cca_dummy'] = np.random.rand()  # Contoh dummy
                # Ganti di sini dengan kode CCA sebenarnya

        features.append(row)

    return pd.DataFrame(features)

# === CONFIG ===
sfreq = 250  # Sampling rate
input_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands'
output_dir = r'C:\laragon\www\skripsi\EEG FEATURES\new2'
os.makedirs(output_dir, exist_ok=True)

# === PROCESS EACH COMMAND FILE ===
all_features_stat = []
all_features_cca = []

for file in os.listdir(input_dir):
    if not file.endswith('.csv'):
        continue

    command_label = file.split('_')[0]  # Extract command label from filename
    file_path = os.path.join(input_dir, file)
    print(f"Extracting from {file_path}...")

    df = pd.read_csv(file_path)

    # Group by subject
    for subject_id, df_subj in df.groupby('subject'):
        # === STAT FEATURES ===
        feats_stat = extract_features_windowed(df_subj, sfreq=sfreq, window_sec=1,
                                               use_stat=True, use_band=False, use_hjorth=False, use_cca=False)
        feats_stat['label'] = command_label
        feats_stat['subject'] = subject_id
        all_features_stat.append(feats_stat)

        stat_file = os.path.join(output_dir, f"{command_label}_subject_{subject_id}_features_stat.csv")
        feats_stat.to_csv(stat_file, index=False)
        print(f"✅ Saved STAT: {command_label} | Subject {subject_id} → {stat_file}")

        # === CCA FEATURES ===
        feats_cca = extract_features_windowed(df_subj, sfreq=sfreq, window_sec=1,
                                              use_stat=False, use_band=False, use_hjorth=False, use_cca=True)
        feats_cca['label'] = command_label
        feats_cca['subject'] = subject_id
        all_features_cca.append(feats_cca)

        cca_file = os.path.join(output_dir, f"{command_label}_subject_{subject_id}_features_cca.csv")
        feats_cca.to_csv(cca_file, index=False)
        print(f"✅ Saved CCA: {command_label} | Subject {subject_id} → {cca_file}")

# === Save Combined Datasets ===
combined_stat = pd.concat(all_features_stat, ignore_index=True)
combined_stat_file = os.path.join(output_dir, "features_stat.csv")
combined_stat.to_csv(combined_stat_file, index=False)
print(f"✅ Combined STAT features saved to {combined_stat_file}")

combined_cca = pd.concat(all_features_cca, ignore_index=True)
combined_cca_file = os.path.join(output_dir, "features_cca.csv")
combined_cca.to_csv(combined_cca_file, index=False)
print(f"✅ Combined CCA features saved to {combined_cca_file}")
