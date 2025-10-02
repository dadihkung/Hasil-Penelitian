import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from sklearn.cross_decomposition import CCA
from scipy.integrate import trapezoid

# === Configuration ===
frequencies = {
    'landing': 7.5,
    'terbang': 5.5,
    'kanan': 11.5,
    'kiri': 9.5,
    'depan': 13.5,
    'belakang': 15.5
}
n_harmonics = 2
sfreq = 250              # Sampling rate
active_duration = 7      # seconds
n_samples = sfreq * active_duration
data_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands'

# === Bandpass Filter ===
def bandpass_filter(data, lowcut=5, highcut=25, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# === Generate Reference Signals for CCA ===
def generate_reference_signals(freq, duration, sfreq, n_harmonics):
    t = np.arange(0, duration, 1/sfreq)
    ref_signals = []
    for h in range(1, n_harmonics + 1):
        ref_signals.append(np.sin(2 * np.pi * freq * h * t))
        ref_signals.append(np.cos(2 * np.pi * freq * h * t))
    return np.array(ref_signals)

# === CCA Feature Extraction ===
def extract_all_cca_features(eeg_data, duration, sfreq, n_harmonics, frequencies):
    feature_dict = {}
    for label, freq in frequencies.items():
        ref = generate_reference_signals(freq, duration, sfreq, n_harmonics)
        cca = CCA(n_components=1)
        cca.fit(eeg_data, ref.T)
        X_c, Y_c = cca.transform(eeg_data, ref.T)
        corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
        feature_dict[f'corr_{label}'] = corr
    return feature_dict

# === Spectral Bandpower Features ===
def extract_bandpower_features(eeg_data, sfreq):
    bandpower_dict = {}
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30)
    }
    for ch in range(eeg_data.shape[1]):
        f, Pxx = welch(eeg_data[:, ch], sfreq, nperseg=sfreq)
        for band, (low, high) in bands.items():
            idx_band = np.logical_and(f >= low, f <= high)
            bp = np.trapz(Pxx[idx_band], f[idx_band])  # area under curve
            bandpower_dict[f'{band}_ch{ch}'] = bp
    return bandpower_dict

# === Main Processing ===
features = []
for command, freq in frequencies.items():
    file_path = os.path.join(data_dir, f'{command}_all_subjects.csv')
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    grouped = df.groupby(['subject', 'loop'])

    for (subject, loop), group in grouped:
        eeg_data = group[['EXG0', 'EXG1']].values

        # ✅ Pad or trim data
        if len(eeg_data) < n_samples:
            pad_length = n_samples - len(eeg_data)
            eeg_data = np.pad(eeg_data, ((0, pad_length), (0, 0)), mode='constant')
        else:
            eeg_data = eeg_data[:n_samples]

        # ✅ Bandpass filtering
        eeg_data = bandpass_filter(eeg_data, fs=sfreq)

        # ✅ Extract CCA features
        cca_features = extract_all_cca_features(
            eeg_data=eeg_data,
            duration=active_duration,
            sfreq=sfreq,
            n_harmonics=n_harmonics,
            frequencies=frequencies
        )

        # ✅ Extract bandpower features
        bandpower_features = extract_bandpower_features(eeg_data, sfreq)

        # ✅ Merge all features
        feature_row = {**cca_features, **bandpower_features}
        feature_row.update({
            'subject': subject,
            'loop': loop,
            'command': command
        })

        features.append(feature_row)

# === Save features to CSV ===
features_df = pd.DataFrame(features)
output_path = os.path.join(data_dir, 'cca_bandpower_features.csv')
features_df.to_csv(output_path, index=False)
print(f"\n✅ Features saved to: {output_path}")
