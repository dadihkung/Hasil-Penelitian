import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.cross_decomposition import CCA

def hjorth_parameters(sig):
    first_deriv = np.diff(sig)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(sig)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    
    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 else 0
    
    return activity, mobility, complexity

def bandpower(sig, sf, band):
    freqs, psd = welch(sig, sf, nperseg=sf*2)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx_band])

def generate_reference_signals(freqs, sf, n_samples):
    t = np.arange(n_samples) / sf
    refs = []
    for f in freqs:
        refs.append(np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=1))
    return refs

def extract_features_windowed(df, sfreq, window_sec=1, 
                              use_stat=True, use_band=True, use_hjorth=True, use_cca=False, 
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

            if use_hjorth:
                act, mob, comp = hjorth_parameters(sig)
                row[f'{ch}_hjorth_activity'] = act
                row[f'{ch}_hjorth_mobility'] = mob
                row[f'{ch}_hjorth_complexity'] = comp

            if use_band:
                row[f'{ch}_delta'] = bandpower(sig, sfreq, (1, 4))
                row[f'{ch}_theta'] = bandpower(sig, sfreq, (4, 8))
                row[f'{ch}_alpha'] = bandpower(sig, sfreq, (8, 12))
                row[f'{ch}_beta'] = bandpower(sig, sfreq, (12, 30))
                row[f'{ch}_gamma'] = bandpower(sig, sfreq, (30, 45))

        if use_cca and target_freqs:
            X = win[['EXG0', 'EXG1']].values
            n_samples = X.shape[0]
            ref_signals = generate_reference_signals(target_freqs, sfreq, n_samples)
            cca = CCA(n_components=1)

            for j, ref in enumerate(ref_signals):
                cca.fit(X, ref)
                X_c, Y_c = cca.transform(X, ref)
                corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                row[f'cca_corr_f{target_freqs[j]}'] = corr

        features.append(row)

    return pd.DataFrame(features)

# === CONFIG ===
sfreq = 250
input_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands'
output_dir = r'C:\laragon\www\skripsi\EEG FEATURES'
os.makedirs(output_dir, exist_ok=True)

# === STEP-WISE EXTRACTION ===
phases = [
    ('features_stat.csv', dict(use_stat=True, use_band=False, use_hjorth=False, use_cca=False)),
    ('features_stat_band.csv', dict(use_stat=True, use_band=True, use_hjorth=False, use_cca=False)),
    ('features_stat_band_hjorth.csv', dict(use_stat=True, use_band=True, use_hjorth=True, use_cca=False)),
    ('features_stat_band_hjorth_cca.csv', dict(use_stat=True, use_band=True, use_hjorth=True, use_cca=True, target_freqs=[8, 10, 12, 15, 20, 30]))
]

for filename, params in phases:
    print(f"\n🚀 Extracting phase: {filename}")
    all_features = []

    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue

        label = file.split('_')[0]
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        feats = extract_features_windowed(df, sfreq=sfreq, window_sec=1, **params)
        feats['label'] = label
        all_features.append(feats)

    phase_df = pd.concat(all_features, ignore_index=True)
    phase_path = os.path.join(output_dir, filename)
    phase_df.to_csv(phase_path, index=False)
    print(f"✅ Saved: {phase_path}")
