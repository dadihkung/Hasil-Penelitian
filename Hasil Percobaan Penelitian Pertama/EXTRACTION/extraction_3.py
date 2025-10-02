import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.cross_decomposition import CCA

# === Generate reference signals for CCA ===
def generate_reference_signals(freqs, sfreq, window_size, harmonics=2):
    """
    Generate reference signals (sine + cosine) for CCA.
    freqs: list of target frequencies
    sfreq: sampling frequency
    window_size: number of samples
    harmonics: number of harmonics to include
    """
    t = np.arange(window_size) / sfreq
    refs = {}
    for f in freqs:
        signals = []
        for h in range(1, harmonics + 1):
            signals.append(np.sin(2 * np.pi * f * h * t))
            signals.append(np.cos(2 * np.pi * f * h * t))
        refs[f] = np.array(signals).T  # shape: (window_size, 2*harmonics)
    return refs

# === CCA Feature Extraction (Multi-channel) ===
def cca_feature(sig, refs, cca_cache=None):
    """
    Compute max canonical correlation between multi-channel EEG signal and reference signals.
    sig: shape (n_samples, n_channels)
    refs: dict of reference signals {freq: array(window_size, 2*harmonics)}
    cca_cache: dict for caching fitted models
    """
    if cca_cache is None:
        cca_cache = {}

    corr_features = {}
    for f, ref in refs.items():
        key = (f, sig.shape[1])
        if key not in cca_cache:
            cca_cache[key] = CCA(n_components=1)

        cca = cca_cache[key]
        cca.fit(sig, ref)
        X_c, Y_c = cca.transform(sig, ref)
        corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
        corr_features[f] = corr
    return corr_features

# === Feature Extraction Function ===
def extract_features_windowed(df, sfreq, window_sec=1, 
                              use_stat=True, use_band=False, use_hjorth=False, use_cca=False, 
                              target_freqs=None):
    window_size = int(sfreq * window_sec)
    refs = None
    if use_cca and target_freqs is not None:
        refs = generate_reference_signals(target_freqs, sfreq, window_size, harmonics=2)

    cca_cache = {}
    features = []

    for i in range(0, len(df) - window_size + 1, window_size):
        win = df.iloc[i:i+window_size]
        row = {
            'subject': win['subject'].iloc[0],
            'loop': win['loop'].iloc[0],
            'start_time': win['time'].iloc[0],
            'end_time': win['time'].iloc[-1],
        }

        # --- Ambil multi-channel (EXG0 + EXG1) ---
        sig_multi = win[['EXG0', 'EXG1']].values

        for idx, ch in enumerate(['EXG0', 'EXG1']):
            sig = sig_multi[:, idx]

            # --- Statistical Features ---
            if use_stat:
                row[f'{ch}_mean'] = np.mean(sig)
                row[f'{ch}_std'] = np.std(sig)
                row[f'{ch}_var'] = np.var(sig)
                row[f'{ch}_skew'] = skew(sig)
                row[f'{ch}_kurtosis'] = kurtosis(sig)

            # --- Bandpower Features ---
            if use_band:
                freqs, psd = welch(sig, sfreq, nperseg=sfreq*5)
                bands = {
                    "delta": (1, 4),
                    "theta": (4, 8),
                    "alpha": (8, 13),
                    "beta": (13, 30)
                }
                for bname, (fmin, fmax) in bands.items():
                    bp = np.trapz(psd[(freqs >= fmin) & (freqs <= fmax)], 
                                  freqs[(freqs >= fmin) & (freqs <= fmax)])
                    row[f"{ch}_{bname}_bandpower"] = bp

            # --- Hjorth Parameters ---
            if use_hjorth:
                diff1 = np.diff(sig)
                diff2 = np.diff(diff1)
                var0 = np.var(sig)
                var1 = np.var(diff1)
                var2 = np.var(diff2)

                activity = var0
                mobility = np.sqrt(var1/var0) if var0 > 0 else 0
                complexity = np.sqrt(var2/var1)/mobility if var1 > 0 and mobility > 0 else 0

                row[f'{ch}_hjorth_activity'] = activity
                row[f'{ch}_hjorth_mobility'] = mobility
                row[f'{ch}_hjorth_complexity'] = complexity

        # --- Real CCA Features (Multi-channel) ---
        if use_cca and refs is not None:
            cca_corrs = cca_feature(sig_multi, refs, cca_cache)
            for f, corr in cca_corrs.items():
                row[f"CCA_{f}Hz"] = corr

        features.append(row)

    return pd.DataFrame(features)

# === CONFIG ===
sfreq = 250  # Sampling rate
window_sec = 7
input_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands2'
output_dir = r'C:\laragon\www\skripsi\EEG FEATURES\new3'
os.makedirs(output_dir, exist_ok=True)

# === PROCESS EACH COMMAND FILE ===
all_features = []

for file in os.listdir(input_dir):
    if not file.endswith('_all_subjects.csv'):
        continue

    command_label = file.split('_')[0]
    file_path = os.path.join(input_dir, file)
    print(f"🔎 Extracting features from {file_path}...")

    df = pd.read_csv(file_path)
    feats = extract_features_windowed(
        df, sfreq=sfreq, window_sec=window_sec,
        use_stat=True, use_band=True, use_hjorth=True, use_cca=True,
        target_freqs=[15.5, 13.5, 11.5, 9.5, 5.5, 7.5]  # sesuaikan dengan stimulusmu
    )

    feats['label'] = command_label
    all_features.append(feats)

    # Save per-command
    output_file = os.path.join(output_dir, f"{command_label}_features.csv")
    feats.to_csv(output_file, index=False)
    print(f"✅ Saved: {command_label} → {output_file}")

# === Save Combined Dataset ===
if all_features:
    all_df = pd.concat(all_features, ignore_index=True)
    combined_output = os.path.join(output_dir, "features_all_commands.csv")
    all_df.to_csv(combined_output, index=False)
    print(f"🎉 Combined features saved to {combined_output}")
else:
    print("⚠️ No command files found in input directory.")
