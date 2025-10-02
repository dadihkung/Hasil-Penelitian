import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, detrend
from sklearn.cross_decomposition import CCA
from datetime import datetime

# ======== CONFIG ========
hari = {
    "Monday": "senin", "Tuesday": "selasa", "Wednesday": "rabu",
    "Thursday": "kamis", "Friday": "jumat", "Saturday": "sabtu", "Sunday": "minggu",
}

now = datetime.now()
fs = 250            # Sampling rate
n_harmonics = 7     # Number of harmonics for CCA
n_subbands = 5      # Number of sub-bands for fbCCA
alpha_fb = 0.8      # Weight factor for fbCCA

input_folders = [r"C:\laragon\www\skripsi\output_cut_selasa_02_09_2025"]

output_folder = f"output_cca_fb_{hari[now.strftime('%A')]}_{now.strftime('%d')}_{now.strftime('%m')}_{now.strftime('%Y')}"
os.makedirs(output_folder, exist_ok=True)

summary_list = []

# ======== FILTER FUNCTION ========
def bandpass_notch_filter(data, lowcut, highcut, fs, order=6):
    data = detrend(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_bp, a_bp = butter(order, [low, high], btype='band')
    bandpassed = filtfilt(b_bp, a_bp, data)
    b_notch, a_notch = iirnotch(50/nyq, 20)
    filtered = filtfilt(b_notch, a_notch, bandpassed)
    return filtered

# ======== fbCCA FUNCTION ========
def fbcca_ssvep(eeg_data, fs, freqs, n_harmonics=7, n_subbands=7, alpha=0.8):
    max_corr = 0
    detected_freq = None
    corr_dict = {}

    fmin = 6
    fmax = 20
    subbands = np.linspace(fmin, fmax, n_subbands+1)

    for f in freqs:
        total_corr = 0
        for i in range(n_subbands):
            low = subbands[i]
            high = subbands[i+1]
            eeg_sub = np.array([bandpass_notch_filter(ch, low, high, fs) for ch in eeg_data])

            N = eeg_sub.shape[1]
            t = np.arange(N)/fs
            ref_signals = []
            for h in range(1, n_harmonics+1):
                ref_signals.append(np.sin(2*np.pi*h*f*t))
                ref_signals.append(np.cos(2*np.pi*h*f*t))
            ref_signals = np.array(ref_signals).T

            cca = CCA(n_components=1)
            U, V = cca.fit_transform(eeg_sub.T, ref_signals)
            corr = np.corrcoef(U.T, V.T)[0,1]

            total_corr += (alpha**i) * corr

        corr_dict[f] = total_corr
        if total_corr > max_corr:
            max_corr = total_corr
            detected_freq = f

    return detected_freq, max_corr, corr_dict

# ======== MAIN LOOP ========
for input_folder in input_folders:
    for subject_folder in sorted(os.listdir(input_folder)):
        subject_path = os.path.join(input_folder, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv')]
        if not csv_files:
            continue

        eeg_data_list = []
        channels_available = []

        for file in csv_files:
            df = pd.read_csv(os.path.join(subject_path, file))
            df.columns = [str(c) for c in df.columns]
            df.rename(columns={"channel_7":"O1","channel_8":"O2"}, inplace=True)

            for ch in ["O1","O2"]:
                if ch in df.columns:
                    eeg_data_list.append(df[ch].values)
                    channels_available.append(ch)

        eeg_data = np.array(eeg_data_list)
        if eeg_data.shape[0] == 0:
            continue

        # Determine target frequency based on subject
        base_name = subject_folder.lower()
        if any(sub in base_name for sub in ["paeko71", "rangga1", "usamah2", "syarif3", "eliza3", "kamil3", "martin3"]):
            target_freq = 10
        elif any(sub in base_name for sub in ["paeko72", "rangga2", "usamah4", "syarif1", "eliza5", "kamil5", "martin1"]):
            target_freq = 14
        elif any(sub in base_name for sub in ["paeko73", "rangga3", "usamah5", "syarif2", "eliza4", "kamil4", "martin2"]):
            target_freq = 12
        elif any(sub in base_name for sub in ["paeko74", "rangga4", "usamah6", "syarif5", "eliza1", "kamil1", "martin5"]):
            target_freq = 8
        else:
            target_freq = 9

        # Frequencies to test
        freqs_to_test = [8, 9, 10, 12, 14]

        # Apply fbCCA
        detected_freq, max_corr, corr_dict = fbcca_ssvep(eeg_data, fs, freqs_to_test,
                                                        n_harmonics=n_harmonics,
                                                        n_subbands=n_subbands,
                                                        alpha=alpha_fb)

        # Save summary
        summary_list.append({
            "subject_folder": subject_folder,
            "channels": ",".join(channels_available),
            "target_freq": target_freq,
            "detected_freq": detected_freq,
            "max_corr": max_corr
        })

        # Plot raw EEG
        plt.figure(figsize=(10,4))
        for i, ch in enumerate(channels_available):
            plt.plot(np.arange(eeg_data.shape[1])/fs, eeg_data[i,:], label=ch)
        plt.title(f"Raw EEG ({subject_folder})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        img_folder = os.path.join(output_folder, subject_folder)
        os.makedirs(img_folder, exist_ok=True)
        plt.savefig(os.path.join(img_folder, f"{subject_folder}_raw.png"))
        plt.close()

        # Plot fbCCA correlations
        plt.figure(figsize=(6,4))
        plt.bar(corr_dict.keys(), corr_dict.values(), color='skyblue')
        plt.axvline(target_freq, color='red', linestyle='--', label=f"Target: {target_freq} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("fbCCA Correlation")
        plt.title(f"fbCCA Correlation ({subject_folder})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_folder, f"{subject_folder}_fbcca_corr.png"))
        plt.close()

        print(f"[✓] {subject_folder} | Target: {target_freq} Hz | Detected: {detected_freq} Hz | Max Corr: {max_corr:.3f}")

# ======== SAVE SUMMARY CSV ========
summary_df = pd.DataFrame(summary_list)
csv_output_path = os.path.join(output_folder, f"fbcca_summary_{hari[now.strftime('%A')]}_{now.strftime('%d_%m_%Y')}.csv")
summary_df.to_csv(csv_output_path, index=False)
print(f"[✓] Summary CSV saved at {csv_output_path}")
