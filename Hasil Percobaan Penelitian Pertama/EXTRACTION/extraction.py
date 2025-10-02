import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Parameters
sfreq = 250  # Sampling frequency
trial_duration_sec = 12  # 7s flicker + 5s rest
flicker_duration_sec = 7  # Use only this part for analysis
stimulus_frequencies = [15.5, 13.5, 11.5, 9.5, 5.5, 7.5]
num_freqs = len(stimulus_frequencies)

# Frequency bands for FBCCA
frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 40)
}

# Load EEG data
file_path = 'EEG BERSIH/preprocessed_eeg_fahri.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Drop sample index column
if 'Sample Index' in df.columns:
    eeg_data = df.drop(columns=['Sample Index'])
else:
    eeg_data = df.copy()

# Convert to numpy array
eeg_data = eeg_data.values

# Calculate samples
samples_per_trial = trial_duration_sec * sfreq
flicker_samples = flicker_duration_sec * sfreq
num_trials = len(eeg_data) // samples_per_trial
if num_trials == 0:
    print("Error: Not enough data for even one trial.")
    exit()

# Time vector for flicker duration
time = np.arange(0, flicker_duration_sec, 1 / sfreq)
if len(time) != flicker_samples:
    # Fix length mismatch due to float rounding
    time = np.linspace(0, flicker_duration_sec, int(flicker_samples), endpoint=False)

# Create reference signals
reference_signals = {}
for freq in stimulus_frequencies:
    reference_signals[freq] = np.vstack((
        np.sin(2 * np.pi * freq * time),
        np.cos(2 * np.pi * freq * time)
    )).T  # shape: (samples, 2)

# Bandpass filter
def bandpass_filter(data, lowcut, highcut, sfreq, order=4):
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# FBCCA analysis
cca = CCA(n_components=1)
cca_data = []
predicted_labels = []
true_labels = []

for i in range(num_trials):
    start_idx = i * samples_per_trial
    end_idx = start_idx + flicker_samples  # only flicker duration

    trial_data = eeg_data[start_idx:end_idx, :]
    if trial_data.shape[0] < flicker_samples:
        print(f"Skipping trial {i}: insufficient data ({trial_data.shape[0]} samples)")
        continue

    trial_scores = []

    for band_name, (lowcut, highcut) in frequency_bands.items():
        try:
            filtered_data = bandpass_filter(trial_data, lowcut, highcut, sfreq)
        except Exception as e:
            print(f"Filter error at trial {i}, band {band_name}: {e}")
            continue

        band_scores = []
        for freq in stimulus_frequencies:
            ref = reference_signals[freq]
            if filtered_data.shape[0] != ref.shape[0]:
                print(f"Warning: Mismatch in sample length at trial {i}, freq {freq}Hz, band {band_name}")
                band_scores.append(0)
                continue
            try:
                cca.fit(filtered_data, ref)
                score = cca.score(filtered_data, ref)
                band_scores.append(score)
            except Exception as e:
                print(f"CCA error at trial {i}, freq {freq} Hz: {e}")
                band_scores.append(0)

        trial_scores.extend(band_scores)

    if len(trial_scores) == 0:
        print(f"Warning: Trial {i} has no valid scores.")
        continue

    label_index = i % num_freqs
    label = stimulus_frequencies[label_index]

    # Defensive check before np.argmax
    if len(trial_scores) == 0:
        predicted_label = None
    else:
        max_idx = np.argmax(trial_scores)
        predicted_label = stimulus_frequencies[max_idx % num_freqs]

    cca_data.append(trial_scores + [label])
    predicted_labels.append(predicted_label)
    true_labels.append(label)

# Save results
if cca_data:
    columns = [f'{band}_{freq}Hz' for band in frequency_bands for freq in stimulus_frequencies] + ['label']
    cca_df = pd.DataFrame(cca_data, columns=columns)
    cca_df.to_csv('extraction_fbcca_for_lda_fahri.csv', index=False)
    print("✅ Saved FBCCA features to 'extraction_fbcca_for_lda_fahri.csv'")
else:
    print("⚠️ No FBCCA data to save.")

# Plot results
if predicted_labels and all(predicted_labels):
    plt.figure(figsize=(10, 5))
    plt.plot(true_labels, label='True Label', marker='o')
    plt.plot(predicted_labels, label='Predicted Label', marker='x')
    plt.title('FBCCA Predicted vs True Labels')
    plt.xlabel('Trial')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No valid predictions to plot.")
