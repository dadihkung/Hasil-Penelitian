import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def plot_eeg_signal(df, title, save_path):
    plt.figure(figsize=(15, 6))
    plt.plot(df['time'], df['EXG0'], label='EXG0')
    plt.plot(df['time'], df['EXG1'], label='EXG1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_windows(df, features, title, save_path):
    plt.figure(figsize=(15, 6))
    
    # Plot original signals
    plt.plot(df['time'], df['EXG0'], 'b-', alpha=0.3, label='EXG0 Raw')
    plt.plot(df['time'], df['EXG1'], 'g-', alpha=0.3, label='EXG1 Raw')
    
    # Plot feature windows
    for _, row in features.iterrows():
        plt.axvspan(row['start_time'], row['end_time'], 
                    color='red', alpha=0.1)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

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

        features.append(row)

    return pd.DataFrame(features)

# === CONFIG ===
sfreq = 250  # Sampling rate
input_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands'
output_dir = r'C:\laragon\www\skripsi\EEG FEATURES\new'
before_plot_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands\BEFORE EKS PLOT'
after_plot_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands\AFTER EKS PLOT'

# List of expected command files (modify according to your actual files)
expected_commands = ['depan', 'belakang', 'kanan', 'kiri', 'terbang', 'landing']

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(before_plot_dir, exist_ok=True)
os.makedirs(after_plot_dir, exist_ok=True)

# === PROCESS EACH COMMAND FILE ===
all_features = []

for file in os.listdir(input_dir):
    if not file.endswith('.csv'):
        continue
    
    # Skip files that don't match our expected command files
    command_label = file.split('_')[0]
    if command_label not in expected_commands:
        print(f"\nSkipping non-command file: {file}")
        continue

    file_path = os.path.join(input_dir, file)
    print(f"\nProcessing {file_path}...")

    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['time', 'EXG0', 'EXG1', 'subject', 'loop']
        if not all(col in df.columns for col in required_columns):
            print(f"⚠️ Skipping {file} - missing required columns")
            continue
        
        # Plot before feature extraction
        before_plot_path = os.path.join(before_plot_dir, f"{command_label}_before.png")
        plot_eeg_signal(df, f'Raw EEG - {command_label}', before_plot_path)
        print(f"✅ Before plot saved: {before_plot_path}")
        
        # Extract features
        feats = extract_features_windowed(df, sfreq=sfreq, window_sec=1,
                                      use_stat=True, use_band=False, use_hjorth=False, use_cca=False)
        feats['label'] = command_label
        
        # Plot after feature extraction
        after_plot_path = os.path.join(after_plot_dir, f"{command_label}_after.png")
        plot_feature_windows(df, feats, f'Feature Windows - {command_label}', after_plot_path)
        print(f"✅ After plot saved: {after_plot_path}")
        
        all_features.append(feats)

        # Save features per command
        output_file = os.path.join(output_dir, f"{command_label}_features_stat.csv")
        feats.to_csv(output_file, index=False)
        print(f"✅ Features saved: {output_file}")
    
    except Exception as e:
        print(f"⚠️ Error processing {file}: {str(e)}")
        continue

# === Save Combined Dataset ===
if all_features:
    all_df = pd.concat(all_features, ignore_index=True)
    combined_output = os.path.join(output_dir, "features_stat.csv")
    all_df.to_csv(combined_output, index=False)
    print(f"\n✅ Combined features saved to {combined_output}")
else:
    print("\n⚠️ No features were extracted!")