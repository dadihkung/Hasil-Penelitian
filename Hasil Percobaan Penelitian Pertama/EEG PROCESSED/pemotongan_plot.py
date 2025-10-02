import os
import pandas as pd
import numpy as np
import mne
from scipy.stats import zscore
import matplotlib.pyplot as plt

def clean_column(col):
    # Replace comma with dot (European decimal format), remove extraneous dots (e.g., thousands separator)
    col = col.astype(str).str.replace('.', '', regex=False)
    col = col.str.replace(',', '.', regex=False)
    col = pd.to_numeric(col, errors='coerce')
    return col

def reject_outliers(df, threshold=5):
    z_scores = np.abs(zscore(df, nan_policy='omit'))
    return df[(z_scores < threshold).all(axis=1)]

def load_and_prepare_raw(file_path, sfreq=250):
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df = df.iloc[:, 0:3]
    df.columns = ['Sample Index', 'EXG0', 'EXG1']

    for col in ['EXG0', 'EXG1']:
        df[col] = clean_column(df[col])
    df = df.dropna(subset=['EXG0', 'EXG1'])
    df[['EXG0', 'EXG1']] = reject_outliers(df[['EXG0', 'EXG1']])

    eeg_data = df[['EXG0', 'EXG1']].dropna().values
    ch_names = ['EXG0', 'EXG1']
    ch_types = ['eeg'] * 2
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data.T, info)

    return raw

def preprocess_raw(raw):
    # Make a copy for preprocessing
    processed_raw = raw.copy()
    
    # Preprocessing steps
    processed_raw.filter(l_freq=1., h_freq=20., fir_design='firwin')
    processed_raw.notch_filter(freqs=50., fir_design='firwin')
    processed_raw.set_eeg_reference('average', projection=False)
    
    return processed_raw

def plot_raw_vs_processed(raw, processed_raw, subject_name, output_dir_mentah, output_dir_bersih):
    # Create directories if they don't exist
    os.makedirs(output_dir_mentah, exist_ok=True)
    os.makedirs(output_dir_bersih, exist_ok=True)
    
    # Plot raw data if provided
    if raw is not None:
        fig = raw.plot(show=False, scalings='auto', title=f'Raw EEG - {subject_name}')
        raw_plot_path = os.path.join(output_dir_mentah, f'raw_{subject_name}.png')
        fig.savefig(raw_plot_path)
        plt.close(fig)
        print(f"  - Raw plot saved: {raw_plot_path}")
    
    # Plot processed data if provided
    if processed_raw is not None:
        fig = processed_raw.plot(show=False, scalings='auto', title=f'Processed EEG - {subject_name}')
        processed_plot_path = os.path.join(output_dir_bersih, f'processed_{subject_name}.png')
        fig.savefig(processed_plot_path)
        plt.close(fig)
        print(f"  - Processed plot saved: {processed_plot_path}")

def extract_command_epochs(raw, command_times, sfreq, subject_name):
    command_epochs = {cmd: [] for cmd in command_times}
    max_time = raw.times[-1]

    for command, time_ranges in command_times.items():
        for i, (tmin, tmax) in enumerate(time_ranges):
            loop_number = i + 1
            if tmin >= max_time:
                continue
            if tmax > max_time:
                tmax = max_time
            try:
                segment = raw.copy().crop(tmin=tmin, tmax=tmax).get_data().T
                df = pd.DataFrame(segment, columns=['EXG0', 'EXG1'])
                df = reject_outliers(df)
                df['subject'] = subject_name
                df['loop'] = loop_number
                df['time'] = np.arange(0, df.shape[0]) / sfreq + tmin
                command_epochs[command].append(df)
            except Exception as e:
                print(f"Error cropping {command} loop {loop_number} for {subject_name}: {e}")

    return command_epochs

# Define command stimulus windows (absolute times in seconds)
command_times = {
    'depan':     [(0, 7), (72, 79), (144, 151), (216, 223), (288, 295)],
    'belakang': [(12, 19), (84, 91), (156, 163), (228, 235), (300, 307)],
    'kanan':    [(24, 31), (96, 103), (168, 175), (240, 247), (312, 319)],
    'kiri':     [(36, 43), (108, 115), (180, 187), (252, 259), (324, 331)],
    'terbang':  [(48, 55), (120, 127), (192, 199), (264, 271), (336, 343)],
    'landing':  [(60, 67), (132, 139), (204, 211), (276, 283), (348, 355)],
}

file_paths = [
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_fahri_simdrone\BrainFlow-RAW_simdrone_fahri_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_sultan_simdrone\BrainFlow-RAW_simdrone_sultan_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_usamah_simdrone\BrainFlow-RAW_simdrone_usamah_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_syarif_simdrone\BrainFlow-RAW_simdrone_syarif_0.csv'
]

# Output directories for plots
output_dir_mentah = r'C:\laragon\www\skripsi\EEG MENTAH\PLOT'
output_dir_bersih = r'C:\laragon\www\skripsi\EEG BERSIH\HASIL PEMOTONGAN'

sfreq = 250

# === Pipeline: Load, plot raw, process, plot processed ===
for file_path in file_paths:
    subject_name = os.path.basename(file_path).split('_')[-2]
    print(f"\nProcessing {subject_name}...")
    
    try:
        # Load raw data
        raw = load_and_prepare_raw(file_path, sfreq=sfreq)
        
        # Plot raw data
        print("Plotting raw data...")
        plot_raw_vs_processed(raw, None, subject_name, output_dir_mentah, output_dir_bersih)
        
        # Preprocess
        print("Preprocessing data...")
        processed_raw = preprocess_raw(raw)
        
        # Plot processed data
        print("Plotting processed data...")
        plot_raw_vs_processed(None, processed_raw, subject_name, output_dir_mentah, output_dir_bersih)
        
    except Exception as e:
        print(f"Error processing {subject_name}: {str(e)}")

print("\nProcessing complete. All plots saved.")