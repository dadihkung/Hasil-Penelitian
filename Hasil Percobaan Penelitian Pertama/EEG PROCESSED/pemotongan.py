import os
import pandas as pd
import numpy as np
import mne
from scipy.stats import zscore

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

    # Preprocessing
    raw.filter(l_freq=1., h_freq=20., fir_design='firwin')
    raw.notch_filter(freqs=50., fir_design='firwin')
    raw.set_eeg_reference('average', projection=False)

    return raw

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

# ==== Command stimulus windows ====
command_times_0 = {
    'depan':     [(0, 7), (72, 79), (144, 151), (216, 223), (288, 295)],
    'belakang': [(12, 19), (84, 91), (156, 163), (228, 235), (300, 307)],
    'kanan':    [(24, 31), (96, 103), (168, 175), (240, 247), (312, 319)],
    'kiri':     [(36, 43), (108, 115), (180, 187), (252, 259), (324, 331)],
    'terbang':  [(48, 55), (120, 127), (192, 199), (264, 271), (336, 343)],
    'landing':  [(60, 67), (132, 139), (204, 211), (276, 283), (348, 355)],
}

command_times_5 = {
    'depan':     [(5, 12), (77, 84), (149, 156), (221, 228), (293, 300)],
    'belakang': [(17, 24), (89, 96), (161, 168), (233, 240), (305, 312)],
    'kanan':    [(29, 36), (101, 108), (173, 180), (245, 252), (317, 324)],
    'kiri':     [(41, 48), (113, 120), (185, 192), (257, 264), (329, 336)],
    'terbang':  [(53, 60), (125, 132), (197, 204), (269, 276), (341, 348)],
    'landing':  [(65, 72), (137, 144), (209, 216), (281, 288), (353, 360)],
}

# ==== Subjects & file paths ====
file_paths = [
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_fahri_simdrone\BrainFlow-RAW_simdrone_fahri_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_sultan_simdrone\BrainFlow-RAW_simdrone_sultan_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_usamah_simdrone\BrainFlow-RAW_simdrone_usamah_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_syarif_simdrone\BrainFlow-RAW_simdrone_syarif_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_didan_simdrone\BrainFlow-RAW_simdrone_didan_0.csv',
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_subagja_simdrone\BrainFlow-RAW_simdrone_subagja_0.csv'
]

sfreq = 250
all_command_data = {cmd: [] for cmd in command_times_0}

# ==== Pipeline: Clean first, then split ====
for file_path in file_paths:
    subject_name = os.path.basename(file_path).split('_')[-2]
    print(f"\nProcessing {subject_name}...")

    # Pilih command_times sesuai subjek
    if subject_name.lower() in ['subagja', 'didan']:
        current_command_times = command_times_5
    else:
        current_command_times = command_times_0

    raw = load_and_prepare_raw(file_path, sfreq=sfreq)
    subject_epochs = extract_command_epochs(raw, current_command_times, sfreq, subject_name)

    for command in current_command_times:
        all_command_data[command].extend(subject_epochs[command])

# ==== Save outputs ====
output_dir = r'C:\laragon\www\skripsi\EEG PROCESSED\commands2'
os.makedirs(output_dir, exist_ok=True)

for command, dfs in all_command_data.items():
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        out_file = os.path.join(output_dir, f"{command}_all_subjects.csv")
        combined_df.to_csv(out_file, index=False)
        print(f"✅ Saved: {command} → {out_file}")
    else:
        print(f"⚠️ No data found for command: {command}")
