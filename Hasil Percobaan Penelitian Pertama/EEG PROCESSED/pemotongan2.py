import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, welch
from scipy.stats import zscore
from scipy.fft import fft, fftfreq
import seaborn as sns

# ==============================
# CONFIGURATION
# ==============================
subject_dirs = {
    "didan": r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_didan_simdrone",
    "fahri": r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_fahri_simdrone",
    "subagja": r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_subagja_simdrone", 
    "sultan": r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_sultan_simdrone",
    "syarif": r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_syarif_simdrone",
    "usamah": r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_usamah_simdrone"
}

# Adjusted stimulus timing with 1-second gaps
command_times_0 = {
    'depan': [(0,7)],        # 0-7s
    'belakang': [(8,15)],     # 8-15s (starts 1s after depan ends)
    'kanan': [(16,23)],       # 16-23s
    'kiri': [(24,31)],        # 24-31s
    'terbang': [(32,39)],     # 32-39s
    'landing': [(40,47)],     # 40-47s
    # Repeat the pattern 4 more times
    'depan': [(48,55)],       # 48-55s
    'belakang': [(56,63)],     # 56-63s
    'kanan': [(64,71)],       # 64-71s
    'kiri': [(72,79)],        # 72-79s
    'terbang': [(80,87)],     # 80-87s
    'landing': [(88,95)],     # 88-95s
    'depan': [(96,103)],      # 96-103s
    'belakang': [(104,111)],   # 104-111s
    'kanan': [(112,119)],     # 112-119s
    'kiri': [(120,127)],      # 120-127s
    'terbang': [(128,135)],   # 128-135s
    'landing': [(136,143)],   # 136-143s
    'depan': [(144,151)],     # 144-151s
    'belakang': [(152,159)],   # 152-159s
    'kanan': [(160,167)],     # 160-167s
    'kiri': [(168,175)],      # 168-175s
    'terbang': [(176,183)],   # 176-183s
    'landing': [(184,191)],   # 184-191s
    'depan': [(192,199)],     # 192-199s
    'belakang': [(200,207)],   # 200-207s
    'kanan': [(208,215)],     # 208-215s
    'kiri': [(216,223)],      # 216-223s
    'terbang': [(224,231)],   # 224-231s
    'landing': [(232,239)],   # 232-239s
    'depan': [(240,247)],     # 240-247s
    'belakang': [(248,255)],   # 248-255s
    'kanan': [(256,263)],     # 256-263s
    'kiri': [(264,271)],      # 264-271s
    'terbang': [(272,279)],   # 272-279s
    'landing': [(280,287)],   # 280-287s
    'depan': [(288,295)],     # 288-295s
    'belakang': [(296,303)],   # 296-303s
    'kanan': [(304,311)],     # 304-311s
    'kiri': [(312,319)],      # 312-319s
    'terbang': [(320,327)],   # 320-327s
    'landing': [(328,335)],   # 328-335s
    'depan': [(336,343)],     # 336-343s
    'belakang': [(344,351)],   # 344-351s
    'kanan': [(352,359)]      # 352-359s
}

command_times_5 = {
    'depan': [(5,12)],        # 5-12s
    'belakang': [(13,20)],     # 13-20s
    'kanan': [(21,28)],       # 21-28s
    'kiri': [(29,36)],        # 29-36s
    'terbang': [(37,44)],     # 37-44s
    'landing': [(45,52)],     # 45-52s
    # Repeat the pattern 4 more times
    'depan': [(53,60)],       # 53-60s
    'belakang': [(61,68)],     # 61-68s
    'kanan': [(69,76)],       # 69-76s
    'kiri': [(77,84)],        # 77-84s
    'terbang': [(85,92)],     # 85-92s
    'landing': [(93,100)],    # 93-100s
    'depan': [(101,108)],     # 101-108s
    'belakang': [(109,116)],   # 109-116s
    'kanan': [(117,124)],     # 117-124s
    'kiri': [(125,132)],      # 125-132s
    'terbang': [(133,140)],   # 133-140s
    'landing': [(141,148)],   # 141-148s
    'depan': [(149,156)],     # 149-156s
    'belakang': [(157,164)],   # 157-164s
    'kanan': [(165,172)],     # 165-172s
    'kiri': [(173,180)],      # 173-180s
    'terbang': [(181,188)],   # 181-188s
    'landing': [(189,196)],   # 189-196s
    'depan': [(197,204)],     # 197-204s
    'belakang': [(205,212)],   # 205-212s
    'kanan': [(213,220)],     # 213-220s
    'kiri': [(221,228)],      # 221-228s
    'terbang': [(229,236)],   # 229-236s
    'landing': [(237,244)],   # 237-244s
    'depan': [(245,252)],     # 245-252s
    'belakang': [(253,260)],   # 253-260s
    'kanan': [(261,268)],     # 261-268s
    'kiri': [(269,276)],      # 269-276s
    'terbang': [(277,284)],   # 277-284s
    'landing': [(285,292)],   # 285-292s
    'depan': [(293,300)],     # 293-300s
    'belakang': [(301,308)],   # 301-308s
    'kanan': [(309,316)],     # 309-316s
    'kiri': [(317,324)],      # 317-324s
    'terbang': [(325,332)],   # 325-332s
    'landing': [(333,340)],   # 333-340s
    'depan': [(341,348)],     # 341-348s
    'belakang': [(349,356)]    # 349-356s
}

target_freqs = {
    'depan': 15.5,
    'belakang': 13.5, 
    'kanan': 11.5,
    'kiri': 9.5,
    'terbang': 5.5,
    'landing': 7.5
}

# Signal processing settings
eeg_channels = [1, 2]  # EXG Channel 1 (O1) and EXG Channel 2 (O2)
fs = 250                # Sampling rate
lowcut = 5              # Bandpass lower cutoff
highcut = 50            # Bandpass upper cutoff
notch_freq = 50         # Notch filter frequency
epoch_length = 7        # Epoch duration in seconds
harmonics = 5

delayed_subjects = ["didan", "subagja"]

# ==============================
# SIGNAL PROCESSING FUNCTIONS
# ==============================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_notch(f0, fs, Q=30):
    nyq = 0.5 * fs
    w0 = f0 / nyq
    bw = w0 / Q
    b, a = butter(2, [w0 - bw/2, w0 + bw/2], btype='bandstop')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, f0, fs, Q=30):
    b, a = butter_notch(f0, fs, Q=Q)
    return filtfilt(b, a, data, axis=0)

# ==============================
# DATA LOADING FUNCTIONS
# ==============================
def load_eeg_txt(subject_dir):
    """Load EEG data from OpenBCI text file"""
    try:
        # Find the text file in directory
        txt_files = [f for f in os.listdir(subject_dir) if f.endswith(".txt") and "OpenBCI-RAW" in f]
        if not txt_files:
            raise FileNotFoundError(f"No OpenBCI text file found in {subject_dir}")
        
        # Read the text file, skipping metadata lines
        with open(os.path.join(subject_dir, txt_files[0]), 'r') as f:
            lines = f.readlines()
        
        # Find where data starts (skip lines starting with %)
        data_lines = [line.strip() for line in lines if not line.startswith('%')]
        
        # Parse the data
        data = []
        for line in data_lines[1:]:  # Skip header line
            parts = line.split(',')
            # Extract EXG channels (columns 1-8 are EXG 0-7)
            exg_data = [float(x.strip()) for x in parts[1:9]]  # EXG Channels 0-7
            data.append(exg_data)
        
        eeg_data = np.array(data)
        
        # Select only the channels we want (O1 and O2)
        eeg_data = eeg_data[:, eeg_channels]
        
        if eeg_data.size == 0:
            raise ValueError("No EEG data found in file")
            
        print(f"Loaded {eeg_data.shape[0]} samples from {txt_files[0]}")
        return eeg_data
        
    except Exception as e:
        print(f"Error loading {subject_dir}: {str(e)}")
        return None

# ==============================
# PROCESSING FUNCTIONS
# ==============================
def preprocess_data(eeg_data):
    """Enhanced preprocessing pipeline"""
    # 1. Detrend
    eeg_data = detrend(eeg_data, axis=0)
    
    # 2. Notch filter at 50Hz
    eeg_data = notch_filter(eeg_data, notch_freq, fs)
    
    # 3. Bandpass filter
    eeg_data = bandpass_filter(eeg_data, lowcut, highcut, fs)
    
    # 4. Z-score normalization
    return zscore(eeg_data, axis=0)

def extract_labeled_epochs(eeg_data, subject_name):
    command_times = command_times_5 if subject_name in delayed_subjects else command_times_0
    samples_per_epoch = epoch_length * fs
    epochs = {}
    
    for command, times in command_times.items():
        command_epochs = []
        for start, end in times:
            start_sample = int(start * fs)
            end_sample = int(end * fs)
            if end_sample <= eeg_data.shape[0]:
                epoch = eeg_data[start_sample:end_sample]
                if epoch.shape[0] == samples_per_epoch:  # Ensure correct length
                    command_epochs.append(epoch)
        
        if command_epochs:
            epochs[command] = np.stack(command_epochs)
            
    return epochs

def analyze_epochs(epochs_dict, subject_name):
    for command, epochs in epochs_dict.items():
        plt.figure(figsize=(15, 5))
        
        # FFT Analysis
        plt.subplot(1, 2, 1)
        fft_vals = np.abs(fft(epochs[0, :, 0]))
        freqs = fftfreq(epochs.shape[1], 1/fs)
        
        # Only plot positive frequencies
        pos_mask = freqs > 0
        plt.plot(freqs[pos_mask], fft_vals[pos_mask][:len(freqs[pos_mask])])
        plt.axvline(x=target_freqs[command], color='r', linestyle='--', label='Target')
        plt.xlim(0, 60)
        plt.title(f"FFT {command} (Expected: {target_freqs[command]}Hz)\nSubject: {subject_name}")
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        
        # Time Domain
        plt.subplot(1, 2, 2)
        time_axis = np.arange(500)/fs
        plt.plot(time_axis, epochs[0, :500, 0])
        plt.title(f"First {500/fs:.1f} seconds\nCommand: {command}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (μV)")
        
        plt.tight_layout()
        plt.show()

# ==============================
# MAIN PROCESSING LOOP
# ==============================
results = {}

for subject_name, subject_dir in subject_dirs.items():
    print(f"\n{'='*40}")
    print(f"Processing {subject_name}")
    print(f"{'='*40}")
    
    # 1. Load data from text file
    eeg_data = load_eeg_txt(subject_dir)
    if eeg_data is None:
        continue
        
    print(f"EEG shape: {eeg_data.shape}")
    
    # 2. Preprocess with enhanced pipeline
    try:
        eeg_data = preprocess_data(eeg_data)
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        continue
    
    # 3. Extract labeled epochs
    epochs_dict = extract_labeled_epochs(eeg_data, subject_name)
    if not epochs_dict:
        print("No valid epochs extracted")
        continue
        
    # 4. Store results
    results[subject_name] = {
        'epochs': epochs_dict,
        'target_freqs': target_freqs
    }
    
    # 5. Generate improved plots
    analyze_epochs(epochs_dict, subject_name)
    
    # Print summary
    print(f"Extracted epochs:")
    for cmd, arr in epochs_dict.items():
        print(f"{cmd}: {arr.shape[0]} epochs")

print("\nProcessing complete!")