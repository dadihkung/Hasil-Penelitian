import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, iirnotch, detrend
from datetime import datetime

# ---------------- Hari dalam bahasa Indonesia ----------------
hari = {
    "Monday": "senin",
    "Tuesday": "selasa",
    "Wednesday": "rabu",
    "Thursday": "kamis",
    "Friday": "jumat",
    "Saturday": "sabtu",
    "Sunday": "minggu",
}

now = datetime.now()
hari_ini = f"{hari[now.strftime('%A')]}_{now.strftime('%d')}_{now.strftime('%m')}_{now.strftime('%Y')}"

# ---------------- PARAMETER ----------------
fs = 250  # Sampling rate (Hz)
lowcut = 7  # Batas bawah bandpass
highcut = 20  # Batas atas bandpass

# ---------------- INPUT & OUTPUT FOLDER ----------------
subject_folder = r"C:\laragon\www\skripsi\output_cut_senin_08_09_2025\fahri_10.0Hz_kiri_bawah_20250908_172946"
output_folder = r"output_fft_{hari_ini}\fahri_10.0Hz_kiri_bawah_20250908_172946"
os.makedirs(output_folder, exist_ok=True)

# ---------------- FILTERING FUNCTION ----------------
def bandpass_notch_filter(data, lowcut, highcut, fs, order=6):
    data = detrend(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Bandpass
    b_bp, a_bp = butter(order, [low, high], btype='band')
    bandpassed = filtfilt(b_bp, a_bp, data)
    # Notch 50 Hz
    notch_freq = 50.0
    Q = 30.0
    b_notch, a_notch = iirnotch(notch_freq / nyq, Q)
    filtered = filtfilt(b_notch, a_notch, bandpassed)
    return filtered

# ---------------- LOOP FILE CSV DI FOLDER SUBJEK ----------------
for file in os.listdir(subject_folder):
    if not file.endswith(".csv"):
        continue
    if not ("O1" in file or "O2" in file):
        continue  # hanya O1 & O2

    file_path = os.path.join(subject_folder, file)
    df = pd.read_csv(file_path)

    # Tentukan channel
    if "O1" in file and "O1" in df.columns:
        signal = df["O1"].values
        channel = "O1"
    elif "O2" in file and "O2" in df.columns:
        signal = df["O2"].values
        channel = "O2"
    else:
        print(f"[!] Kolom channel tidak ditemukan di {file}")
        continue

    # Target freq khusus usamah = 14 Hz
    target_freq = 10.0

    # Base filename
    base_filename = file.replace(".csv", "")

    # ---------------- FILTER ----------------
    filtered = bandpass_notch_filter(signal, lowcut, highcut, fs)

    # ---------------- FFT ----------------
    N = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(N, 1 / fs)
    xf = xf[:N // 2]
    yf = 2.0 / N * np.abs(yf[:N // 2])

    # Cari peak dalam rentang target
    mask = (xf >= lowcut) & (xf <= highcut)
    peak_idx = np.argmax(yf[mask])
    peak_freq = xf[mask][peak_idx]
    peak_amp = yf[mask][peak_idx]

    # ---------------- PLOT RAW SIGNAL ----------------
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(signal))/fs, signal, color='gray', label=f"Raw ({channel})")
    plt.title(f"Raw EEG Signal ({channel}) - {base_filename}")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo")
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    raw_image_output_path = os.path.join(output_folder, f"{base_filename}_raw.png")
    plt.tight_layout()
    plt.savefig(raw_image_output_path)
    plt.close()
    print(f"[✓] {file} Raw signal plot saved.")

    # ---------------- PLOT FFT ----------------
    plt.figure(figsize=(8,4))
    plt.plot(xf, yf, color='blue', label="FFT Signal")
    plt.plot(peak_freq, peak_amp, 'ro', label=f"Peak: {peak_freq:.2f} Hz")
    plt.axvline(x=target_freq, color='gray', linestyle='--', linewidth=1, label=f"Target: {target_freq:.2f} Hz")
    plt.xlim(0,30)
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Amplitudo")
    plt.title(f"FFT EEG Signal ({channel}) - {base_filename}")
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    fft_image_output_path = os.path.join(output_folder, f"{base_filename}_fft.png")
    plt.tight_layout()
    plt.savefig(fft_image_output_path)
    plt.close()
    print(f"[✓] {file} FFT plot saved. Peak: {peak_freq:.2f} Hz | Target: {target_freq:.2f} Hz")
