import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, iirnotch, detrend
from datetime import datetime

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

# ======== PARAMETERS ========
fs = 250  # sampling rate
lowcut = 7
highcut = 20

input_folder = r"C:\laragon\www\skripsi\output_cut_sabtu_30_08_2025"
output_folder = f"output_fft_{hari[now.strftime('%A')]}_{now.strftime('%d')}_{now.strftime('%m')}_{now.strftime('%Y')}"
os.makedirs(output_folder, exist_ok=True)

summary_list = []  # to store results for CSV

# ======== FILTERING FUNCTION ========
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
    Q = 20.0
    b_notch, a_notch = iirnotch(notch_freq / nyq, Q)
    filtered = filtfilt(b_notch, a_notch, bandpassed)

    return filtered

# ======== LOOP FILE CSV ========
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue
        if not ("O1" in file or "O2" in file):
            continue

        file_path = os.path.join(root, file)
        df = pd.read_csv(file_path)

        channels = []
        if "O1" in file and "O1" in df.columns:
            channels.append("O1")
        if "O2" in file and "O2" in df.columns:
            channels.append("O2")
        if not channels:
            print(f"[!] Kolom channel tidak ditemukan di {file}")
            continue

        if "usamah2" in file.lower():
            target_freq = 10
        elif "usamah4" in file.lower():
            target_freq = 14
        elif "usamah5" in file.lower():
            target_freq = 12
        elif "usamah6" in file.lower():
            target_freq = 8
        else:
            target_freq = 9

        # Output folder
        relative_path = os.path.relpath(root, input_folder)
        base_filename = file.replace(".csv", "")
        image_folder = os.path.join(output_folder, relative_path)
        os.makedirs(image_folder, exist_ok=True)

        for channel in channels:
            signal = df[channel].values

            # Filtering
            filtered = bandpass_notch_filter(signal, lowcut, highcut, fs)

            # FFT with zero-padding
            N = len(filtered)
            N_fft = 16 * N
            yf = fft(filtered, n=N_fft)
            xf = fftfreq(N_fft, 1/fs)[:N_fft//2]
            yf = 2.0 / N * np.abs(yf[:N_fft//2])

            # Peak detection ±2 Hz around target
            mask = (xf >= target_freq-2) & (xf <= target_freq+2)
            peak_idx = np.argmax(yf[mask])
            peak_freq = xf[mask][peak_idx]
            peak_amp = yf[mask][peak_idx]

            # Append to summary
            summary_list.append({
                "filename": file,
                "channel": channel,
                "target_freq": target_freq,
                "peak_freq": peak_freq,
                "peak_amplitude": peak_amp
            })

            # Plot raw signal
            plt.figure(figsize=(10,4))
            plt.plot(np.arange(len(signal))/fs, signal, color='gray', label=f"Raw Signal ({channel})")
            plt.title(f"Raw EEG Signal ({channel})")
            plt.xlabel("Waktu (detik)")
            plt.ylabel("Amplitudo")
            plt.grid(True)
            plt.legend(loc='upper right', fontsize=10)
            raw_image_output_path = os.path.join(image_folder, f"{base_filename}_{channel}_raw.png")
            plt.tight_layout()
            plt.savefig(raw_image_output_path)
            plt.close()

            # Plot FFT
            plt.figure(figsize=(8,4))
            plt.plot(xf, yf, color='blue', label="FFT Signal")
            plt.plot(peak_freq, peak_amp, 'ro', label=f"Peak: {peak_freq:.2f} Hz")
            plt.axvline(x=target_freq, color='gray', linestyle='--', linewidth=1, label=f"Target: {target_freq} Hz")
            plt.xlim(0,30)
            plt.xlabel("Frekuensi (Hz)")
            plt.ylabel("Amplitudo")
            plt.grid(True)
            plt.title(f"FFT EEG Signal ({channel})")
            plt.legend(loc='upper right', fontsize=10)
            fft_image_output_path = os.path.join(image_folder, f"{base_filename}_{channel}_fft.png")
            plt.tight_layout()
            plt.savefig(fft_image_output_path)
            plt.close()

            print(f"[✓] {file} ({channel}) FFT saved. Peak: {peak_freq:.2f} Hz | Target: {target_freq}")

# ======== SAVE SUMMARY CSV ========
summary_df = pd.DataFrame(summary_list)
csv_output_path = os.path.join(output_folder, f"fft_summary_{hari[now.strftime('%A')]}_{now.strftime('%d_%m_%Y')}.csv")
summary_df.to_csv(csv_output_path, index=False)
print(f"[✓] Summary CSV saved at {csv_output_path}")
