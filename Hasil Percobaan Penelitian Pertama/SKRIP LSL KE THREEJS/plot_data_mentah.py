import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Baca file input (gantilah dengan path file Anda)
# Contoh format file: CSV dengan kolom waktu dan channel-channel EEG
file_path = r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_usamah_simdrone\BrainFlow-RAW_simdrone_usamah_0.csv'  # Ganti dengan path file Anda
df = pd.read_csv(file_path)

# Jika file tidak memiliki kolom waktu, buat waktu berdasarkan sampling rate
if 'time' not in df.columns:
    sfreq = 250  # Sampling rate (Hz)
    durasi = len(df) / sfreq  # Durasi dalam detik
    time = np.arange(0, durasi, 1/sfreq)
else:
    time = df['time'].values

# Daftar channel (sesuaikan dengan file Anda)
channels = ['AF3', 'F7', 'F3', 'FC5', 'P7', 'O1', 'O2', 'P8', 'FC6', 'F4', 'F8', 'AF4']

# 2. Membuat plot dengan offset untuk setiap channel
offset = 50  # Jarak vertikal antara channel
fig, ax = plt.subplots(figsize=(12, 6))

for i, ch in enumerate(channels):
    if ch in df.columns:  # Pastikan channel ada di dataframe
        # Plot data channel dengan offset
        ax.plot(time, df[ch] + i*offset, label=ch)

# 3. Konfigurasi plot
yticks = [i*offset for i in range(len(channels))]
ax.set_yticks(yticks)
ax.set_yticklabels(channels)
ax.set_xlabel('Waktu (detik)')
ax.set_ylabel('Channel')
ax.set_title('Data EEG Mentah per Channel')

plt.tight_layout()
plt.show()