import pandas as pd
import os
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

# Tanggal sekarang
now = datetime.now()
sekarang = f"{hari[now.strftime('%A')]}_{now.strftime('%d')}_{now.strftime('%m')}_{now.strftime('%Y')}"

# ---------------- Daftar file input EEG ----------------
input_files = [
    r"C:\laragon\www\skripsi\data_mentah2\fahri\fahri_10.0Hz_kiri_bawah_20250908_172946.csv"
]

# Folder output
output_filename = f"output_cut_{sekarang}"

# ---------------- Konfigurasi channel EEG ----------------
# Cyton 8-channel OpenBCI
o1_index = 6
o2_index = 7

for input_filename in input_files:
    basename = os.path.splitext(os.path.basename(input_filename))[0]

    # Baca CSV hanya 8 kolom pertama (EEG channels) saja
    df = pd.read_csv(input_filename, usecols=range(8), header=0)
    
    # Rename O1/O2
    df.rename(columns={f"CH_{o1_index}": "O1",
                       f"CH_{o2_index}": "O2"}, inplace=True)

    # Buat folder output
    output_dir = os.path.join(output_filename, basename)
    os.makedirs(output_dir, exist_ok=True)

    # Simpan O1 dan O2 masing-masing
    o1_file = os.path.join(output_dir, f"{basename}_O1.csv")
    o2_file = os.path.join(output_dir, f"{basename}_O2.csv")

    df[["O1"]].to_csv(o1_file, index=False)
    df[["O2"]].to_csv(o2_file, index=False)

    print(f"[✓] Disimpan: {o1_file} dan {o2_file}")
