import pandas as pd
import os
from datetime import datetime

hari = {
    "Monday" : "senin",
    "Tuesday" : "selasa",
    "Wednesday" : "rabu",
    "Thursday" : "kamis",
    "Friday" : "jumat",
    "Saturday" : "sabtu",
    "Sunday" : "minggu",
}

now = datetime.now()
sekarang = f"{hari[now.strftime('%A')]}_{now.strftime('%d')}_{now.strftime('%m')}_{now.strftime('%Y')}"

# Nama file input EEG
input_filename = r"C:\laragon\www\skripsi\data_mentah_brainflow\usamah7_recording_20250830_115027.csv"

# input_filename = f"output_eeg_{sekarang}/usamah_kanan_bawah_raw.csv"
# input_filename = f"output_eeg_{sekarang}/usamah_kiri_atas_raw.csv"
# input_filename = f"output_eeg_{sekarang}/usamah_kiri_bawah_raw.csv"
output_filename = f"output_cut_{sekarang}"
# Ambil nama dasar file (tanpa .csv)
basename = os.path.splitext(os.path.basename(input_filename))[0]

# Baca file EEG
df = pd.read_csv(input_filename)

# Ubah nama kolom menjadi string agar bisa diganti
df.columns = [str(col) for col in df.columns]

# Ganti nama kolom 1 dan 2 menjadi O1 dan O2
df.rename(columns={"channel_7": "O1", "channel_8": "O2"}, inplace=True)

# Ambil hanya kolom O1 dan O2
df_eeg = df[["O1", "O2"]]

# Buat folder output sesuai nama file
output_dir = os.path.join(output_filename, basename)
os.makedirs(output_dir, exist_ok=True)

# Simpan O1 dan O2 tanpa segmentasi
df_eeg[["O1"]].to_csv(os.path.join(output_dir, f"{basename}_O1.csv"), index=False)
df_eeg[["O2"]].to_csv(os.path.join(output_dir, f"{basename}_O2.csv"), index=False)

print(f"Disimpan: {basename}_O1.csv dan {basename}_O2.csv") 

