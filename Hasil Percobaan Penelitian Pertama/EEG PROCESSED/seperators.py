import pandas as pd

# Load semicolon-separated CSV
df = pd.read_csv(
    r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_fahri_simdrone\BrainFlow-RAW_simdrone_fahri_0.csv',
    sep=';',
    low_memory=False
)

# Save as tab-separated CSV (but keep .csv extension)
df.to_csv(r'C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_fahri_simdrone\BrainFlow-RAW_simdrone_fahri_0.csv', sep='\t', index=False)
