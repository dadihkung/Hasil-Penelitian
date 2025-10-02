import pandas as pd
import joblib

# ====== CONFIG ======
csv_path = r"C:\laragon\www\skripsi\EEG FEATURES\new3\kanan_features.csv"
model_path = r"C:\laragon\www\skripsi\model_rf.joblib"
scaler_path = r"C:\laragon\www\skripsi\scaler.joblib"
encoder_path = r"C:\laragon\www\skripsi\label_encoder.joblib"

# ====== LOAD MODEL + TOOLS ======
print("🔎 Loading model, scaler, and label encoder...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# ====== LOAD CSV ======
df = pd.read_csv(csv_path)

# Urutkan fitur sesuai training
if hasattr(scaler, "feature_names_in_"):
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
elif hasattr(model, "feature_names_in_"):
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
else:
    raise ValueError("Tidak bisa mendeteksi daftar fitur dari model/scaler.")

# ====== SCALE & PREDICT ======
X_scaled = scaler.transform(df)
predictions = model.predict(X_scaled)
predicted_labels = label_encoder.inverse_transform(predictions)

# ====== SIMPAN KE CSV ======
df['predicted_frequency'] = predicted_labels
output_csv = "prediksi_frequency.csv"
df.to_csv(output_csv, index=False)

print(f"✅ Hasil prediksi frekuensi disimpan di {output_csv}")
