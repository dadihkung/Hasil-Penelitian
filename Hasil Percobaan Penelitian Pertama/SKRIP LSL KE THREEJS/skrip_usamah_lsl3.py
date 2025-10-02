import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop
import joblib
from scipy.stats import skew, kurtosis
import time
import pyautogui
import win32gui

# -----------------------------
# 🔧 Load model, scaler, encoder
# -----------------------------
model = joblib.load(r'C:\laragon\www\skripsi\model_rf.joblib')
scaler = joblib.load(r'C:\laragon\www\skripsi\scaler.joblib')
label_encoder = joblib.load(r'C:\laragon\www\skripsi\label_encoder.joblib')

# -----------------------------
# 🔍 Cari stream EEG
# -----------------------------
print("🔍 Mencari stream EEG...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("Tidak ada EEG stream ditemukan.")
inlet = StreamInlet(streams[0])
print("✅ EEG stream ditemukan!")

# -----------------------------
# ⏱ Parameter
# -----------------------------
sfreq = 50  # misal 50Hz, ubah sesuai device
window_size = 250  # 5 detik x 50Hz
channels_to_use = [0, 1, 2, 3, 4]  # channel 1 & 2 (index 0 & 1)
target_window_title = "Drone Simulator"

# -----------------------------
# 🔢 Fungsi ekstraksi fitur
# -----------------------------
def extract_features(data):
    features = []
    for ch_data in data.T:
        features.extend([
            np.mean(ch_data),
            np.std(ch_data),
            np.var(ch_data),
            skew(ch_data),
            kurtosis(ch_data)
        ])
    return np.array(features).reshape(1, -1)

# -----------------------------
# 🎮 Fungsi kirim keyboard
# -----------------------------
key_mapping = {
    'w': 'w',
    'a': 'a',
    's': 's',
    'd': 'd',
    'q': 'q',
    'Shift': 'shift'
}

def send_key(label):
    # cek apakah label ada mapping-nya
    key = key_mapping.get(label)
    if not key:
        return
    # aktifkan window target
    def enumHandler(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            if target_window_title in win32gui.GetWindowText(hwnd):
                win32gui.SetForegroundWindow(hwnd)
    win32gui.EnumWindows(enumHandler, None)
    # kirim key
    pyautogui.press(key)

# -----------------------------
# 🔄 Loop prediksi real-time
# -----------------------------
buffer = []

print("🚀 Mulai prediksi real-time...")
while True:
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample)
    
    # pakai hanya channel 1 & 2
    buffer_ch = np.array(buffer)[:, channels_to_use]

    if len(buffer_ch) >= window_size:
        window_data = buffer_ch[-window_size:]
        features = extract_features(window_data)
        features_scaled = scaler.transform(features)
        pred_label_encoded = model.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_label_encoded])[0]

        # tampilkan prediksi
        print(f"Prediksi: {pred_label}")

        # kirim keyboard
        send_key(pred_label)

        # reset buffer (atau sliding window)
        buffer = buffer[-window_size:]
    
    time.sleep(0.01)  # sedikit delay agar CPU tidak max
