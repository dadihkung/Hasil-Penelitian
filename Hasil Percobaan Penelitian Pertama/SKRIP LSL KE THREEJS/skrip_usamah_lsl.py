import time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from pylsl import StreamInlet, resolve_byprop
from pynput.keyboard import Controller, Key

# ====== LOAD TRAINED MODEL & SCALER ======
model = joblib.load("model_rf.joblib")          # Or "model_xgb.joblib"
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# ====== KEYBOARD CONTROL SETUP ======
keyboard = Controller()

command_to_key = {
    "forward": 'w',
    "backward": 's',
    "left": 'a',
    "right": 'd',
    "up": Key.up,
    "down": Key.down,
    "takeoff": Key.space,
    "land": 'l'
}

# ====== FEATURE EXTRACTION FUNCTION ======
def extract_features_windowed(df, sfreq=250):
    row = {}
    for ch in df.columns:
        sig = df[ch].values
        row[f'{ch}_mean'] = np.mean(sig)
        row[f'{ch}_std'] = np.std(sig)
        row[f'{ch}_var'] = np.var(sig)
        row[f'{ch}_skew'] = skew(sig)
        row[f'{ch}_kurtosis'] = kurtosis(sig)
    return row

# ====== CONNECT TO LSL STREAM ======
print("Looking for an EEG stream on the network...")
streams = resolve_byprop('type', 'EEG')
if not streams:
    print("No EEG stream found. Exiting.")
    exit()

inlet = StreamInlet(streams[0])
sfreq = int(inlet.info().nominal_srate())
print(f"Connected to LSL EEG stream at {sfreq} Hz")

buffer_seconds = 1
buffer_samples = int(sfreq * buffer_seconds)
channels = inlet.info().channel_count()
print(f"Expecting {channels} channels")

# Buffer to store EEG samples before feature extraction
data_buffer = np.empty((0, channels))

print("Starting real-time EEG classification... Press Ctrl+C to stop.")

try:
    while True:
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample:
            data_buffer = np.vstack([data_buffer, sample])

        # When buffer is filled for 1 second, extract features and classify
        if data_buffer.shape[0] >= buffer_samples:
            window_data = data_buffer[-buffer_samples:, :]
            ch_names = [f'EXG{i}' for i in range(channels)]
            df_eeg = pd.DataFrame(window_data, columns=ch_names)

            # Extract features
            feats_dict = extract_features_windowed(df_eeg, sfreq)
            feats_df = pd.DataFrame([feats_dict])

            # Scale & predict
            feats_scaled = scaler.transform(feats_df)
            pred_class = model.predict(feats_scaled)[0]
            command = label_encoder.inverse_transform([pred_class])[0]

            print(f"🛫 Predicted Command: {command}")

            # Send key press if mapped
            if command in command_to_key:
                key_to_press = command_to_key[command]
                keyboard.press(key_to_press)
                keyboard.release(key_to_press)

            # Clear buffer after processing
            data_buffer = np.empty((0, channels))

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
