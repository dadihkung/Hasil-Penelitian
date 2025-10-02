import time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes
from pynput.keyboard import Controller, Key

# ====== LOAD TRAINED MODEL & SCALER ======
model = joblib.load("model_rf.joblib")          # Or "model_xgb.joblib"
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# ====== KEYBOARD CONTROL SETUP ======
keyboard = Controller()

# Map your predicted labels to keys used by the simulator
command_to_key = {
    "forward": 'w',
    "backward": 's',
    "left": 'a',
    "right": 'd',
    "up": Key.up,        # example: arrow up
    "down": Key.down,    # example: arrow down
    "takeoff": Key.space,
    "land": 'l'
}

# ====== REAL-TIME FEATURE EXTRACTION ======
def extract_features_windowed(df, sfreq=250, window_sec=1):
    window_size = int(sfreq * window_sec)
    win = df.iloc[-window_size:]
    row = {}
    for ch in df.columns:
        sig = win[ch].values
        row[f'{ch}_mean'] = np.mean(sig)
        row[f'{ch}_std'] = np.std(sig)
        row[f'{ch}_var'] = np.var(sig)
        row[f'{ch}_skew'] = skew(sig)
        row[f'{ch}_kurtosis'] = kurtosis(sig)
    return row

# ====== CONNECT TO CYTON ======
params = BrainFlowInputParams()
params.serial_port = 'COM3'  # ⚠ Change this to your actual Cyton port
board = BoardShim(BoardShim.CYTON_BOARD, params)

sfreq = 250
window_sec = 1

print("📡 Preparing session...")
board.prepare_session()
board.start_stream()

print("✅ Streaming started... Press Ctrl+C to stop.")

try:
    while True:
        data = board.get_board_data()
        if data.shape[1] < sfreq * window_sec:
            time.sleep(0.1)
            continue

        eeg_channels = board.get_eeg_channels(BoardShim.CYTON_BOARD)
        eeg_data = data[eeg_channels, :].T
        ch_names = [f'EXG{i}' for i in range(len(eeg_channels))]
        df_eeg = pd.DataFrame(eeg_data, columns=ch_names)

        # Bandpass & notch filter
        for ch_idx in range(len(eeg_channels)):
            DataFilter.perform_bandpass(df_eeg[ch_names[ch_idx]].values,
                                        sfreq, 1.0, 20.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(df_eeg[ch_names[ch_idx]].values,
                                        sfreq, 50.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

        # Feature extraction
        feats_dict = extract_features_windowed(df_eeg, sfreq, window_sec)
        feats_df = pd.DataFrame([feats_dict])

        # Scaling & prediction
        feats_scaled = scaler.transform(feats_df)
        pred_class = model.predict(feats_scaled)[0]
        command = label_encoder.inverse_transform([pred_class])[0]

        # Output
        print(f"🛫 Predicted Command: {command}")

        # Send key press to simulator if mapping exists
        if command in command_to_key:
            key_to_press = command_to_key[command]
            keyboard.press(key_to_press)
            keyboard.release(key_to_press)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    board.stop_stream()
    board.release_session()
