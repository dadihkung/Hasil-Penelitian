#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def plot_fft_two_channels(fft_csv_path_ch7, fft_csv_path_ch8, blink_hz=None):
    """
    Plot FFT / PSD for EEG channels 7 and 8 on the same figure.
    
    Parameters:
    - fft_csv_path_ch7: path to FFT CSV of channel 7
    - fft_csv_path_ch8: path to FFT CSV of channel 8
    - blink_hz: optional stimulus frequency to highlight
    """
    # Load channel 7
    df7 = pd.read_csv(fft_csv_path_ch7)
    freqs7 = df7["freq_hz"].values
    power7 = df7["power"].values

    # Load channel 8
    df8 = pd.read_csv(fft_csv_path_ch8)
    freqs8 = df8["freq_hz"].values
    power8 = df8["power"].values

    plt.figure(figsize=(12, 6))
    plt.plot(freqs7, power7, label="Channel 7 (O1)", color="blue")
    plt.plot(freqs8, power8, label="Channel 8 (O2)", color="green")

    # Highlight stimulus frequency
    if blink_hz is not None:
        plt.axvline(blink_hz, color="red", linestyle="--", label=f"{blink_hz} Hz Stimulus")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("FFT / PSD of EEG Channels 7 & 8")
    plt.xlim(0, 60)  # EEG interesting range <60 Hz
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace these paths with your actual FFT CSVs
    fft_ch7_path = r"C:\laragon\www\skripsi\data_mentah_brainflow\usamah2_recording_20250830_100733_fft_ch7.csv"
    fft_ch8_path = r"C:\laragon\www\skripsi\data_mentah_brainflow\usamah2_recording_20250830_100733_fft_ch8.csv"

    # Stimulus frequency (Hz)
    stim_freq = 10

    plot_fft_two_channels(fft_ch7_path, fft_ch8_path, blink_hz=stim_freq)
