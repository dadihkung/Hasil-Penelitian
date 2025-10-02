#!/usr/bin/env python3
"""
Single-file app: OpenBCI Cyton (8-ch) + BrainFlow + 10 Hz blinking stimulus (10 s)

- Opening screen says "PRESS SPACE TO START"
- On Space: connects to Cyton (BrainFlow), starts stream & recording (or continues without board if not available)
- Shows one centered square blinking at 10 Hz for ~10 seconds
- Inserts markers at stimulus START and END (if board available)
- Stops streaming, saves CSV, shows a completion message
- Always writes a sidecar META JSON (fs, blink_hz, duration, stim start/end indices & times)
- Exports the stimulus epoch and FFT/PSD for EEG channels 7 and 8 to CSV

Notes:
- This script is written to be tolerant of older BrainFlow installs (no WindowFunctions import).
- If markers are not captured, we fallback to taking the last `duration` seconds of data and
  write the recorded wall-clock stimulus start/end into the META JSON.
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ----------------------- CLI Args -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenBCI Cyton 10 Hz Blink (BrainFlow) – Single File")
    p.add_argument("--serial-port", required=False, default="COM16",
                   help="Serial port for Cyton (e.g., COM3, /dev/ttyUSB0). Optional for testing without hardware.")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Stimulus duration in seconds (default: 10.0)")
    p.add_argument("--blink-hz", type=float, default=9.0,
                   help="Blink frequency in Hz (default: 10.0)")
    p.add_argument("--outfile", default=None,
                   help="Optional output CSV path; if omitted a timestamped file is created.")
    p.add_argument("--width", type=int, default=900, help="Window width (default: 900)")
    p.add_argument("--height", type=int, default=700, help="Window height (default: 700)")
    p.add_argument("--welch", action="store_true",
                   help="Use Welch PSD (numpy windowed average) instead of simple rFFT.")
    return p.parse_args()


# ----------------------- BrainFlow Helpers -----------------------
class CytonSession:
    def __init__(self, serial_port: str):
        self.board_id = BoardIds.CYTON_BOARD.value
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board = None
        self.sampling_rate = None
        self.connected = False

    def start(self):
        try:
            self.board = BoardShim(self.board_id, self.params)
            self.board.prepare_session()
            self.board.start_stream(45000, streamer_params="")
            # get sampling rate even if start_stream failed - wrap in try
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.connected = True
        except Exception as e:
            print("[WARN] Could not connect to board, running without hardware:", e)
            self.board = None
            self.connected = False
            try:
                self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            except Exception:
                self.sampling_rate = 250.0

    def insert_marker(self, value: float):
        if self.board is not None:
            try:
                self.board.insert_marker(value)
            except Exception:
                pass

    def stop_and_get_data(self):
        # return numpy array shaped (channels x samples) or empty array
        if self.board is None:
            return np.array([])
        try:
            self.board.stop_stream()
            data = self.board.get_board_data()
            self.board.release_session()
            return data if data is not None else np.array([])
        except Exception:
            return np.array([])


def save_brainflow_csv(data: np.ndarray, board_id: int, out_path: Path):
    """
    Save raw board data to CSV. If no data, still create header-only file.
    Uses BoardShim.get_board_descr when available to get column names.
    """
    try:
        descr = BoardShim.get_board_descr(board_id)
        # some brainflow versions use 'channels'/'num_channels' keys; fallback to channels
        if "channels" in descr and "num_channels" in descr:
            channel_labels = [descr["channels"].get(idx, f"channel_{idx}") for idx in range(descr["num_channels"])]
        else:
            # fallback to channels if present
            channel_count = descr.get("num_r", data.shape[0] if data is not None else 0)
            channel_labels = [descr.get("channels", {}).get(idx, f"channel_{idx}") for idx in range(channel_count)]
    except Exception:
        channel_labels = [f"channel_{i}" for i in range(max(1, (data.shape[0] if data is not None else 16)))]

    header = ",".join(channel_labels)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        if data is not None and data.size > 0:
            # BrainFlow data shape is (channels, samples) -> write samples as channels
            np.savetxt(f, data.T, delimiter=",", fmt="%.10f")
    return out_path


# ----------------------- FFT / Export Helpers -----------------------
def _find_epoch_indices_from_markers(data: np.ndarray, board_id: int):
    """
    Return (start_idx, end_idx) in samples using BrainFlow marker channel:
    - start: first sample where marker==1.0
    - end: first sample where marker==2.0 after start
    Returns (None, None) if markers not found.
    """
    if data is None or data.size == 0:
        return None, None
    try:
        marker_channel = BoardShim.get_marker_channel(board_id)
    except Exception:
        return None, None
    if marker_channel is None or marker_channel < 0 or marker_channel >= data.shape[0]:
        return None, None

    markers = data[marker_channel, :]
    start_candidates = np.where(np.isclose(markers, 1.0))[0]
    if start_candidates.size == 0:
        return None, None
    start_idx = int(start_candidates[0])

    end_candidates = np.where(np.isclose(markers[start_idx:], 2.0))[0]
    if end_candidates.size == 0:
        return None, None
    end_idx = int(start_idx + end_candidates[0])

    if end_idx <= start_idx:
        return None, None
    return start_idx, end_idx


def _rfft_power(x: np.ndarray, fs: float):
    """Hanning + rFFT → power spectrum"""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([]), np.array([])
    x = x - np.mean(x)
    win = np.hanning(len(x))
    xw = x * win
    spec = np.fft.rfft(xw)
    power = (np.abs(spec) ** 2) / np.sum(win ** 2)
    freqs = np.fft.rfftfreq(len(xw), d=1.0 / fs)
    return freqs, power


def export_epoch_and_fft(
    data: np.ndarray,
    board_id: int,
    fs: float,
    out_path: Path,
    blink_hz: float,
    use_welch: bool,
    stim_duration_s: float,
):
    """
    Exports epoch & FFT CSVs for EEG channels 7 & 8 (if available).
    If markers missing, fallback to using the last `stim_duration_s` seconds of data.
    Returns dict mapping channel_index -> {'epoch': path, 'fft': path}
    """
    base = out_path.with_suffix("")
    results = {}

    try:
        eeg_chs = BoardShim.get_eeg_channels(board_id)
        # eeg_chs is a list of channel indices that correspond to EEG channels
    except Exception:
        eeg_chs = list(range(min(8, data.shape[0] if data is not None else 8)))

    # Select channel channels corresponding to EEG channel 7 & 8
    # CORRECT (last two channels)
    if len(eeg_chs) >= 8:
        target_indices = [eeg_chs[6], eeg_chs[7]]  # zero-based: ch7 & ch8
        user_friendly_names = [7, 8]
    else:
        # fallback: last two available EEG channels
        target_indices = eeg_chs[-2:] if len(eeg_chs) >= 2 else []
        user_friendly_names = list(range(len(eeg_chs) - len(target_indices) + 1, len(eeg_chs) + 1))


    # Find epoch using markers (preferred)
    start_idx, end_idx = _find_epoch_indices_from_markers(data, board_id)

    # If markers missing, fallback: take last `stim_duration_s` seconds
    if start_idx is None or end_idx is None:
        n_samples = data.shape[1] if data is not None and data.size > 0 else 0
        seg_len = int(round(stim_duration_s * fs))
        if n_samples >= seg_len and seg_len > 0:
            start_idx = max(0, n_samples - seg_len)
            end_idx = n_samples
        else:
            # take all available
            start_idx = 0
            end_idx = n_samples

    # Export each chosen channel
    for ch_channel, friendly in zip(target_indices, user_friendly_names):
        # guard if ch_channel out of bounds
        if data is None or data.size == 0 or ch_channel < 0 or ch_channel >= data.shape[0]:
            # create header-only files
            epoch_file = Path(str(base) + f"_epoch_ch{friendly}.csv")
            fft_file = Path(str(base) + f"_fft_ch{friendly}.csv")
            epoch_file.write_text("sample,value\n", encoding="utf-8")
            fft_file.write_text("freq_hz,power\n", encoding="utf-8")
            results[friendly] = {"epoch": str(epoch_file), "fft": str(fft_file)}
            continue

        sig = data[ch_channel, start_idx:end_idx].astype(float)

        epoch_file = Path(str(base) + f"_epoch_ch{friendly}.csv")
        fft_file = Path(str(base) + f"_fft_ch{friendly}.csv")

        # write epoch
        with epoch_file.open("w", encoding="utf-8") as f:
            f.write("sample,value\n")
            for i, v in enumerate(sig):
                f.write(f"{i},{v:.10f}\n")

        # compute PSD
        if use_welch:
            # simple windowed average Welch-like using rFFT (no scipy)
            seg_len = int(min(len(sig), max(int(fs), 256)))
            step = seg_len // 2 if seg_len >= 4 else seg_len
            if seg_len < 4 or len(sig) < 4:
                freqs, power = _rfft_power(sig, fs)
            else:
                psds = []
                # sliding windows
                for offset in range(0, max(1, len(sig) - seg_len + 1), step):
                    _, p = _rfft_power(sig[offset:offset + seg_len], fs)
                    psds.append(p)
                if len(psds) == 0:
                    freqs, power = _rfft_power(sig, fs)
                else:
                    power = np.mean(np.vstack(psds), axis=0)
                    freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
        else:
            freqs, power = _rfft_power(sig, fs)

        # write FFT CSV
        with fft_file.open("w", encoding="utf-8") as f:
            f.write("freq_hz,power\n")
            # guard empty arrays
            if freqs.size == 0:
                pass
            else:
                for fr, pw in zip(freqs, power):
                    f.write(f"{fr:.6f},{pw:.10f}\n")

        results[friendly] = {"epoch": str(epoch_file), "fft": str(fft_file)}

    return results, (start_idx, end_idx)


def write_meta_json(out_path: Path, meta: dict):
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(meta_path)


# ----------------------- Graphics (Pygame) -----------------------
class BlinkApp:
    BG = (255, 255, 255)
    TEXT = (0, 0, 0)
    SQUARE = (0, 0, 0)

    def __init__(self, width=900, height=700, blink_hz=9.0, duration=10.0):
        pygame.init()
        pygame.display.set_caption("8 Hz Blink – Cyton Recorder")
        # use fullscreen to match your previous code behavior; can be changed
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)
        self.small_font = pygame.font.SysFont(None, 28)
        self.running = True
        self.blink_hz = blink_hz
        self.duration = duration
        self.square_size = int(min(self.width, self.height) * 0.5)
        self.square_visible = False
        self._last_toggle = 0.0
        self.toggle_interval = 1.0 / (2.0 * self.blink_hz)
        self.phase = "intro"
        self.result_text = None
        # store wall-clock times for fallback
        self.stim_wall_start = None
        self.stim_wall_end = None

    def draw_intro(self):
        self.screen.fill(self.BG)
        title = self.font.render("PRESS SPACE TO START", True, self.TEXT)
        hint = self.small_font.render("EEG recording will begin (board optional).", True, self.TEXT)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, self.height // 2 - 20)))
        self.screen.blit(hint, hint.get_rect(center=(self.width // 2, self.height // 2 + 30)))
        pygame.display.flip()

    def draw_stim(self):
        self.screen.fill(self.BG)
        if self.square_visible:
            x = (self.width - self.square_size) // 2
            y = (self.height - self.square_size) // 2
            pygame.draw.rect(self.screen, self.SQUARE, pygame.Rect(x, y, self.square_size, self.square_size))
        pygame.display.flip()

    def draw_done(self):
        self.screen.fill(self.BG)
        msg = self.font.render("Recording complete.", True, self.TEXT)
        self.screen.blit(msg, msg.get_rect(center=(self.width // 2, self.height // 2 - 40)))
        if self.result_text:
            y = self.height // 2 + 10
            for line in self.result_text.split("\n"):
                surf = self.small_font.render(line, True, self.TEXT)
                self.screen.blit(surf, surf.get_rect(center=(self.width // 2, y)))
                y += 30
        quit_hint = self.small_font.render("Press ESC or close window to exit.", True, self.TEXT)
        self.screen.blit(quit_hint, quit_hint.get_rect(center=(self.width // 2, self.height - 40)))
        pygame.display.flip()

    def run_intro(self):
        while self.running and self.phase == "intro":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        return True
            self.draw_intro()
            self.clock.tick(60)
        return False

    def run_stim(self, cyton: CytonSession):
        """
        Runs the stimulus; returns (stim_wall_start, stim_wall_end).
        Also inserts markers into board when possible.
        """
        start_time = time.perf_counter()
        self._last_toggle = start_time
        self.square_visible = True

        # record wall-clock start
        self.stim_wall_start = time.time()
        cyton.insert_marker(1.0)

        while self.running and (time.perf_counter() - start_time) < self.duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            now = time.perf_counter()
            if (now - self._last_toggle) >= self.toggle_interval:
                self.square_visible = not self.square_visible
                self._last_toggle = now

            self.draw_stim()
            self.clock.tick(120)

        # stimulus end
        self.square_visible = False
        self.draw_stim()
        cyton.insert_marker(2.0)
        self.stim_wall_end = time.time()
        return (self.stim_wall_start, self.stim_wall_end)

    def run_done(self, text: str):
        self.phase = "done"
        self.result_text = text
        while self.running and self.phase == "done":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
            self.draw_done()
            self.clock.tick(60)

    def quit(self):
        pygame.quit()


# ----------------------- Main -----------------------
def main():
    args = parse_args()

    # Prepare output path
    if args.outfile:
        out_path = Path(args.outfile).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(r"C:\laragon\www\skripsi\data_mentah_brainflow") / f"usamah7_recording_{ts}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    app = BlinkApp(width=args.width, height=args.height, blink_hz=args.blink_hz, duration=args.duration)
    cyton = CytonSession(serial_port=args.serial_port)

    try:
        if not app.run_intro():
            app.quit()
            return 0

        # Start board (attempt). If fails, script continues but sampling_rate set in CytonSession.start()
        cyton.start()

        # Set phase and run stimulus — capture wall-clock stim times for fallback
        app.phase = "stim"
        stim_wall_start, stim_wall_end = app.run_stim(cyton)

        # Stop & grab raw board data
        data = cyton.stop_and_get_data()
        save_brainflow_csv(data, cyton.board_id, out_path)

        # sampling rate fallback if None
        fs = float(cyton.sampling_rate) if cyton.sampling_rate else 250.0

        # attempt to find indices from marker channel
        start_idx, end_idx = _find_epoch_indices_from_markers(data, cyton.board_id)

        # If markers were missing, export_epoch_and_fft will fallback to last N seconds.
        results, (epoch_start_idx, epoch_end_idx) = export_epoch_and_fft(
            data=data,
            board_id=cyton.board_id,
            fs=fs,
            out_path=out_path,
            blink_hz=args.blink_hz,
            use_welch=args.welch,
            stim_duration_s=args.duration,
        )

        # Try to map markers to timestamp channel for meta times; if not found, use wall-clock times
        stim_start_time = None
        stim_end_time = None
        try:
            ts_channel = BoardShim.get_timestamp_channel(cyton.board_id)
            if data is not None and data.size > 0 and ts_channel is not None and 0 <= ts_channel < data.shape[0]:
                ts_vec = data[ts_channel, :]
                # If ts_vec looks like epoch (values >> 1e6), it's likely unix epoch seconds
                if np.nanmax(ts_vec) > 1e6:
                    # when markers present, map indices to timestamps; else pick timestamps at epoch indices
                    if start_idx is not None and end_idx is not None:
                        stim_start_time = float(ts_vec[start_idx])
                        stim_end_time = float(ts_vec[end_idx - 1]) if end_idx - 1 < ts_vec.size else float(ts_vec[-1])
                    else:
                        # use epoch indices (fallback)
                        n = ts_vec.size
                        sidx = max(0, n - int(round(args.duration * fs)))
                        stim_start_time = float(ts_vec[sidx]) if sidx < n else None
                        stim_end_time = float(ts_vec[-1]) if n > 0 else None
                else:
                    # ts_vec likely relative seconds (small numbers). We'll not attempt mapping to wall-clock.
                    # As markers mapping is unreliable here, prefer wall-clock fallback
                    stim_start_time = None
                    stim_end_time = None
        except Exception:
            pass

        # If we couldn't get times from timestamp channel, use the wall-clock times recorded during run_stim
        if stim_start_time is None:
            stim_start_time = float(stim_wall_start) if stim_wall_start is not None else None
        if stim_end_time is None:
            stim_end_time = float(stim_wall_end) if stim_wall_end is not None else None

        # Final start/end indices: prefer marker-derived if available, else epoch_start_idx/epoch_end_idx
        final_start_idx = int(start_idx) if start_idx is not None else int(epoch_start_idx) if epoch_start_idx is not None else None
        final_end_idx = int(end_idx) if end_idx is not None else int(epoch_end_idx) if epoch_end_idx is not None else None

        meta = {
            "board_id": int(cyton.board_id),
            "connected": bool(cyton.connected),
            "sampling_rate_hz": float(fs),
            "blink_hz": float(args.blink_hz),
            "duration_s": float(args.duration),
            "stimulus": {
                "start_idx": final_start_idx,
                "end_idx": final_end_idx,
                "start_time": stim_start_time,
                "end_time": stim_end_time
            },
            "files": {
                "raw_csv": str(out_path)
            }
        }

        # attach exported epoch/fft files info
        meta["files"].update(results)

        meta_path = write_meta_json(out_path, meta)

        msg = f"Saved EEG data to:\n{str(out_path)}\nSaved meta to:\n{meta_path}\nEpoch & FFT CSV exported for channels 7 and 8 (if available)."
        if not cyton.connected:
            msg += "\n(Note: No board detected, raw CSV likely contains only headers.)"
        app.run_done(msg)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
    finally:
        try:
            if cyton.board is not None and cyton.board.board is not None:
                try:
                    cyton.board.stop_stream()
                except Exception:
                    pass
                try:
                    cyton.board.release_session()
                except Exception:
                    pass
        except Exception:
            pass
        app.quit()

    return 0


if __name__ == "__main__":
    time.sleep(5)
    sys.exit(main())
