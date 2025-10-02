#!/usr/bin/env python3
"""
Single-trial SSVEP Recording with Cyton (BrainFlow)

- Fullscreen display
- All boxes blink simultaneously
- User specifies which box the subject is focusing on for filename
- Countdown 5s, then boxes blink for --duration
- Spacebar starts the session
- Inserts markers at stim START and END
- Saves raw CSV + META JSON in organized folder
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-trial SSVEP Recorder")
    p.add_argument("--serial-port", default="COM16", help="Serial port for Cyton (e.g. COM3, /dev/ttyUSB0)")
    p.add_argument("--duration", type=float, default=10.0, help="Stimulus duration (seconds)")
    p.add_argument("--subject", required=True, help="Subject/session name")
    p.add_argument("--target", default="tengah",
                   choices=["kiri_atas", "kanan_atas", "kiri_bawah", "kanan_bawah", "tengah"],
                   help="Which box the subject is focusing on (for filename)")
    return p.parse_args()


# ---------------- Board ----------------
class CytonSession:
    def __init__(self, serial_port):
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
            self.board.start_stream(45000)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.connected = True
        except Exception as e:
            print("[WARN] Could not connect to board:", e)
            self.connected = False
            self.sampling_rate = 250.0

    def insert_marker(self, val):
        if self.connected and self.board is not None:
            try:
                self.board.insert_marker(val)
            except Exception:
                pass

    def stop_and_get_data(self):
        if not self.connected or self.board is None:
            return np.array([])
        try:
            self.board.stop_stream()
            data = self.board.get_board_data()
            self.board.release_session()
            return data
        except Exception:
            return np.array([])


# ---------------- File Helpers ----------------
def save_csv(data, board_id, path: Path):
    """Save BrainFlow data with proper headers for all channels"""
    try:
        desc = BoardShim.get_board_descr(board_id)
        all_channels = desc["all_channels"]
        labels = []

        for ch in all_channels:
            if ch in desc.get("eeg_channels", []):
                labels.append(f"EEG_{ch}")
            elif ch in desc.get("accel_channels", []):
                labels.append(f"ACC_{ch}")
            elif ch in desc.get("analog_channels", []):
                labels.append(f"ANALOG_{ch}")
            elif ch == desc.get("marker_channel"):
                labels.append("MARKER")
            elif ch == desc.get("timestamp_channel"):
                labels.append("TIMESTAMP")
            else:
                labels.append(f"CH_{ch}")
    except Exception:
        labels = [f"CH_{i}" for i in range(data.shape[0] if data.size > 0 else 8)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(labels) + "\n")
        if data.size > 0:
            np.savetxt(f, data.T, delimiter=",", fmt="%.6f")
    return path


# ---------------- Pygame ----------------
class BlinkApp:
    BG = (0, 0, 0)
    TEXT = (255, 255, 255)
    ACTIVE = (255, 0, 0)

    def __init__(self, duration):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("SSVEP Single Trial")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 80)
        self.duration = duration
        self.running = True

        rw, rh = int(self.width * 0.25), int(self.height * 0.2)
        self.boxes = {
            "kiri_atas": {"freq": 8.0, "rect": (0, 0, rw, rh)},
            "kanan_atas": {"freq": 9.0, "rect": (self.width - rw, 0, rw, rh)},
            "kiri_bawah": {"freq": 10.0, "rect": (0, self.height - rh, rw, rh)},
            "kanan_bawah": {"freq": 12.0, "rect": (self.width - rw, self.height - rh, rw, rh)},
            "tengah": {"freq": 14.0, "rect": ((self.width - rw) // 2, (self.height - rh) // 2, rw, rh)},
        }

    def run_intro(self):
        while self.running:
            self.screen.fill(self.BG)
            txt = self.font.render("PRESS SPACE TO START", True, self.TEXT)
            self.screen.blit(txt, txt.get_rect(center=self.screen.get_rect().center))
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        return True
                    elif e.key == pygame.K_ESCAPE:
                        self.running = False
            self.clock.tick(60)
        return False

    def run_countdown(self, secs=5):
        for t in range(secs, 0, -1):
            self.screen.fill(self.BG)
            txt = self.font.render(f"Get Ready: {t}", True, self.TEXT)
            self.screen.blit(txt, txt.get_rect(center=self.screen.get_rect().center))
            pygame.display.flip()
            pygame.time.wait(1000)

    def run_trial(self, cyton: CytonSession):
        intervals = {name: 1 / (2 * info["freq"]) for name, info in self.boxes.items()}
        last_toggle = {name: time.perf_counter() for name in self.boxes}
        visible = {name: True for name in self.boxes}

        self.run_countdown(5)
        start = time.perf_counter()
        cyton.insert_marker(1.0)

        while self.running and (time.perf_counter() - start) < self.duration:
            now = time.perf_counter()
            for name in self.boxes:
                if now - last_toggle[name] >= intervals[name]:
                    visible[name] = not visible[name]
                    last_toggle[name] = now

            self.screen.fill(self.BG)
            for name, info in self.boxes.items():
                color = self.ACTIVE if visible[name] else self.BG
                pygame.draw.rect(self.screen, color, pygame.Rect(*info["rect"]))
            pygame.display.flip()

            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self.running = False
            self.clock.tick(120)

        cyton.insert_marker(2.0)


# ---------------- Main ----------------
def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq = BlinkApp(0).boxes[args.target]["freq"]

    base_dir = Path("data_mentah2") / args.subject
    base_dir.mkdir(parents=True, exist_ok=True)
    out_csv = base_dir / f"{args.subject}_{freq}Hz_{args.target}_{ts}.csv"
    out_meta = out_csv.with_suffix(".meta.json")

    app = BlinkApp(duration=args.duration)
    cyton = CytonSession(args.serial_port)

    try:
        if not app.run_intro():
            return 0
        cyton.start()
        print(f"[INFO] Running trial for subject: {args.subject}, focus: {args.target} ({freq} Hz)")
        app.run_trial(cyton)
        data = cyton.stop_and_get_data()
        save_csv(data, cyton.board_id, out_csv)

        meta = {
            "board_id": cyton.board_id,
            "connected": cyton.connected,
            "fs": cyton.sampling_rate,
            "duration": args.duration,
            "target": args.target,
            "freq": freq,
            "file": str(out_csv),
        }
        out_meta.write_text(json.dumps(meta, indent=2))
        print(f"[✓] Saved CSV: {out_csv}")
        print(f"[✓] Saved META: {out_meta}")
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
