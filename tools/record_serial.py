"""Records the ESP32's raw ECG stream to CSV for Objective 10 (experimental signals).

    python tools/record_serial.py COM5 --seconds 120 --out data/captures/rest_01.csv

Columns: raw_mv, filtered_mv, leads_off. Sent at 250 Hz, so use the firmware's 921600 baud.
"""
import argparse
from pathlib import Path

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--out", default="data/captures/capture.csv")
    a = ap.parse_args()
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    want = a.seconds * 250
    got = 0
    with serial.Serial(a.port, 921600, timeout=2) as port, open(path, "w") as f:
        f.write("raw_mv,filtered_mv,leads_off\n")
        port.write(b"r")  # Switch the firmware to the raw stream
        try:
            while got < want:
                line = port.readline().decode("ascii", "ignore").strip()
                if not line.startswith("R,"):
                    continue
                f.write(line[2:] + "\n")
                got += 1
                if got % 2500 == 0:
                    print(f"{got / 250:.0f} s")
        finally:
            port.write(b"r")
    print(f"Saved {got} samples to {path}")


if __name__ == "__main__":
    main()
