"""Runs a capture from tools/record_serial.py through the pipeline and reports the result.

    python -m ml.eval_capture data/captures/rest_01.csv
"""
import sys
from collections import Counter

import numpy as np

from .common import rf_features
from .core import CLASSES, GOOD_SQ, READY, hrv, rf_predict, rhythm_rule, run_core
from .train_cnn import TFLITE, Int8Model


def capture_hrv(beats, classes):
    """Apply the live device's interval selection, retaining rejected-beat gaps.

    `classes` contains one post-rule prediction per classifiable beat. The device
    excludes PVC/Other neighbours; AFib and bundle-branch classes remain eligible.
    """
    ready = (beats["flags"] & (READY | GOOD_SQ)) == (READY | GOOD_SQ)
    if len(classes) != int(ready.sum()):
        raise ValueError("Expected one class per classifiable beat")
    previous, index, intervals = -1, 0, []
    for beat, usable in zip(beats, ready):
        if not usable:
            previous = -1
            continue
        current = int(classes[index])
        index += 1
        if previous >= 0 and previous not in (1, 5) and current not in (1, 5):
            intervals.append(beat["rr_pre_ms"])
        previous = current
    return hrv(intervals)


def main():
    data = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 3 or len(data) < 125 or not np.isfinite(data).all():
        raise SystemExit("Capture must contain at least 125 finite rows: raw_mv,filtered_mv,leads_off")
    beats, sq, _ = run_core(data[:, 0], leads_off=data[:, 2] > 0)
    ok = beats[(beats["flags"] & (READY | GOOD_SQ)) == (READY | GOOD_SQ)]
    print(f"Duration {len(data) / 250:.0f} s, beats detected {len(beats)}, analysed {len(ok)}")
    print(f"Good signal {sq['good'].mean() * 100:.0f}% of the time, median QRS {np.median(sq['amp_mv']):.2f} mV, "
          f"median SNR {np.median(sq['snr_db']):.1f} dB, leads off {sq['leads_off'].mean() * 100:.0f}%")
    if len(ok) == 0:
        print("No beats passed quality checks. Check electrode contact and AD8232_GAIN.")
        return
    cnn = rhythm_rule(Int8Model(TFLITE).predict(ok["wave"], ok["ctx"])[0], ok["ctx"])
    rf = rhythm_rule(rf_predict(rf_features({"ctx": ok["ctx"], "morph": ok["morph"]})), ok["ctx"])
    print("CNN:", {CLASSES[k]: v for k, v in Counter(cnn.tolist()).items()})
    print("RF: ", {CLASSES[k]: v for k, v in Counter(rf.tolist()).items()})
    print("HRV:", {k: round(v, 1) for k, v in capture_hrv(beats, cnn).items()})


if __name__ == "__main__":
    main()
