"""Evaluates signal quality assessment with real recorded noise (Key Improvement).

Noise from the MIT-BIH Noise Stress Test Database (electrode motion, muscle artefact,
baseline wander) is added to ECG records at controlled SNRs, using the database's
own SNR definition. Reports how often the signal is accepted at each noise level.

    python -m ml.sqa_eval            (tuning set: DS1)
    python -m ml.sqa_eval test       (final result: DS2)
"""
import json
import sys

import numpy as np
import wfdb

from .core import FS, ROOT, run_core
from .dataset import DS1, DS2, RAW, ensure_mitdb, load_mit, to_fs

SNRS = [24, 18, 12, 6, 0, -6]
NOISES = {"em": "electrode motion", "ma": "muscle artefact", "bw": "baseline wander"}


def load_noise(name):
    return to_fs(wfdb.rdrecord(str(RAW / "nst" / name)).p_signal[:, 0], 360)


def signal_power(sig, idx):
    """Mean squared peak-to-peak QRS amplitude over 8, as defined for the NST database."""
    amps = []
    for r in idx[(idx > 30) & (idx < len(sig) - 30)][::4]:
        seg = sig[r - 25: r + 25]
        amps.append(seg.max() - seg.min())
    return np.mean(np.square(amps)) / 8


def add_noise(sig, idx, noise, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(noise) - len(sig)) if len(noise) > len(sig) else 0
    n = np.resize(noise[start:], len(sig))
    n = n - np.mean(n)
    gain = np.sqrt(signal_power(sig, idx) / (np.var(n) * 10 ** (snr_db / 10)))
    return sig + gain * n


def main():
    final = len(sys.argv) > 1 and sys.argv[1] == "test"
    records = DS2 if final else DS1
    base = ensure_mitdb()
    noises = {k: load_noise(k) for k in NOISES}
    rows, clean = [], []
    data = [(r, *load_mit(r, base)[:2]) for r in records]
    for rec, sig, idx in data:
        clean.append(run_core(sig)[1]["good"].mean())
    print(f"clean signal accepted: {np.mean(clean) * 100:.1f}%  min record {np.min(clean) * 100:.1f}%", flush=True)
    for name in NOISES:
        for snr in SNRS:
            acc, fails = [], []
            for rec, sig, idx in data:
                sq = run_core(add_noise(sig, idx, noises[name], snr, seed=rec))[1]
                acc.append(sq["good"].mean())
                fails.append([1 - sq[k].mean() for k in ("amp_ok", "snr_ok", "base_ok", "morph_ok")])
            f = np.mean(fails, axis=0)
            rows.append({"noise": name, "snr_db": snr, "accepted": float(np.mean(acc)),
                         "fail_amp": float(f[0]), "fail_snr": float(f[1]), "fail_base": float(f[2]), "fail_morph": float(f[3])})
            print(f"{name} {snr:4d} dB accepted {np.mean(acc) * 100:5.1f}%  fail amp {f[0]:.2f} snr {f[1]:.2f} base {f[2]:.2f} morph {f[3]:.2f}", flush=True)
    if final:
        out = ROOT / "results"
        out.mkdir(exist_ok=True)
        (out / "sqa.json").write_text(json.dumps({"clean_accepted": float(np.mean(clean)), "clean_min_record": float(np.min(clean)), "noise": rows}, indent=1))


if __name__ == "__main__":
    main()
