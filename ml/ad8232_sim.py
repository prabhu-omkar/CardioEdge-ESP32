"""Bench simulation of the analog chain: electrodes -> AD8232 -> ESP32 ADC -> firmware.

Test patients' ECGs are passed through a model of the module's filters, gain, mains
pickup and the 12-bit ADC's noise and non-linearity, then converted back the way
firmware/src/main.cpp does it. This is a model, so it estimates robustness to these
effects. It does not replace testing with real hardware.

    python -m ml.ad8232_sim
"""
import json

import numpy as np
from scipy import signal

from .common import RESULTS, metrics
from .core import FS, GOOD_SQ, READY, rhythm_rule, run_core
from .dataset import DELAY, DS2, ensure_mitdb, load_mit, match_beats
from .eval_qrs import score_record
from .train_cnn import TFLITE, Int8Model

GAIN = 1100.0
FULL_SCALE_MV = 3100.0
OVERSAMPLE = 4


def simulate(mv, seed=0, hum_uv=30.0, adc_noise_lsb=6.0, inl_lsb=10.0, gain_error=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(len(mv)) / FS
    x = mv + hum_uv / 1000 * np.sin(2 * np.pi * 50 * t + rng.random() * 6.28)
    # Module pass band of roughly 0.5 to 40 Hz, which AD8232 boards set with discrete parts
    x = signal.sosfilt(signal.butter(1, 0.5, "hp", fs=FS, output="sos"), x)
    x = signal.sosfilt(signal.butter(2, 40, "lp", fs=FS, output="sos"), x)
    out_mv = 1500.0 + x * GAIN * gain_error
    code = out_mv / FULL_SCALE_MV * 4095
    code += inl_lsb * np.sin(np.pi * np.clip(code, 0, 4095) / 4095)  # Bowed ADC transfer curve
    code += rng.normal(0, adc_noise_lsb / np.sqrt(OVERSAMPLE), len(code))
    code = np.clip(np.round(code * OVERSAMPLE) / OVERSAMPLE, 0, 4095)
    return (code * FULL_SCALE_MV / 4095 / GAIN).astype(np.float32)


CONDITIONS = {
    "Ideal (no front end)": None,
    "Typical AD8232 + ESP32": {},
    "Noisy ADC, strong mains pickup": {"hum_uv": 100.0, "adc_noise_lsb": 12.0, "inl_lsb": 20.0},
    "Gain 30% lower than assumed": {"gain_error": 0.7},
}


def main():
    cnn = Int8Model(TFLITE)
    base = ensure_mitdb()
    data = [(r, *load_mit(r, base)) for r in DS2]
    res = {}
    for name, kw in CONDITIONS.items():
        ys, ps, tp, fp, fn, kept, total = [], [], 0, 0, 0, 0, 0
        for rec, sig, idx, lab in data:
            s = sig if kw is None else simulate(sig, seed=rec, **kw)
            b = run_core(s)[0]
            a, c, d = score_record(b["r_index"] - DELAY, idx, 5 * FS)
            tp += a; fp += c; fn += d
            total += len(b)
            b = b[(b["flags"] & (READY | GOOD_SQ)) == (READY | GOOD_SQ)]
            kept += len(b)
            m = match_beats(b["r_index"] - DELAY, idx)
            y = np.where(m >= 0, lab[np.maximum(m, 0)], -1)
            p = rhythm_rule(cnn.predict(b["wave"], b["ctx"])[0], b["ctx"])
            ok = y >= 0
            ys.append(y[ok]); ps.append(p[ok])
        mt = metrics(np.concatenate(ys), np.concatenate(ps))
        res[name] = {"sensitivity": tp / (tp + fn), "ppv": tp / (tp + fp), "beats_analysed": kept / total,
                     "accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"]}
        print(name, res[name], flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ad8232_sim.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
