"""Python binding to the firmware DSP core, plus MIT-BIH style record loading.

The same C++ the ESP32 runs is compiled into a shared library here, so training
data, evaluation and the device all share one implementation.
"""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "firmware" / "lib" / "ecg_core"
BUILD_DIR = ROOT / "build"
DLL = BUILD_DIR / ("ecg_core.dll" if sys.platform == "win32" else "libecg_core.so")

FS = 250
WAVE_PRE, WAVE_LEN = 60, 160
N_CTX, N_MORPH = 13, 32
CLASSES = ["Normal", "PVC", "AFib", "LBBB", "RBBB", "Other"]
SQA_HOP = FS // 2

BEAT_DTYPE = np.dtype(
    [
        ("r_index", "<i4"),
        ("flags", "u1"),
        ("rr_pre_ms", "<f4"),
        ("rr_post_ms", "<f4"),
        ("ctx", "<f4", N_CTX),
        ("morph", "<f4", N_MORPH),
        ("wave", "<f4", WAVE_LEN),
    ],
    align=True,
)
SQA_DTYPE = np.dtype(
    [
        ("good", "?"),
        ("leads_off", "?"),
        ("amp_ok", "?"),
        ("snr_ok", "?"),
        ("base_ok", "?"),
        ("morph_ok", "?"),
        ("amp_mv", "<f4"),
        ("snr_db", "<f4"),
        ("base_mv", "<f4"),
        ("morph", "<f4"),
        ("kurtosis", "<f4"),
    ],
    align=True,
)
READY, GOOD_SQ = 1, 2


def build_core(force=False):
    sources = list(CORE_DIR.glob("*.cpp"))
    headers = list(CORE_DIR.glob("*.h")) + [ROOT / "firmware" / "src" / "rf_predict.h", ROOT / "firmware" / "src" / "model_rf.h"]
    if DLL.exists() and not force:
        newest = max(p.stat().st_mtime for p in sources + headers)
        if DLL.stat().st_mtime > newest:
            return DLL
    BUILD_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "ziglang", "c++", "-shared", "-O2", "-std=c++17", "-DECG_BUILD_SHARED", "-Wno-everything", f"-I{CORE_DIR}",
           "-o", str(DLL)] + [str(s) for s in sources]
    env = os.environ.copy()
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(BUILD_DIR / "zig-global-cache"))
    env.setdefault("ZIG_LOCAL_CACHE_DIR", str(BUILD_DIR / "zig-local-cache"))
    subprocess.run(cmd, check=True, env=env)
    return DLL


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(build_core()))
        _lib.ecg_run.restype = ctypes.c_int
        assert _lib.ecg_beat_size() == BEAT_DTYPE.itemsize, "Beat struct layout mismatch"
        assert _lib.ecg_sqa_size() == SQA_DTYPE.itemsize, "SqaStatus struct layout mismatch"
    return _lib


def run_core(mv, leads_off=None, want_filtered=False, bypass_sqa=False):
    """Run a signal in mV through the device pipeline. Returns (beats, sqa, filtered)."""
    mv = np.ascontiguousarray(mv, dtype=np.float32)
    if mv.ndim != 1 or not np.isfinite(mv).all():
        raise ValueError("ECG samples must be a finite one-dimensional array")
    n = len(mv)
    max_beats = n // 40 + 8
    beats = np.zeros(max_beats, BEAT_DTYPE)
    max_sq = n // SQA_HOP + 2
    sq = np.zeros(max_sq, SQA_DTYPE)
    filt = np.zeros(n, np.float32) if want_filtered else None
    lo = None
    if leads_off is not None:
        lo = np.ascontiguousarray(leads_off, dtype=np.uint8)
        if lo.shape != mv.shape:
            raise ValueError("leads_off must have one entry per ECG sample")
    nb = lib().ecg_run(
        mv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n),
        lo.ctypes.data_as(ctypes.c_void_p) if lo is not None else None,
        beats.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(max_beats),
        sq.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(max_sq),
        filt.ctypes.data_as(ctypes.c_void_p) if filt is not None else None,
        ctypes.c_int(int(bypass_sqa)),
    )
    return beats[:nb], sq[: n // SQA_HOP], filt


def hrv(rr_ms):
    rr = np.ascontiguousarray(rr_ms, dtype=np.float32)
    out = np.zeros(5, np.float32)
    lib().ecg_hrv(rr.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(rr)), out.ctypes.data_as(ctypes.c_void_p))
    return dict(zip(["hr", "mean_nn", "sdnn", "rmssd", "pnn50"], out.tolist()))


def rhythm(cls):
    """Device verdict after each beat. Returns (labels, alerts)."""
    c = np.asarray(cls, dtype=np.int8)
    label, alert = np.zeros(len(c), np.int8), np.zeros(len(c), np.uint8)
    lib().ecg_rhythm(c.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(c)), label.ctypes.data_as(ctypes.c_void_p),
                     alert.ctypes.data_as(ctypes.c_void_p))
    return label, alert.astype(bool)


def rhythm_rule(cls, ctx):
    """The device's final per-beat class: the classifier output after the AF rhythm rule."""
    out = np.array(cls, dtype=np.int8)
    # ascontiguousarray: a field of a structured array is a strided view, unsafe to pass to C
    c = np.ascontiguousarray(ctx, dtype=np.float32)
    lib().ecg_rhythm_rule(out.ctypes.data_as(ctypes.c_void_p), c.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(out)))
    return out


def rf_predict(features):
    """Random Forest as compiled for the device, from firmware/src/rf_predict.h."""
    f = np.ascontiguousarray(features, dtype=np.float32)
    out = np.zeros(len(f), np.int8)
    lib().ecg_rf_predict(f.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(f)), out.ctypes.data_as(ctypes.c_void_p))
    return out
