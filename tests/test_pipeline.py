"""Tests run against the firmware code itself, through the shared library.

    python -m pytest tests -q
"""
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from ml import core
from ml.core import CLASSES, FS, GOOD_SQ, READY, ROOT, rhythm, run_core

BEATS = ROOT / "data" / "beats"


def synthetic_ecg(seconds=60, hr=70, amp=1.0, seed=0, jitter=0.0):
    """QRS, P and T as Gaussians on a regular or jittered rhythm."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    t = np.arange(n) / FS
    beats, now = [], 0.5
    while now < seconds - 0.5:
        beats.append(now)
        now += (60 / hr) * (1 + rng.normal(0, jitter))
    x = np.zeros(n)
    for b in beats:
        x += amp * np.exp(-((t - b) ** 2) / (2 * 0.012 ** 2))
        x += -0.15 * amp * np.exp(-((t - b + 0.03) ** 2) / (2 * 0.01 ** 2))
        x += 0.12 * amp * np.exp(-((t - b + 0.17) ** 2) / (2 * 0.03 ** 2))
        x += 0.28 * amp * np.exp(-((t - b - 0.28) ** 2) / (2 * 0.06 ** 2))
    return x.astype(np.float32), np.array(beats)


def filtered(x):
    return run_core(x, want_filtered=True)[2]


# ------------------------------------------------------------ filters


def tone_gain(freq):
    t = np.arange(20 * FS) / FS
    x = np.sin(2 * np.pi * freq * t).astype(np.float32)
    y = filtered(x)[10 * FS:]
    return np.sqrt(2 * np.mean(y ** 2))


def test_passband_keeps_qrs_frequencies():
    for f in (2, 5, 10, 20, 30):
        assert 0.9 < tone_gain(f) < 1.1, f


def test_baseline_wander_removed():
    assert tone_gain(0.1) < 0.05


def test_mains_50hz_removed():
    assert tone_gain(50) < 0.02


def test_high_frequency_noise_removed():
    assert tone_gain(90) < 0.02


# ------------------------------------------------------------ detector


@pytest.mark.parametrize("hr", [40, 60, 90, 130, 180])
def test_detects_every_beat(hr):
    x, truth = synthetic_ecg(60, hr)
    r = run_core(x)[0]["r_index"]
    # Allow the start-up period; every later beat must match one detection
    late = truth[truth > 5] * FS
    errors = [np.min(np.abs(r - t)) for t in late[:-5]]
    assert np.max(errors) < 10


def test_detection_independent_of_gain():
    x, _ = synthetic_ecg(40)
    a = len(run_core(x * 0.3)[0])
    b = len(run_core(x * 3.0)[0])
    assert abs(a - b) <= 1


def test_reports_no_beats_on_flat_signal():
    assert len(run_core(np.zeros(30 * FS, np.float32))[0]) == 0


def test_no_nan_on_extreme_input():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 50, 60 * FS).astype(np.float32)
    b, sq, f = run_core(x, want_filtered=True)
    assert np.isfinite(f).all()
    assert np.isfinite(b["wave"]).all() and np.isfinite(b["ctx"]).all() and np.isfinite(b["morph"]).all()


# ------------------------------------------------------------ signal quality


def test_clean_signal_is_good():
    x, _ = synthetic_ecg(40)
    sq = run_core(x)[1]
    assert sq["good"][20:].mean() > 0.95


def test_flat_signal_is_bad():
    assert not run_core(np.zeros(20 * FS, np.float32))[1]["good"].any()


def test_white_noise_is_bad():
    x = np.random.default_rng(1).normal(0, 0.5, 30 * FS).astype(np.float32)
    assert run_core(x)[1]["good"].mean() < 0.05


def test_leads_off_forces_bad_and_recovers():
    x, _ = synthetic_ecg(60)
    off = np.zeros(len(x), bool)
    off[25 * FS: 32 * FS] = True
    sq = run_core(x, leads_off=off)[1]
    hop = core.SQA_HOP
    assert not sq["good"][26 * FS // hop: 31 * FS // hop].any()
    assert sq["good"][45 * FS // hop:].mean() > 0.9


def test_noise_burst_blocks_classification():
    x, _ = synthetic_ecg(60)
    y = x.copy()
    y[25 * FS: 35 * FS] += np.random.default_rng(2).normal(0, 0.6, 10 * FS).astype(np.float32)
    b = run_core(y)[0]
    ok = b[(b["flags"] & (READY | GOOD_SQ)) == (READY | GOOD_SQ)]
    inside = (ok["r_index"] > 26 * FS) & (ok["r_index"] < 34 * FS)
    assert inside.sum() == 0


def test_bypass_lets_noise_through():
    x, _ = synthetic_ecg(60)
    y = x.copy()
    y[25 * FS: 35 * FS] += np.random.default_rng(2).normal(0, 0.6, 10 * FS).astype(np.float32)
    b = run_core(y, bypass_sqa=True)[0]
    inside = (b["r_index"] > 26 * FS) & (b["r_index"] < 34 * FS)
    assert ((b["flags"][inside] & GOOD_SQ) > 0).any()


# ------------------------------------------------------------ features


def test_wave_is_normalised_and_centred():
    x, _ = synthetic_ecg(30, amp=2.5)
    b = run_core(x)[0]
    b = b[(b["flags"] & READY) > 0]
    assert np.allclose(np.abs(b["wave"]).max(axis=1), 1.0, atol=1e-3)
    assert (np.abs(b["wave"]).argmax(axis=1) == core.WAVE_PRE).mean() > 0.95


def test_rr_features_on_regular_rhythm():
    x, _ = synthetic_ecg(40, hr=60)
    b = run_core(x)[0]
    b = b[(b["flags"] & READY) > 0]
    assert np.allclose(b["ctx"][:, 0], 1.0, atol=0.02)  # RR equal to the median RR
    assert b["ctx"][:, 5].max() < 0.02                  # Coefficient of variation


def test_rhythm_features_do_not_depend_on_heart_rate():
    slow = run_core(synthetic_ecg(60, hr=60)[0])[0]
    fast = run_core(synthetic_ecg(60, hr=115)[0])[0]
    keep = [0, 1, 2, 3, 4, 5, 6, 7, 10]  # Turning points and entropy react to 1-sample rounding
    a = np.median(slow["ctx"][20:][:, keep], axis=0)
    b = np.median(fast["ctx"][20:][:, keep], axis=0)
    assert np.allclose(a, b, atol=0.03)


def test_irregular_rhythm_raises_irregularity_features():
    reg = run_core(synthetic_ecg(60, jitter=0.0)[0])[0]
    irr = run_core(synthetic_ecg(60, jitter=0.2, seed=3)[0])[0]
    assert irr["ctx"][:, 5].mean() > 5 * max(reg["ctx"][:, 5].mean(), 0.01)


# ------------------------------------------------------------ HRV, compared with direct formulas


def test_hrv_matches_definitions():
    rr = np.random.default_rng(4).normal(800, 40, 50).astype(np.float32)
    s = core.hrv(rr)
    assert s["mean_nn"] == pytest.approx(rr.mean(), rel=1e-4)
    assert s["sdnn"] == pytest.approx(rr.std(ddof=1), rel=1e-3)
    assert s["rmssd"] == pytest.approx(np.sqrt(np.mean(np.diff(rr) ** 2)), rel=1e-3)
    assert s["pnn50"] == pytest.approx(100 * np.mean(np.abs(np.diff(rr)) > 50), rel=1e-3)
    assert s["hr"] == pytest.approx(60000 / rr.mean(), rel=1e-4)


def test_hrv_window_keeps_time_order_after_wrapping():
    rr = np.random.default_rng(5).normal(800, 40, 90).astype(np.float32)
    s = core.hrv(rr)
    assert s["rmssd"] == pytest.approx(np.sqrt(np.mean(np.diff(rr[-60:]) ** 2)), rel=1e-3)


def test_hrv_rejects_impossible_intervals():
    assert core.hrv([800, 810, 20, 5000, 790])["mean_nn"] == pytest.approx(800, abs=0.01)


# ------------------------------------------------------------ rhythm verdict


def test_rhythm_rule_only_touches_narrow_beats():
    regular = np.zeros((6, core.N_CTX), np.float32)
    irregular = regular.copy()
    irregular[:, 7], irregular[:, 10] = 0.6, 0.15
    classes = np.arange(6)
    assert core.rhythm_rule(classes, regular).tolist() == [0, 1, 0, 3, 4, 5]
    assert core.rhythm_rule(classes, irregular).tolist() == [2, 1, 2, 3, 4, 2]


def test_c_calls_accept_fields_of_structured_arrays():
    b = run_core(synthetic_ecg(60, jitter=0.2, seed=3)[0])[0]
    cls = np.zeros(len(b), np.int8)
    assert core.rhythm_rule(cls, b["ctx"]).tolist() == core.rhythm_rule(cls, b["ctx"].copy()).tolist()


def verdict(seq):
    return rhythm(seq)


def test_no_verdict_until_enough_beats():
    assert verdict([0, 0, 0, 0])[0].tolist() == [-1, -1, -1, -1]


def test_normal_rhythm_no_alert():
    label, alert = verdict([0] * 20)
    assert label[-1] == 0 and not alert.any()


def test_single_pvc_does_not_alarm():
    _, alert = verdict([0] * 10 + [1] + [0] * 10)
    assert not alert.any()


def test_frequent_pvcs_alarm():
    label, alert = verdict([0, 0, 1] * 8)
    assert label[-1] == 1 and alert[-1]


@pytest.mark.parametrize("cls", [2, 3, 4])
def test_sustained_abnormal_rhythm_alarms(cls):
    label, alert = verdict([cls] * 12)
    assert label[-1] == cls and alert[-1]


def test_brief_abnormal_verdict_does_not_alarm():
    _, alert = verdict([0] * 12 + [2] * 7 + [0] * 12)
    assert not alert.any()


def test_alert_clears_when_rhythm_returns_to_normal():
    label, alert = verdict([2] * 12 + [0] * 12)
    assert alert[11] and not alert[-1]


# ------------------------------------------------------------ data integrity


@pytest.mark.skipif(not (BEATS / "train.npz").exists(), reason="dataset not built")
def test_no_patient_in_both_training_and_test():
    train = set(np.load(BEATS / "train.npz")["rec"]) | set(np.load(BEATS / "val.npz")["rec"])
    test = set()
    for n in ("test_mit", "test_af", "test_ptb"):
        test |= set(np.load(BEATS / f"{n}.npz")["rec"])
    assert not train & test


def test_ds1_and_ds2_are_the_published_split():
    from ml.dataset import DS1, DS2
    assert len(DS1) == 22 and len(DS2) == 22 and not set(DS1) & set(DS2)
    assert not {102, 104, 107, 217} & set(DS1 + DS2)


# ------------------------------------------------------------ deployed models


@pytest.mark.skipif(not (ROOT / "models" / "rf.pkl").exists(), reason="model not trained")
def test_firmware_rf_matches_trained_forest():
    import pickle
    from ml.common import load_split, rf_features
    from ml.train_rf import hard_vote
    rf = pickle.load(open(ROOT / "models" / "rf.pkl", "rb"))
    X = rf_features(load_split("test_mit"))[:20000]
    assert (core.rf_predict(X) == hard_vote(rf, X)).mean() > 0.9999


@pytest.mark.skipif(not (ROOT / "models" / "cnn_int8.tflite").exists(), reason="model not trained")
def test_int8_header_matches_tflite_file_and_uses_only_registered_ops():
    import tensorflow as tf
    tfl = (ROOT / "models" / "cnn_int8.tflite").read_bytes()
    head = (ROOT / "firmware" / "src" / "model_cnn.h").read_text()
    body = head[head.index("{") + 1: head.index("}")]
    assert bytes(int(v) for v in re.findall(r"\d+", body)) == tfl

    i = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    i.allocate_tensors()
    used = {o["op_name"] for o in i._get_ops_details()}
    src = (ROOT / "firmware" / "src" / "classifiers.cpp").read_text()
    registered = {m.upper() for m in re.findall(r"ops\.Add(\w+)\(", src)}
    aliases = {"CONV_2D": "CONV2D", "MAX_POOL_2D": "MAXPOOL2D", "FULLY_CONNECTED": "FULLYCONNECTED"}
    for op in used:
        assert aliases.get(op, op) in registered, op


@pytest.mark.skipif(not (ROOT / "models" / "cnn_int8.tflite").exists(), reason="model not trained")
def test_dense_weights_are_not_per_channel():
    import tensorflow as tf
    i = tf.lite.Interpreter(model_path=str(ROOT / "models" / "cnn_int8.tflite"),
                            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    i.allocate_tensors()
    tensors = {t["index"]: t for t in i.get_tensor_details()}
    for op in i._get_ops_details():
        if op["op_name"] == "FULLY_CONNECTED":
            weights = tensors[op["inputs"][1]]
            assert len(weights["quantization_parameters"]["scales"]) == 1


def test_dashboard_header_is_current():
    html = (ROOT / "firmware" / "web" / "dashboard.html").read_text(encoding="utf8")
    assert html in (ROOT / "firmware" / "src" / "dashboard_html.h").read_text(encoding="utf8")


@pytest.mark.parametrize("invalid", [-1, 6, 127])
def test_invalid_class_clears_alarm_and_requires_fresh_history(invalid):
    labels, alerts = rhythm([2] * 12 + [invalid] + [2] * 4)
    assert alerts[11]
    assert labels[12:].tolist() == [-1] * 5
    assert not alerts[12:].any()


def test_capture_hrv_excludes_ectopic_neighbours_and_rejected_gaps():
    from ml.eval_capture import capture_hrv
    b = np.zeros(9, core.BEAT_DTYPE)
    b["flags"] = READY | GOOD_SQ
    b["flags"][5] = 0
    b["rr_pre_ms"] = [800, 810, 400, 1200, 790, 600, 1100, 820, 830]
    # Only intervals 810, 790, 820, 830 have eligible adjacent classifications.
    actual = capture_hrv(b, [0, 0, 1, 0, 0, 0, 0, 0])
    expected = core.hrv([810, 790, 820, 830])
    assert actual == pytest.approx(expected)


def test_emulator_validation_rejects_incomplete_and_mismatched_runs():
    import copy
    import json
    from tools.run_qemu_selftest import validate_result
    good = json.loads((ROOT / "results" / "qemu_selftest_baseline_20261105.json").read_text())
    assert validate_result(good) == []
    bad = copy.deepcopy(good)
    bad["done"] = False
    bad["vectors"]["cnn_match"] = "239"
    bad["clips"][0]["esp32"]["analysed"] = "0"
    assert len(validate_result(bad)) >= 3
    assert validate_result({})


def test_emulator_timeout_stops_a_silent_process():
    import subprocess
    import sys
    import time
    from tools.run_qemu_selftest import collect_output
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    start = time.monotonic()
    assert collect_output(process, timeout=0.2) == []
    assert process.poll() is not None
    assert time.monotonic() - start < 5



def test_raw_signal_and_lead_flags_accept_strided_capture_columns():
    x, _ = synthetic_ecg(30)
    columns = np.zeros((len(x), 3), np.float32)
    columns[:, 0] = x
    columns[10 * FS:15 * FS, 2] = 1
    a = run_core(columns[:, 0], leads_off=columns[:, 2], want_filtered=True)
    b = run_core(x.copy(), leads_off=columns[:, 2].copy(), want_filtered=True)
    for actual, expected in zip(a, b):
        np.testing.assert_array_equal(actual, expected)


def test_core_rejects_wrong_lead_flag_length():
    with pytest.raises(ValueError, match="one entry"):
        run_core(np.zeros(100), leads_off=np.zeros(10))
