"""Final evaluation on patients never seen in training. Run once, after all tuning.

Produces results/evaluation.json:
  beat metrics for both classifiers on each test set (the int8 CNN and the device RF)
  rhythm verdict accuracy on full recordings, in time order
  the effect of SQA: the same noisy recordings with the gate on and off

    python -m ml.evaluate
"""
import json
import time

import numpy as np

from .common import RESULTS, load_split, metrics, rf_features
from .core import CLASSES, FS, GOOD_SQ, READY, rf_predict, rhythm, rhythm_rule, run_core
from .dataset import AFDB_TEST, DELAY, DS2, ensure_mitdb, load_afdb, load_mit, match_beats
from .sqa_eval import add_noise, load_noise
from .train_cnn import TFLITE, Int8Model

SQA_SNRS = [12, 6, 0, -6]


def features(b):
    return np.hstack([b["ctx"], b["morph"]]).astype(np.float32)


def classifiable(b):
    return b[(b["flags"] & (READY | GOOD_SQ)) == (READY | GOOD_SQ)]


def beat_tables(cnn):
    sets = {"MIT-BIH DS2": ["test_mit"], "AF database": ["test_af"], "PTB-XL fold 10": ["test_ptb"],
            "All test patients": ["test_mit", "test_af", "test_ptb"]}
    out = {}
    for name, parts in sets.items():
        d = load_split(*parts)
        p_cnn = rhythm_rule(cnn.predict(d["wave"], d["ctx"])[0], d["ctx"])
        p_rf = rhythm_rule(rf_predict(rf_features(d)), d["ctx"])
        out[name] = {"beats": len(d["y"]), "patients": int(len(np.unique(d["rec"]))),
                     "cnn": metrics(d["y"], p_cnn), "rf": metrics(d["y"], p_rf)}
        print(name, len(d["y"]), f"CNN acc {out[name]['cnn']['accuracy']:.4f} F1 {out[name]['cnn']['macro_f1']:.4f}  "
              f"RF acc {out[name]['rf']['accuracy']:.4f} F1 {out[name]['rf']['macro_f1']:.4f}", flush=True)
    return out


def rhythm_tables(cnn):
    """Device verdict against the true content of the same 10-beat window."""
    base = ensure_mitdb()
    truth, pred_c, pred_r = [], [], []
    records = [("mit", r) for r in DS2] + [("af", r) for r in AFDB_TEST]
    for kind, rec in records:
        sig, idx, lab = load_mit(rec, base) if kind == "mit" else load_afdb(rec)
        b = classifiable(run_core(sig)[0])
        m = match_beats(b["r_index"] - DELAY, idx)
        y = np.where(m >= 0, lab[np.maximum(m, 0)], -1)
        lc = rhythm(rhythm_rule(cnn.predict(b["wave"], b["ctx"])[0], b["ctx"]))[0]
        lr = rhythm(rhythm_rule(rf_predict(features(b)), b["ctx"]))[0]
        for i in range(10, len(y)):
            w = y[i - 9: i + 1]
            if (w < 0).any():
                continue
            counts = np.bincount(w, minlength=len(CLASSES))
            major = int(counts.argmax())
            if counts[major] < 8 and counts[1] < 3:
                continue  # Mixed window with no clear truth
            t = 1 if counts[1] >= 3 and major == 0 else major
            truth.append(t); pred_c.append(lc[i]); pred_r.append(lr[i])
    truth, pred_c, pred_r = map(np.array, (truth, pred_c, pred_r))
    return {"windows": len(truth), "cnn": metrics(truth, pred_c), "rf": metrics(truth, pred_r)}


def sqa_ab(cnn):
    """The middle third of each DS2 record is corrupted, since electrode trouble comes and
    goes, and the same recording is processed with the quality gate on and off.
    False alarms are counted on records whose true rhythm is almost entirely normal."""
    base = ensure_mitdb()
    noises = {k: load_noise(k) for k in ("em", "ma")}
    data = [(r, *load_mit(r, base)) for r in DS2]
    rows = []
    for nname, noise in noises.items():
        for snr in SQA_SNRS:
            y_all = {"on": [], "off": []}
            p_all = {"on": [], "off": []}
            alarms = {"on": 0, "off": 0}
            counts = {"on": 0, "off": 0}
            hours = 0.0
            for rec, sig, idx, lab in data:
                n = len(sig)
                mixed = sig.copy()
                noisy = add_noise(sig, idx, noise, snr, seed=rec)
                mixed[n // 3: 2 * n // 3] = noisy[n // 3: 2 * n // 3]
                normal_rec = np.mean(lab == 0) > 0.95
                if normal_rec:
                    hours += n / FS / 3600 / 3
                for mode in ("on", "off"):
                    b = classifiable(run_core(mixed, bypass_sqa=mode == "off")[0])
                    p = rhythm_rule(cnn.predict(b["wave"], b["ctx"])[0], b["ctx"])
                    in_noise = (b["r_index"] > n // 3) & (b["r_index"] < 2 * n // 3)
                    m = match_beats(b["r_index"] - DELAY, idx)
                    y = np.where(m >= 0, lab[np.maximum(m, 0)], -1)
                    s = in_noise & (y >= 0)
                    y_all[mode].append(y[s]); p_all[mode].append(p[s])
                    counts[mode] += int(in_noise.sum())
                    if normal_rec:
                        alert = rhythm(p)[1]
                        alarms[mode] += int(np.sum(alert[1:] & ~alert[:-1] & in_noise[1:]))
            row = {"noise": nname, "snr_db": snr, "beats_analysed_in_noise": counts["on"] / max(counts["off"], 1)}
            for mode in ("on", "off"):
                mt = metrics(np.concatenate(y_all[mode]), np.concatenate(p_all[mode]))
                row[f"sqa_{mode}"] = {"accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"], "beats": counts[mode],
                                      "false_alarms_per_hour": alarms[mode] / max(hours, 1e-9)}
            rows.append(row)
            print(row, flush=True)
    return rows


def main():
    cnn = Int8Model(TFLITE)
    t0 = time.time()
    res = {"beats": beat_tables(cnn)}
    res["rhythm"] = rhythm_tables(cnn)
    print("rhythm", res["rhythm"]["windows"], res["rhythm"]["cnn"]["accuracy"], res["rhythm"]["rf"]["accuracy"], flush=True)
    res["sqa_ab"] = sqa_ab(cnn)
    res["seconds"] = time.time() - t0
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "evaluation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
