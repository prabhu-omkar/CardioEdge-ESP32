"""Evaluates the R-peak detector against the MIT-BIH beat annotations (Objective 4).

Uses the AAMI EC38 / ANSI 150 ms matching window and skips the first 5 s of each
record, which is the detector's start-up period.

    python -m ml.eval_qrs
"""
import json

import numpy as np

from .core import FS, ROOT, run_core
from .dataset import DS1, DS2, DELAY, MATCH_TOL, ensure_mitdb, load_mit


def score_record(r_index, ann_idx, skip):
    r = np.sort(r_index[r_index >= skip])
    a = ann_idx[ann_idx >= skip]
    i = j = tp = 0
    while i < len(r) and j < len(a):
        d = r[i] - a[j]
        if abs(d) <= MATCH_TOL:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return tp, len(r) - tp, len(a) - tp


def main():
    base = ensure_mitdb()
    rows = []
    offsets = []
    for rec in sorted(DS1 + DS2):
        sig, idx, _ = load_mit(rec, base)
        beats, _, _ = run_core(sig)
        r = beats["r_index"]
        offsets.append(np.median([r[k] - idx[np.argmin(np.abs(idx - r[k]))] for k in range(0, len(r), 5)]))
        tp, fp, fn = score_record(r - DELAY, idx, 5 * FS)
        rows.append({"record": rec, "tp": tp, "fp": fp, "fn": fn,
                     "se": tp / max(tp + fn, 1), "ppv": tp / max(tp + fp, 1)})
        print(f"{rec}: Se={rows[-1]['se']:.4f} PPV={rows[-1]['ppv']:.4f} fp={fp} fn={fn}", flush=True)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    res = {"beats": tp + fn, "tp": tp, "fp": fp, "fn": fn, "sensitivity": tp / (tp + fn), "ppv": tp / (tp + fp),
           "der": (fp + fn) / (tp + fn), "median_offset_samples": float(np.median(offsets)), "records": rows}
    print({k: v for k, v in res.items() if k != "records"})
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "qrs_detection.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
