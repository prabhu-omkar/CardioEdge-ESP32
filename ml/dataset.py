"""Builds the beat dataset by running every record through the device DSP core.

Beats are the ones the device itself detects, each labelled from the database
annotations. Splits are by patient so no person appears in both training and testing:
  MIT-BIH Arrhythmia  de Chazal DS1 / DS2 inter-patient split
  MIT-BIH AF          whole records assigned to train, validation or test
  PTB-XL              the database's patient-safe folds: 1-8 train, 9 validation, 10 test
  INCART              training only, since one patient can have several records

MIT-BIH alone has one or two LBBB and RBBB patients per side, which is why PTB-XL is
needed: without it those classes fail on any new person.

    python -m ml.dataset
"""
import zipfile
from math import gcd

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from .core import CLASSES, FS, GOOD_SQ, READY, ROOT, run_core

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "beats"

# Paced records 102, 104, 107, 217 are excluded, as in the AAMI recommendation.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
VAL_MIT = [109, 124, 203, 215, 223]  # Held out of DS1 for tuning, then folded back in

# AF database: 00735 and 03665 have no signal files. Records are assigned to whole
# patients, a held-out group forms the AF part of the test set.
AFDB = ["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091", "05121", "05261", "06426", "06453",
        "06995", "07162", "07859", "07879", "07910", "08215", "08219", "08378", "08405", "08434", "08455"]
AFDB_TEST = ["04043", "05121", "06453", "07879", "08219", "08434"]
AFDB_VAL = ["04048", "05261", "07162"]
AFDB_CAP_PER_CLASS = 5000

INCART_NORMAL_CAP = 1500
PTB_SELECTION = ROOT / "ml" / "ptbxl_selection.csv"  # Chosen by label quality, see docs/DATASETS.md
PTB_LOOPS = 4  # 10 s records are looped so the rhythm history and template can form

MATCH_TOL = int(0.15 * FS)  # AAMI EC38 beat matching window, 150 ms
DELAY = 3  # Group delay of the core's filter chain at QRS frequencies, in samples

SUPRAVENT_ETC = set("aAJSeEjFf")  # Everything else from the annotation set maps to Other
BEAT_SYMBOLS = set("NLRVaAJSeEjFf/Q")


def ensure_mitdb():
    dest = RAW / "mitdb"
    if not dest.exists():
        with zipfile.ZipFile(RAW / "mitdb.zip") as z:
            z.extractall(dest)
    hea = next(dest.rglob("100.hea"))
    return hea.parent


def to_fs(sig, fs_in):
    fs_in = int(round(fs_in))
    if fs_in == FS:
        return sig.astype(np.float32)
    g = gcd(FS, fs_in)
    return resample_poly(sig, FS // g, fs_in // g).astype(np.float32)


def rhythm_at(ann):
    """Rhythm label in force at each annotation, from the '+' rhythm-change markers."""
    out, cur = [], "(N"
    for sym, aux in zip(ann.symbol, ann.aux_note):
        if sym == "+" and aux:
            cur = aux.rstrip("\x00").strip()
        out.append(cur)
    return out


def map_label(sym, rhythm):
    if sym == "V":
        return 1
    if sym == "L":
        return 3
    if sym == "R":
        return 4
    if sym == "N":
        if rhythm.startswith("(AFIB"):
            return 2
        if rhythm.startswith("(AFL"):
            return 5
        return 0
    if sym in SUPRAVENT_ETC:
        return 5
    return -1


def load_mit(rec, base=None):
    """Returns (signal_mV at FS, annotated sample indices at FS, labels)."""
    base = base or ensure_mitdb()
    r = wfdb.rdrecord(str(base / str(rec)))
    ch = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = to_fs(r.p_signal[:, ch], r.fs)
    ann = wfdb.rdann(str(base / str(rec)), "atr")
    rhythm = rhythm_at(ann)
    idx, lab = [], []
    for s, sym, rh in zip(ann.sample, ann.symbol, rhythm):
        if sym not in BEAT_SYMBOLS:
            continue
        idx.append(int(round(s * FS / r.fs)))
        lab.append(map_label(sym, rh))
    return sig, np.array(idx), np.array(lab)


def load_afdb(rec):
    base = RAW / "afdb" / rec
    r = wfdb.rdrecord(str(base))
    sig = to_fs(r.p_signal[:, 0], r.fs)
    rh = wfdb.rdann(str(base), "atr")
    qrs = wfdb.rdann(str(base), "qrs")
    # Rhythm segments, expressed in samples at r.fs
    starts = rh.sample
    notes = [a.rstrip("\x00").strip() for a in rh.aux_note]
    pos = np.searchsorted(starts, qrs.sample, side="right") - 1
    lab = np.full(len(qrs.sample), -1)
    for i, p in enumerate(pos):
        if p < 0:
            continue
        if notes[p].startswith("(AFIB"):
            lab[i] = 2
        elif notes[p] == "(N":
            lab[i] = 0
    return sig, (qrs.sample * FS / r.fs).astype(int), lab


def load_incart(rec):
    base = RAW / "incart" / rec
    r = wfdb.rdrecord(str(base), channel_names=["II"])
    sig = to_fs(r.p_signal[:, 0], r.fs)
    ann = wfdb.rdann(str(base), "atr")
    rhythm = rhythm_at(ann)
    idx, lab = [], []
    for s, sym, rh in zip(ann.sample, ann.symbol, rhythm):
        if sym in BEAT_SYMBOLS:
            idx.append(int(round(s * FS / r.fs)))
            # Rhythm markers are too sparse here to trust for N beats: without a marker that
            # ends the episode, one AF note would label the rest of the record. Drop those beats.
            lab.append(-1 if sym == "N" and rh != "(N" else map_label(sym, rh))
    return sig, np.array(idx), np.array(lab)


def ptb_beats(row, name):
    """Beats from a looped 10 s lead II record. The whole record carries one label."""
    r = wfdb.rdrecord(str(RAW / "ptbxl" / row.filename_hr), channel_names=["II"])
    sig = to_fs(r.p_signal[:, 0], r.fs)
    n = len(sig)
    beats, _, _ = run_core(np.tile(sig, PTB_LOOPS))
    # Third pass only, clear of the start-up period and of the join between loops
    keep = (beats["r_index"] >= 2 * n + FS // 2) & (beats["r_index"] < 3 * n - FS // 2)
    labels = np.where(keep, CLASSES.index(row.lab), -1)
    return pack(beats, labels, name)


def match_beats(r_index, ann_idx, tol=MATCH_TOL):
    """Index of the nearest annotation for each detected beat, -1 if none is in tolerance."""
    pos = np.searchsorted(ann_idx, r_index)
    pos = np.clip(pos, 1, len(ann_idx) - 1)
    left, right = ann_idx[pos - 1], ann_idx[pos]
    pick = np.where(r_index - left <= right - r_index, pos - 1, pos)
    return np.where(np.abs(ann_idx[pick] - r_index) <= tol, pick, -1)


def beats_for(sig, ann_idx, ann_lab):
    beats, _, _ = run_core(sig)
    m = match_beats(beats["r_index"] - DELAY, ann_idx)
    labels = np.where(m >= 0, ann_lab[np.maximum(m, 0)], -1)
    return beats, labels


def pack(beats, labels, rec):
    keep = (labels >= 0) & ((beats["flags"] & (READY | GOOD_SQ)) == (READY | GOOD_SQ))
    return {
        "wave": beats["wave"][keep].astype(np.float32),
        "ctx": beats["ctx"][keep].astype(np.float32),
        "morph": beats["morph"][keep].astype(np.float32),
        "y": labels[keep].astype(np.int8),
        "rec": np.full(keep.sum(), rec, dtype="U8"),
        "r": beats["r_index"][keep],
    }


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def cap(d, per_class, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(d["y"]):
        idx = np.flatnonzero(d["y"] == c)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return {k: v[keep] for k, v in d.items()}


def summarize(name, d):
    counts = {CLASSES[c]: int((d["y"] == c).sum()) for c in range(len(CLASSES))}
    print(f"{name:10s} n={len(d['y']):7d} patients={len(np.unique(d['rec'])):3d} {counts}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = ensure_mitdb()
    parts = {"train": [], "val": [], "test_mit": [], "test_af": []}
    for rec in DS1 + DS2:
        sig, idx, lab = load_mit(rec, base)
        beats, labels = beats_for(sig, idx, lab)
        p = pack(beats, labels, str(rec))
        name = "test_mit" if rec in DS2 else ("val" if rec in VAL_MIT else "train")
        parts[name].append(p)
    for i, rec in enumerate(AFDB):
        sig, idx, lab = load_afdb(rec)
        beats, labels = beats_for(sig, idx, lab)
        p = cap(pack(beats, labels, rec), AFDB_CAP_PER_CLASS, i)
        name = "test_af" if rec in AFDB_TEST else ("val" if rec in AFDB_VAL else "train")
        parts[name].append(p)
        print("afdb", rec, len(p["y"]), flush=True)
    for rec in sorted(p.stem for p in (RAW / "incart").glob("*.dat")):
        sig, idx, lab = load_incart(rec)
        beats, labels = beats_for(sig, idx, lab)
        p = pack(beats, labels, rec)
        rng = np.random.default_rng(int(rec[1:]))
        normal = np.flatnonzero(p["y"] == 0)
        drop = rng.permutation(normal)[INCART_NORMAL_CAP:]
        parts["train"].append({k: np.delete(v, drop, axis=0) for k, v in p.items()})
    print("incart done", flush=True)
    sel = pd.read_csv(PTB_SELECTION)
    sel = sel[sel.lab != "AFib"]  # Too few clean AF records; AF comes from the AF database
    parts["test_ptb"] = []
    for row in sel.itertuples():
        name = "test_ptb" if row.strat_fold == 10 else ("val" if row.strat_fold == 9 else "train")
        parts[name].append(ptb_beats(row, f"p{row.patient_id:.0f}"))
    for name, ps in parts.items():
        d = concat(ps)
        summarize(name, d)
        np.savez_compressed(OUT / f"{name}.npz", **d)


if __name__ == "__main__":
    main()
