"""Shared data loading and metrics for the two classifiers."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .core import CLASSES, ROOT

BEATS = ROOT / "data" / "beats"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_split(*names):
    parts = [np.load(BEATS / f"{n}.npz") for n in names]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def source(rec):
    """Database of each beat, from its record name."""
    return np.array(["incart" if r[0] == "I" else "ptbxl" if r[0] == "p" else "mit" if len(r) == 3 else "afdb" for r in rec])


def sample_weights(y, rec, alpha=0.5):
    """Softened class balance, plus equal weight for each database within a class.

    Full class balance tells the model rare classes are as common as Normal, which they
    are not. Database balance stops a class being recognised by where its data came from.
    """
    y = np.asarray(y).astype(int)
    src = source(rec)
    n = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = (n.max() / np.maximum(n[y], 1)) ** alpha
    for c in range(len(CLASSES)):
        k = y == c
        names = np.unique(src[k])
        for s in names:
            j = k & (src == s)
            w[j] *= k.sum() / (len(names) * j.sum())
    return w / w.mean()


def rf_features(d):
    return np.hstack([d["ctx"], d["morph"]]).astype(np.float32)


def metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "macro_f1": float(f[present].mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
