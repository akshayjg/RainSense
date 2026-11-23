import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data import augment, load_clips
from .features import CLIP_SECONDS, FEATURE_NAMES, WINDOW_SECONDS, compute_features


def export(scaler, clf, temperature=1.0, bias=None):
    """Plain-JSON softmax model, used by predict.py and the Android library. Calibration is folded into coef/intercept."""
    binary = len(clf.classes_) == 2
    # sklearn stores one row for binary problems; expand to one row per class
    coef = np.vstack([-clf.coef_[0], clf.coef_[0]]) / 2 if binary else clf.coef_
    intercept = np.array([-clf.intercept_[0], clf.intercept_[0]]) / 2 if binary else clf.intercept_
    return {
        "classes": [str(c) for c in clf.classes_],
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": (coef / temperature).tolist(),
        "intercept": (intercept / temperature + (0 if bias is None else np.asarray(bias))).tolist(),
        "temperature": float(temperature),
    }


def _logits(model, F):
    return ((F - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"]).T + np.array(model["intercept"])


def _softmax(L):
    p = np.exp(L - L.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _nll(P, yi, w):
    return float(-np.sum(w * np.log(P[np.arange(len(yi)), yi] + 1e-12)) / w.sum())


def _balanced_weights(yi):
    """Weights that make every class count equally, i.e. probabilities calibrated for an even prior.
    The city aggregator can then apply the local base rate."""
    counts = np.bincount(yi)
    return 1.0 / counts[yi]


def _reliability(p_rain, is_rain, w, bins=5):
    """[(mean predicted, observed fraction under an even prior, count)] per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_rain, edges) - 1, 0, bins - 1)
    return [(float(np.average(p_rain[m], weights=w[m])), float(np.average(is_rain[m], weights=w[m])), int(m.sum()))
            for m in (idx == b for b in range(bins)) if m.any()]


def train(data_dir, out_dir, seed=0):
    C, y, groups = load_clips(data_dir)
    print(f"{len(C)} windows of {WINDOW_SECONDS}s from {len(set(groups))} files, classes: {sorted(set(y))}")
    F = np.array([compute_features(c) for c in C])
    rng = np.random.default_rng(seed)

    def fit(idx):
        aug = augment([C[i] for i in idx], y[idx], rng)
        X = np.vstack([F[idx], [compute_features(c) for c in aug]])
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(scaler.transform(X), np.concatenate([y[idx], y[idx]]))
        return scaler, clf

    # split by source file so windows from one recording never land in both train and test
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(F, y, groups))
    scaler, clf = fit(tr)
    raw = export(scaler, clf)
    classes = raw["classes"]
    yi = np.array([classes.index(c) for c in y[te]])
    L = _logits(raw, F[te])
    P_raw = _softmax(L)
    pred = np.array(classes)[P_raw.argmax(axis=1)]
    acc = float((pred == y[te]).mean())
    print(classification_report(y[te], pred, zero_division=0))
    print("confusion matrix (rows=true):\n", confusion_matrix(y[te], pred, labels=classes))

    # calibration: a temperature plus a per-class offset, fit on held-out data (even prior). Keeps the model a plain softmax.
    w = _balanced_weights(yi)
    k = len(classes)
    res = minimize(lambda x: _nll(_softmax(L / np.exp(x[0]) + x[1:]), yi, w), np.zeros(k + 1), method="L-BFGS-B")
    T, bias = float(np.exp(res.x[0])), res.x[1:]
    P_cal = _softmax(L / T + bias)
    print(f"calibration: T={T:.2f}, held-out log-loss {_nll(P_raw, yi, w):.3f} -> {_nll(P_cal, yi, w):.3f}")

    # clip-level result = median over the 1 s windows of each 5 s chunk of a held-out file
    clip_true, clip_pred = [], []
    for g in np.unique(groups[te]):
        rows = np.flatnonzero(groups[te] == g)
        for k in range(0, len(rows) - CLIP_SECONDS + 1, CLIP_SECONDS):
            r = rows[k:k + CLIP_SECONDS]
            clip_true.append(y[te][r[0]])
            clip_pred.append(classes[int(np.median(P_cal[r], axis=0).argmax())])
    clip_acc = float(np.mean(np.array(clip_true) == np.array(clip_pred))) if clip_true else None
    print(f"held-out accuracy: 1 s window {acc:.3f}" + (f", {CLIP_SECONDS} s clip (median) {clip_acc:.3f}" if clip_acc is not None else ""))

    metrics = {
        "windows": len(C), "files": int(len(set(groups))), "classes": classes,
        "heldout_window_accuracy": acc,
        "heldout_window_balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "heldout_clip_accuracy": clip_acc,
        "temperature": T, "bias": bias.tolist(),
        "logloss_raw": _nll(P_raw, yi, w), "logloss_calibrated": _nll(P_cal, yi, w),
    }
    if "no_rain" in classes:
        is_rain = y[te] != "no_rain"
        nr = classes.index("no_rain")
        metrics["reliability_raw"] = _reliability(1 - P_raw[:, nr], is_rain, w)
        metrics["reliability_calibrated"] = _reliability(1 - P_cal[:, nr], is_rain, w)

    # refit on everything for the shipped model, keeping the held-out calibration
    scaler, clf = fit(np.arange(len(C)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(export(scaler, clf, T, bias)))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out / 'model.json'} and metrics.json")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    train(a.data, a.out)
