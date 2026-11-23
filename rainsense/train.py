import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .data import augment, list_wavs, load_clips
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


# loud street sounds used to interrupt held-out rain clips in the noise test
STREET = {"car_horn", "siren", "engine", "train", "footsteps", "helicopter", "airplane"}


def _category(path):
    """'esc50_sea_waves_1234.wav' -> 'sea_waves'; names without a <source>_<category>_<id> pattern -> 'other'."""
    parts = Path(path).stem.split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else "other"


def _calibrate(L, yi):
    """A temperature plus a per-class offset that minimise even-prior log-loss. Keeps the model a plain softmax."""
    w = _balanced_weights(yi)
    res = minimize(lambda x: _nll(_softmax(L / np.exp(x[0]) + x[1:]), yi, w), np.zeros(L.shape[1] + 1), method="L-BFGS-B")
    return float(np.exp(res.x[0])), res.x[1:]


def train(data_dir, out_dir, seed=0, folds=5):
    C, y, groups = load_clips(data_dir)
    cats = np.array([_category(w) for w in list_wavs(data_dir)])[groups]
    print(f"{len(C)} windows of {WINDOW_SECONDS}s from {len(set(groups))} files, classes: {sorted(set(y))}")
    F = np.array([compute_features(c) for c in C])
    rng = np.random.default_rng(seed)

    def fit(idx):
        aug = augment([C[i] for i in idx], y[idx], rng)
        X = np.vstack([F[idx], [compute_features(c) for c in aug]])
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(scaler.transform(X), np.concatenate([y[idx], y[idx]]))
        return scaler, clf

    classes = sorted(set(y))  # LogisticRegression orders classes_ the same way
    yi = np.array([classes.index(c) for c in y])
    rain = classes.index("rain") if "rain" in classes else None
    L_raw, P_cal = np.zeros((len(C), len(classes))), np.zeros((len(C), len(classes)))
    clip_rows, clip_fold = [], []
    noise = {"median": [], "mean": [], "noise_window": []}

    # Every file is tested exactly once, never in the same fold as its own training data. Calibration is fitted on a
    # grouped quarter of each training part, so the test fold never influences the numbers reported for it.
    split = lambda n: StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(split(folds).split(F, y, groups)):
        fi, ci = next(split(4).split(F[tr], y[tr], groups[tr]))
        model = export(*fit(tr[fi]))
        T, b = _calibrate(_logits(model, F[tr[ci]]), yi[tr[ci]])
        L_raw[te] = _logits(model, F[te])
        P_cal[te] = _softmax(L_raw[te] / T + b)
        first = len(clip_rows)
        for g in np.unique(groups[te]):  # clip = median over the 1 s windows of each 5 s chunk of a test file
            rows = np.flatnonzero(groups == g)
            clip_rows += [rows[s:s + CLIP_SECONDS] for s in range(0, len(rows) - CLIP_SECONDS + 1, CLIP_SECONDS)]
        clip_fold += [f] * (len(clip_rows) - first)

        # noise test: two of the five windows of a test rain clip replaced by a street sound at twice its loudness
        street = [C[i] for i in te if cats[i] in STREET]
        if rain is None or not street:
            continue
        p_rain = lambda ws: _softmax(_logits(model, np.array([compute_features(w) for w in ws])) / T + b)[:, rain]
        for r in clip_rows[first:]:
            if yi[r[0]] != rain:
                continue
            seg = [C[i] for i in r]
            for j in (1, 3):
                n = street[rng.integers(len(street))]
                seg[j] = np.clip(n * 2 * np.sqrt(np.mean(seg[j] ** 2) / (np.mean(n ** 2) + 1e-12)), -1, 1)
            P = p_rain(seg)
            noise["median"].append(np.median(P) >= 0.5)
            noise["mean"].append(np.mean(P) >= 0.5)
            noise["noise_window"].append(P[1] >= 0.5)

    pred = P_cal.argmax(axis=1)
    print(classification_report(yi, pred, target_names=classes, zero_division=0))
    print("confusion matrix (rows=true):\n", confusion_matrix(yi, pred))
    clip_true = np.array([yi[r[0]] for r in clip_rows])
    clip_pred = np.array([np.median(P_cal[r], axis=0).argmax() for r in clip_rows])
    ok, clip_fold = clip_true == clip_pred, np.array(clip_fold)
    w = _balanced_weights(yi)
    acc = float((pred == yi).mean())
    metrics = {
        "evaluation": f"{folds}-fold cross-validation grouped by source file; calibration fitted inside each training fold",
        "windows": len(C), "files": int(len(set(groups))), "clips": len(clip_rows), "classes": classes,
        "window_accuracy": acc,
        "window_balanced_accuracy": float(balanced_accuracy_score(yi, pred)),
        "clip_accuracy": float(ok.mean()),
        "clip_accuracy_per_fold": [float(ok[clip_fold == f].mean()) for f in range(folds)],
        "clip_balanced_accuracy": float(balanced_accuracy_score(clip_true, clip_pred)),
        "clip_recall": {c: [float(ok[clip_true == i].mean()), int((clip_true == i).sum())] for i, c in enumerate(classes)},
        "logloss_raw": _nll(_softmax(L_raw), yi, w), "logloss_calibrated": _nll(P_cal, yi, w),
    }
    if "no_rain" in classes:
        nr, m = classes.index("no_rain"), y == "no_rain"
        metrics["reliability_raw"] = _reliability(1 - _softmax(L_raw)[:, nr], y != "no_rain", w)
        metrics["reliability_calibrated"] = _reliability(1 - P_cal[:, nr], y != "no_rain", w)
        metrics["false_alarm_by_category"] = sorted(
            ((c, float(np.mean(pred[m & (cats == c)] != nr)), int(np.sum(m & (cats == c)))) for c in np.unique(cats[m])),
            key=lambda t: -t[1])
    if noise["median"]:
        metrics["street_noise_2s_detected"] = {k: [float(np.mean(v)), len(v)] for k, v in noise.items()}

    # shipped model: fit on everything; calibration fitted on the pooled out-of-fold logits
    T, bias = _calibrate(L_raw, yi)
    metrics |= {"temperature": T, "bias": bias.tolist()}
    print(f"cross-validated: 1 s window {acc:.3f}, {CLIP_SECONDS} s clip (median) {ok.mean():.3f} over {len(clip_rows)} clips, "
          f"log-loss {metrics['logloss_raw']:.3f} -> {metrics['logloss_calibrated']:.3f}")
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
