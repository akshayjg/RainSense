import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .features import FEATURE_NAMES


def train(data_dir, out_dir, seed=0):
    X, y, groups = load_dataset(data_dir)
    print(f"{len(X)} clips, classes: {sorted(set(y))}")

    # split by source file so clips from one recording never land in both train and test
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X[tr]), y[tr])

    pred = clf.predict(scaler.transform(X[te]))
    acc = float((pred == y[te]).mean())
    print(classification_report(y[te], pred, zero_division=0))
    print("confusion matrix (rows=true):\n", confusion_matrix(y[te], pred, labels=clf.classes_))

    # refit on everything for the shipped model
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X), y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf}, out / "model.joblib")
    binary = len(clf.classes_) == 2
    model = {  # plain JSON: used by predict.py and copied into the Android app
        "classes": [str(c) for c in clf.classes_],
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        # sklearn stores one row for binary problems; expand to one row per class
        "coef": (np.vstack([-clf.coef_[0], clf.coef_[0]]) / 2 if binary else clf.coef_).tolist(),
        "intercept": (np.array([-clf.intercept_[0], clf.intercept_[0]]) / 2 if binary else clf.intercept_).tolist(),
    }
    (out / "model.json").write_text(json.dumps(model))
    print(f"held-out accuracy: {acc:.3f}  -> saved {out / 'model.json'}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    train(a.data, a.out)
