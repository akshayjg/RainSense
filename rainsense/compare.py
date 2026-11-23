"""Compare rain detectors on real labelled recordings: rain vs no_rain, cross-validated by file.

Expects data/<class>/<time>_<place>.wav, the layout the app's "Record ..." buttons save.
Every class except no_rain counts as rain. Detectors compared:
  handcrafted (19)   the current features.py + logistic regression
  yamnet rain score  YAMNet's built-in Rain/Raindrop/Rain-on-surface score (1 number per clip)
  yamnet embedding   YAMNet's 1024-d embedding + logistic regression
YAMNet needs: pip install -r requirements-yamnet.txt   (or pass --no-yamnet)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import clips, load_mono_16k
from .features import compute_features

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
RAIN_CLASSES = {"Rain", "Raindrop", "Rain on surface"}


def load_yamnet():
    """Returns clip -> (rain score, 1024-d embedding)."""
    import tensorflow_hub as hub  # heavy, so only imported when YAMNet is used

    model = hub.load(YAMNET_URL)
    with open(model.class_map_path().numpy().decode()) as f:
        rain_idx = [i for i, row in enumerate(csv.DictReader(f)) if row["display_name"] in RAIN_CLASSES]

    def run(clip):
        scores, emb, _ = model(clip.astype(np.float32))
        return float(scores.numpy()[:, rain_idx].max(axis=1).mean()), emb.numpy().mean(axis=0)

    return run


def evaluate(features, is_rain, groups, places, n_splits=5):
    """features: {name: (n_clips, d)}. Prints AUC / balanced accuracy per place; returns {name: (auc, bal_acc)} overall."""
    files_per_class = min(len(set(groups[is_rain == c])) for c in (True, False))
    if files_per_class < 2:
        raise SystemExit("Need at least 2 recordings of rain and 2 of no_rain.")
    cv = StratifiedGroupKFold(n_splits=min(n_splits, files_per_class), shuffle=True, random_state=0)
    probs = {}
    for name, X in features.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        probs[name] = cross_val_predict(clf, np.asarray(X), is_rain, groups=groups, cv=cv, method="predict_proba")[:, 1]

    print("cells: AUC / balanced accuracy (places with one class show plain accuracy)")
    print(f"{'place':<10}{'clips':>6}{'rain':>6}" + "".join(f"{n:>22}" for n in features))
    overall = {}
    for place in ["all"] + sorted(set(places)):
        m = np.ones(len(is_rain), bool) if place == "all" else places == place
        y = is_rain[m]
        cells = []
        for name, p in probs.items():
            if len(set(y)) < 2:
                cells.append(f"- / {np.mean((p[m] > 0.5) == y):.2f}")
                continue
            auc, acc = roc_auc_score(y, p[m]), balanced_accuracy_score(y, p[m] > 0.5)
            cells.append(f"{auc:.2f} / {acc:.2f}")
            if place == "all":
                overall[name] = (auc, acc)
        print(f"{place:<10}{m.sum():>6}{y.sum():>6}" + "".join(f"{c:>22}" for c in cells))
    return overall


def main(data, use_yamnet=True):
    yamnet = load_yamnet() if use_yamnet else None
    feats = {"handcrafted (19)": [], "yamnet rain score": [], "yamnet embedding": []}
    labels, groups, places = [], [], []
    for g, wav in enumerate(sorted(Path(data).glob("*/*.wav"))):
        place = wav.stem.rsplit("_", 1)[1] if "_" in wav.stem else "unknown"
        for c in clips(load_mono_16k(wav)):
            feats["handcrafted (19)"].append(compute_features(c))
            if yamnet:
                score, emb = yamnet(c)
                feats["yamnet rain score"].append([score])
                feats["yamnet embedding"].append(emb)
            labels.append(wav.parent.name)
            groups.append(g)
            places.append(place)
    labels = np.array(labels)
    if "no_rain" not in labels:
        raise SystemExit(f"No no_rain/ clips found under {data}")
    feats = {k: v for k, v in feats.items() if v}
    return evaluate(feats, labels != "no_rain", np.array(groups), np.array(places))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-yamnet", action="store_true")
    a = ap.parse_args()
    main(a.data, use_yamnet=not a.no_yamnet)
