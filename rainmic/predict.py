"""Classify a wav file using model.json (same math as the Android app)."""
import argparse
import json

import numpy as np

from .data import load_mono_16k, clips
from .features import compute_features


def predict_features(model, f):
    z = (np.asarray(f) - np.array(model["mean"])) / np.array(model["scale"])
    logits = np.array(model["coef"]) @ z + np.array(model["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return dict(zip(model["classes"], p.tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="models/model.json")
    a = ap.parse_args()
    model = json.load(open(a.model))
    y = load_mono_16k(a.wav)
    for i, c in enumerate(clips(y)):
        probs = predict_features(model, compute_features(c))
        best = max(probs, key=probs.get)
        print(f"clip {i} ({i * 5}-{i * 5 + 5}s): {best}  " + "  ".join(f"{k}={v:.2f}" for k, v in probs.items()))
