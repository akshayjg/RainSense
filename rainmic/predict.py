"""Classify a wav file using model.json (same math as the Android library)."""
import argparse
import json

import numpy as np

from .data import clips, load_mono_16k
from .features import CLIP_SECONDS, compute_features


def predict_features(model, f):
    z = (np.asarray(f) - np.array(model["mean"])) / np.array(model["scale"])
    logits = np.array(model["coef"]) @ z + np.array(model["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return dict(zip(model["classes"], p.tolist()))


def predict_clip(model, clip):
    """Median of the per-window probabilities, so one loud second (a horn, a shout) can't flip the result."""
    P = np.array([list(predict_features(model, compute_features(w)).values()) for w in clips(clip)])
    p = np.median(P, axis=0)
    return dict(zip(model["classes"], (p / p.sum()).tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="models/model.json")
    a = ap.parse_args()
    with open(a.model) as f:
        model = json.load(f)
    for i, c in enumerate(clips(load_mono_16k(a.wav), CLIP_SECONDS)):
        probs = predict_clip(model, c)
        best = max(probs, key=probs.get)
        print(f"clip {i} ({i * CLIP_SECONDS}-{(i + 1) * CLIP_SECONDS}s): {best}  " + "  ".join(f"{k}={v:.2f}" for k, v in probs.items()))
