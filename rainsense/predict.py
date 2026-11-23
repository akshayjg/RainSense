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


def label(model, probs):
    """no_rain unless P(rain) = 1 - P(no_rain) reaches the model's rain_threshold, then the likeliest rain class."""
    if "no_rain" not in probs:
        return max(probs, key=probs.get)
    if 1 - probs["no_rain"] < model.get("rain_threshold", 0.5):
        return "no_rain"
    return max((k for k in probs if k != "no_rain"), key=probs.get)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="models/model.json")
    a = ap.parse_args()
    with open(a.model) as f:
        model = json.load(f)
    for i, c in enumerate(clips(load_mono_16k(a.wav), CLIP_SECONDS)):
        probs = predict_clip(model, c)
        best = label(model, probs)
        print(f"clip {i} ({i * CLIP_SECONDS}-{(i + 1) * CLIP_SECONDS}s): {best}  " + "  ".join(f"{k}={v:.2f}" for k, v in probs.items()))
