from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .features import SR, CLIP_SECONDS, compute_features


def load_mono_16k(path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return y.astype(np.float32)


def clips(y: np.ndarray, seconds: int = CLIP_SECONDS):
    n = seconds * SR
    for i in range(0, len(y) - n + 1, n):
        yield y[i:i + n]


def load_dataset(root):
    """root/<class_name>/*.wav -> X, labels (str), groups (file index)"""
    root = Path(root)
    X, labels, groups = [], [], []
    files = 0
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav in sorted(cls_dir.glob("*.wav")):
            y = load_mono_16k(wav)
            for c in clips(y):
                X.append(compute_features(c))
                labels.append(cls_dir.name)
                groups.append(files)
            files += 1
    if not X:
        raise SystemExit(f"No >= {CLIP_SECONDS}s wav clips found under {root}")
    return np.array(X), np.array(labels), np.array(groups)
