from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly

from .features import SR, WINDOW_SECONDS


def load_mono_16k(path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return y.astype(np.float32)


def clips(y: np.ndarray, seconds: int = WINDOW_SECONDS):
    n = seconds * SR
    for i in range(0, len(y) - n + 1, n):
        yield y[i:i + n]


def load_clips(root):
    """root/<class_name>/*.wav -> 1 s windows, labels (str), groups (file index)"""
    root = Path(root)
    C, labels, groups = [], [], []
    files = 0
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav in sorted(cls_dir.glob("*.wav")):
            for c in clips(load_mono_16k(wav)):
                C.append(c)
                labels.append(cls_dir.name)
                groups.append(files)
            files += 1
    if not C:
        raise SystemExit(f"No >= {WINDOW_SECONDS}s wav clips found under {root}")
    return C, np.array(labels), np.array(groups)


def augment(C, labels, rng):
    """One extra copy per window, imitating street conditions:
    rain windows are (half the time) mixed with a random no-rain window at -3..10 dB SNR (traffic, speech, wind);
    the rest are low-passed at 0.8-2.5 kHz like a phone in a pocket or bag. Random gain on everything."""
    noise = [c for c, lab in zip(C, labels) if lab == "no_rain"]
    out = []
    for c, lab in zip(C, labels):
        if lab != "no_rain" and noise and rng.random() < 0.5:
            n = noise[rng.integers(len(noise))]
            snr_db = rng.uniform(-3, 10)
            x = c + n * np.sqrt(np.mean(c ** 2) / (np.mean(n ** 2) + 1e-12) / 10 ** (snr_db / 10))
        else:
            b, a = butter(4, rng.uniform(800, 2500), fs=SR)
            x = lfilter(b, a, c)
        out.append(np.clip(x * rng.uniform(0.3, 1.5), -1, 1).astype(np.float32))
    return out
