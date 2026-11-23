"""Synthetic rain / non-rain audio generator.

ONLY for smoke-testing the pipeline. Real accuracy needs real recordings
(see README). Folder layout produced: <out>/<class>/<n>.wav
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .features import SR

CLASSES = ["no_rain", "light", "moderate", "heavy"]
RATES = {"light": 25, "moderate": 90, "heavy": 260}      # drops per second
AMPS = {"light": 0.25, "moderate": 0.5, "heavy": 0.9}


def _shaped_noise(n, rng, f_lo, f_hi, tilt=0.0):
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / SR)
    w = ((f >= f_lo) & (f <= f_hi)).astype(float) * (np.maximum(f, 1.0) ** tilt)
    x = np.fft.irfft(spec * w, n)
    return x / (np.std(x) + 1e-9)


def rain(n, kind, rng):
    base = _shaped_noise(n, rng, 800, 7500, tilt=-0.2) * 0.05 * AMPS[kind]
    drops = np.zeros(n)
    k = int(RATES[kind] * n / SR)
    pos = rng.integers(0, n - 200, k)
    for p in pos:
        L = rng.integers(40, 160)
        t = np.arange(L)
        click = rng.standard_normal(L) * np.exp(-t / rng.uniform(6, 25))
        drops[p:p + L] += click * rng.uniform(0.2, 1.0) * AMPS[kind] * 0.25
    return base + drops


def no_rain(n, rng):
    kind = rng.integers(0, 5)
    t = np.arange(n) / SR
    if kind == 0:    # wind
        x = _shaped_noise(n, rng, 20, 600, tilt=-1.0) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t))
    elif kind == 1:  # fan / AC hum
        f0 = rng.choice([50, 100, 120])
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 5)) * 0.3 + _shaped_noise(n, rng, 50, 2500) * 0.2
    elif kind == 2:  # speech-like
        env = np.clip(np.sin(2 * np.pi * rng.uniform(2, 5) * t), 0, None)
        x = _shaped_noise(n, rng, 200, 3200) * env
    elif kind == 3:  # traffic rumble
        x = _shaped_noise(n, rng, 30, 1500, tilt=-0.7)
    else:            # quiet room
        x = _shaped_noise(n, rng, 20, 4000, tilt=-1.0) * 0.2
    return x * rng.uniform(0.02, 0.15) / (np.std(x) + 1e-9)


def generate(out: Path, n_files: int = 30, seconds: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = seconds * SR
    for cls in CLASSES:
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            y = no_rain(n, rng) if cls == "no_rain" else rain(n, cls, rng)
            y = y * rng.uniform(0.3, 1.5)   # random gain so loudness alone is not the cue
            sf.write(d / f"{i:03d}.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    generate(Path(a.out), a.n)
    print(f"wrote synthetic dataset to {a.out}")
