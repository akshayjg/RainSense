"""Held-out numbers for the demo slides. Trains on 75% of ESC-50 files, evaluates the exported (shipped-form) model on the rest."""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from rainmic.data import clips, load_mono_16k
from rainmic.features import SR, compute_features
from rainmic.predict import predict_features
from rainmic.train import train

DATA, WORK = Path(sys.argv[1]), Path(sys.argv[2])
files = [w for d in sorted(p for p in DATA.iterdir() if p.is_dir()) for w in sorted(d.glob("*.wav"))]
labels = np.array([f.parent.name for f in files])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(files, labels, np.arange(len(files))))

tdir = WORK / "train_split"
shutil.rmtree(tdir, ignore_errors=True)
for i in tr:
    (tdir / labels[i]).mkdir(parents=True, exist_ok=True)
    shutil.copy(files[i], tdir / labels[i] / files[i].name)
if not (WORK / "model_split" / "model.json").exists():
    train(tdir, WORK / "model_split")
model = json.loads((WORK / "model_split" / "model.json").read_text())
R = model["classes"].index("rain")


def p_rain(w):
    return predict_features(model, compute_features(w))["rain"]


def cat(f):
    return f.stem.split("_", 2)[1] if f.parent.name == "no_rain" else "rain"


win = defaultdict(list)            # category -> per-window rain probs
clip_med = defaultdict(list)       # category -> per-5s-clip median rain probs
held = {}                          # file -> windows
for i in te:
    f = files[i]
    W = list(clips(load_mono_16k(f)))
    held[f] = W
    P = [p_rain(w) for w in W]
    win[cat(f)] += P
    clip_med[cat(f)] += [float(np.median(P[k:k + 5])) for k in range(0, len(P) - 4, 5)]


def rate(d, keys, rain):
    v = np.concatenate([d[k] for k in keys])
    return float(np.mean(v >= 0.5) if rain else np.mean(v < 0.5)), len(v)


confusers = [k for k in win if k != "rain"]
out = {
    "heldout_files": {"rain": int(sum(labels[te] == "rain")), "no_rain": int(sum(labels[te] == "no_rain"))},
    "window": {"rain_recall": rate(win, ["rain"], True), "no_rain_recall": rate(win, confusers, False)},
    "clip_median": {"rain_recall": rate(clip_med, ["rain"], True), "no_rain_recall": rate(clip_med, confusers, False)},
    "false_alarm_by_confuser_window": sorted(((k, float(np.mean(np.array(win[k]) >= 0.5)), len(win[k])) for k in confusers), key=lambda t: -t[1]),
}

# interruption test: 5 s of held-out rain with one second replaced by a loud held-out street sound
rng = np.random.default_rng(0)
street = [w for f, W in held.items() if cat(f) in {"car_horn", "siren", "engine", "train", "footsteps", "helicopter", "airplane"} for w in W]
res = {"single_window_at_interruption": [], "mean": [], "median": [], "clean_median": [], "mean_2s": [], "median_2s": []}
for f, W in held.items():
    if cat(f) != "rain":
        continue
    for k in range(0, len(W) - 4, 5):
        seg = W[k:k + 5]
        res["clean_median"].append(np.median([p_rain(w) for w in seg]))
        n = street[rng.integers(len(street))]
        n = n * 2 * np.sqrt(np.mean(seg[2] ** 2) / (np.mean(n ** 2) + 1e-12))
        mixed = [seg[0], seg[1], np.clip(n, -1, 1), seg[3], seg[4]]
        P = [p_rain(w) for w in mixed]
        res["single_window_at_interruption"].append(P[2])
        res["mean"].append(np.mean(P))
        res["median"].append(np.median(P))
        n2 = street[rng.integers(len(street))]
        n2 = n2 * 2 * np.sqrt(np.mean(seg[3] ** 2) / (np.mean(n2 ** 2) + 1e-12))
        P2 = [p_rain(w) for w in [seg[0], np.clip(n, -1, 1), seg[2], np.clip(n2, -1, 1), seg[4]]]
        res["mean_2s"].append(np.mean(P2))
        res["median_2s"].append(np.median(P2))
out["interruption_detect_rate"] = {k: (float(np.mean(np.array(v) >= 0.5)), len(v)) for k, v in res.items()}
(WORK / "analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
