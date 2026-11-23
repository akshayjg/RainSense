"""Import public labelled audio into <out>/rain/ and <out>/no_rain/ for training.

  python -m rainmic.datasets esc50 --out data                      # downloads ESC-50 (~600 MB), CC BY-NC 3.0
  python -m rainmic.datasets fsd50k /path/to/FSD50K --out data     # a local FSD50K copy, CC0 / CC-BY / CC-BY-NC per clip

Licences are non-commercial for part of this data: fine for research and a prototype, not for a commercial model.
"""
import argparse
import csv
import io
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
# street and household confusers; "thunderstorm" is left out because those clips usually contain rain too
ESC50_NO_RAIN = {
    "wind", "engine", "car_horn", "train", "siren", "footsteps", "water_drops", "pouring_water", "sea_waves",
    "crackling_fire", "helicopter", "airplane", "toilet_flush", "vacuum_cleaner", "keyboard_typing",
    "insects", "crickets", "chirping_birds", "laughing", "church_bells",
}
# shortcut: label names typed from the FSD50K vocabulary; check FSD50K.ground_truth/vocabulary.csv if a class comes up empty
FSD_RAIN = {"Rain", "Raindrop"}
FSD_NO_RAIN = {
    "Traffic_noise_and_roadway_noise", "Vehicle", "Car", "Car_passing_by", "Bus", "Truck", "Motorcycle", "Engine",
    "Vehicle_horn_and_car_horn_and_honking", "Siren", "Wind", "Speech", "Walk_and_footsteps", "Drip",
    "Water_tap_and_faucet", "Stream", "Ocean", "Crowd", "Bird",
}


def _trim_silence(y, thresh=1e-4):
    """ESC-50 pads short clips with digital silence; padding labelled 'rain' would teach silence = rain."""
    nz = np.flatnonzero(np.abs(y) > thresh)
    return y[nz[0]:nz[-1] + 1] if len(nz) else y[:0]


def import_esc50_zip(zpath, out):
    """Clips from the same Freesound source are joined into one file, so the train/test split (by file) can't leak."""
    out = Path(out)
    with zipfile.ZipFile(zpath) as z:
        prefix = next(n for n in z.namelist() if n.endswith("meta/esc50.csv"))[:-len("meta/esc50.csv")]
        meta = list(csv.DictReader(io.TextIOWrapper(z.open(prefix + "meta/esc50.csv"), encoding="utf-8")))
        by_src = defaultdict(list)
        for r in meta:
            cls = "rain" if r["category"] == "rain" else "no_rain" if r["category"] in ESC50_NO_RAIN else None
            if cls:
                by_src[(cls, r["category"], r["src_file"])].append(r["filename"])
        counts = defaultdict(int)
        for (cls, cat, src), names in sorted(by_src.items()):
            parts, sr = [], None
            for n in sorted(names):
                y, sr = sf.read(io.BytesIO(z.read(prefix + "audio/" + n)), dtype="float32")
                parts.append(_trim_silence(y))
            y = np.concatenate(parts)
            if len(y) == 0:
                continue
            (out / cls).mkdir(parents=True, exist_ok=True)
            sf.write(out / cls / f"esc50_{cat}_{src}.wav", y, sr)
            counts[cls] += 1
    return dict(counts)


def import_fsd50k(root, out, max_per_class=1500):
    root, out = Path(root), Path(out)
    counts = defaultdict(int)
    for split in ("dev", "eval"):
        gt = root / "FSD50K.ground_truth" / f"{split}.csv"
        if not gt.exists():
            continue
        with open(gt, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels = set(r["labels"].split(","))
                cls = "rain" if labels & FSD_RAIN else "no_rain" if labels & FSD_NO_RAIN and "Thunder" not in labels else None
                src = root / f"FSD50K.{split}_audio" / f"{r['fname']}.wav"
                if cls is None or counts[cls] >= max_per_class or not src.exists():
                    continue
                (out / cls).mkdir(parents=True, exist_ok=True)
                shutil.copy(src, out / cls / f"fsd50k_{r['fname']}.wav")
                counts[cls] += 1
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["esc50", "fsd50k"])
    ap.add_argument("path", nargs="?", help="FSD50K root (fsd50k) or an already-downloaded ESC-50 zip (esc50)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    if a.dataset == "esc50":
        zpath = Path(a.path or "esc50.zip")
        if not zpath.exists():
            print(f"downloading ESC-50 to {zpath} ...")
            urllib.request.urlretrieve(ESC50_URL, zpath)
        print(import_esc50_zip(zpath, a.out))
    else:
        if not a.path:
            ap.error("fsd50k needs the path to your FSD50K folder")
        print(import_fsd50k(a.path, a.out))
