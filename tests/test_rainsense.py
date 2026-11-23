import json
import math
from pathlib import Path

import numpy as np
from rainsense.features import compute_features, FEATURE_NAMES, SR, BAND_BINS, N_FFT, N_BANDS, FMIN, FMAX
from rainsense.synth import generate
from rainsense.train import train
from rainsense.predict import predict_clip, predict_features
from rainsense.data import augment
from rainsense.datasets import import_esc50_zip, import_fsd50k
from rainsense.weather import check, lookup

FIXTURES = Path(__file__).parent / "fixtures"


def test_feature_shape():
    f = compute_features(np.random.randn(5 * SR) * 0.1)
    assert f.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(f))


def test_band_bins_match_formula():
    edges = np.geomspace(FMIN, FMAX, N_BANDS + 1)
    idx = np.floor(edges / (SR / N_FFT) + 0.5).astype(int)
    expect = [(int(idx[i]), max(int(idx[i + 1]), int(idx[i]) + 1)) for i in range(N_BANDS)]
    assert expect == BAND_BINS


def test_pipeline(tmp_path):
    generate(tmp_path / "d", n_files=16, seconds=10, seed=1)
    acc = train(tmp_path / "d", tmp_path / "m")
    assert acc > 0.7
    model = json.load(open(tmp_path / "m" / "model.json"))
    assert len(model["coef"]) == len(model["classes"]) == 4
    assert model["temperature"] > 0
    metrics = json.load(open(tmp_path / "m" / "metrics.json"))
    assert metrics["clip_accuracy"] is not None
    assert metrics["clips"] == 4 * 16 * 2  # every 10 s file gives two 5 s clips and is tested exactly once
    assert len(metrics["clip_accuracy_per_fold"]) == 5
    p = predict_features(model, np.zeros(len(FEATURE_NAMES)))
    assert abs(sum(p.values()) - 1) < 1e-6


def test_compare_evaluate():
    from rainsense.compare import evaluate
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 3)            # 10 files, 3 clips each
    is_rain = groups % 2 == 0
    places = np.array(["hand", "pocket"])[groups % 4 // 2]
    X = rng.standard_normal((30, 3))
    X[:, 0] += is_rain * 5                           # one informative feature
    out = evaluate({"sep": X, "noise": rng.standard_normal((30, 3))}, is_rain, groups, places)
    assert out["sep"][0] > 0.95 and out["sep"][1] > 0.9
    assert out["noise"][0] < 0.9


def parity_signal(n=SR):
    """Same LCG noise + tones as ParityTest.kt, so Python and Kotlin see identical samples."""
    s, y = 12345, np.empty(n)
    for i in range(n):
        s = (s * 1664525 + 1013904223) % 2 ** 32
        y[i] = (s / 2 ** 32 - 0.5) * 0.2 + 0.3 * math.sin(2 * math.pi * 440 * i / SR) + 0.05 * math.sin(2 * math.pi * 3000 * i / SR)
    return y


def test_parity_fixture_is_current():
    # android/rainsense/src/test/.../ParityTest.kt checks the Kotlin extractor against this same file
    expect = np.loadtxt(FIXTURES / "parity_features.txt")
    assert np.allclose(compute_features(parity_signal()), expect, atol=1e-9)


def test_predict_clip_uses_median():
    # toy model: "rain" iff loud (rms_db is feature 17); 2 of 5 windows are loud bursts
    k = len(FEATURE_NAMES)
    w = np.zeros(k); w[17] = 1.0
    model = {"classes": ["no_rain", "rain"], "mean": [0.0] * k, "scale": [1.0] * k,
             "coef": [(-w).tolist(), w.tolist()], "intercept": [30.0, -30.0]}
    rng = np.random.default_rng(0)
    clip = rng.standard_normal(5 * SR) * 0.001
    clip[:2 * SR] *= 500
    assert predict_clip(model, clip)["rain"] < 0.05   # a mean over windows would give ~0.4


def test_augment_keeps_shape_and_range():
    rng = np.random.default_rng(0)
    C = [rng.standard_normal(SR).astype(np.float32) * 0.1 for _ in range(4)]
    out = augment(C, np.array(["rain", "rain", "no_rain", "no_rain"]), rng)
    assert len(out) == 4 and all(o.shape == (SR,) and np.abs(o).max() <= 1 for o in out)


def test_import_esc50_zip(tmp_path):
    import io, zipfile, soundfile as sf
    zp = tmp_path / "esc.zip"
    rows = "filename,fold,target,category,esc10,src_file,take\n"
    with zipfile.ZipFile(zp, "w") as z:
        for i, (cat, src) in enumerate([("rain", "1"), ("rain", "1"), ("engine", "2"), ("dog", "3")]):
            b = io.BytesIO(); sf.write(b, np.r_[np.full(100, 0.5), np.zeros(50)], 44100, format="WAV")
            z.writestr(f"ESC-50-master/audio/{i}.wav", b.getvalue())
            rows += f"{i}.wav,1,0,{cat},False,{src},A\n"
        z.writestr("ESC-50-master/meta/esc50.csv", rows)
    assert import_esc50_zip(zp, tmp_path / "d") == {"rain": 1, "no_rain": 1}   # same source joined; dog ignored
    y, _ = sf.read(tmp_path / "d" / "rain" / "esc50_rain_1.wav")
    assert len(y) == 200   # trailing silence trimmed from both clips


def test_import_fsd50k(tmp_path):
    import soundfile as sf
    gt, audio = tmp_path / "FSD50K.ground_truth", tmp_path / "FSD50K.dev_audio"
    gt.mkdir(); audio.mkdir()
    (gt / "dev.csv").write_text('fname,labels,mids,split\n1,"Rain,Water",x,train\n2,"Speech,Human_voice",x,train\n3,Guitar,x,train\n')
    for f in "123":
        sf.write(audio / f"{f}.wav", np.zeros(10), 16000)
    assert import_fsd50k(tmp_path, tmp_path / "d") == {"rain": 1, "no_rain": 1}


def test_weather_check(tmp_path):
    for label, stamp in [("rain", "20261121-141200"), ("no_rain", "20261121-143100"), ("rain", "20261121-150000")]:
        (tmp_path / label).mkdir(exist_ok=True)
        (tmp_path / label / f"{stamp}_hand.wav").write_bytes(b"")
    precip = lookup({"minutely_15": {"time": ["2026-11-21T14:00", "2026-11-21T14:30", "2026-11-21T15:00"],
                                     "precipitation": [0.4, 0.0, 0.0]}})
    assert check(tmp_path, precip) == (2, 3)   # the 15:00 "rain" recording disagrees


def test_category_from_filename():
    from rainsense.train import _category
    assert _category("esc50_sea_waves_1234.wav") == "sea_waves"
    assert _category("esc50_engine_99.wav") == "engine"
    assert _category("20251121-101010_park.wav") == "other"
