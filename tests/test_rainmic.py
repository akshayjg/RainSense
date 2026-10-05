import json
import numpy as np
from rainmic.features import compute_features, FEATURE_NAMES, SR, BAND_BINS, N_FFT, N_BANDS, FMIN, FMAX
from rainmic.synth import generate
from rainmic.train import train
from rainmic.predict import predict_features


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
    p = predict_features(model, np.zeros(len(FEATURE_NAMES)))
    assert abs(sum(p.values()) - 1) < 1e-6
