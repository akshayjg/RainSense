"""Feature extraction (numpy only, so it can be mirrored exactly in Kotlin).

Audio: mono, 16 kHz, float in [-1, 1]. Frames of 1024 samples, hop 512, periodic Hann.
Features (19):
  16 log-spaced band log-energies (100 Hz - 8 kHz, dB, mean over frames)
  spectral flatness over 1-8 kHz (mean over frames)
  clip RMS level (dB)
  std-dev over frames of the 2-8 kHz frame energy (dB)  -> how "steady" the sound is
"""
import numpy as np

SR = 16000
N_FFT = 1024
HOP = 512
N_BANDS = 16
FMIN = 100.0
FMAX = 8000.0
CLIP_SECONDS = 5

# FFT-bin ranges [lo, hi) per band (floor(edge/bin_width + 0.5)). Hard-coded so the
# Android port can copy them verbatim; tests check they match the formula.
BAND_BINS = [(6, 8), (8, 11), (11, 15), (15, 19), (19, 25), (25, 33), (33, 44), (44, 57), (57, 75), (75, 99), (99, 130), (130, 171), (171, 225), (225, 296), (296, 389), (389, 512)]
FLAT_LO, FLAT_HI = 64, 513      # ~1 kHz - 8 kHz
HF_LO, HF_HI = 128, 513         # ~2 kHz - 8 kHz

FEATURE_NAMES = [f"band_{i}" for i in range(N_BANDS)] + ["flatness", "rms_db", "hf_var_db"]


def _hann():
    n = np.arange(N_FFT)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)


def compute_features(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    spec = np.abs(np.fft.rfft(frames * _hann(), axis=1)) ** 2  # (T, 513)

    bands = np.array([spec[:, lo:hi].mean(axis=1) for lo, hi in BAND_BINS]).T  # (T, 16)
    band_db = 10 * np.log10(bands.mean(axis=0) + 1e-10)

    flat_part = spec[:, FLAT_LO:FLAT_HI] + 1e-12
    flatness = np.mean(np.exp(np.mean(np.log(flat_part), axis=1)) / np.mean(flat_part, axis=1))

    rms_db = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-8)

    hf_db = 10 * np.log10(spec[:, HF_LO:HF_HI].sum(axis=1) + 1e-10)
    hf_var = np.std(hf_db)

    return np.concatenate([band_db, [flatness, rms_db, hf_var]])
