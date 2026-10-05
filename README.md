# rain-mic

Detect rainfall (and rough intensity) from a phone microphone.

Raindrops produce broadband, impulsive noise, mostly 1-8 kHz. This repo has:

- **`rainmic/`**: Python pipeline: feature extraction, training, prediction, and a synthetic data generator.
- **`android/`**: Kotlin app that records a 5 s clip every minute in a foreground service, extracts the same 19 features, runs the exported model on-device, and appends results to `rain_log.csv`. No audio leaves the phone.

The model is a small multinomial logistic regression on 19 hand-crafted spectral features (16 log-spaced band energies, spectral flatness, RMS level, and variability of 2-8 kHz energy). It is exported as plain JSON (`model.json`), so there's no TFLite dependency. Once you have lots of real data, swapping in a CNN on mel spectrograms is the natural next step.

## Quick start (Python)

```bash
pip install -r requirements.txt
python -m rainrainmic.synth --out data     # demo only: synthetic audio
python -m rainmic.train --data data --out models
python -m rainmic.predict some_clip.wav --model models/model.json
pytest -q
```

> The synthetic data only proves the pipeline runs. The accuracy it reports means nothing for real rain. `models/model.json` in this repo is trained on it, so **retrain on real recordings before trusting it.**

## Training on real data

Put WAV files (any sample rate; they're resampled to 16 kHz mono) in class folders:

```
data/
  no_rain/   *.wav      # include lots of confusers: wind, traffic, fans, AC, speech, taps, showers
  light/     *.wav
  moderate/  *.wav
  heavy/     *.wav
```

- Start with just `no_rain/` and `rain/` (any two or more folder names work).
- Record on the phones you'll deploy on. Mics and cases differ a lot.
- Label against a nearby rain gauge or weather station if you want intensity classes.
- Files are cut into non-overlapping 5 s clips; train/test splitting is by file to avoid leakage.

## Android app

1. Copy your trained model: `cp models/model.json android/app/src/main/assets/model.json`
2. Open `android/` in Android Studio (it will generate the Gradle wrapper), run on a device.
3. Tap **Start monitoring**, grant mic + notification permission.

Notes:
- Android requires a visible notification for background mic use (foreground service).
- Log file: `/data/data/com.example.rainmic/files/rain_log.csv` (pull via Device Explorer).
- `FeatureExtractor.kt` is a hand port of `rainmic/features.py`. If you change one, change the other. Constants (band bins, frame size) must match.
- The Kotlin code has not been compiled or run on a device in the environment this repo was generated in. Expect to fix minor build issues, and sanity-check parity (see below).

### Checking Python/Kotlin parity

Record a WAV, print its features in Python (`compute_features`), then log the Kotlin features for the same samples. They should agree to ~1e-3 dB. If they don't, the classifier will silently misbehave.

## Limitations

- Phone/mic variability, surface type (tin roof vs leaves vs window), and wind/traffic false positives are the main error sources.
- Intensity is only reliable if calibrated against a gauge at your site.
- Indoors it works poorly.

## License

MIT
