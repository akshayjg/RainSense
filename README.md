# rain-mic

Detect rain from a phone microphone, on the phone, as a library any app can embed. Many phones reporting together give a street-level rain map (see [docs/OUTLINE.md](docs/OUTLINE.md)).

Raindrops produce broadband, impulsive noise, mostly 1-8 kHz. This repo has:

- **`rainmic/`**: Python pipeline: dataset import, augmentation, training with calibration, prediction, detector comparison, and weather-based validation.
- **`android/rainsense/`**: the embeddable Android library (**Pitterpat**). One call: `Pitterpat.sample(context)`.
- **`android/app/`**: a demo/test harness for the library (monitoring service, "sample now", labelled data collection).

The model is a small logistic regression on 19 hand-crafted spectral features per 1 s window, exported as plain JSON (`model.json`). No TFLite, no network, no audio leaves the phone.

## Quick start (Python)

```bash
pip install -r requirements.txt
python -m rainmic.datasets esc50 --out data          # real rain + 20 kinds of confusers (~600 MB download)
python -m rainmic.train --data data --out models     # writes model.json + metrics.json
python -m rainmic.predict some_clip.wav --model models/model.json
pytest -q
```

`models/model.json` is trained on ESC-50 (rain vs 20 confuser categories). On held-out source files: 87% accuracy on 5 s clips (153 of 175, only 12 of them rain; [diagram/carousel/src/analysis.json](diagram/carousel/src/analysis.json)) and 82% balanced accuracy per 1 s window ([models/metrics.json](models/metrics.json)). Weak spots: sea waves, vacuum cleaners and toilet flushes are often mistaken for rain. **ESC-50 is CC BY-NC**, so this model is for research and demos, not commercial use.

## How the model is built

- **Real data.** `rainmic.datasets esc50` downloads ESC-50; `rainmic.datasets fsd50k <path>` imports a local FSD50K copy. Both write `data/rain/` and `data/no_rain/`. ESC-50 clips from one source are joined into one file, so the train/test split can't leak.
- **Augmentation** (`data.augment`). Each training window gets one extra copy: rain mixed with a random no-rain sound at -3 to 10 dB SNR, or low-passed like a phone in a pocket, plus random gain.
- **1 s windows + median.** The model scores 1 s windows; a 5 s clip's result is the median over its 5 windows, so a horn or a shout can't flip it.
- **Calibration.** A temperature and per-class offset are fit on held-out data and folded into the exported weights, so `rainProb` is calibrated for an even rain/no-rain prior. The aggregator can apply the local base rate.
- **Validation against real weather.** `python -m rainmic.weather --lat .. --lon .. data/` (or `rain_log.csv`) compares labels or predictions with Open-Meteo 15-minute precipitation (free, no key, last 92 days).

## Training on your own recordings

Put WAV files (any sample rate; resampled to 16 kHz mono) in class folders: `data/no_rain/*.wav`, `data/rain/*.wav` (or `light/`, `moderate/`, `heavy/`). Record on the phones you'll deploy on, and include plenty of confusers.

The demo app's **Record ...** buttons save labelled 15 s WAVs through the same mic path the detector uses, named `<time>_<place>.wav`:

```bash
adb pull /sdcard/Android/data/com.example.rainmic/files/recordings data
python -m rainmic.weather --lat 52.37 --lon 4.90 data     # check the labels against real weather
```

`python -m rainmic.compare --data data` compares the hand-crafted model with YAMNet (`pip install -r requirements-yamnet.txt`, Python 3.11/3.12), broken down by phone placement.

## Android library

```kotlin
when (val r = Pitterpat.sample(context)) {          // suspend; call while the host app is in the foreground
    is RainResult.Reading -> report(r.rainProb)      // 0..1, calibrated
    is RainResult.Skipped -> log(r.reason)           // e.g. "covered and dark (probably in a pocket or bag)"
}
```

- Add `implementation(project(":rainsense"))`. The host app asks for `RECORD_AUDIO` at runtime; the library declares it.
- **Context gate** (before recording): skips when on Wi-Fi (probably indoors), moving faster than ~25 km/h (uses the host's location permission if it has one), or when the proximity sensor is covered and it's dark (pocket or bag). After recording, an all-silent clip (mic busy) is skipped too. `sample(context, useGate = false)` turns the gate off.
- Records with the `UNPROCESSED` mic source when supported (no gain control or noise suppression), else `VOICE_RECOGNITION`.
- The model ships in the library's assets: `cp models/model.json android/rainsense/src/main/assets/model.json`.

Build and test (Android SDK + JDK 17+; Android Studio generates the Gradle wrapper):

```bash
gradle -p android :rainsense:testDebugUnitTest :app:assembleDebug
```

`ParityTest.kt` checks the Kotlin feature extractor against `tests/fixtures/parity_features.txt`, the same fixture the Python tests check, so the two can't drift apart silently.

## Limitations

- Trained on ESC-50 clips, not phone recordings. Phone mics, pockets and surfaces (tin roof, leaves, umbrella) differ; collect real recordings before trusting it in the field.
- Steady broadband noise (sea, vacuum cleaners, flushing) is confused with rain.
- Intensity classes need labels calibrated against a rain gauge.

## License

MIT for the code. The shipped model is trained on ESC-50 (CC BY-NC 3.0).
