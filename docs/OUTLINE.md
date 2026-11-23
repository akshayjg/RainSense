# Rain-mic: crowdsourced rain sensing (outline)

Prototype outline. Not a spec.

## 1. Goal

Build a street-level rain map from phones people already carry. The detector is a small module inside an existing, widely used app. It listens for a few seconds at natural moments, like waiting at a bus stop or for a cab, and reports whether it hears rain.

- **Output:** for each ~500 m grid cell and each 10-minute window, *dry* or *rain*. Coarse intensity (light/heavy) comes later.
- **Not goals:** intensity from a single phone, always-on background listening, background use on iOS.

Why it can work: a single phone is often wrong (it may be in a pocket, in a vehicle, or next to traffic). Many phones in the same area agreeing is reliable.

## 2. Data flow

```
Host app open ──> trigger ──> gate checks ──> record 5 s ──> classify on phone
                                   │                              │
                               (skip clip)          {cell, time, rain_prob, quality}
                                                                  │
                                                     host app backend ──> aggregator ──> rain map / API
```

No audio ever leaves the phone.

## 3. Components

| # | Component | Runs on | Exists today? |
|---|-----------|---------|---------------|
| 1 | Detector module | phone | Mostly: `FeatureExtractor`, `RainClassifier`, `Audio.kt` |
| 2 | Trigger (when to sample) | phone, host app decides | No |
| 3 | Gate (skip bad clips) | phone | No |
| 4 | Model | phone | Prototype (logistic regression); YAMNet being evaluated |
| 5 | Report | phone → host backend | No |
| 6 | Aggregator | server | No |
| 7 | Map / API output | server | No |

**3.1 Detector module.** This is the part you own. It's a library with one call. It has no UI, asks for no permissions and does no networking: the host app already handles those.

```kotlin
// sketch
object Pitterpat {
    suspend fun sample(context: Context): RainReading?   // null = skipped (gated out, mic busy, silent)
}
data class RainReading(val rainProb: Float, val quality: Float, val timeMs: Long)
```

**3.2 Trigger.** The host app calls `sample()` when its app is open and the user is probably outside: waiting for a ride, at a transit stop, or a delivery rider between orders. Rate-limit it to at most once every few minutes per device.

**3.3 Gate.** Skip the clip when any of these is true:
- speed > ~20 km/h (in a vehicle)
- proximity sensor covered and dark (in a pocket or bag)
- connected to Wi-Fi (probably indoors)
- the clip is silent (mic in use by something else)
- later: YAMNet says "inside, small room"

**3.4 Model.** Start with the current 19-feature logistic regression. Switch to YAMNet if `rainmic.compare` shows it does better on real recordings.

**3.5 Report.** Send `{cell_id, time_bucket, rain_prob, quality}`. No audio, no user ID, location rounded to the grid cell.

**3.6 Aggregator.** For each cell and 10-minute window, require at least *k* reports (e.g. 3). Take the quality-weighted mean of `rain_prob`, then threshold it to get rain or dry. Intensity comes later: calibrate the cell's rain fraction against rain gauges or radar.

**3.7 Output.** A map layer, plus a simple lookup for the host app ("is it raining at this pickup point?").

## 4. Constraints

- **Mic only while the app is open.** Android 11+ blocks background mic access, and iOS effectively forbids it.
- **Disclosure.** Users must be clearly told the app listens briefly, and app store policy reviews apply.
- **Privacy.** All processing happens on the phone, location is coarse, and reports are anonymous.
- **Battery.** 5 s of audio plus a tiny model every few minutes costs very little. Check this on low-end phones.

## 5. Prototype plan

| Step | What | Done when |
|------|------|-----------|
| M1 | Per-clip detection: record labelled clips in real situations (hand, pocket, umbrella, roadside), run `rainmic.compare` | ≥ ~75% balanced accuracy on clips that pass the gate |
| M2 | Package the detector as a library with the `sample()` API above, plus the gate | A test app can call it in one line |
| M3 | Field test: 5–10 phones, a simple aggregator (Python + SQLite), compare to radar or the nearest weather station | The map broadly agrees with radar during real rain |
| M4 | Pitch to a host app, using M3's results as evidence | |

## 6. Open questions

- Which host app? Delivery, ride-hailing and transit apps fit best.
- Who runs the aggregator: you or the host app?
- What ground truth do you use for intensity calibration (gauges, radar, weather stations)?
- What happens when the host app itself uses the mic (calls, voice search)?
- iOS: only while the app is open, or skip iOS entirely for the prototype?
