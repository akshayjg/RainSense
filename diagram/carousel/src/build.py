"""Builds slides.html: a 9-slide technical overview of RainSense (1080x1350, LinkedIn document format).
Every number comes from models/metrics.json; the spectrograms are computed from ESC-50 audio.
Usage: python build.py <repo> <dir holding data_esc50/>"""
import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

REPO, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO))
from rainsense.data import load_mono_16k  # noqa: E402

BRAND = "rainsense"
N = 9
AUTHORS = []  # e.g. ["Your Name", "Collaborator 1"]; shown on the cover when filled in
metrics = json.loads((REPO / "models" / "metrics.json").read_text())
op = {o["threshold"]: o for o in metrics["operating_points"]}
chosen = op[metrics["rain_threshold"]]
noise = metrics["street_noise_2s_detected"]
DATA = SCRATCH / "data_esc50"


def top(label, i, inv=False):
    return (f'<header class="top"><span><b class="brand">{BRAND}</b><i class="sep">/</i>{label}</span>'
            f'<span>{i:02d} / {N:02d}</span></header>')


def pct(x):
    return f"{round(x * 100)}%"


def fig(n, text):
    return f'<p class="fig"><b>Fig. {n}</b> {text}</p>'


def band(r, c):
    """Illustrative rain band crossing an 8x8 neighbourhood grid."""
    return max(0.0, 1 - abs((c - 0.8 * r) - 1.5) / 3.2)


slides = []

# 1 cover -----------------------------------------------------------
deco = "".join(f'<i style="--v:{band(r, c):.2f}"></i>' for r in range(8) for c in range(8))
byline = " · ".join(AUTHORS + ["github.com/akshayjg/RainSense"])
slides.append(f'''
<section class="slide inv cover r">
  {top("technical overview", 1, True)}
  <h1>RainSense</h1>
  <p class="subtitle">Street-level rainfall sensing from smartphone audio</p>
  <p class="lede">On-device acoustic classification, calibrated for crowdsourced aggregation into hyper-local rain maps.</p>
  <div class="deco">{deco}</div>
  <dl class="keyband">
    <div><dt>{pct(metrics["clip_accuracy"])}</dt><dd>clip accuracy<br>5-fold CV</dd></div>
    <div><dt>{pct(chosen["false_alarms"])}</dt><dd>false-alarm rate<br>at threshold {metrics["rain_threshold"]:.2f}</dd></div>
    <div><dt>1e-6</dt><dd>Python–Kotlin<br>feature parity</dd></div>
    <div><dt>0 B</dt><dd>audio leaves<br>the device</dd></div>
  </dl>
  <p class="byline mono">{byline}</p>
</section>''')

# 2 problem ---------------------------------------------------------
fine = "".join(f'<i class="{"rain" if band(r, c) >= 0.5 else ""}" style="--v:{band(r, c):.2f}"></i>' for r in range(8) for c in range(8))
slides.append(f'''
<section class="slide r">
  {top("1 · problem", 2)}
  <h2>Rainfall is reported per city. It falls per street.</h2>
  <p class="lede">Forecasts and gauge networks resolve rain at the scale of districts. Phones already on the street form a dense, untapped sensor network.</p>
  <div class="cmp">
    <div><h3>conventional reporting</h3><div class="coarse"><span>One value for the area</span></div></div>
    <div><h3>rainsense</h3><div class="fine">{fine}</div></div>
  </div>
  {fig(1, "Illustration. The same neighbourhood at district resolution and at the resolution of pooled phone readings; outlined cells are classified as raining.")}
</section>''')

# 3 architecture ----------------------------------------------------
stages = [("Context gate", "Wi-Fi, speed above 25 km/h, proximity and light sensors"),
          ("Capture", "5 s mono at 16 kHz, unprocessed microphone source"),
          ("Features", "19 spectral features per 1 s window"),
          ("Model", "Softmax regression with folded-in calibration"),
          ("Decision", f"Median of 5 windows, threshold {metrics['rain_threshold']:.2f}")]
rows = "".join(f'<li><span class="n mono">{i + 1:02d}</span><b>{t}</b><span class="d">{d}</span></li>' for i, (t, d) in enumerate(stages))
slides.append(f'''
<section class="slide r">
  {top("2 · architecture", 3)}
  <h2>Inference runs entirely on the device.</h2>
  <p class="lede">Only a calibrated probability leaves the phone. Aggregation pools readings by map cell and time.</p>
  <div class="arch">
    <div class="tier"><h3>on device · Kotlin library</h3><ol class="stages">{rows}</ol></div>
    <div class="arrow mono">{{ cell, slot, p }}</div>
    <div class="tier srv"><h3>aggregation · design</h3>
      <ol class="stages"><li><span class="n mono">06</span><b>Pooling</b><span class="d">Map cell × 10 min, at least 3 readings</span></li>
      <li><span class="n mono">07</span><b>Rain map</b><span class="d">Cells shrink as phone density grows</span></li></ol></div>
  </div>
  {fig(2, "Pipeline. Stages 01–05 ship in the Android library behind a single call, RainSense.sample(context).")}
</section>''')


# 4 signal ----------------------------------------------------------
def spectrogram(path, secs=5, rows=128, cols=240, n=1024):
    y = load_mono_16k(path)[:secs * 16000]
    hop = (len(y) - n) // (cols - 1)
    frames = np.stack([y[i * hop:i * hop + n] * np.hanning(n) for i in range(cols)])
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / 16000)
    edges = np.linspace(0, 8000, rows + 1)  # linear axis, so the high-frequency share is visible
    S = np.stack([P[:, (f >= lo) & (f < hi)].mean(axis=1) for lo, hi in zip(edges[:-1], edges[1:])])
    return 10 * np.log10(S + 1e-12)


# two typical recordings: broadband rain, and an engine whose energy falls ~26 dB above 4 kHz
specs = [("Rain", spectrogram(DATA / "rain" / "esc50_rain_101676.wav")),
         ("Engine noise", spectrogram(DATA / "no_rain" / "esc50_engine_106487.wav"))]
vmax = max(s.max() for _, s in specs)
RANGE_DB = 45
STOPS = [(0, (13, 18, 36)), (.45, (26, 75, 224)), (.8, (143, 176, 255)), (1, (255, 255, 255))]


def png_data_uri(S):
    """Spectrogram as a PNG data URI (high frequencies at the top). Plain zlib, no imaging library needed."""
    v = np.clip((S - (vmax - RANGE_DB)) / RANGE_DB, 0, 1)[::-1]
    rgb = np.stack([np.interp(v, [t for t, _ in STOPS], [c[k] for _, c in STOPS]) for k in range(3)], axis=-1).astype(np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def spec_svg(S):
    return f'<img class="spec" alt="" src="{png_data_uri(S)}">'


spec_html = "".join(f'<div><h3>{name}</h3><div class="specbox">{spec_svg(S)}</div>'
                    f'<div class="ax mono"><span>0 s</span><span>0–8 kHz</span><span>5 s</span></div></div>'
                    for name, S in specs)
slides.append(f'''
<section class="slide r">
  {top("3 · signal", 4)}
  <h2>Rain has a broadband signature.</h2>
  <p class="lede">Droplet impacts spread energy across the whole band up to 8 kHz. Engine and traffic noise fall away sharply above 1 kHz.</p>
  <div class="specs">{spec_html}</div>
  <dl class="facts">
    <div><dt>16</dt><dd>log band energies<br>100 Hz – 8 kHz</dd></div>
    <div><dt>3</dt><dd>flatness, level,<br>2–8 kHz variability</dd></div>
    <div><dt>1 s</dt><dd>analysis window,<br>median over 5</dd></div>
  </dl>
  {fig(3, "Log-power spectrograms of two ESC-50 recordings, 5 s each, frequency rising upward, shared 45 dB colour range.")}
</section>''')

# 5 method ----------------------------------------------------------
method = [("Data", f"ESC-50: {metrics['files']} recordings, rain against 20 confusable classes; {metrics['windows']:,} one-second windows"),
          ("Augmentation", "Street noise mixed at −3 to 10 dB SNR, pocket low-pass at 0.8–2.5 kHz, random gain"),
          ("Validation", "5-fold cross-validation grouped by source recording; no source appears in both training and test"),
          ("Calibration", f"Temperature and class offset fitted inside each training fold (deployed T = {metrics['temperature']:.2f})"),
          ("Decision rule", f"Threshold {metrics['rain_threshold']:.2f}, the lowest that holds clip false alarms at or below 10%")]
mrows = "".join(f'<li><span class="n mono">{i + 1:02d}</span><b>{t}</b><span class="d">{d}</span></li>' for i, (t, d) in enumerate(method))
slides.append(f'''
<section class="slide r">
  {top("4 · method", 5)}
  <h2>Evaluated so the test data never shapes the model.</h2>
  <p class="lede">Every recording is scored exactly once, by a model and a calibration fitted without it.</p>
  <ol class="stages method mid">{mrows}</ol>
</section>''')

# 6 results ---------------------------------------------------------
X0, Y0, CW, CH = 120, 540, 730, 480
xmax = 0.25
pts = [(o["false_alarms"], o["rain_caught"], t) for t, o in sorted(op.items())]


def P(fa, rc):
    return X0 + min(fa, xmax) / xmax * CW, Y0 - rc * CH


line = " ".join(f"{P(fa, rc)[0]:.0f},{P(fa, rc)[1]:.0f}" for fa, rc, _ in pts)
dots = "".join(f'<circle cx="{P(fa, rc)[0]:.0f}" cy="{P(fa, rc)[1]:.0f}" r="{12 if t == metrics["rain_threshold"] else 7}" '
               f'class="{"sel" if t == metrics["rain_threshold"] else "pt"}"/>' for fa, rc, t in pts)
tlabels = "".join(f'<text x="{P(fa, rc)[0] + 14:.0f}" y="{P(fa, rc)[1] + 30:.0f}" class="tl">{t:.2f}</text>'
                  for fa, rc, t in pts if t in (0.5, 0.6, 0.8, 0.9))
cx, cy = P(chosen["false_alarms"], chosen["rain_caught"])
capx = X0 + 0.10 / xmax * CW
grid = "".join(f'<line x1="{X0}" y1="{Y0 - g * CH:.0f}" x2="{X0 + CW}" y2="{Y0 - g * CH:.0f}" class="grid"/>'
               f'<text x="{X0 - 16}" y="{Y0 - g * CH + 7:.0f}" class="tick" text-anchor="end">{int(g * 100)}%</text>' for g in (0, .25, .5, .75, 1))
grid += "".join(f'<text x="{X0 + g / xmax * CW:.0f}" y="{Y0 + 36}" class="tick" text-anchor="middle">{int(round(g * 100))}%</text>'
                for g in (0, .05, .1, .15, .2, .25))
lo, hi = min(metrics["clip_accuracy_per_fold"]), max(metrics["clip_accuracy_per_fold"])
slides.append(f'''
<section class="slide r">
  {top("5 · results", 6)}
  <h2>An operating point chosen for the map.</h2>
  <p class="lede">Many phones report per cell, so a false alarm costs more than a missed reading. The threshold trades recall for precision.</p>
  <svg class="roc" viewBox="0 0 904 620">
    {grid}
    <line x1="{capx:.0f}" y1="{Y0 - CH}" x2="{capx:.0f}" y2="{Y0}" class="cap"/>
    <text x="{capx + 10:.0f}" y="{Y0 - CH + 26}" class="tl">10% cap</text>
    <polyline points="{line}" class="curve"/>{dots}{tlabels}
    <text x="{cx - 20:.0f}" y="{cy - 24:.0f}" class="sl" text-anchor="end">deployed {metrics["rain_threshold"]:.2f}</text>
    <text x="{X0 + CW / 2}" y="{Y0 + 78}" class="axis" text-anchor="middle">non-rain clips flagged as rain</text>
    <text transform="translate(30 {Y0 - CH / 2}) rotate(-90)" class="axis" text-anchor="middle">rain clips detected</text>
  </svg>
  <dl class="facts">
    <div><dt>{pct(metrics["clip_accuracy"])}</dt><dd>clip accuracy,<br>folds {pct(lo)}–{pct(hi)}</dd></div>
    <div><dt>{pct(metrics["clip_balanced_accuracy"])}</dt><dd>balanced<br>accuracy</dd></div>
    <div><dt>{metrics["logloss_raw"]:.2f} → {metrics["logloss_calibrated"]:.2f}</dt><dd>log-loss,<br>with calibration</dd></div>
  </dl>
  {fig(4, f"Detection versus false alarms across decision thresholds, {metrics['clips']} cross-validated 5 s clips.")}
</section>''')


# 7 robustness ------------------------------------------------------
def hbar(label, v, n, strong=False):
    return (f'<div class="hb{" strong" if strong else ""}"><span class="hb-l">{label}</span>'
            f'<span class="hb-t"><i style="width:{v * 100:.1f}%"></i></span><span class="hb-v mono">{round(v * n)}/{n}</span></div>')


slides.append(f'''
<section class="slide r">
  {top("6 · robustness", 7)}
  <h2>Median aggregation rejects transient noise.</h2>
  <p class="lede">In each held-out rain clip, two of five windows were replaced with horns, sirens, engines or aircraft at twice the rain's loudness.</p>
  <div class="chart mid">
    <div class="chart-h"><b>Rain clips still detected</b><span class="mono">{noise["median"][1]} clips</span></div>
    {hbar("Median of 5 windows", *noise["median"], True)}
    {hbar("Mean of 5 windows", *noise["mean"])}
    {hbar("Single window on the noise", *noise["noise_window"])}
  </div>
  {fig(5, "Noise-injection test, run inside each cross-validation fold using test-fold audio only.")}
</section>''')

# 8 engineering -----------------------------------------------------
slides.append(f'''
<section class="slide r">
  {top("7 · engineering", 8)}
  <h2>Built to ship, not only to demo.</h2>
  <p class="lede">The model is trained in Python and runs in Kotlin. Both are held to the same numbers.</p>
  <ol class="stages method mid">
    <li><span class="n mono">01</span><b>Feature parity</b><span class="d">All 19 features agree between Python and Kotlin to within 1e-6 on a shared fixture</span></li>
    <li><span class="n mono">02</span><b>Continuous integration</b><span class="d">Python tests, Android unit tests and an APK build on every push</span></li>
    <li><span class="n mono">03</span><b>Portable model</b><span class="d">Plain-JSON weights with calibration folded in; no ML runtime on the phone</span></li>
    <li><span class="n mono">04</span><b>Privacy by design</b><span class="d">Audio is scored in memory and discarded; reports carry a cell, a time slot and a probability</span></li>
  </ol>
</section>''')

# 9 next ------------------------------------------------------------
slides.append(f'''
<section class="slide inv r">
  {top("8 · next", 9, True)}
  <h2>From benchmark to street.</h2>
  <p class="lede">The library, model and evaluation are open source.</p>
  <ol class="next mid">
    <li>Field recordings across devices, placements and surfaces</li>
    <li>Benchmark against a pretrained audio model (YAMNet)</li>
    <li>Intensity estimation, calibrated against rain gauges</li>
    <li>Pilot deployment inside an app used outdoors</li>
  </ol>
  <p class="cta mono">github.com/akshayjg/RainSense</p>
</section>''')

CSS = (Path(__file__).parent / "slides.css").read_text(encoding="utf-8")
html = f'''<!doctype html><html lang="en" data-radius="sharp"><head><meta charset="utf-8">
<title>RainSense technical overview</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400..700&family=JetBrains+Mono:wght@400..500&display=block" rel="stylesheet">
<style>{CSS}</style></head><body>
{"".join(slides)}
<script>const s=new URLSearchParams(location.search).get("s");if(s)document.querySelectorAll(".slide").forEach((e,i)=>{{if(i+1!=+s)e.remove()}})</script>
</body></html>'''
(Path(__file__).parent / "slides.html").write_text(html, encoding="utf-8")
print("slides:", len(slides))
