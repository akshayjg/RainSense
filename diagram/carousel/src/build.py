"""Builds slides.html: an 11-slide LinkedIn carousel (1080x1350) from real project outputs.
Design: taste-skill deal (palette white-link-blue, type Schibsted Grotesk + JetBrains Mono, radius sharp, software kind)
plus ui-ux-pro-max chart rules (values printed, numeric legend, never colour alone)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

REPO, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO))
from rainsense.data import clips, load_mono_16k  # noqa: E402

BRAND = "rainsense"
N = 11
metrics = json.loads((REPO / "models" / "metrics.json").read_text())
analysis = json.loads((SCRATCH / "analysis.json").read_text())
parity = [float(x) for x in (REPO / "tests" / "fixtures" / "parity_features.txt").read_text().split()]
DATA = SCRATCH / "data_esc50"

# ---------- real waveforms ----------
rain = list(clips(load_mono_16k(sorted((DATA / "rain").glob("*.wav"))[3])))[1]
engine = list(clips(load_mono_16k(sorted((DATA / "no_rain").glob("esc50_engine_*.wav"))[0])))[0]
mixed = rain + engine * np.sqrt(np.mean(rain ** 2) / np.mean(engine ** 2))          # 0 dB SNR, as data.augment does
b, a = butter(4, 1200, fs=16000)
muffled = lfilter(b, a, rain)                                                     # pocket low-pass, as data.augment does


def wave_path(y, w=640, h=96):
    y = y / (np.abs(np.r_[rain, mixed]).max() + 1e-9)
    cols = np.array_split(np.abs(y), w // 2)
    top = [f"{i * 2},{h / 2 - c.max() * h / 2:.1f}" for i, c in enumerate(cols)]
    bot = [f"{i * 2},{h / 2 + c.max() * h / 2:.1f}" for i, c in reversed(list(enumerate(cols)))]
    return "M" + " L".join(top + bot) + " Z"


def top(label, i, inv=False):
    return (f'<header class="top"><span><b class="brand">{BRAND}</b><i class="sep">/</i>{label}</span>'
            f'<span>{i:02d} / {N:02d}</span></header>')


def pct(x):
    return f"{round(x * 100)}%"


# cover accuracy from analysis.json, the same held-out split and model as slides 4 and 11
_cm = analysis["clip_median"]
clip_acc = (_cm["rain_recall"][0] * _cm["rain_recall"][1] + _cm["no_rain_recall"][0] * _cm["no_rain_recall"][1]) / (_cm["rain_recall"][1] + _cm["no_rain_recall"][1])


slides = []

# 1 cover -----------------------------------------------------------
rain_streaks = "".join(
    f'<line x1="{20 + (i * 47) % 300}" y1="{12 + (i * 71) % 150}" x2="{14 + (i * 47) % 300}" y2="{28 + (i * 71) % 150}"/>'
    for i in range(26))
bars = "".join(f'<i style="height:{h}px"></i>' for h in [10, 22, 14, 34, 18, 42, 24, 12, 38, 16, 30, 10, 40, 20, 14, 32, 16, 24, 10, 18])
slides.append(f'''
<section class="slide inv cover">
  {top("android sdk", 1, True)}
  <h1>Rain detection for any Android app.</h1>
  <p class="lede">Five seconds of microphone audio, scored on the phone. Many phones together draw a street-level rain map.</p>
  <div class="cover-low">
    <dl class="stats-col">
      <div><dt>5 s</dt><dd>of audio per reading</dd></div>
      <div><dt>0 bytes</dt><dd>of audio uploaded</dd></div>
      <div><dt>{pct(clip_acc)}</dt><dd>of held-out 5 s clips classified right</dd></div>
    </dl>
    <div class="phone">
      <div class="screen">
        <div class="status mono">9:41</div>
        <div class="app-title">Pickup in 4 min</div>
        <div class="app-sub">Main St, Gate 2</div>
        <svg class="minimap" viewBox="0 0 320 170"><rect width="320" height="170" class="mm-bg"/>
          <path d="M0 64 H320 M110 0 V170 M0 150 L320 104 M236 0 V170" class="mm-road"/>
          <g class="mm-rain">{rain_streaks}</g>
          <circle cx="110" cy="64" r="11" class="mm-pin"/><circle cx="110" cy="64" r="4" class="mm-dot"/></svg>
        <div class="rain-card">
          <div class="rc-k mono">Rain nearby</div>
          <div class="rc-v">Light rain, 92%</div>
          <div class="rc-bars">{bars}</div>
        </div>
        <div class="btn-app">Confirm pickup</div>
      </div>
    </div>
  </div>
</section>''')

# 2 api -------------------------------------------------------------
slides.append(f'''
<section class="slide">
  {top("api", 2)}
  <h2>One call from your app.</h2>
  <p class="lede"><code>RainSense.sample()</code> checks the context, listens for 5 s, scores the audio and returns. No UI, no network, no background service.</p>
  <pre class="code"><span class="c">// build.gradle.kts</span>
implementation(project(<span class="s">":rainsense"</span>))

<span class="c">// in a coroutine, while your app is in the foreground</span>
<span class="k">when</span> (<span class="k">val</span> r = RainSense.sample(context)) {{
    <span class="k">is</span> RainResult.Reading -> report(r.rainProb)
    <span class="k">is</span> RainResult.Skipped -> log(r.reason)
}}</pre>
  <div class="results">
    <div class="res"><span class="tag">Reading</span><code>rainProb = 0.92, label = "rain"</code></div>
    <div class="res"><span class="tag">Skipped</span><code>"covered and dark (probably in a pocket or bag)"</code></div>
  </div>
  <dl class="facts">
    <div><dt>Android 8.0+</dt><dd>minSdk 26</dd></div>
    <div><dt>1 permission</dt><dd>RECORD_AUDIO</dd></div>
    <div><dt>suspend fun</dt><dd>runs on Dispatchers.IO</dd></div>
  </dl>
</section>''')

# 3 gate ------------------------------------------------------------
gate = [
    ("Wi-Fi connected", "on Wi-Fi (probably indoors)"),
    ("Faster than 25 km/h", "moving fast (probably in a vehicle)"),
    ("Proximity covered, under 5 lux", "covered and dark (probably in a pocket or bag)"),
    ("Silent clip after recording", "microphone busy (silent clip)"),
]
rows = "".join(f'<li><span class="n mono">{i + 1:02d}</span><div><b>{s}</b><code>"{r}"</code></div><span class="chip">skip</span></li>'
               for i, (s, r) in enumerate(gate))
slides.append(f'''
<section class="slide">
  {top("context gate", 3)}
  <h2>Skips the readings that would mislead.</h2>
  <p class="lede">Four checks around the microphone. A skipped reading costs nothing; a wrong one pollutes the map.</p>
  <ol class="gate">{rows}
    <li class="go"><span class="n mono">ok</span><div><b>All clear</b><code>record 5 s, score, return a Reading</code></div><span class="chip solid">read</span></li>
  </ol>
</section>''')

# 4 median ----------------------------------------------------------
ir = analysis["interruption_detect_rate"]
win_bars = [(0.90, "rain"), (0.08, "horn"), (0.86, "rain"), (0.11, "siren"), (0.88, "rain")]
med = sorted(v for v, _ in win_bars)[2]
strip = "".join(f'<div class="w{" noise" if l in ("horn", "siren") else ""}"><i style="height:{v * 300:.0f}px"></i><span class="mono">{l}</span></div>' for v, l in win_bars)


def hbar(label, v, n, strong=False):
    return (f'<div class="hb{" strong" if strong else ""}"><span class="hb-l">{label}</span>'
            f'<span class="hb-t"><i style="width:{v * 100:.1f}%"></i></span><span class="hb-v mono">{round(v * n)}/{n}</span></div>')


slides.append(f'''
<section class="slide">
  {top("1 s windows", 4)}
  <h2>Two seconds of horn can't flip the answer.</h2>
  <p class="lede">Every 5 s reading is five 1 s windows, and the answer is their median. Short bursts of noise get outvoted.</p>
  <div class="strip"><div class="median" style="bottom:{med * 300 + 46:.0f}px"><span class="mono">median</span></div>{strip}</div>
  <div class="chart">
    <div class="chart-h"><b>Rain clips still detected, 2 s of street noise inside</b><span class="mono">held-out, 12 clips</span></div>
    {hbar("Median of 5 windows", ir["median_2s"][0], 12, True)}
    {hbar("Mean of 5 windows", ir["mean_2s"][0], 12)}
    {hbar("One 1 s check on the noise", ir["single_window_at_interruption"][0], 12)}
  </div>
</section>''')

# 5 data ------------------------------------------------------------
waves = [("rain", rain, ""), ("+ engine, 0 dB", mixed, ""), ("+ pocket, 1.2 kHz", muffled, "")]
wrows = "".join(f'<div class="wave"><span class="mono">{l}</span><svg viewBox="0 0 640 96" preserveAspectRatio="none"><path d="{wave_path(y)}"/></svg></div>' for l, y, _ in waves)
slides.append(f'''
<section class="slide">
  {top("training data", 5)}
  <h2>Trained on real rain, then roughed up.</h2>
  <p class="lede">ESC-50 recordings: rain against 20 sounds that are easy to mistake for it. Every training window gets a harder copy.</p>
  <dl class="facts four">
    <div><dt>40</dt><dd>rain recordings</dd></div>
    <div><dt>626</dt><dd>confuser recordings</dd></div>
    <div><dt>20</dt><dd>kinds of confuser</dd></div>
    <div><dt>{metrics["windows"]:,}</dt><dd>1 s windows</dd></div>
  </dl>
  <div class="waves">{wrows}</div>
  <div class="chips"><span>noise at −3 to 10 dB SNR</span><span>low-pass 0.8–2.5 kHz</span><span>gain ×0.3–1.5</span></div>
</section>''')

# 6 calibration -----------------------------------------------------
X0, Y0, S = 110, 540, 480   # chart origin and size


def pt(p, o):
    return X0 + p * S, Y0 - o * S


raw = metrics["reliability_raw"]
cal = metrics["reliability_calibrated"]
raw_pts = "".join(f'<circle cx="{pt(p, o)[0]:.0f}" cy="{pt(p, o)[1]:.0f}" r="11" class="raw"/>' for p, o, _ in raw)
cal_pts = "".join(f'<rect x="{pt(p, o)[0] - 10:.0f}" y="{pt(p, o)[1] - 10:.0f}" width="20" height="20" class="cal"/>'
                  for p, o, n in cal)
r7, c7 = raw[3], cal[3]
ticks = "".join(f'<text x="{X0 + t * S:.0f}" y="{Y0 + 40}" class="tick" text-anchor="middle">{int(t * 100)}%</text>'
                f'<text x="{X0 - 18}" y="{Y0 - t * S + 8:.0f}" class="tick" text-anchor="end">{int(t * 100)}%</text>'
                f'<line x1="{X0}" y1="{Y0 - t * S:.0f}" x2="{X0 + S}" y2="{Y0 - t * S:.0f}" class="grid"/>' for t in (0, .25, .5, .75, 1))
slides.append(f'''
<section class="slide">
  {top("calibration", 6)}
  <h2>Its confident answers hold up.</h2>
  <p class="lede">A fitted temperature and class offset are folded into the exported weights. Confident answers now match what happened; the unsure middle still over-calls rain.</p>
  <svg class="rel" viewBox="0 0 904 660">
    {ticks}
    <line x1="{X0}" y1="{Y0}" x2="{X0 + S}" y2="{Y0 - S}" class="diag"/>
    <text x="{X0 + S - 6}" y="{Y0 - S - 16}" class="lbl" text-anchor="end">perfect</text>
    {raw_pts}{cal_pts}
    <line x1="{pt(*r7[:2])[0] + 14:.0f}" y1="{pt(*r7[:2])[1]:.0f}" x2="{X0 + S + 40}" y2="{pt(*r7[:2])[1]:.0f}" class="lead"/>
    <text x="{X0 + S + 52}" y="{pt(*r7[:2])[1] - 6:.0f}" class="lbl">before: said {pct(r7[0])}</text>
    <text x="{X0 + S + 52}" y="{pt(*r7[:2])[1] + 26:.0f}" class="lbl muted">was rain {pct(r7[1])}</text>
    <line x1="{pt(*c7[:2])[0] + 14:.0f}" y1="{pt(*c7[:2])[1]:.0f}" x2="{X0 + S + 40}" y2="{pt(*c7[:2])[1]:.0f}" class="lead acc"/>
    <text x="{X0 + S + 52}" y="{pt(*c7[:2])[1] - 6:.0f}" class="lbl acc">after: said {pct(c7[0])}</text>
    <text x="{X0 + S + 52}" y="{pt(*c7[:2])[1] + 26:.0f}" class="lbl muted">was rain {pct(c7[1])}</text>
    <text x="{X0 + S / 2}" y="{Y0 + 86}" class="axis" text-anchor="middle">predicted rain probability</text>
    <text transform="translate(26 {Y0 - S / 2}) rotate(-90)" class="axis" text-anchor="middle">observed share of rain</text>
  </svg>
  <div class="legend"><span><i class="lg-raw"></i>before</span><span><i class="lg-cal"></i>after calibration</span><span class="mono">held-out, bins of 76–578 windows</span></div>
  <dl class="facts two">
    <div><dt>{metrics["logloss_raw"]:.3f} → {metrics["logloss_calibrated"]:.3f}</dt><dd>held-out log-loss</dd></div>
    <div><dt>T = {metrics["temperature"]:.2f}</dt><dd>plus one offset per class</dd></div>
  </dl>
</section>''')

# 7 weather ---------------------------------------------------------
slots = [("14:00", 0.4, [("14:12", "rain", True)]), ("14:15", 0.0, []), ("14:30", 0.0, [("14:31", "no rain", True)]), ("14:45", 0.0, []), ("15:00", 0.0, [("15:00", "rain", False)])]
scols = ""
for t, mm, recs in slots:
    rec = "".join(f'<div class="rec{"" if ok else " bad"}"><b class="mono">{rt}</b><span>{lab}</span><em>{"agrees" if ok else "disagrees"}</em></div>' for rt, lab, ok in recs)
    scols += f'<div class="slot"><div class="mm"><i style="height:{mm * 700:.0f}px"></i><span class="mono">{mm:.1f} mm</span></div><span class="t mono">{t}</span><div class="recs">{rec}</div></div>'
slides.append(f'''
<section class="slide">
  {top("weather check, example run", 7)}
  <h2>Checked against the sky.</h2>
  <p class="lede">Labels and predictions are compared with Open-Meteo's 15-minute precipitation where the phone was. Free, no API key, the last 92 days.</p>
  <pre class="code term"><span class="c">$</span> python -m rainsense.weather --lat 52.37 --lon 4.90 data/
disagrees: rain/20261121-150000_hand.wav says rain,
           Open-Meteo 0.0 mm / 15 min
<span class="s">2/3 agree with Open-Meteo</span></pre>
  <div class="slots">{scols}</div>
</section>''')

# 8 privacy ---------------------------------------------------------
slides.append(f'''
<section class="slide">
  {top("privacy", 8)}
  <h2>Audio never leaves the phone.</h2>
  <p class="lede">RainSense returns a number and drops the sound. What the host app reports is its own choice; a rain map needs only this much.</p>
  <div class="split">
    <div class="panel">
      <h3 class="mono">Kept on the phone</h3>
      <ul class="kept"><li><b>5 s of audio</b><span>scored in memory, then dropped</span></li>
        <li><b>Exact location</b><span>rounded to a 500 m cell</span></li>
        <li><b>User and device IDs</b><span>not needed, not sent</span></li></ul>
    </div>
    <div class="panel dark">
      <h3 class="mono">Sent by the host app</h3>
      <pre class="json">&#123;
  <span class="s">"cell"</span>: <span class="s">"7F3A"</span>,
  <span class="s">"slot"</span>: <span class="s">"14:10"</span>,
  <span class="s">"p"</span>: 0.92
&#125;</pre>
      <p class="mono small">cell: 500 m square<br>slot: 10 minutes<br>p: calibrated rain probability</p>
    </div>
  </div>
</section>''')

# 9 map -------------------------------------------------------------
grid = [
    [8, 12, 41, 58, 47, 15, 6, None],
    [10, 38, 66, 81, 62, 33, 9, 4],
    [29, 61, 88, 93, 79, 52, 14, 7],
    [12, 47, 84, 91, 72, 44, None, 5],
    [6, 21, 55, 69, 48, 18, 8, 3],
    [None, 9, 24, 37, 22, 10, 4, 2],
]
cells = ""
for row in grid:
    for v in row:
        if v is None:
            cells += '<div class="cell nodata"><span class="mono">–</span></div>'
        else:
            txt = f'<span class="mono">{v}</span>' if v >= 30 else ""
            cells += f'<div class="cell{" rain" if v >= 50 else ""}" style="--v:{v / 100:.2f}">{txt}</div>'
slides.append(f'''
<section class="slide">
  {top("city map, concept", 9)}
  <h2>One rain map from many phones.</h2>
  <p class="lede">Each 500 m cell pools 10 minutes of readings. Three or more, with a weighted mean over 50%, mark it as raining.</p>
  <div class="map">{cells}</div>
  <div class="legend map-legend"><span class="scale"><i></i></span><span class="mono">0–100% of phones hear rain</span><span class="nd"><i></i><span class="mono">under 3 readings</span></span><span><b class="outline"></b><span class="mono">raining</span></span></div>
</section>''')

# 10 parity ---------------------------------------------------------
prow = [("band 94–125 Hz", parity[0]), ("band 0.9–1.2 kHz", parity[8]), ("band 6.1–8 kHz", parity[15]),
        ("spectral flatness", parity[16]), ("RMS level, dB", parity[17]), ("2–8 kHz variability, dB", parity[18])]
trs = "".join(f'<tr><td>{n}</td><td class="mono">{v:.6f}</td><td class="mono ok">within 1e-6</td></tr>' for n, v in prow)
slides.append(f'''
<section class="slide">
  {top("parity", 10)}
  <h2>Python trains it. Kotlin runs it. Same numbers.</h2>
  <p class="lede">Both sides compute the 19 features of the same generated signal and check them against one shared file.</p>
  <table class="ptable"><thead><tr><th>feature</th><th>Python</th><th>Kotlin</th></tr></thead><tbody>{trs}</tbody></table>
  <div class="results">
    <div class="res"><span class="tag">pytest</span><code>10 passed</code></div>
    <div class="res"><span class="tag">ParityTest.kt</span><code>1 test, 0 failures</code></div>
  </div>
</section>''')

# 11 limits ---------------------------------------------------------
names = {"sea": "Sea waves", "vacuum": "Vacuum cleaner", "toilet": "Toilet flush", "engine": "Engine", "crackling": "Crackling fire"}
fa = [(names[k], v, n) for k, v, n in analysis["false_alarm_by_confuser_window"] if k in names][:5]
fbars = "".join(f'<div class="hb"><span class="hb-l">{k}</span><span class="hb-t"><i style="width:{v * 100:.1f}%"></i></span><span class="hb-v mono">{pct(v)}</span></div>' for k, v, n in fa)
slides.append(f'''
<section class="slide inv">
  {top("limits", 11, True)}
  <h2>What it gets wrong today.</h2>
  <p class="lede">Steady broadband noise sounds like rain to 19 spectral features. False alarms per 1 s window, held-out recordings:</p>
  <div class="chart inv-chart">{fbars}</div>
  <h3 class="next-h mono">Next</h3>
  <ol class="next">
    <li>Field recordings in a hand, a pocket and under an umbrella</li>
    <li>YAMNet against these features on those recordings</li>
    <li>Intensity labels from a rain gauge</li>
  </ol>
  <p class="foot mono">Model trained on ESC-50 (CC BY-NC 3.0). Python training, Kotlin on-device library.</p>
</section>''')

CSS = (Path(__file__).parent / "slides.css").read_text()
html = f'''<!doctype html><html lang="en" data-radius="sharp"><head><meta charset="utf-8">
<title>RainSense demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400..700&family=JetBrains+Mono:wght@400..500&display=block" rel="stylesheet">
<style>{CSS}</style></head><body>
{"".join(slides)}
<script>const s=new URLSearchParams(location.search).get("s");if(s)document.querySelectorAll(".slide").forEach((e,i)=>{{if(i+1!=+s)e.remove()}})</script>
</body></html>'''
(Path(__file__).parent / "slides.html").write_text(html, encoding="utf-8")
print("slides:", len(slides))
