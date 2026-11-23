"""Builds slides.html: a 6-slide LinkedIn carousel (1080x1350) from real project outputs.
Design: taste-skill deal (palette white-link-blue, type Schibsted Grotesk + JetBrains Mono, radius sharp, software kind)
plus ui-ux-pro-max chart rules (values printed, numeric legend, never colour alone)."""
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1])

BRAND = "rainsense"
N = 6
metrics = json.loads((REPO / "models" / "metrics.json").read_text())
false_alarms = next(o["false_alarms"] for o in metrics["operating_points"] if o["threshold"] == metrics["rain_threshold"])


def top(label, i, inv=False):
    return (f'<header class="top"><span><b class="brand">{BRAND}</b><i class="sep">/</i>{label}</span>'
            f'<span>{i:02d} / {N:02d}</span></header>')


def pct(x):
    return f"{round(x * 100)}%"




slides = []

# 1 cover -----------------------------------------------------------
rain_streaks = "".join(
    f'<line x1="{20 + (i * 47) % 300}" y1="{12 + (i * 71) % 150}" x2="{14 + (i * 47) % 300}" y2="{28 + (i * 71) % 150}"/>'
    for i in range(26))
bars = "".join(f'<i style="height:{h}px"></i>' for h in [10, 22, 14, 34, 18, 42, 24, 12, 38, 16, 30, 10, 40, 20, 14, 32, 16, 24, 10, 18])
slides.append(f'''
<section class="slide inv cover">
  {top("hyper-local rain", 1, True)}
  <h1>Rain, street by street.</h1>
  <p class="lede">Weather apps tell you about your city. RainSense tells you about your street: five seconds of phone audio, scored on the device, pooled from many phones.</p>
  <div class="cover-low">
    <dl class="stats-col">
      <div><dt>5 s</dt><dd>of audio per reading</dd></div>
      <div><dt>0 bytes</dt><dd>of audio uploaded</dd></div>
      <div><dt>1 call</dt><dd>from any Android app</dd></div>
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

# 2 idea ------------------------------------------------------------
fine = ""
for r in range(8):
    for c in range(8):
        v = max(0.0, 1 - abs((c - 0.8 * r) - 1.5) / 3.2)  # a rain band crossing the area diagonally
        fine += f'<i class="{"rain" if v >= 0.5 else ""}" style="--v:{v:.2f}"></i>'
slides.append(f'''
<section class="slide">
  {top("the idea", 2)}
  <h2>Your weather app sees the city. RainSense sees the street.</h2>
  <p class="lede">Forecasts and rain gauges give one answer for a whole area. Phones already out on the street can tell one block from the next.</p>
  <div class="cmp">
    <div><h3>weather app</h3><div class="coarse"><span>“Rain in your area”</span></div><p>One answer for everyone in it</p></div>
    <div><h3>rainsense</h3><div class="fine">{fine}</div><p>Raining on this street, dry on the next</p></div>
  </div>
  <p class="foot-light mono">Illustration: the same neighbourhood, two resolutions</p>
</section>''')

# 3 how -------------------------------------------------------------
steps = [("Listen", "The phone records five seconds of sound."),
         ("Check", "Readings from pockets, vehicles and indoors are skipped."),
         ("Decide", "The phone works out whether it hears rain."),
         ("Share", "Only a probability is sent. The audio is deleted.")]
rows = "".join(f'<li><span class="n mono">{i + 1:02d}</span><div><b>{t}</b><code>{d}</code></div><span></span></li>'
               for i, (t, d) in enumerate(steps))
slides.append(f'''
<section class="slide">
  {top("how it works", 3)}
  <h2>Five seconds, on the phone.</h2>
  <p class="lede">Any Android app can add RainSense with a single call.</p>
  <ol class="gate mid">{rows}</ol>
</section>''')

# 4 map ---------------------------------------------------------------
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
  {top("city map, concept", 4)}
  <h2>One rain map from many phones.</h2>
  <p class="lede">Readings pool into map cells every 10 minutes. More phones, smaller cells: down to a single street.</p>
  <div class="map">{cells}</div>
  <div class="legend map-legend"><span class="scale"><i></i></span><span class="mono">% of phones hearing rain</span><span class="nd"><i></i><span class="mono">under 3 readings</span></span><span><b class="outline"></b><span class="mono">raining</span></span></div>
</section>''')

# 5 results ---------------------------------------------------------
slides.append(f'''
<section class="slide">
  {top("results", 5)}
  <h2>Tested on public recordings.</h2>
  <p class="lede">Five-fold cross-validation on the ESC-50 dataset: rain, and 20 everyday sounds often mistaken for it.</p>
  <ul class="big mid">
    <li><b>{pct(metrics["clip_accuracy"])}</b><span>of 5-second clips classified correctly</span></li>
    <li><b>{pct(false_alarms)}</b><span>of non-rain sounds mistaken for rain</span></li>
    <li><b>{metrics["files"]}</b><span>recordings, each tested on a model that never saw it</span></li>
  </ul>
</section>''')

# 6 next ------------------------------------------------------------
slides.append(f'''
<section class="slide inv">
  {top("next", 6, True)}
  <h2>Next: real streets.</h2>
  <p class="lede">RainSense is open source.</p>
  <ol class="next mid">
    <li>Field tests on real phones</li>
    <li>Rain intensity, calibrated against rain gauges</li>
    <li>A pilot inside an app people use outdoors</li>
  </ol>
  <p class="cta mono">github.com/akshayjg/RainSense</p>
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
