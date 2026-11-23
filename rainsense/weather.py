"""Check labels and predictions against real weather: Open-Meteo precipitation (free, no API key).

  python -m rainsense.weather --lat 52.37 --lon 4.90 data/            # recordings saved as <yyyyMMdd-HHmmss>_<place>.wav
  python -m rainsense.weather --lat 52.37 --lon 4.90 rain_log.csv     # the demo app's monitoring log

Uses 15-minute precipitation for the last 92 days at one location (where the phone was).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

API = "https://api.open-meteo.com/v1/forecast"
RAIN_MM = 0.1   # >= 0.1 mm in a 15-minute slot counts as raining


def fetch(lat, lon, past_days=92):
    q = urlencode({"latitude": lat, "longitude": lon, "minutely_15": "precipitation",
                   "past_days": past_days, "forecast_days": 1, "timezone": "auto"})
    with urlopen(f"{API}?{q}", timeout=30) as r:
        return json.load(r)


def lookup(resp):
    m = resp["minutely_15"]
    return dict(zip(m["time"], m["precipitation"]))


def slot(t: datetime) -> str:
    return t.replace(minute=t.minute - t.minute % 15, second=0).strftime("%Y-%m-%dT%H:%M")


def _items(path):
    """(name, time, says_rain) from a recordings folder or the app's rain_log.csv."""
    path = Path(path)
    if path.is_dir():
        for wav in sorted(path.glob("*/*.wav")):
            try:
                t = datetime.strptime(wav.stem.split("_")[0], "%Y%m%d-%H%M%S")
            except ValueError:
                continue
            yield f"{wav.parent.name}/{wav.name}", t, wav.parent.name != "no_rain"
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            ts, label = line.split(",")[:2]
            if label != "skipped":
                yield ts, datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"), label != "no_rain"


def check(path, precip):
    rows = [(name, said, precip[slot(t)]) for name, t, said in _items(path) if precip.get(slot(t)) is not None]
    for name, said, mm in rows:
        if said != (mm >= RAIN_MM):
            print(f"disagrees: {name} says {'rain' if said else 'no rain'}, Open-Meteo {mm} mm / 15 min")
    agree = sum(said == (mm >= RAIN_MM) for _, said, mm in rows)
    print(f"{agree}/{len(rows)} agree with Open-Meteo")
    return agree, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="recordings folder or rain_log.csv")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    a = ap.parse_args()
    check(a.path, lookup(fetch(a.lat, a.lon)))
