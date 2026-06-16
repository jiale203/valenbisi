"""
Build the Valenbisi training history from open data.

Sources
-------
* Historical station snapshots: github.com/ceferra/valenbici  (daily zips of
  15-minute CSV snapshots — open data, mirrors valencia.opendatasoft.com).
* Weather: Open-Meteo archive API (keyless).

Output
------
data/history.parquet     hourly station availability + weather (training table)
data/stations.parquet    per-station metadata (id, name, coords, capacity)

Usage
-----
python download_data.py            # curated multi-season sample (~35 days)
python download_data.py --smoke    # 3 days, for a quick pipeline test
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
import data_prep as dp  # noqa: E402

REPO = "ceferra/valenbici"
BRANCH = "master"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# Seed months spread across seasons / years; we take a handful of available days
# from each so the model sees every weekday, weekend and season.
SEED_MONTHS = [(2024, 1), (2024, 4), (2024, 7), (2024, 10), (2025, 3)]
DAYS_PER_MONTH = 7

os.makedirs("data", exist_ok=True)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def available_dates() -> list[datetime]:
    import json
    import re
    tree = json.loads(_get(TREE_URL))["tree"]
    out = []
    for f in tree:
        m = re.match(r"(\d{2})-(\d{2})-(\d{4})\.zip$", f["path"])
        if m:
            d, mo, y = map(int, m.groups())
            try:
                out.append(datetime(y, mo, d))
            except ValueError:
                pass
    return sorted(out)


def pick_sample(dates: list[datetime], smoke: bool) -> list[datetime]:
    if smoke:
        # first 3 available days of the first seed month
        y, mo = SEED_MONTHS[0]
        days = [d for d in dates if (d.year, d.month) == (y, mo)]
        return days[:3]

    by_month = defaultdict(list)
    for d in dates:
        by_month[(d.year, d.month)].append(d)
    chosen: list[datetime] = []
    for ym in SEED_MONTHS:
        days = by_month.get(ym, [])
        if not days:
            continue
        # evenly spread DAYS_PER_MONTH across the month's available days
        step = max(1, len(days) // DAYS_PER_MONTH)
        chosen.extend(days[::step][:DAYS_PER_MONTH])
    return sorted(chosen)


def parse_day(date: datetime) -> pd.DataFrame:
    """Download one daily zip and parse its hourly (:00) snapshots."""
    fname = date.strftime("%d-%m-%Y") + ".zip"
    try:
        blob = _get(f"{RAW}/{fname}")
    except Exception as e:                       # noqa: BLE001 -- skip missing days
        print(f"   ! skip {fname}: {e}")
        return pd.DataFrame()

    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        # one snapshot per hour: filenames contain _HH-00-
        names = [n for n in z.namelist()
                 if n.lower().endswith(".csv") and "-00-" in os.path.basename(n)]
        # keep only the top-of-hour file (MM == 00)
        names = [n for n in names if dp.snapshot_datetime(n)
                 and dp.snapshot_datetime(n).minute == 0]
        for n in sorted(names):
            with z.open(n) as fh:
                tmp = os.path.join("/tmp", os.path.basename(n))
                with open(tmp, "wb") as out:
                    out.write(fh.read())
            try:
                snap = dp.parse_snapshot(tmp)
                if not snap.empty:
                    frames.append(snap)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_weather(start: datetime, end: datetime) -> pd.DataFrame:
    import json
    lat, lon = dp.CITY_CENTER
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
        "&hourly=temperature_2m,precipitation&timezone=Europe%2FMadrid"
    )
    try:
        h = json.loads(_get(url))["hourly"]
        t = pd.to_datetime(h["time"])
        return pd.DataFrame({
            "date": t.date,
            "hour": t.hour,
            "temperature": h["temperature_2m"],
            "precipitation": h["precipitation"],
        })
    except Exception as e:                       # noqa: BLE001
        print(f"   ! weather fetch failed ({e}); continuing without weather")
        return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 3-day sample")
    args = ap.parse_args()

    print(">> Listing available dates in", REPO)
    dates = available_dates()
    print(f"   {len(dates)} daily archives ({dates[0]:%Y-%m-%d} … {dates[-1]:%Y-%m-%d})")

    sample = pick_sample(dates, args.smoke)
    print(f">> Downloading {len(sample)} day(s)")
    frames = []
    for i, d in enumerate(sample, 1):
        day = parse_day(d)
        if not day.empty:
            frames.append(day)
        print(f"   [{i:>2}/{len(sample)}] {d:%Y-%m-%d}  rows={len(day)}")

    if not frames:
        raise SystemExit("No data parsed — aborting.")
    hist = pd.concat(frames, ignore_index=True)
    hist = dp.add_time_features(hist)
    hist = dp.add_geo_features(hist)

    # weather over the ACTUAL parsed date range (archives can straddle midnight),
    # merged by (date, hour)
    d_start = datetime.combine(min(hist["date"]), datetime.min.time())
    d_end = datetime.combine(max(hist["date"]), datetime.min.time())
    wx = fetch_weather(d_start, d_end)
    if not wx.empty:
        hist = hist.merge(wx, on=["date", "hour"], how="left")
    for c in ["temperature", "precipitation"]:
        if c not in hist:
            hist[c] = 0.0
    hist["temperature"] = hist["temperature"].fillna(hist["temperature"].median()
                                                     if hist["temperature"].notna().any() else 18.0)
    hist["precipitation"] = hist["precipitation"].fillna(0.0)
    hist["is_rain"] = (hist["precipitation"] > 0.1).astype(int)
    hist["is_empty"] = (hist["bikes_available"] == 0).astype(int)
    hist["is_full"] = (hist["docks_free"].fillna(
        hist["capacity"] - hist["bikes_available"]) == 0).astype(int)

    hist.to_parquet("data/history.parquet", index=False)
    print(f">> Saved data/history.parquet  rows={len(hist):,}  "
          f"stations={hist['station_id'].nunique()}  "
          f"days={hist['date'].nunique()}")

    stations = (hist.groupby("station_id")
                .agg(name=("name", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
                     lat=("lat", "median"), lon=("lon", "median"),
                     capacity=("capacity", "median"))
                .reset_index())
    stations["capacity"] = stations["capacity"].round().astype(int)
    stations["dist_center_km"] = dp.haversine_km(
        stations["lat"], stations["lon"], *dp.CITY_CENTER)
    stations.to_parquet("data/stations.parquet", index=False)
    print(f">> Saved data/stations.parquet  stations={len(stations)}")
    print("Done.")


if __name__ == "__main__":
    main()
