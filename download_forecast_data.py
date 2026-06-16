"""
Build the *short-term forecasting* dataset: contiguous days at full 15-minute
resolution, so lag features (bikes now, 15/30/60 min ago, ...) are well defined.

Unlike download_data.py (which samples scattered days for the "typical profile"
model), this downloads a **consecutive block** of days — lags need an unbroken
time series per station.

Sources: github.com/ceferra/valenbici (snapshots) + Open-Meteo (weather).

Output
------
data/history_15min.parquet   long table: station x 15-min timestamp + weather
data/slot_profile.parquet    station x hour x weekday -> mean/median bikes
                             (a serveable proxy for "same time yesterday/last week")

Usage
-----
python download_forecast_data.py --start 2025-03-01 --days 60     # full run (Colab)
python download_forecast_data.py --start 2025-03-01 --days 14 --smoke
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
import data_prep as dp  # noqa: E402

REPO, BRANCH = "ceferra/valenbici", "master"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
os.makedirs("data", exist_ok=True)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_day_full(date: datetime) -> pd.DataFrame:
    """Download one daily zip and parse ALL 15-minute snapshots."""
    fname = date.strftime("%d-%m-%Y") + ".zip"
    try:
        blob = _get(f"{RAW}/{fname}")
    except Exception as e:                       # noqa: BLE001
        print(f"   ! skip {fname}: {e}")
        return pd.DataFrame()
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in sorted(x for x in z.namelist() if x.lower().endswith(".csv")):
            tmp = os.path.join("/tmp", os.path.basename(n))
            with z.open(n) as fh, open(tmp, "wb") as out:
                out.write(fh.read())
            try:
                snap = dp.parse_snapshot(tmp)
                if not snap.empty:
                    frames.append(snap)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_weather(start: datetime, end: datetime) -> pd.DataFrame:
    import json
    lat, lon = dp.CITY_CENTER
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
           "&hourly=temperature_2m,precipitation&timezone=Europe%2FMadrid")
    try:
        h = json.loads(_get(url))["hourly"]
        t = pd.to_datetime(h["time"])
        return pd.DataFrame({"date": t.date, "hour": t.hour,
                             "temperature": h["temperature_2m"],
                             "precipitation": h["precipitation"]})
    except Exception as e:                       # noqa: BLE001
        print(f"   ! weather failed ({e}); continuing without")
        return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-03-01", help="first day YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=60, help="number of consecutive days")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(args.days)]
    print(f">> Downloading {len(dates)} consecutive days from {args.start} "
          f"(15-min resolution)")

    frames = []
    for i, d in enumerate(dates, 1):
        day = parse_day_full(d)
        if not day.empty:
            frames.append(day)
        if i % 5 == 0 or i == len(dates):
            print(f"   [{i:>3}/{len(dates)}] {d:%Y-%m-%d}  cum_rows="
                  f"{sum(len(f) for f in frames):,}")
    if not frames:
        raise SystemExit("No data parsed — aborting.")

    hist = pd.concat(frames, ignore_index=True)
    # snap to a clean 15-minute grid and de-duplicate
    hist["dt"] = pd.to_datetime(hist["snapshot_dt"]).dt.floor("15min")
    hist = (hist.sort_values("dt")
            .drop_duplicates(subset=["station_id", "dt"], keep="last"))
    hist = dp.add_time_features(hist, "dt")
    hist = dp.add_geo_features(hist)

    wx = fetch_weather(min(hist["date"]), max(hist["date"]))
    if not wx.empty:
        hist = hist.merge(wx, on=["date", "hour"], how="left")
    for c in ("temperature", "precipitation"):
        if c not in hist:
            hist[c] = 0.0
    hist["temperature"] = hist["temperature"].fillna(hist["temperature"].median())
    hist["precipitation"] = hist["precipitation"].fillna(0.0)
    hist["is_rain"] = (hist["precipitation"] > 0.1).astype(int)

    keep = ["station_id", "name", "lat", "lon", "capacity", "bikes_available",
            "docks_free", "occupancy", "dt", "hour", "dayofweek", "is_weekend",
            "month", "date", "dist_center_km", "temperature", "precipitation",
            "is_rain"]
    hist[keep].to_parquet("data/history_15min.parquet", index=False)
    print(f">> Saved data/history_15min.parquet  rows={len(hist):,}  "
          f"stations={hist['station_id'].nunique()}  "
          f"span={hist['dt'].min()} … {hist['dt'].max()}")

    # serveable slot profile: typical bikes by station x hour x weekday
    slot = (hist.groupby(["station_id", "hour", "dayofweek"])
            .agg(slot_mean=("bikes_available", "mean"),
                 slot_capacity=("capacity", "median")).reset_index())
    slot.to_parquet("data/slot_profile.parquet", index=False)
    print(f">> Saved data/slot_profile.parquet  rows={len(slot):,}")
    print("Done.")


if __name__ == "__main__":
    main()
