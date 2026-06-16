"""
Shared parsing & feature engineering for the Valenbisi availability app.

`download_data.py` (data build), `train.py` (training) and `streamlit_app.py`
(serving) all import this module so the feature definitions are identical at
train and serve time — no train/serve skew.

Public API
----------
parse_snapshot(path)         -> tidy DataFrame for one 15-min station snapshot CSV
snapshot_datetime(filename)  -> datetime parsed from the snapshot filename
add_time_features(df, col)   -> adds hour / dow / cyclical / month features
add_geo_features(df)         -> adds distance-to-centre
make_feature_row(...)        -> single-row feature frame for live prediction
FEATURES / all_feature_names() / TARGET_REG / TARGET_CLF
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

# Plaça de l'Ajuntament — Valencia city centre (lat, lon)
CITY_CENTER = (39.4699, -0.3763)

# Raw Inside/ceferra valenbici CSV columns (";"-separated, Spanish labels)
RAW_COLS = {
    "Direccion": "name",
    "Numero": "station_id",
    "Activo": "active",
    "Bicis_disponibles": "bikes_available",
    "Espacios_libres": "docks_free",
    "Espacios_totales": "capacity",
    "geo_point_2d": "geo_point_2d",
}

# --------------------------------------------------------------------------- #
# Model feature definition
# --------------------------------------------------------------------------- #
FEATURES = {
    "numeric": [
        "hour", "hour_sin", "hour_cos", "dayofweek", "is_weekend", "month",
        "capacity", "dist_center_km", "lat", "lon",
        "temperature", "precipitation", "is_rain",
    ],
    "categorical": ["station_id"],
}
TARGET_REG = "bikes_available"
TARGET_CLF = "is_empty"          # bikes_available == 0  (a stockout for renters)


def all_feature_names() -> list[str]:
    return FEATURES["numeric"] + FEATURES["categorical"]


# --------------------------------------------------------------------------- #
# Parsing raw snapshots
# --------------------------------------------------------------------------- #
_FNAME_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})")


def snapshot_datetime(filename: str) -> datetime | None:
    """Extract the snapshot time from e.g. valenbici_01-06-2025_08-00-05.csv."""
    m = _FNAME_RE.search(os.path.basename(filename))
    if not m:
        return None
    d, mo, y, hh, mm, ss = map(int, m.groups())
    try:
        return datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None


def _split_geo(series: pd.Series) -> pd.DataFrame:
    """geo_point_2d is 'lat,lon' (sometimes with spaces). Robust to bad rows."""
    parts = (series.astype(str).str.split(",", n=1, expand=True)
             .reindex(columns=[0, 1]))   # guarantee both columns exist
    lat = pd.to_numeric(parts[0].astype(str).str.strip(), errors="coerce")
    lon = pd.to_numeric(parts[1].astype(str).str.strip(), errors="coerce")
    return pd.DataFrame({"lat": lat.values, "lon": lon.values}, index=series.index)


def parse_snapshot(path: str) -> pd.DataFrame:
    """Read one station snapshot CSV into a tidy, validated DataFrame."""
    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8",
                         engine="python", on_bad_lines="skip")
    except Exception:                            # noqa: BLE001 -- skip unreadable
        return pd.DataFrame()
    df = df.rename(columns={c: RAW_COLS[c] for c in df.columns if c in RAW_COLS})
    needed = set(RAW_COLS.values())
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    for c in ["bikes_available", "docks_free", "capacity", "station_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    geo = _split_geo(df["geo_point_2d"])
    df = pd.concat([df.drop(columns=["geo_point_2d"]), geo], axis=1)

    dt = snapshot_datetime(path)
    df["snapshot_dt"] = dt

    # Validity filters (plan: Activo == 'T', capacity > 0, 0 <= bikes <= capacity)
    df = df[df["active"].astype(str).str.upper().str.startswith("T")]
    df = df.dropna(subset=["station_id", "capacity", "bikes_available",
                           "lat", "lon"])
    df = df[(df["capacity"] > 0)
            & (df["bikes_available"] >= 0)
            & (df["bikes_available"] <= df["capacity"])]
    df["station_id"] = df["station_id"].astype(int)
    df["occupancy"] = df["bikes_available"] / df["capacity"]
    return df[["station_id", "name", "lat", "lon", "capacity",
               "bikes_available", "docks_free", "occupancy", "snapshot_dt"]]


# --------------------------------------------------------------------------- #
# Feature engineering (applied to the tidy history table)
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_time_features(df: pd.DataFrame, dt_col: str = "snapshot_dt") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["date"] = dt.dt.date
    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist_center_km"] = haversine_km(df["lat"], df["lon"], *CITY_CENTER)
    return df


# --------------------------------------------------------------------------- #
# Serve-time single-row builder (used by the Streamlit app)
# --------------------------------------------------------------------------- #
def make_feature_row(station: dict, hour: int, dayofweek: int, month: int,
                     temperature: float, precipitation: float) -> pd.DataFrame:
    """Build a one-row feature frame for a hypothetical scenario.

    `station` must contain station_id, lat, lon, capacity.
    """
    row = {
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dayofweek": dayofweek,
        "is_weekend": int(dayofweek >= 5),
        "month": month,
        "capacity": station["capacity"],
        "dist_center_km": haversine_km(station["lat"], station["lon"], *CITY_CENTER),
        "lat": station["lat"],
        "lon": station["lon"],
        "temperature": temperature,
        "precipitation": precipitation,
        "is_rain": int(precipitation > 0.1),
        "station_id": int(station["station_id"]),
    }
    return pd.DataFrame([row])[all_feature_names()]
