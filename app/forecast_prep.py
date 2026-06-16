"""
Feature builder for the **short-term (next-30-min) availability forecaster**.

Shared by `train_forecast.py` (training) and `streamlit_app.py` (live serving) so
the features are identical at train and serve time.

Idea
----
Predict a station's bikes `HORIZON_MIN` minutes into the future using:
  * `bikes_now`  — the current observed bikes (lag-0). At serve time this comes
                   straight from the CityBikes live API.
  * `slot_mean`  — the station's typical bikes at the *target* hour/weekday
                   (a serveable stand-in for "same time yesterday / last week").
  * `delta_now_vs_slot` — how unusual the current state is vs typical.
  * target-time calendar + weather + station geometry.

This is fully serveable from `live bikes + slot_profile + scenario`, yet far more
accurate than the abstract "typical profile" model, because it is anchored on the
station's current state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_MIN = 30          # forecast horizon in minutes
STEP_MIN = 15             # snapshot cadence
HORIZON_STEPS = HORIZON_MIN // STEP_MIN

# numeric features fed to BOTH models (the net additionally uses a station embedding)
NUM_FC = [
    "bikes_now", "occ_now", "slot_mean", "slot_mean_now", "delta_now_vs_slot",
    "t_hour_sin", "t_hour_cos", "t_dayofweek", "t_is_weekend", "t_month",
    "temperature", "precipitation", "is_rain",
    "capacity", "dist_center_km", "lat", "lon",
]
TARGET = "target_bikes"


def all_feature_names_fc() -> list[str]:
    return list(NUM_FC)


# --------------------------------------------------------------------------- #
# Training-table construction
# --------------------------------------------------------------------------- #
def build_supervised(hist: pd.DataFrame, slot: pd.DataFrame,
                     horizon_steps: int = HORIZON_STEPS) -> pd.DataFrame:
    """Turn the 15-min history into a leakage-safe supervised table.

    Target = bikes at t + horizon (kept only when that exact snapshot exists, so
    gaps never leak). All features use information available at time t.
    """
    df = hist.sort_values(["station_id", "dt"]).copy()
    df["dt"] = pd.to_datetime(df["dt"])

    # target: bikes `horizon` steps in the future, aligned by exact timestamp
    fut = df[["station_id", "dt", "bikes_available"]].rename(
        columns={"bikes_available": TARGET})
    fut["dt"] = fut["dt"] - pd.Timedelta(minutes=STEP_MIN * horizon_steps)
    df = df.merge(fut, on=["station_id", "dt"], how="inner")

    df["bikes_now"] = df["bikes_available"].astype(float)
    df["occ_now"] = df["occupancy"].astype(float)

    tt = df["dt"] + pd.Timedelta(minutes=STEP_MIN * horizon_steps)
    df["t_hour"] = tt.dt.hour
    df["t_dayofweek"] = tt.dt.dayofweek
    df["t_is_weekend"] = (df["t_dayofweek"] >= 5).astype(int)
    df["t_month"] = tt.dt.month
    df["t_hour_sin"] = np.sin(2 * np.pi * df["t_hour"] / 24)
    df["t_hour_cos"] = np.cos(2 * np.pi * df["t_hour"] / 24)

    df = _join_slots(df, slot)
    df["delta_now_vs_slot"] = df["bikes_now"] - df["slot_mean_now"]
    return df


def _join_slots(df: pd.DataFrame, slot: pd.DataFrame) -> pd.DataFrame:
    sm = slot.rename(columns={"hour": "t_hour", "dayofweek": "t_dayofweek",
                              "slot_mean": "slot_mean"})[
        ["station_id", "t_hour", "t_dayofweek", "slot_mean"]]
    df = df.merge(sm, on=["station_id", "t_hour", "t_dayofweek"], how="left")
    smn = slot.rename(columns={"slot_mean": "slot_mean_now"})[
        ["station_id", "hour", "dayofweek", "slot_mean_now"]]
    df = df.merge(smn, on=["station_id", "hour", "dayofweek"], how="left")
    df["slot_mean"] = df["slot_mean"].fillna(df["bikes_now"])
    df["slot_mean_now"] = df["slot_mean_now"].fillna(df["bikes_now"])
    return df


# --------------------------------------------------------------------------- #
# Serve-time single-row builder (used by the live app)
# --------------------------------------------------------------------------- #
def slot_lookup(slot: pd.DataFrame) -> dict:
    """(station_id, hour, dayofweek) -> mean bikes, for O(1) serve-time lookup."""
    return {(int(r.station_id), int(r.hour), int(r.dayofweek)): float(r.slot_mean)
            for r in slot.itertuples(index=False)}


def make_live_row(station: dict, bikes_now: float, now_dt, target_dt,
                  temperature: float, precipitation: float,
                  slot_idx: dict) -> pd.DataFrame:
    """Build one feature row for a live next-horizon prediction.

    `station` needs station_id, lat, lon, capacity, dist_center_km.
    """
    sid = int(station["station_id"])
    cap = float(station["capacity"])
    th, tdow = target_dt.hour, target_dt.weekday()
    nh, ndow = now_dt.hour, now_dt.weekday()
    slot_mean = slot_idx.get((sid, th, tdow), bikes_now)
    slot_mean_now = slot_idx.get((sid, nh, ndow), bikes_now)
    row = {
        "bikes_now": bikes_now,
        "occ_now": bikes_now / max(cap, 1),
        "slot_mean": slot_mean,
        "slot_mean_now": slot_mean_now,
        "delta_now_vs_slot": bikes_now - slot_mean_now,
        "t_hour_sin": np.sin(2 * np.pi * th / 24),
        "t_hour_cos": np.cos(2 * np.pi * th / 24),
        "t_dayofweek": tdow,
        "t_is_weekend": int(tdow >= 5),
        "t_month": target_dt.month,
        "temperature": temperature,
        "precipitation": precipitation,
        "is_rain": int(precipitation > 0.1),
        "capacity": cap,
        "dist_center_km": station.get("dist_center_km", 0.0),
        "lat": station["lat"],
        "lon": station["lon"],
    }
    return pd.DataFrame([row])[NUM_FC]
