"""
ValenWheel — live availability, forecasting & rebalancing for Valenbisi (València).

A Streamlit app that loads models trained in Colab and turns Valenbisi open data
into four city tools:

  1. Live Now          — real-time station status (CityBikes API) on a map.
  2. Availability Forecast — predict bikes/docks for any station, day, hour &
                         weather scenario, with stockout probability (SHAP-explained).
  3. Rebalancing Planner — where bikes/docks will run out + suggested moves.
  4. Station Clusters  — KMeans archetypes of station behaviour (commuter flows).

City problem: Valenbisi bikes (or free docks) are often unavailable when/where you
need them. This app helps residents plan trips and the operator plan rebalancing.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import data_prep as dp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_PATH = os.path.join(ROOT, "models", "forecast_model.pkl")
CLF_PATH = os.path.join(ROOT, "models", "stockout_model.pkl")
META_PATH = os.path.join(ROOT, "models", "model_meta.json")
PROF_PATH = os.path.join(ROOT, "data", "station_profiles.parquet")
BANDS_PATH = os.path.join(ROOT, "data", "hourly_profiles.parquet")
CITYBIKES = "http://api.citybik.es/v2/networks/valenbisi"

GREEN = "#179C7D"
WARM = "#F4A259"
INK = "#22333B"
RED = "#E5484D"
BLUE = "#3E7CB1"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
# typical Valencia temperature by month (°C) for sensible weather defaults
MONTH_TEMP = {1: 12, 2: 13, 3: 15, 4: 17, 5: 20, 6: 24,
              7: 27, 8: 27, 9: 24, 10: 20, 11: 15, 12: 12}

st.set_page_config(page_title="ValenWheel · Valenbisi intelligence",
                   page_icon="🚲", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.5rem; max-width: 1260px;}}
      h1, h2, h3 {{color: {INK} !important;}}
      .vw-hero {{
        background: linear-gradient(120deg, {GREEN} 0%, #16a085 45%, {WARM} 135%);
        color: white; padding: 1.35rem 1.8rem; border-radius: 16px;
        margin-bottom: 1.0rem;
      }}
      .vw-hero h1 {{color: white; margin: 0 0 .2rem 0; font-size: 1.95rem;}}
      .vw-hero p {{margin: 0; opacity: .95; font-size: 1.01rem;}}
      div[data-testid="stMetricValue"] {{color: {GREEN};}}
      .stTabs [data-baseweb="tab"] {{font-size: 1.0rem; padding: 8px 14px;}}
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_reg():
    return joblib.load(REG_PATH)


@st.cache_resource(show_spinner=False)
def load_clf():
    return joblib.load(CLF_PATH)


@st.cache_data(show_spinner=False)
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_profiles():
    return pd.read_parquet(PROF_PATH)


@st.cache_data(show_spinner=False)
def load_bands():
    return pd.read_parquet(BANDS_PATH)


@st.cache_resource(show_spinner=False)
def get_explainer(_reg):
    import shap
    return shap.TreeExplainer(_reg.named_steps["model"])


@st.cache_data(ttl=120, show_spinner=False)
def fetch_live():
    """Real-time station status from CityBikes. Returns (df, ok, timestamp)."""
    try:
        req = urllib.request.Request(CITYBIKES, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        sts = data["network"]["stations"]
        rows = []
        for s in sts:
            extra = s.get("extra") or {}
            fb = s.get("free_bikes") or 0
            es = s.get("empty_slots") or 0
            rows.append({
                "name": str(s.get("name", "")).lstrip("_").replace("_", " ").title(),
                "free_bikes": fb, "empty_slots": es,
                "lat": s["latitude"], "lon": s["longitude"],
                "capacity": extra.get("slots") or (fb + es),
                "address": extra.get("address", ""),
            })
        df = pd.DataFrame(rows)
        df["capacity"] = df["capacity"].replace(0, np.nan).fillna(
            df["free_bikes"] + df["empty_slots"]).clip(lower=1)
        df["occupancy"] = (df["free_bikes"] / df["capacity"]).clip(0, 1)
        ts = sts[0].get("timestamp") if sts else None
        return df, True, ts
    except Exception:                                # noqa: BLE001 -- offline fallback
        return None, False, None


def live_fallback(profiles: pd.DataFrame) -> pd.DataFrame:
    """If the live API is down, synthesise 'now' from the typical hourly curve."""
    hh = datetime.now().hour
    occ = profiles[f"occ_h{hh:02d}"].fillna(profiles[[f"occ_h{h:02d}"
                   for h in range(24)]].mean(axis=1))
    df = profiles[["name", "lat", "lon", "capacity"]].copy()
    df["occupancy"] = occ.clip(0, 1).values
    df["free_bikes"] = (df["occupancy"] * df["capacity"]).round().astype(int)
    df["empty_slots"] = (df["capacity"] - df["free_bikes"]).astype(int)
    df["address"] = ""
    return df


def occ_color(occ: float) -> list[int]:
    """Diverging colour: 0 bikes -> red, balanced -> green, 0 docks -> blue."""
    occ = float(np.clip(occ, 0, 1))
    if occ <= 0.5:
        t = occ / 0.5
        return [int(229 - 95 * t), int(72 + 84 * t), int(77 + 48 * t), 200]   # red->green
    t = (occ - 0.5) / 0.5
    return [int(23 + 39 * t), int(156 - 32 * t), int(125 + 52 * t), 200]      # green->blue


reg = load_reg()
clf = load_clf()
meta = load_meta()
profiles = load_profiles()
bands = load_bands()
profiles = profiles.sort_values("name").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Hero + sidebar
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="vw-hero">
      <h1>🚲 ValenWheel</h1>
      <p>Live availability, forecasting & rebalancing for <b>Valenbisi</b> —
      {meta['n_stations']} stations · {meta['n_rows']:,} observations ·
      forecast MAE ≈ {meta['regressor_test']['mae_bikes']:.1f} bikes ·
      stockout AUC {meta['stockout_test']['roc_auc']:.2f}.</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write("**Problem.** Valenbisi riders often find *no bike* or *no free dock*. "
             "ValenWheel forecasts availability and tells the operator where to "
             "rebalance.")
    st.markdown("---")
    st.markdown("**Data** · ceferra/valenbici (Valencia open data) + Open-Meteo + "
                "CityBikes live API")
    st.caption(f"History window: {meta['data_window']}")
    st.markdown(f"**Forecast model** · {meta['best_regressor']}")
    st.markdown(f"**Stockout model** · {meta['stockout_classifier']}")
    st.caption(f"Trained {meta['trained_on']}")
    st.markdown("---")
    st.caption("DS methods: feature engineering (cyclical time, geo, weather), "
               "time-aware model selection (4 regressors), a calibrated stockout "
               "classifier, SHAP explainability, KMeans station clustering, and a "
               "greedy rebalancing matcher.")

tab_live, tab_fc, tab_reb, tab_clu, tab_model = st.tabs(
    ["🚲 Live Now", "🔮 Availability Forecast", "♻️ Rebalancing Planner",
     "🧭 Station Clusters", "🤖 Model & Method"])


# =========================================================================== #
# TAB 1 — Live Now
# =========================================================================== #
with tab_live:
    st.subheader("Real-time station status")
    live, ok, ts = fetch_live()
    if not ok:
        st.warning("Live CityBikes API unreachable — showing the *typical* pattern "
                   "for this hour instead.")
        live = live_fallback(profiles)
    else:
        st.caption(f"Source: CityBikes · last update {ts}")

    empty_n = int((live["free_bikes"] == 0).sum())
    full_n = int((live["empty_slots"] == 0).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stations", f"{len(live)}")
    k2.metric("Bikes available now", f"{int(live['free_bikes'].sum()):,}")
    k3.metric("Empty stations 🚫🚲", f"{empty_n}")
    k4.metric("Full stations 🚫🅿️", f"{full_n}")

    flt = st.radio("Show", ["All", "Empty (no bikes)", "Full (no docks)"],
                   horizontal=True)
    view = live.copy()
    if flt.startswith("Empty"):
        view = view[view["free_bikes"] == 0]
    elif flt.startswith("Full"):
        view = view[view["empty_slots"] == 0]
    view["color"] = view["occupancy"].apply(occ_color)
    view["radius"] = 40 + view["capacity"] * 2

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer(
            "ScatterplotLayer", view, get_position="[lon, lat]",
            get_fill_color="color", get_radius="radius", pickable=True,
            opacity=0.85, stroked=True, get_line_color=[255, 255, 255])],
        initial_view_state=pdk.ViewState(latitude=39.465, longitude=-0.375,
                                         zoom=12, pitch=0),
        map_style="light",
        tooltip={"text": "{name}\nBikes: {free_bikes}  Free docks: {empty_slots}"},
    ), width="stretch")
    st.caption("🔴 no bikes · 🟢 balanced · 🔵 no free docks. Dot size = capacity.")

    st.markdown("##### Stations needing attention right now")
    worst = live.assign(
        problem=np.where(live["free_bikes"] == 0, "No bikes",
                np.where(live["empty_slots"] == 0, "No docks", "OK")))
    worst = worst[worst["problem"] != "OK"][
        ["name", "free_bikes", "empty_slots", "capacity", "problem"]]
    st.dataframe(worst.reset_index(drop=True), hide_index=True,
                 width="stretch", height=240)


# =========================================================================== #
# Helpers for forecasting
# =========================================================================== #
def station_row(name: str) -> dict:
    r = profiles[profiles["name"] == name].iloc[0]
    return {"station_id": int(r["station_id"]), "lat": r["lat"],
            "lon": r["lon"], "capacity": int(r["capacity"]), "name": name}


def predict_scenario(stat: dict, hour: int, dow: int, month: int,
                     temp: float, precip: float):
    X = dp.make_feature_row(stat, hour, dow, month, temp, precip)
    bikes = float(np.clip(reg.predict(X)[0], 0, stat["capacity"]))
    p_empty = float(clf.predict_proba(X)[0, 1])
    docks = max(0.0, stat["capacity"] - bikes)
    return bikes, docks, p_empty, X


NICE = {"hour": "Hour", "hour_sin": "Hour", "hour_cos": "Hour",
        "dayofweek": "Day of week", "is_weekend": "Weekend", "month": "Month",
        "capacity": "Capacity", "dist_center_km": "Dist. to centre",
        "lat": "Latitude", "lon": "Longitude", "temperature": "Temperature",
        "precipitation": "Precipitation", "is_rain": "Raining",
        "station_id": "Station"}


def shap_reasons(X: pd.DataFrame) -> pd.Series:
    expl = get_explainer(reg)
    pre = reg.named_steps["pre"]
    Xt = pre.transform(X)
    out = pre.get_feature_names_out()
    sv = expl.shap_values(Xt)[0]
    agg: dict[str, float] = {}
    for name, val in zip(out, sv):
        src = name.split("__", 1)[-1]
        for feat in dp.all_feature_names():
            if src == feat or src.startswith(feat + "_"):
                key = NICE.get(feat, feat)
                agg[key] = agg.get(key, 0.0) + float(val)
                break
    return pd.Series(agg)


@st.cache_data(show_spinner=False)
def predict_all_stations(hour: int, dow: int, month: int, temp: float, precip: float):
    """Vectorised prediction for every station under one scenario (one model call)."""
    p = profiles
    cap = p["capacity"].to_numpy()
    dist = (p["dist_center_km"].to_numpy() if "dist_center_km" in p
            else dp.haversine_km(p["lat"], p["lon"], *dp.CITY_CENTER).to_numpy())
    X = pd.DataFrame({
        "hour": hour, "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24), "dayofweek": dow,
        "is_weekend": int(dow >= 5), "month": month, "capacity": cap,
        "dist_center_km": dist, "lat": p["lat"].to_numpy(), "lon": p["lon"].to_numpy(),
        "temperature": temp, "precipitation": precip, "is_rain": int(precip > 0.1),
        "station_id": p["station_id"].astype(int).to_numpy(),
    })[dp.all_feature_names()]
    bikes = np.clip(reg.predict(X), 0, cap)
    p_empty = clf.predict_proba(X)[:, 1]
    docks = np.clip(cap - bikes, 0, None)
    return pd.DataFrame({
        "name": p["name"].to_numpy(), "lat": p["lat"].to_numpy(),
        "lon": p["lon"].to_numpy(), "capacity": cap.astype(int),
        "pred_bikes": bikes.round(1), "pred_docks": docks.round(1),
        "p_empty": p_empty.round(3)})


# =========================================================================== #
# TAB 2 — Availability Forecast
# =========================================================================== #
with tab_fc:
    st.subheader("Forecast availability for any station & scenario")
    c = st.columns([1.15, 1])
    with c[0]:
        sname = st.selectbox("Station", profiles["name"].tolist())
        cc = st.columns(2)
        day = cc[0].selectbox("Day", WEEKDAYS, index=2)
        hour = cc[1].slider("Hour", 0, 23, 8)
        month = cc[0].selectbox("Month", list(range(1, 13)),
                                index=datetime.now().month - 1,
                                format_func=lambda m: datetime(2024, m, 1).strftime("%B"))
        weather = cc[1].radio("Weather", ["Clear", "Light rain", "Heavy rain"],
                              horizontal=False)
        temp = st.slider("Temperature (°C)", 0, 40, MONTH_TEMP[month])
        precip = {"Clear": 0.0, "Light rain": 1.0, "Heavy rain": 5.0}[weather]
        dow = WEEKDAYS.index(day)
        stat = station_row(sname)

    bikes, docks, p_empty, X = predict_scenario(stat, hour, dow, month, temp, precip)
    with c[1]:
        st.markdown("#### Predicted at this time")
        m = st.columns(3)
        m[0].metric("Bikes available", f"{bikes:.0f}", help=f"of {stat['capacity']} docks")
        m[1].metric("Free docks", f"{docks:.0f}")
        m[2].metric("P(no bikes)", f"{p_empty*100:.0f}%")
        if p_empty > 0.5 or bikes < 1:
            st.error("High risk of **no bikes** — plan an alternative station.")
        elif docks < 2:
            st.warning("Likely **no free docks** — hard to return a bike here.")
        else:
            st.success("Good chance of finding a bike **and** a free dock.")

    # 24h predicted curve + empirical band
    st.markdown("##### Predicted day profile vs typical range")
    hours = list(range(24))
    preds = [predict_scenario(stat, h, dow, month, temp, precip)[0] for h in hours]
    band = bands[(bands["station_id"] == stat["station_id"])
                 & (bands["is_weekend"] == int(dow >= 5))].set_index("hour")
    band = band.reindex(hours)
    fig = go.Figure()
    if band["p90_bikes"].notna().any():
        fig.add_trace(go.Scatter(x=hours, y=band["p90_bikes"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hours, y=band["p10_bikes"], fill="tonexty",
                                 fillcolor="rgba(23,156,125,0.15)", line=dict(width=0),
                                 name="Typical range (p10–p90)"))
    fig.add_trace(go.Scatter(x=hours, y=preds, line=dict(color=GREEN, width=3),
                             name="Predicted bikes"))
    fig.add_vline(x=hour, line_dash="dash", line_color=WARM)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", xaxis_title="Hour",
                      yaxis_title="Bikes available", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Why this prediction? (SHAP)")
    cs = shap_reasons(X).sort_values()
    cs = pd.concat([cs.head(5), cs.tail(5)]).drop_duplicates()
    colors = [GREEN if v >= 0 else RED for v in cs.values]
    fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h",
                           marker_color=colors,
                           text=[f"{v:+.1f}" for v in cs.values], textposition="outside"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white",
                      xaxis_title="Impact on predicted bikes (SHAP)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green pushes availability up, red pushes it down.")


# =========================================================================== #
# TAB 3 — Rebalancing Planner
# =========================================================================== #
with tab_reb:
    st.subheader("Where will bikes & docks run out — and what to move?")
    cc = st.columns(4)
    r_day = cc[0].selectbox("Day", WEEKDAYS, index=0, key="reb_day")
    r_hour = cc[1].slider("Hour", 0, 23, 8, key="reb_hour")
    r_month = cc[2].selectbox("Month", list(range(1, 13)),
                              index=datetime.now().month - 1, key="reb_month",
                              format_func=lambda m: datetime(2024, m, 1).strftime("%b"))
    r_weather = cc[3].radio("Weather", ["Clear", "Rain"], horizontal=True, key="reb_w")
    r_dow = WEEKDAYS.index(r_day)
    r_precip = 0.0 if r_weather == "Clear" else 3.0

    fc = predict_all_stations(r_hour, r_dow, r_month, MONTH_TEMP[r_month], r_precip).copy()
    # need bikes if predicted very low; need docks if predicted nearly full
    fc["need_bikes"] = (fc["pred_bikes"] < 2) | (fc["p_empty"] > 0.5)
    fc["need_docks"] = fc["pred_docks"] < 2
    fc["status"] = np.where(fc["need_bikes"], "Needs bikes",
                   np.where(fc["need_docks"], "Needs docks", "OK"))
    cmap = {"Needs bikes": [229, 72, 77, 220], "Needs docks": [62, 124, 177, 220],
            "OK": [180, 190, 195, 90]}
    fc["color"] = fc["status"].map(cmap)
    fc["radius"] = 50 + fc["capacity"] * 2

    k = st.columns(3)
    k[0].metric("Stations likely empty", int(fc["need_bikes"].sum()))
    k[1].metric("Stations likely full", int(fc["need_docks"].sum()))
    k[2].metric("Scenario", f"{r_day[:3]} {r_hour:02d}:00 · {r_weather}")

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", fc, get_position="[lon, lat]",
                          get_fill_color="color", get_radius="radius", pickable=True,
                          stroked=True, get_line_color=[255, 255, 255])],
        initial_view_state=pdk.ViewState(latitude=39.465, longitude=-0.375, zoom=12),
        map_style="light",
        tooltip={"text": "{name}\n{status}\nbikes~{pred_bikes} docks~{pred_docks}"}),
        width="stretch")
    st.caption("🔴 likely no bikes (deliver bikes) · 🔵 likely no docks (remove bikes).")

    # greedy nearest-neighbour move suggestions (full -> empty)
    st.markdown("##### Suggested rebalancing moves")
    donors = fc[fc["need_docks"]].copy()        # full: take bikes from here
    receivers = fc[fc["need_bikes"]].copy()     # empty: bring bikes to here
    moves = []
    used = set()
    for _, rec in receivers.iterrows():
        avail = donors[~donors["name"].isin(used)]
        if avail.empty:
            break
        dists = dp.haversine_km(rec["lat"], rec["lon"], avail["lat"], avail["lon"])
        j = dists.idxmin()
        don = donors.loc[j]
        qty = int(max(1, min(don["pred_bikes"] - don["capacity"] / 2,
                             don["capacity"] / 2 - rec["pred_bikes"], 10)))
        moves.append({"From (full)": don["name"], "To (empty)": rec["name"],
                      "Distance (km)": round(float(dists.min()), 2),
                      "Bikes to move": max(1, qty)})
        used.add(don["name"])
    if moves:
        st.dataframe(pd.DataFrame(moves), hide_index=True, width="stretch")
        st.caption("Greedy nearest-neighbour matching of predicted-full → "
                   "predicted-empty stations.")
    else:
        st.info("No rebalancing needed for this scenario — supply looks balanced.")


# =========================================================================== #
# TAB 4 — Station Clusters
# =========================================================================== #
with tab_clu:
    st.subheader("Station behaviour archetypes (KMeans)")
    st.write("Stations clustered by their **24-hour occupancy signature**. This "
             "reveals commuter flows: residential stations fill overnight and empty "
             "in the morning; central/work stations do the opposite.")

    occ_cols = [f"occ_h{h:02d}" for h in range(24)]
    palette = [GREEN, WARM, BLUE, "#9B5DE5", "#E5484D"]
    cl_names = sorted(profiles["cluster_name"].unique())
    cl_color = {n: palette[i % len(palette)] for i, n in enumerate(cl_names)}
    cl_rgb = {n: [int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 200]
              for n, c in cl_color.items()}
    pmap = profiles.copy()
    pmap["color"] = pmap["cluster_name"].map(cl_rgb)
    pmap["radius"] = 60 + pmap["capacity"] * 2

    cc = st.columns([1.2, 1])
    with cc[0]:
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", pmap, get_position="[lon, lat]",
                              get_fill_color="color", get_radius="radius",
                              pickable=True, stroked=True, get_line_color=[255, 255, 255])],
            initial_view_state=pdk.ViewState(latitude=39.465, longitude=-0.375, zoom=12),
            map_style="light",
            tooltip={"text": "{name}\n{cluster_name}"}),
            width="stretch")
    with cc[1]:
        fig = go.Figure()
        for n in cl_names:
            curve = profiles[profiles["cluster_name"] == n][occ_cols].mean()
            fig.add_trace(go.Scatter(x=list(range(24)), y=curve.values, name=n,
                                     line=dict(color=cl_color[n], width=3)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="white", xaxis_title="Hour",
                          yaxis_title="Mean occupancy", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    summ = (profiles.groupby("cluster_name")
            .agg(stations=("station_id", "size"),
                 avg_capacity=("capacity", "mean"),
                 chronic_empty=("chronic_empty_pct", "mean"),
                 chronic_full=("chronic_full_pct", "mean"))
            .round(1).reset_index())
    st.dataframe(summ, hide_index=True, width="stretch")
    st.caption(f"k={meta['clustering']['k']} chosen by silhouette "
               f"({meta['clustering']['silhouette']}). Chronic % = share of all "
               "observations with zero bikes / zero docks.")


# =========================================================================== #
# TAB 5 — Model & Method
# =========================================================================== #
with tab_model:
    st.subheader("How the models were built")
    m = st.columns(4)
    m[0].metric("Forecast model", meta["best_regressor"])
    m[1].metric("Forecast MAE", f"{meta['regressor_test']['mae_bikes']:.2f} bikes")
    m[2].metric("Forecast R²", f"{meta['regressor_test']['r2']:.2f}")
    m[3].metric("Stockout ROC-AUC", f"{meta['stockout_test']['roc_auc']:.2f}")

    st.markdown("##### Regressor model selection (time-ordered hold-out)")
    lb = pd.DataFrame(meta["regressor_leaderboard"]).sort_values("mae_bikes")
    fig = px.bar(lb, x="mae_bikes", y="model", orientation="h", text="mae_bikes",
                 color="mae_bikes", color_continuous_scale=[GREEN, WARM, RED],
                 labels={"mae_bikes": "MAE (bikes, lower=better)", "model": ""})
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                      coloraxis_showscale=False, plot_bgcolor="white",
                      yaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, width="stretch")

    cc = st.columns(2)
    with cc[0]:
        st.markdown("##### Stockout classifier")
        ct = meta["stockout_test"]
        st.write(f"- **ROC-AUC**: {ct['roc_auc']}")
        st.write(f"- **PR-AUC**: {ct['pr_auc']} (base rate {ct['base_rate']})")
        st.write(f"- **Brier score**: {ct['brier']} (lower = better calibrated)")
    with cc[1]:
        st.markdown("##### Global feature importance (SHAP)")
        feats = dp.all_feature_names()
        samp = profiles.sample(min(120, len(profiles)), random_state=0)
        # build representative scenarios across hours for importance
        recs = []
        for _, r in samp.iterrows():
            for h in (8, 14, 19):
                recs.append(dp.make_feature_row(
                    {"station_id": int(r["station_id"]), "lat": r["lat"],
                     "lon": r["lon"], "capacity": int(r["capacity"])},
                    h, 2, datetime.now().month, 20, 0.0))
        Xb = pd.concat(recs, ignore_index=True)
        expl = get_explainer(reg)
        pre = reg.named_steps["pre"]
        sv = np.array(expl.shap_values(pre.transform(Xb)))
        out = pre.get_feature_names_out()
        imp = {}
        for j, name in enumerate(out):
            src = name.split("__", 1)[-1]
            for feat in feats:
                if src == feat or src.startswith(feat + "_"):
                    key = NICE.get(feat, feat)
                    imp[key] = imp.get(key, 0.0) + np.abs(sv[:, j]).mean()
                    break
        imp_s = pd.Series(imp).sort_values()
        fig = px.bar(x=imp_s.values, y=imp_s.index, orientation="h",
                     color=imp_s.values, color_continuous_scale=[GREEN, RED],
                     labels={"x": "mean |SHAP|", "y": ""})
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_showscale=False, plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Methodology & data-science pipeline"):
        st.markdown(
            f"""
- **Sources.** Historical 15-min station snapshots from
  [github.com/ceferra/valenbici](https://github.com/ceferra/valenbici) (Valencia
  open data, {meta['data_window']}), enriched with **Open-Meteo** weather and a
  **CityBikes** live feed. {meta['n_rows']:,} hourly observations, {meta['n_stations']} stations.
- **Features.** Cyclical hour (sin/cos), day-of-week, weekend, month, station
  capacity & location, distance to centre, temperature, precipitation, rain flag,
  station id (one-hot). Shared `data_prep.py` at train **and** serve time → no skew.
- **Forecasting.** {meta['best_regressor']} selected from Ridge / RandomForest /
  GradientBoosting / HistGradientBoosting via a **time-ordered hold-out** (train on
  earlier dates, test on later) to avoid leakage. Target = bikes available.
- **Stockout risk.** A HistGradientBoosting **classifier** estimates P(no bikes),
  reported with ROC-AUC, PR-AUC and Brier score.
- **Clustering.** **KMeans** on each station's 24-h occupancy signature
  (k={meta['clustering']['k']} by silhouette) → commuter-flow archetypes.
- **Rebalancing.** A greedy nearest-neighbour matcher pairs predicted-full donor
  stations with predicted-empty receivers.
            """)
    st.caption("Built with Streamlit · scikit-learn · SHAP · pydeck · Plotly. "
               "Models trained in Google Colab, served from models/*.pkl.")
