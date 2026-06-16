# 🚲 ValenWheel — Valenbisi availability, forecasting & rebalancing

An interactive **Streamlit** app that tackles a daily València problem: **a
Valenbisi rider often finds no bike to take, or no free dock to return one.**
ValenWheel combines a **live feed**, a **trained forecasting model**, a
**stockout-risk classifier** and a **rebalancing planner** into one tool for
both residents and the bike-share operator.

| Tab | What it does | DS method |
|-----|--------------|-----------|
| 🚲 **Live Now** | Real-time map of every station (bikes / free docks), flags empty & full | Live API + geospatial viz |
| 🔮 **Availability Forecast** | A **live next-30-min forecast** anchored on current availability, *plus* a typical-pattern explorer for any station/day/hour/weather | **XGBoost-GPU** + **PyTorch** short-term forecaster, gradient-boosted **regression**, **SHAP**, a **classifier** |
| ♻️ **Rebalancing Planner** | Maps where bikes/docks will run out at a chosen time and suggests **moves** | Model-driven + greedy nearest-neighbour matching |
| 🧭 **Station Clusters** | Groups stations by their 24-h behaviour (commuter source/sink, leisure…) | **KMeans** clustering |
| 🤖 **Model & Method** | Leaderboard, metrics, SHAP importance, methodology | Time-aware model selection |

## Live app
👉 **(https://valenbisi-3erh2tubk7hurcmnbmamgj.streamlit.app/)**

## Data
- **Historical (training):** [`github.com/ceferra/valenbici`](https://github.com/ceferra/valenbici)
  — daily archives of **15-minute station snapshots** (open data, mirrors
  valencia.opendatasoft.com). We sample several weeks across seasons.
- **Weather:** [Open-Meteo](https://open-meteo.com/) archive API (keyless) —
  hourly temperature & precipitation.
- **Live:** [CityBikes](https://citybik.es/) `valenbisi` network (real-time).

Raw archives are **not** committed (see `.gitignore`); they are re-downloaded by
`download_data.py`. The small trained artifacts **are** committed so the app runs
immediately.

## Data-science pipeline
1. **Parsing & cleaning** (`app/data_prep.py`, shared by training **and** the app
   → no train/serve skew): parse `;`-separated snapshots, validate
   (`Activo == T`, `capacity > 0`, `0 ≤ bikes ≤ capacity`), derive occupancy.
2. **Feature engineering:** cyclical hour (sin/cos), day-of-week, weekend, month,
   capacity, distance-to-centre, lat/lon, temperature, precipitation, rain flag,
   station id.
3. **Forecasting:** Ridge / RandomForest / GradientBoosting / HistGradientBoosting
   compared with a **time-ordered hold-out** (train past, test future); winner
   ships. Target = bikes available.
4. **Stockout risk:** HistGradientBoosting **classifier** for P(no bikes), scored
   with ROC-AUC / PR-AUC / Brier.
5. **Clustering:** **KMeans** on each station's 24-h occupancy curve (k by
   silhouette) → behaviour archetypes.
6. **Explainability:** **SHAP** (global + per-prediction).
7. **Rebalancing:** greedy nearest-neighbour matching of predicted-full →
   predicted-empty stations.

### ⚡ Short-term (next-30-min) forecaster — the GPU model
A second, stronger forecaster predicts a station's bikes **30 minutes ahead**,
*anchored on its current availability* (from the live API) plus its typical slot
profile and weather. Two GPU-trained models (see `train_forecast.py` /
`notebook/train_forecast_gpu.ipynb`):
- **XGBoost** (`device="cuda"`) — served live in the app.
- **PyTorch station-embedding MLP** — a deep-learning showcase (`nn.Embedding`
  per station + MLP).

Both are scored against **persistence** and **slot-mean** baselines on a
time-ordered hold-out, trained on full **15-minute** snapshots
(`download_forecast_data.py` builds a contiguous block so lag/`bikes_now`
features are well defined). Feed it more consecutive days in Colab for an even
stronger model.

## Project layout
```
valenbisi/
├── app/
│   ├── streamlit_app.py        # the 5-tab app
│   └── data_prep.py            # shared parsing + feature engineering
├── download_data.py            # seasonal sample (hourly) for the profile models
├── download_forecast_data.py   # contiguous 15-min block for the short-term forecaster
├── train.py                    # profile regressor + stockout classifier + clustering
├── train_forecast.py           # XGBoost-GPU + PyTorch net (next-30-min forecaster)
├── notebook/
│   ├── train_model.ipynb           # Colab: profile models
│   └── train_forecast_gpu.ipynb    # Colab (T4 GPU): short-term forecaster
├── models/   forecast_model.pkl, stockout_model.pkl, model_meta.json,
│             forecast30_xgb.pkl, forecast30_net.pt, forecast30_meta.json
├── data/     stations.parquet, station_profiles.parquet, hourly_profiles.parquet,
│             slot_profile.parquet
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
python download_data.py                               # seasonal hourly sample
python train.py                                       # profile models + clustering
python download_forecast_data.py --start 2025-03-01 --days 14   # 15-min block
python train_forecast.py                              # next-30-min forecaster
streamlit run app/streamlit_app.py
```

## Retrain in Google Colab
Open `notebook/train_model.ipynb` in Colab → *Runtime ▸ Run all*. It clones the
repo, downloads the data, runs the exact same pipeline, prints the leaderboard +
SHAP, and lets you download the trained artifacts to drop back into the repo.

## Deploy (Streamlit Community Cloud — free)
1. Push this repo to GitHub (artifacts in `models/` + `data/*.parquet` are small
   and committed; only raw archives are ignored).
2. **share.streamlit.io** → *New app* → pick the repo.
3. **Main file path** = `app/streamlit_app.py` → Deploy.
4. Paste the URL into the *Live app* section above.

---
Built with Streamlit · scikit-learn · SHAP · pydeck · Plotly.
