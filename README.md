# 🚲 ValenWheel — Valenbisi availability, forecasting & rebalancing

An interactive **Streamlit** app that tackles a daily València problem: **a
Valenbisi rider often finds no bike to take, or no free dock to return one.**
ValenWheel combines a **live feed**, a **trained forecasting model**, a
**stockout-risk classifier** and a **rebalancing planner** into one tool for
both residents and the bike-share operator.

| Tab | What it does | DS method |
|-----|--------------|-----------|
| 🚲 **Live Now** | Real-time map of every station (bikes / free docks), flags empty & full | Live API + geospatial viz |
| 🔮 **Availability Forecast** | Predicts bikes & free docks for any station, day, hour & **weather scenario**, with stockout probability | Gradient-boosted **regression** + **SHAP** + a **classifier** |
| ♻️ **Rebalancing Planner** | Maps where bikes/docks will run out at a chosen time and suggests **moves** | Model-driven + greedy nearest-neighbour matching |
| 🧭 **Station Clusters** | Groups stations by their 24-h behaviour (commuter source/sink, leisure…) | **KMeans** clustering |
| 🤖 **Model & Method** | Leaderboard, metrics, SHAP importance, methodology | Time-aware model selection |

## Live app
👉 **[Deployed on Streamlit Community Cloud]](https://valenbisi-3erh2tubk7hurcmnbmamgj.streamlit.app/)[(#)**

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

## Project layout
```
valenbisi/
├── app/
│   ├── streamlit_app.py        # the 5-tab app
│   └── data_prep.py            # shared parsing + feature engineering
├── download_data.py            # fetch ceferra/valenbici + Open-Meteo → parquet
├── train.py                    # model selection + classifier + clustering → artifacts
├── notebook/train_model.ipynb  # Google Colab training notebook
├── models/   forecast_model.pkl, stockout_model.pkl, model_meta.json
├── data/     history.parquet, stations.parquet, station_profiles.parquet, hourly_profiles.parquet
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
python download_data.py      # build data/history.parquet (sample of open data)
python train.py              # train models + write artifacts
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
