"""
Train the Valenbisi availability models and produce every artifact the app uses.

Run locally:   python train.py        (needs data/history.parquet — see download_data.py)
Run in Colab:  notebook/train_model.ipynb runs this same script.

Outputs:
    models/forecast_model.pkl    regressor pipeline -> predicts bikes_available
    models/stockout_model.pkl    classifier pipeline -> P(station empty)
    models/model_meta.json       leaderboard, metrics, feature list, data window
    data/station_profiles.parquet  per-station cluster + 24h occupancy curve + chronic rates
    data/hourly_profiles.parquet   station x hour x weekend empirical bands (for the app)

Evaluation note
---------------
We predict the *typical* availability of a station for a given hour / weekday /
month / weather — a planning tool, not a live next-step forecast. The honest test
is therefore a hold-out of **whole random days** (GroupShuffleSplit by date): no
within-day leakage, and no year-long fleet-drift artefact that a forward split
would introduce.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
import data_prep as dp  # noqa: E402

HIST = "data/history.parquet"
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), dp.FEATURES["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             dp.FEATURES["categorical"]),
        ],
        remainder="drop",
    )


def day_split(df: pd.DataFrame, test_size=0.2, seed=42):
    """Hold out whole random days (no intra-day leakage)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df["date"].astype(str)))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def main():
    print(">> Loading", HIST)
    df = pd.read_parquet(HIST)
    feats = dp.all_feature_names()
    print(f"   rows={len(df):,}  stations={df['station_id'].nunique()}  "
          f"days={df['date'].nunique()}")

    train_df, test_df = day_split(df)
    Xtr, ytr = train_df[feats], train_df[dp.TARGET_REG]
    Xte, yte = test_df[feats], test_df[dp.TARGET_REG]
    print(f"   train={len(train_df):,} ({train_df['date'].nunique()} days)  "
          f"test={len(test_df):,} ({test_df['date'].nunique()} days)")

    # leaderboard on a common subsample, evaluated on the same held-out days
    sub = train_df.sample(min(60000, len(train_df)), random_state=42)
    Xs, ys = sub[feats], sub[dp.TARGET_REG]

    candidates = {
        "Ridge (baseline)": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=60, max_depth=14, min_samples_leaf=40,
            n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=None,
            l2_regularization=1.0, random_state=42),
    }

    print(">> Regressor model selection (hold-out days)")
    leaderboard = []
    for name, est in candidates.items():
        pipe = Pipeline([("pre", make_preprocessor()), ("model", est)])
        pipe.fit(Xs, ys)
        pred = np.clip(pipe.predict(Xte), 0, None)
        res = {"model": name,
               "mae_bikes": round(float(mean_absolute_error(yte, pred)), 3),
               "r2": round(float(r2_score(yte, pred)), 4)}
        leaderboard.append(res)
        print(f"   {name:24} MAE={res['mae_bikes']:.2f} bikes  R2={res['r2']:.3f}")

    best_name = min(leaderboard, key=lambda d: d["mae_bikes"])["model"]
    print(">> Best regressor:", best_name)

    # refit winner on full train -> honest metrics, then on all data -> ship
    reg = Pipeline([("pre", make_preprocessor()), ("model", candidates[best_name])])
    reg.fit(Xtr, ytr)
    pred_te = np.clip(reg.predict(Xte), 0, None)
    reg_test = {
        "mae_bikes": round(float(mean_absolute_error(yte, pred_te)), 3),
        "r2": round(float(r2_score(yte, pred_te)), 4),
        "mae_docks": round(float(mean_absolute_error(
            test_df["docks_free"].fillna(test_df["capacity"] - yte),
            np.clip(test_df["capacity"] - pred_te, 0, None))), 3),
        "naive_mae_bikes": round(float(mean_absolute_error(
            yte, np.full(len(yte), ytr.mean()))), 3),
    }
    print(f"   hold-out: MAE={reg_test['mae_bikes']} bikes  R2={reg_test['r2']}  "
          f"(naive-mean MAE={reg_test['naive_mae_bikes']})")
    reg.fit(df[feats], df[dp.TARGET_REG])
    joblib.dump(reg, "models/forecast_model.pkl")
    print(">> Saved models/forecast_model.pkl")

    # stockout classifier: P(station empty)
    clf = Pipeline([("pre", make_preprocessor()),
                    ("model", HistGradientBoostingClassifier(
                        max_iter=400, learning_rate=0.06,
                        l2_regularization=1.0, random_state=42))])
    clf.fit(Xtr, train_df[dp.TARGET_CLF])
    proba = clf.predict_proba(Xte)[:, 1]
    clf_test = {
        "roc_auc": round(float(roc_auc_score(test_df[dp.TARGET_CLF], proba)), 4),
        "pr_auc": round(float(average_precision_score(test_df[dp.TARGET_CLF], proba)), 4),
        "brier": round(float(brier_score_loss(test_df[dp.TARGET_CLF], proba)), 4),
        "base_rate": round(float(test_df[dp.TARGET_CLF].mean()), 4),
    }
    print(f"   stockout classifier: ROC-AUC={clf_test['roc_auc']} "
          f"PR-AUC={clf_test['pr_auc']} Brier={clf_test['brier']}")
    clf.fit(df[feats], df[dp.TARGET_CLF])
    joblib.dump(clf, "models/stockout_model.pkl")
    print(">> Saved models/stockout_model.pkl")

    # ---- station clustering on the 24h occupancy signature -----------------
    curve = df.pivot_table(index="station_id", columns="hour",
                           values="occupancy", aggfunc="mean").reindex(columns=range(24))
    # fill any missing hour with that station's own daily mean
    curve = curve.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0.4)
    Z = curve.sub(curve.mean(axis=1), axis=0)          # centre each station's shape
    Zs = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-9)
    best_k, best_sil = 4, -1.0
    for k in (3, 4, 5):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Zs)
        sil = float(silhouette_score(Zs, labels))
        print(f"   KMeans k={k}: silhouette={sil:.3f}")
        if sil > best_sil:
            best_k, best_sil = k, sil
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Zs)
    cluster_of = pd.Series(km.labels_, index=curve.index, name="cluster")

    # Clusters capture the *shape* of a station's day (we centred each curve), so
    # name them by when each cluster is fullest relative to its own daily mean.
    centroids = Z.groupby(cluster_of).mean()          # cluster x hour (centred)

    def shape_name(row: pd.Series) -> str:
        peak = int(np.asarray(row.values).argmax())
        if peak <= 6:
            return "Morning-emptying (residential)"
        if peak <= 11:
            return "Late-morning full"
        if peak <= 16:
            return "Midday full (centre/leisure)"
        return "Evening full (commute return)"

    from collections import Counter
    base = {c: shape_name(centroids.loc[c]) for c in centroids.index}
    counts = Counter(base.values())
    amp = centroids.max(axis=1) - centroids.min(axis=1)   # daily swing per cluster
    names = {}
    for c in centroids.index:
        nm = base[c]
        if counts[nm] > 1:   # distinguish same-shape clusters by how strongly they swing
            grp = sorted((cc for cc in centroids.index if base[cc] == nm),
                         key=lambda cc: amp[cc], reverse=True)
            tag = ["pronounced", "mild", "weak"][grp.index(c)] if grp.index(c) < 3 \
                else f"#{grp.index(c)}"
            nm = f"{nm} — {tag}"
        names[c] = nm

    # ---- per-station profile table -----------------------------------------
    stations = pd.read_parquet("data/stations.parquet").set_index("station_id")
    chronic = df.groupby("station_id").agg(
        chronic_empty_pct=("is_empty", lambda s: round(100 * s.mean(), 1)),
        chronic_full_pct=("is_full", lambda s: round(100 * s.mean(), 1)))
    prof_tbl = stations.join(cluster_of).join(chronic)
    prof_tbl["cluster_name"] = prof_tbl["cluster"].map(names)
    for h in range(24):
        prof_tbl[f"occ_h{h:02d}"] = curve[h]
    prof_tbl = prof_tbl.reset_index()
    prof_tbl.to_parquet("data/station_profiles.parquet", index=False)
    print(f">> Saved data/station_profiles.parquet  k={best_k}")

    # ---- empirical hourly bands (station x hour x weekend) for the app ------
    bands = (df.groupby(["station_id", "hour", "is_weekend"])
             .agg(mean_bikes=("bikes_available", "mean"),
                  p10_bikes=("bikes_available", lambda s: np.percentile(s, 10)),
                  p90_bikes=("bikes_available", lambda s: np.percentile(s, 90)))
             .reset_index())
    bands.to_parquet("data/hourly_profiles.parquet", index=False)
    print(">> Saved data/hourly_profiles.parquet")

    meta = {
        "trained_on": str(date.today()),
        "data_source": "ceferra/valenbici (Valencia open data) + Open-Meteo",
        "data_window": f"{df['date'].min()} … {df['date'].max()}",
        "n_rows": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "n_days": int(df["date"].nunique()),
        "eval": "GroupShuffleSplit by day (20% held-out days)",
        "features": dp.FEATURES,
        "best_regressor": best_name,
        "regressor_leaderboard": leaderboard,
        "regressor_test": reg_test,
        "stockout_classifier": "HistGradientBoostingClassifier",
        "stockout_test": clf_test,
        "clustering": {"algo": "KMeans", "k": int(best_k),
                       "silhouette": round(best_sil, 3),
                       "names": {str(k): v for k, v in names.items()}},
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(">> Saved models/model_meta.json")
    print(f"\nDone. Regressor={best_name} (MAE {reg_test['mae_bikes']} bikes vs "
          f"naive {reg_test['naive_mae_bikes']}, R² {reg_test['r2']}) · "
          f"stockout AUC {clf_test['roc_auc']}")


if __name__ == "__main__":
    main()
