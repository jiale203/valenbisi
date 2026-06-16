"""
Train the short-term (next-30-min) Valenbisi forecaster — two models:

  1. XGBoost   (GPU-accelerated when a CUDA device is present)  -> served by the app
  2. PyTorch station-embedding MLP (the deep-learning showcase, GPU when present)

Both anchor on the station's *current* availability (`bikes_now`) plus its typical
slot profile + weather, so they massively beat the abstract "typical profile"
model. We report each against two naive baselines (persistence, slot-mean).

Run locally (CPU ok):  python train_forecast.py
Run in Colab (T4 GPU):  notebook/train_forecast_gpu.ipynb (set runtime to GPU)

Outputs:
    models/forecast_short.pkl    served model — sklearn HistGradientBoosting (app loads this)
    models/forecast_net.pt       PyTorch state_dict (+ _cfg.json) — GPU showcase
    models/forecast_meta.json    metrics, baselines, horizon, features
    (data/slot_profile.parquet is produced by download_forecast_data.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
import forecast_prep as fp  # noqa: E402

HIST = "data/history_15min.parquet"
SLOT = "data/slot_profile.parquet"
os.makedirs("models", exist_ok=True)


def metrics(y, pred, cap=None):
    pred = np.clip(pred, 0, cap) if cap is not None else np.clip(pred, 0, None)
    return {"mae": round(float(mean_absolute_error(y, pred)), 4),
            "r2": round(float(r2_score(y, pred)), 4)}


def time_split(df, frac=0.8):
    df = df.sort_values("dt").reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


# --------------------------------------------------------------------------- #
# PyTorch station-embedding MLP
# --------------------------------------------------------------------------- #
def train_net(tr, te, feats, device, epochs=25, emb_dim=16, batch=4096):
    import torch
    import torch.nn as nn

    sta_index = {s: i for i, s in enumerate(sorted(tr["station_id"].unique()))}
    n_sta = len(sta_index)
    scaler = StandardScaler().fit(tr[feats].values)

    def to_tensors(d):
        Xn = torch.tensor(scaler.transform(d[feats].values), dtype=torch.float32)
        Xs = torch.tensor(d["station_id"].map(sta_index).fillna(0).astype(int).values,
                          dtype=torch.long)
        y = torch.tensor(d[fp.TARGET].values, dtype=torch.float32)
        return Xn, Xs, y

    Xn_tr, Xs_tr, y_tr = to_tensors(tr)
    Xn_te, Xs_te, y_te = to_tensors(te)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_sta, emb_dim)
            self.mlp = nn.Sequential(
                nn.Linear(len(feats) + emb_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 1))

        def forward(self, xn, xs):
            return self.mlp(torch.cat([xn, self.emb(xs)], dim=1)).squeeze(1)

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss()
    n = len(y_tr)
    Xn_tr, Xs_tr, y_tr = Xn_tr.to(device), Xs_tr.to(device), y_tr.to(device)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = net(Xn_tr[idx], Xs_tr[idx])
            loss = lossf(out, y_tr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"     epoch {ep+1:>2}/{epochs}  loss={tot/n:.4f}")

    net.eval()
    with torch.no_grad():
        pred = net(Xn_te.to(device), Xs_te.to(device)).cpu().numpy()
    m = metrics(y_te.numpy(), pred, cap=te["capacity"].values)

    # persist
    torch.save(net.state_dict(), "models/forecast_net.pt")
    cfg = {"features": feats, "emb_dim": emb_dim, "n_stations": n_sta,
           "station_index": {str(k): v for k, v in sta_index.items()},
           "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    with open("models/forecast_net_cfg.json", "w") as f:
        json.dump(cfg, f)
    return m


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--no-net", action="store_true", help="skip the PyTorch model")
    args = ap.parse_args()

    print(">> Loading", HIST)
    hist = pd.read_parquet(HIST)
    slot = pd.read_parquet(SLOT)
    df = fp.build_supervised(hist, slot)
    feats = fp.all_feature_names_fc()
    print(f"   supervised rows={len(df):,}  stations={df['station_id'].nunique()}  "
          f"horizon={fp.HORIZON_MIN}min  features={len(feats)}")

    tr, te = time_split(df)
    yte = te[fp.TARGET].values
    cap = te["capacity"].values
    print(f"   train={len(tr):,}  test={len(te):,} (time-ordered)")

    # naive baselines on the held-out tail
    base = {
        "persistence": metrics(yte, te["bikes_now"].values, cap),
        "slot_mean": metrics(yte, te["slot_mean"].values, cap),
    }
    print(f">> Baselines  persistence MAE={base['persistence']['mae']} "
          f"R2={base['persistence']['r2']} | slot MAE={base['slot_mean']['mae']}")

    # ---- SERVED model: scikit-learn HistGradientBoosting -------------------
    # Pure-sklearn so the deployed app needs no extra dependency (xgboost has no
    # wheel on very new Python versions and would break the cloud build).
    from sklearn.ensemble import HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=42)
    hgb.fit(tr[feats], tr[fp.TARGET])
    served_m = metrics(yte, hgb.predict(te[feats]), cap)
    print(f">> Served HistGradientBoosting  MAE={served_m['mae']} bikes  "
          f"R2={served_m['r2']}")
    joblib.dump({"model": hgb, "features": feats, "horizon_min": fp.HORIZON_MIN},
                "models/forecast_short.pkl")
    print(">> Saved models/forecast_short.pkl")

    # ---- SHOWCASE: XGBoost on GPU (optional; not required to serve) ---------
    device, xgb_m = "cpu", None
    try:
        import xgboost as xgb
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:                            # noqa: BLE001
            device = "cpu"
        xreg = xgb.XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.0, tree_method="hist", device=device,
            early_stopping_rounds=40, eval_metric="mae")
        xreg.fit(tr[feats], tr[fp.TARGET],
                 eval_set=[(te[feats], te[fp.TARGET])], verbose=False)
        xgb_m = metrics(yte, xreg.predict(te[feats]), cap)
        print(f">> [showcase] XGBoost ({device})  MAE={xgb_m['mae']}  R2={xgb_m['r2']}")
    except Exception as e:                           # noqa: BLE001
        print(f">> [showcase] XGBoost skipped ({type(e).__name__})")

    # ---- SHOWCASE: PyTorch station-embedding net (optional) ----------------
    net_m = None
    if not args.no_net:
        try:
            print(">> [showcase] Training PyTorch station-embedding MLP")
            net_m = train_net(tr, te, feats, device, epochs=args.epochs)
            print(f">> [showcase] Net  MAE={net_m['mae']}  R2={net_m['r2']}")
        except Exception as e:                       # noqa: BLE001
            print(f">> [showcase] Net skipped ({type(e).__name__})")

    meta = {
        "trained_on": str(date.today()),
        "task": f"forecast bikes_available {fp.HORIZON_MIN} min ahead",
        "data_source": "ceferra/valenbici 15-min snapshots + Open-Meteo",
        "data_window": f"{hist['dt'].min()} … {hist['dt'].max()}",
        "n_rows": int(len(df)), "n_stations": int(df["station_id"].nunique()),
        "horizon_min": fp.HORIZON_MIN, "features": feats,
        "served_model": "HistGradientBoostingRegressor", "served": served_m,
        "baselines": base, "xgboost": xgb_m, "xgboost_device": device, "net": net_m,
    }
    with open("models/forecast_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(">> Saved models/forecast_meta.json")

    cands = [("persistence", base["persistence"]["mae"]), ("served-HGB", served_m["mae"])]
    cands += [("XGBoost", xgb_m["mae"])] if xgb_m else []
    cands += [("Net", net_m["mae"])] if net_m else []
    best = min(cands, key=lambda t: t[1])
    print(f"\nDone. Served MAE {served_m['mae']} bikes vs persistence "
          f"{base['persistence']['mae']}. Best overall: {best[0]} = {best[1]}.")


if __name__ == "__main__":
    main()
