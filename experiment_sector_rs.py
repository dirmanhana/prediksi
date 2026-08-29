"""
experiment_sector_rs.py
=======================
Eksperimen A/B: apakah fitur RELATIVE STRENGTH vs SEKTOR menaikkan AUC?

Fitur baru:
  - sector_ret_1 : median ret_1 saham sesektor pada tanggal yang sama
  - rs_sector_1  : ret_1 - sector_ret_1  (outperform vs sektor, 1 hari)
  - rs_sector_5  : ret_5 - median ret_5 sesektor (5 hari)

Metode: walk-forward bulanan (sama seperti walk_forward.py), dibandingkan
baseline (42 fitur) vs +RS (45 fitur) pada lipatan yang sama.

Output: data/metrics_rs_experiment.json + print per bulan
"""

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from train_xgb import FEATURE_COLS, add_features

BASE = os.path.dirname(os.path.abspath(__file__))
PQ = os.path.join(BASE, "data", "history_id_5y.parquet")
LIST_FILE = os.path.join(BASE, "data", "stocks_id.csv")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
RS_COLS = ["sector_ret_1", "rs_sector_1", "rs_sector_5"]

RANDOM_STATE = 42
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
}


def load_and_build():
    print("Load parquet + fitur...")
    df = pd.read_parquet(PQ)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df[(df["close"] > 0) & (df["open"] > 0)].sort_values(["ticker", "tanggal"])

    feat = add_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # makro (asof mundur, konsisten dgn produksi)
    import predict_daily as pd_
    macro = pd_.load_macro()
    if macro is not None:
        feat = feat.sort_values("tanggal")
        macro_s = macro[["tanggal"] + MACRO_COLS].sort_values("tanggal")
        feat["tanggal"] = feat["tanggal"].astype("datetime64[us]")
        macro_s["tanggal"] = macro_s["tanggal"].astype("datetime64[us]")
        feat = pd.merge_asof(feat, macro_s, on="tanggal", direction="backward")

    # ---- fitur relative strength vs sektor ----
    meta = pd.read_csv(LIST_FILE)
    sector_map = dict(zip(meta["name"].astype(str), meta["sector"].astype(str)))
    feat["sector"] = feat["ticker"].map(sector_map).fillna("UNKNOWN")
    feat = feat.sort_values("tanggal")
    g = feat.groupby(["tanggal", "sector"])
    feat["sector_ret_1"] = g["ret_1"].transform("median")
    feat["rs_sector_1"] = feat["ret_1"] - feat["sector_ret_1"]
    feat["rs_sector_5"] = feat["ret_5"] - g["ret_5"].transform("median")

    # target terakhir per saham = NaN (belum ada besok)
    last_idx = feat.groupby("ticker")["tanggal"].idxmax()
    feat.loc[last_idx, "target"] = np.nan
    return feat


def walkforward(feat, cols, label):
    dates = np.sort(feat["tanggal"].unique())
    test_start = pd.Timestamp(dates[int(len(dates) * 0.80)])
    test = feat[feat["tanggal"] > test_start].copy()
    # buang baris target NaN (baris terakhir per saham — belum ada "besok")
    test = test.dropna(subset=["target"])
    test["bulan"] = test["tanggal"].dt.to_period("M")
    months = sorted(test["bulan"].unique())

    monthly = []  # {bulan, auc, prob[], y[]}
    for i, m in enumerate(months):
        t0f = time.time()
        train_df = feat[feat["tanggal"] < m.start_time].dropna(subset=["target"])
        tdates = np.sort(train_df["tanggal"].unique())
        vcut = tdates[int(len(tdates) * 0.80)]
        tr = train_df[train_df["tanggal"] <= vcut]
        va = train_df[train_df["tanggal"] > vcut]
        te = test[test["bulan"] == m]
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(tr[cols], tr["target"],
                  eval_set=[(va[cols], va["target"])], verbose=False)
        prob = model.predict_proba(te[cols])[:, 1]
        monthly.append({
            "bulan": str(m), "auc": round(float(roc_auc_score(te["target"], prob)), 4),
            "prob": prob, "y": te["target"].to_numpy(),
        })
        print(f"  [{i + 1}/{len(months)}] {label} {m}: auc={monthly[-1]['auc']:.4f} "
              f"({time.time() - t0f:.0f}s)", flush=True)
    return monthly


def agg(monthly):
    prob = np.concatenate([mth["prob"] for mth in monthly])
    y = np.concatenate([mth["y"] for mth in monthly])
    return y, prob


def main():
    t0 = time.time()
    feat = load_and_build()
    print(f"Fitur siap: {len(feat):,} baris, {feat['ticker'].nunique()} saham "
          f"({time.time()-t0:.0f}s)")

    base_cols = FEATURE_COLS + MACRO_COLS
    rs_cols = base_cols + RS_COLS

    print(f"\n=== WALK-FORWARD BASELINE ({len(base_cols)} fitur) ===")
    m_base = walkforward(feat, base_cols, "baseline")
    print(f"\n=== WALK-FORWARD + RS ({len(rs_cols)} fitur) ===")
    m_rs = walkforward(feat, rs_cols, "rs")

    y_base, p_base = agg(m_base)
    y_rs, p_rs = agg(m_rs)
    assert len(y_base) == len(y_rs) and (y_base == y_rs).all()

    print("\n=== PERBANDINGAN (OOS, lipatan sama) ===")
    print(f"{'bulan':10s} {'AUC base':>9s} {'AUC +RS':>9s} {'delta':>7s}")
    for mb, mr in zip(m_base, m_rs):
        d = mr["auc"] - mb["auc"]
        print(f"{mb['bulan']:10s} {mb['auc']:9.4f} {mr['auc']:9.4f} {d:+7.4f}")

    ab, ar = roc_auc_score(y_base, p_base), roc_auc_score(y_rs, p_rs)
    print("-" * 38)
    print(f"{'AGREGAT':10s} {ab:9.4f} {ar:9.4f} {ar-ab:+7.4f}")

    verdict = ("RS MEMBANTU" if ar - ab >= 0.003
               else "RS NETRAL (tanpa efek jelas)" if ar - ab > -0.003
               else "RS MERUGIKAN")
    print(f"\nKesimpulan: {verdict} (delta {ar-ab:+.4f})")

    with open(os.path.join(BASE, "data", "metrics_rs_experiment.json"), "w") as f:
        json.dump({
            "n": int(len(y_base)),
            "auc_baseline": round(ab, 4),
            "auc_with_rs": round(ar, 4),
            "delta": round(ar - ab, 4),
            "verdict": verdict,
            "rs_cols": RS_COLS,
            "monthly": [{"bulan": mb["bulan"], "base": mb["auc"], "rs": mr["auc"]}
                        for mb, mr in zip(m_base, m_rs)],
        }, f, indent=2)
    print("Hasil: data/metrics_rs_experiment.json")


if __name__ == "__main__":
    main()
