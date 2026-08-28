"""
walk_forward.py
===============
Walk-forward validation untuk XGBoost (dengan fitur makro).

Ide: setiap bulan di periode test, model DILATIH ULANG hanya dengan data
sebelum bulan tersebut (expanding window), lalu memprediksi bulan berikutnya.
Ini simulasi penggunaan nyata: model tidak pernah melihat masa depan.

Output:
  - data/walkforward_preds.csv   (ticker, tanggal, prob) — prediksi out-of-sample
  - data/metrics_walkforward.json
"""

import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from train_xgb import FEATURE_COLS  # reuse

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data", "features_macro.db")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
FULL_COLS = FEATURE_COLS + MACRO_COLS

OUT_PROBS = os.path.join(BASE, "data", "walkforward_preds.csv")
OUT_METRICS = os.path.join(BASE, "data", "metrics_walkforward.json")

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


def main():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM features_macro ORDER BY ticker, tanggal", conn)
    conn.close()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    print(f"Fitur: {len(df):,} baris, {df['ticker'].nunique()} saham")

    # periode test dimulai setelah cutoff 80% tanggal (sama dengan sebelumnya)
    dates = np.sort(df["tanggal"].unique())
    test_start = pd.Timestamp(dates[int(len(dates) * 0.80)])
    print(f"Periode test walk-forward dimulai: {test_start.date()}")

    # batas bulanan dalam periode test
    test_df = df[df["tanggal"] > test_start].copy()
    test_df["bulan"] = test_df["tanggal"].dt.to_period("M")
    months = sorted(test_df["bulan"].unique())
    print(f"Jumlah langkah bulanan: {len(months)} ({months[0]} s/d {months[-1]})")

    preds, monthly = [], []
    t0 = time.time()

    for i, m in enumerate(months):
        # data pelatihan: SEMUA sebelum bulan ini (expanding window)
        train_df = df[df["tanggal"] < m.start_time]
        # validasi kronologis: 20% tanggal terakhir dari window pelatihan
        tdates = np.sort(train_df["tanggal"].unique())
        vcut = tdates[int(len(tdates) * 0.80)]
        tr = train_df[train_df["tanggal"] <= vcut]
        va = train_df[train_df["tanggal"] > vcut]
        # prediksi: bulan ini
        te = test_df[test_df["bulan"] == m]

        model = xgb.XGBClassifier(**PARAMS)
        model.fit(
            tr[FULL_COLS], tr["target"],
            eval_set=[(va[FULL_COLS], va["target"])],
            verbose=False,
        )
        prob = model.predict_proba(te[FULL_COLS])[:, 1]

        tmp = te[["ticker", "tanggal"]].copy()
        tmp["prob"] = prob
        preds.append(tmp)

        auc = roc_auc_score(te["target"], prob)
        monthly.append({"bulan": str(m), "n": len(te),
                        "auc": round(auc, 4),
                        "base_rate": round(te["target"].mean(), 4)})
        elapsed = (time.time() - t0) / 60
        print(f"  [{i + 1}/{len(months)}] {m}: n={len(te):,} auc={auc:.4f} "
              f"({elapsed:.1f} menit)")

    allp = pd.concat(preds, ignore_index=True)
    allp.to_csv(OUT_PROBS, index=False)

    # ---- evaluasi agregat ----
    y = df.set_index(["ticker", "tanggal"]).loc[
        pd.MultiIndex.from_frame(allp[["ticker", "tanggal"]])
    ]["target"].to_numpy()
    prob = allp["prob"].to_numpy()

    base = y.mean()
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    print("\n=== WALK-FORWARD: hasil out-of-sample ===")
    print(f"  ROC-AUC : {roc_auc_score(y, prob):.4f}")
    print(f"  Akurasi : {accuracy_score(y, (prob >= best_t).astype(int)):.4f} "
          f"(baseline: {max(base, 1 - base):.4f})")
    print(f"  F1 (th={best_t:.2f}): {best_f1:.4f}")
    print(f"  Total prediksi OOS: {len(y):,}")

    print("\n=== AUC per bulan ===")
    for mrow in monthly:
        print(f"  {mrow['bulan']}: auc={mrow['auc']:.4f}  "
              f"base_rate_naik={mrow['base_rate']:.3f}  n={mrow['n']:,}")

    with open(OUT_METRICS, "w") as f:
        json.dump({
            "auc": round(float(roc_auc_score(y, prob)), 4),
            "accuracy": round(float(accuracy_score(y, (prob >= best_t).astype(int))), 4),
            "f1": round(float(best_f1), 4),
            "optimal_threshold": round(best_t, 2),
            "baseline_accuracy": round(float(max(base, 1 - base)), 4),
            "base_rate_naik": round(float(base), 4),
            "n": int(len(y)),
            "monthly": monthly,
        }, f, indent=2)
    print(f"\nPrediksi: {OUT_PROBS}\nMetrik  : {OUT_METRICS}")


if __name__ == "__main__":
    main()
