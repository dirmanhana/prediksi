"""
train_xgb_macro.py
==================
Train XGBoost prediksi arah besok DENGAN fitur makro (IHSG, USD/IDR).
Split waktu sama dengan train_xgb.py agar bisa dibandingkan.

Output:
  - data/metrics_xgb_macro.json
  - data/xgb_macro_test_probs.csv  (proba test utk ensemble)
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

from train_xgb import FEATURE_COLS, best_threshold, evaluate  # reuse

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data", "features_macro.db")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
FULL_COLS = FEATURE_COLS + MACRO_COLS
OUT_METRICS = os.path.join(BASE, "data", "metrics_xgb_macro.json")
OUT_PROBS = os.path.join(BASE, "data", "xgb_macro_test_probs.csv")

RANDOM_STATE = 42


def time_split(df):
    dates = np.sort(df["tanggal"].unique())
    n = len(dates)
    cut_train = dates[int(n * 0.68)]
    cut_valid = dates[int(n * 0.80)]
    train = df[df["tanggal"] <= cut_train]
    valid = df[(df["tanggal"] > cut_train) & (df["tanggal"] <= cut_valid)]
    test = df[df["tanggal"] > cut_valid]
    return train, valid, test


def main():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM features_macro ORDER BY ticker, tanggal", conn)
    conn.close()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    print(f"Fitur: {len(df):,} baris, {df['ticker'].nunique()} saham, "
          f"{len(FULL_COLS)} fitur ({len(FEATURE_COLS)} teknikal + {len(MACRO_COLS)} makro)")

    train, valid, test = time_split(df)
    print(f"Train : {len(train):,} | Valid : {len(valid):,} | Test : {len(test):,}")

    X_train, y_train = train[FULL_COLS], train["target"]
    X_valid, y_valid = valid[FULL_COLS], valid["target"]
    X_test, y_test = test[FULL_COLS], test["target"]

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 100,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(f"Early stopping: iterasi {model.best_iteration}")

    y_prob_valid = model.predict_proba(X_valid)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    th = best_threshold(y_valid, y_prob_valid)
    print(f"Threshold optimal validasi: {th:.2f}")

    print("\n--- XGBoost + MAKRO (TEST) ---")
    m_test = evaluate(y_test, y_prob_test, label="TEST + makro", threshold=th)

    # bandingkan dengan versi tanpa makro
    old = json.load(open(os.path.join(BASE, "data", "metrics.json")))["test"]
    print("\n--- Perbandingan (TEST) ---")
    print(f"{'versi':28s} {'AUC':>6s} {'acc':>6s} {'f1':>6s}")
    print(f"{'XGBoost (tanpa makro)':28s} {old['roc_auc']:6.3f} {old['accuracy']:6.3f} {old['f1']:6.3f}")
    print(f"{'XGBoost (+ makro)':28s} {m_test['roc_auc']:6.3f} {m_test['accuracy']:6.3f} {m_test['f1']:6.3f}")

    with open(OUT_METRICS, "w") as f:
        json.dump({"test": m_test, "optimal_threshold": th,
                   "best_iteration": model.best_iteration}, f, indent=2)
    print(f"Metrik: {OUT_METRICS}")

    # simpan proba test utk ensemble
    out = test[["ticker", "tanggal"]].copy()
    out["prob"] = y_prob_test
    out.to_csv(OUT_PROBS, index=False)
    print(f"Proba test: {OUT_PROBS} ({len(out):,} baris)")


if __name__ == "__main__":
    main()
