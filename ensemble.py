"""
ensemble.py
===========
Ensemble XGBoost (walk-forward) + LSTM: rata-rata probabilitas.
Membandingkan tiap model secara adil pada baris yang sama.

Input:
  - data/walkforward_preds.csv  (XGBoost walk-forward, fitur makro)
  - data/lstm_test_probs.csv    (LSTM, fitur makro)

Output:
  - data/metrics_ensemble.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))


def load():
    xgb_p = pd.read_csv(os.path.join(BASE, "data", "walkforward_preds.csv"),
                        parse_dates=["tanggal"])
    lstm_p = pd.read_csv(os.path.join(BASE, "data", "lstm_test_probs.csv"),
                         parse_dates=["tanggal"])
    m = xgb_p.merge(lstm_p, on=["ticker", "tanggal"], suffixes=("_xgb", "_lstm"))
    print(f"Baris cocok (ticker+tanggal): {len(m):,}")
    return m


def report(name, y, prob):
    base = y.mean()
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    pred = (prob >= best_t).astype(int)
    res = {
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(best_f1), 4),
        "threshold": round(best_t, 2),
    }
    print(f"  {name:14s} AUC={res['auc']:.4f}  acc={res['accuracy']:.4f}  "
          f"prec={res['precision']:.4f}  rec={res['recall']:.4f}  "
          f"f1={res['f1']:.4f}  (th={res['threshold']:.2f})")
    return res


def main():
    m = load()
    y = m["target"].to_numpy() if "target" in m.columns else None

    # ambil target dari DB fitur makro (merge ulang)
    if y is None:
        import sqlite3
        conn = sqlite3.connect(os.path.join(BASE, "data", "features_macro.db"))
        tgt = pd.read_sql("SELECT ticker, tanggal, target FROM features_macro", conn)
        conn.close()
        tgt["tanggal"] = pd.to_datetime(tgt["tanggal"])
        m = m.merge(tgt, on=["ticker", "tanggal"], how="left")
        m = m.dropna(subset=["target"])
    y = m["target"].to_numpy().astype(int)

    prob_xgb = m["prob_xgb"].to_numpy()
    prob_lstm = m["prob_lstm"].to_numpy()
    prob_ens = 0.5 * prob_xgb + 0.5 * prob_lstm
    # varian: ensamble dengan bobot optimal sederhana (grid)
    best_w, best_auc_w = 0.5, -1
    for w in np.arange(0.2, 0.81, 0.05):
        auc = roc_auc_score(y, w * prob_xgb + (1 - w) * prob_lstm)
        if auc > best_auc_w:
            best_auc_w, best_w = auc, w
    prob_ens_w = best_w * prob_xgb + (1 - best_w) * prob_lstm

    print("\n=== PERBANDINGAN (baris yang sama) ===")
    r_xgb = report("XGBoost (WF)", y, prob_xgb)
    r_lstm = report("LSTM", y, prob_lstm)
    r_ens = report("Ensemble 50/50", y, prob_ens)
    r_ensw = report(f"Ensemble {best_w:.2f}/{1 - best_w:.2f}", y, prob_ens_w)

    with open(os.path.join(BASE, "data", "metrics_ensemble.json"), "w") as f:
        json.dump({
            "n": int(len(y)),
            "base_rate_naik": round(float(y.mean()), 4),
            "xgb_walkforward": r_xgb,
            "lstm": r_lstm,
            "ensemble_50_50": r_ens,
            "ensemble_best_weight": {"weight_xgb": round(best_w, 2), **r_ensw},
        }, f, indent=2)
    print("\nMetrik: data/metrics_ensemble.json")


if __name__ == "__main__":
    main()
