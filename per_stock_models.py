"""
per_stock_models.py
===================
Model PER-SAham untuk 20 saham IDX paling likuid (nilai transaksi tertinggi).
Fitur makro disertakan (IHSG, USD/IDR) — biasanya lebih berguna di level per-saham.

1. Klasifikasi: arah besok (naik/turun), WALK-FORWARD bulanan (retrain per bulan)
   -> AUC out-of-sample per saham + agregat
2. Regresi: prediksi return besok (return_1d.shift(-1)), split 80/20
3. Backtest: strategi "long jika prediksi return > 0" vs Buy & Hold
   (biaya transaksi 0.2% round-trip, 1 hari hold)

Output:
  - data/per_stock_results.csv    (ringkasan tiap saham)
  - data/per_stock_walkforward_preds.csv
  - data/metrics_per_stock.json
"""

import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from train_xgb import FEATURE_COLS  # reuse

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data", "features_macro.db")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
FULL_COLS = FEATURE_COLS + MACRO_COLS

TOP_N = 20
COST_RT = 0.002          # biaya round-trip (masuk+keluar) 0.2%
RANDOM_STATE = 42

PARAMS_CLF = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 600,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
}
PARAMS_REG = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 600,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
}


def select_liquid(df, top_n=TOP_N):
    """20 saham dengan rata-rata nilai transaksi harian terbesar (12 bulan terakhir)."""
    last = df[df["tanggal"] >= df["tanggal"].max() - pd.Timedelta(days=365)]
    val = last.assign(value=last["close"] * last["volume"]).groupby("ticker")["value"].mean()
    return val.nlargest(top_n).index.tolist(), val


def clf_walkforward(stock_df, cols):
    """Klasifikasi walk-forward bulanan utk satu saham -> (probs, y, months)."""
    dates = np.sort(stock_df["tanggal"].unique())
    test_start = dates[int(len(dates) * 0.80)]
    test = stock_df[stock_df["tanggal"] > test_start].copy()
    test["bulan"] = test["tanggal"].dt.to_period("M")

    probs, ys, months = [], [], []
    for m in sorted(test["bulan"].unique()):
        tr_df = stock_df[stock_df["tanggal"] < m.start_time]
        if len(tr_df) < 300:
            continue
        tdates = np.sort(tr_df["tanggal"].unique())
        vcut = tdates[int(len(tdates) * 0.80)]
        tr, va = tr_df[tr_df["tanggal"] <= vcut], tr_df[tr_df["tanggal"] > vcut]
        te = test[test["bulan"] == m]

        model = xgb.XGBClassifier(**PARAMS_CLF)
        model.fit(tr[cols], tr["target"], eval_set=[(va[cols], va["target"])], verbose=False)
        probs.append(model.predict_proba(te[cols])[:, 1])
        ys.append(te["target"].to_numpy())
        months.append(te["bulan"].to_numpy())

    return np.concatenate(probs), np.concatenate(ys), np.concatenate(months)


def reg_backtest(stock_df, cols):
    """Regresi return besok + backtest strategi long jika pred > 0."""
    d = stock_df.copy()
    d["ret_tomorrow"] = d.groupby("ticker")["return_1d"].shift(-1)
    d = d.dropna(subset=["ret_tomorrow"])
    dates = np.sort(d["tanggal"].unique())
    cut = dates[int(len(dates) * 0.80)]
    tr, te = d[d["tanggal"] <= cut], d[d["tanggal"] > cut]

    if len(tr) < 300 or len(te) < 60:
        return None

    # validasi utk early stopping
    tdates = np.sort(tr["tanggal"].unique())
    vcut = tdates[int(len(tdates) * 0.80)]
    va = tr[tr["tanggal"] > vcut]
    tr2 = tr[tr["tanggal"] <= vcut]

    model = xgb.XGBRegressor(**PARAMS_REG)
    model.fit(tr2[cols], tr2["ret_tomorrow"],
              eval_set=[(va[cols], va["ret_tomorrow"])], verbose=False)
    pred = model.predict(te[cols])
    y = te["ret_tomorrow"].to_numpy()

    # --- backtest ---
    signal = pred > 0
    # biaya transaksi saat sinyal berubah (masuk/keluar)
    change = signal.astype(int)
    cost = np.abs(np.diff(change, prepend=0)).astype(float) * (COST_RT / 2)
    strat_daily = np.where(signal, y, 0.0) - cost
    strat_total = np.prod(1 + strat_daily) - 1
    buyhold_total = np.prod(1 + y) - 1

    trades = int(change.sum())
    hit_rate = float((y[signal] > 0).mean()) if trades else np.nan
    return {
        "auc": None,
        "corr": float(np.corrcoef(pred, y)[0, 1]),
        "strat_return": float(strat_total),
        "buyhold_return": float(buyhold_total),
        "trades": trades,
        "hit_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else None,
        "n_test": int(len(y)),
        "n_train": int(len(tr2)),
    }


def main():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM features_macro ORDER BY ticker, tanggal", conn)
    conn.close()
    df["tanggal"] = pd.to_datetime(df["tanggal"])

    liquid, val = select_liquid(df)
    print(f"20 saham paling likuid (rata-rata nilai transaksi/hari, 12 bln terakhir):")
    for i, tk in enumerate(liquid, 1):
        print(f"  {i:2d}. {tk:6s}  Rp {val[tk] / 1e9:10,.1f} M")

    rows = []
    wf_all = {"ticker": [], "tanggal": [], "prob": [], "target": []}
    t0 = time.time()

    for i, tk in enumerate(liquid, 1):
        sd = df[df["ticker"] == tk].copy()
        if len(sd) < 400:
            print(f"  [{i}/20] {tk}: data kurang ({len(sd)})")
            continue

        # 1) klasifikasi walk-forward
        try:
            probs, ys, months = clf_walkforward(sd, FULL_COLS)
            auc = roc_auc_score(ys, probs)
            base = ys.mean()
        except Exception as e:
            print(f"  [{i}/20] {tk}: clf error {e}")
            auc, probs, ys = None, None, None

        # 2) regresi + backtest
        reg = reg_backtest(sd, FULL_COLS)

        if auc is not None:
            wf_all["ticker"].extend([tk] * len(ys))
            wf_all["tanggal"].extend(sd[sd["tanggal"] > np.sort(sd["tanggal"].unique())[int(len(sd["tanggal"].unique()) * 0.80)]]["tanggal"].tolist()[:len(ys)])
            wf_all["prob"].extend(probs.tolist())
            wf_all["target"].extend(ys.tolist())

        r = {"ticker": tk, "rows": len(sd),
             "clf_auc": round(auc, 4) if auc is not None else None,
             "clf_base_rate": round(base, 4) if auc is not None else None}
        if reg:
            r.update({
                "corr_pred_ret": round(reg["corr"], 4),
                "strat_return_pct": round(reg["strat_return"] * 100, 2),
                "buyhold_return_pct": round(reg["buyhold_return"] * 100, 2),
                "trades": reg["trades"],
                "hit_rate": reg["hit_rate"],
            })
        rows.append(r)

        status = f"auc={auc:.3f}" if auc is not None else "auc=?"
        if reg:
            status += f" | strat={r['strat_return_pct']:+.1f}% bh={r['buyhold_return_pct']:+.1f}%"
        print(f"  [{i}/{len(liquid)}] {tk:6s} {status}  ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(BASE, "data", "per_stock_results.csv"), index=False)

    wf = pd.DataFrame(wf_all)
    wf.to_csv(os.path.join(BASE, "data", "per_stock_walkforward_preds.csv"), index=False)

    # ---- agregat ----
    print("\n" + "=" * 66)
    print("RINGKASAN 20 SAHAM LIKUID")
    print("=" * 66)
    print(res.to_string(index=False))

    clf_aucs = res["clf_auc"].dropna()
    print(f"\n--- Klasifikasi (walk-forward OOS) ---")
    print(f"  Rata-rata AUC/saham : {clf_aucs.mean():.4f}")
    print(f"  Median AUC/saham    : {clf_aucs.median():.4f}")
    print(f"  Saham AUC > 0.55    : {(clf_aucs > 0.55).sum()}/{len(clf_aucs)}")

    beat = (res["strat_return_pct"] > res["buyhold_return_pct"]).sum()
    print(f"\n--- Backtest (regresi, biaya 0.2% round-trip) ---")
    print(f"  Strategi menang vs Buy&Hold: {beat}/{len(res)} saham")
    print(f"  Rata-rata return strategi  : {res['strat_return_pct'].mean():+.2f}%")
    print(f"  Rata-rata return buy&hold  : {res['buyhold_return_pct'].mean():+.2f}%")

    # AUC agregat (gabung semua prediksi OOS)
    agg_auc = roc_auc_score(wf["target"], wf["prob"])
    print(f"\n--- AUC agregat (gabung semua saham) : {agg_auc:.4f} ---")

    with open(os.path.join(BASE, "data", "metrics_per_stock.json"), "w") as f:
        json.dump({
            "avg_auc_per_stock": round(float(clf_aucs.mean()), 4),
            "median_auc_per_stock": round(float(clf_aucs.median()), 4),
            "aggregate_auc": round(float(agg_auc), 4),
            "strategy_beats_buyhold": int(beat),
            "n_stocks": int(len(res)),
            "cost_round_trip": COST_RT,
        }, f, indent=2)
    print("\nFile: data/per_stock_results.csv, data/per_stock_walkforward_preds.csv, "
          "data/metrics_per_stock.json")


if __name__ == "__main__":
    main()
