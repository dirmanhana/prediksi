"""
experiment_foreign_ff.py
========================
Eksperimen A/B: apakah fitur FOREIGN FLOW (net asing, sumber resmi idx.co.id)
menaikkan AUC?

Fitur baru:
  - ff_buy_ratio  : foreign buy / volume (proporsi beli asing)
  - ff_sell_ratio : foreign sell / volume (proporsi jual asing)
  - ff_net_ratio  : (foreign buy - foreign sell) / volume

Metode: walk-forward bulanan (sama seperti experiment_sector_rs.py), baseline
(45 fitur: 34 teknikal + 8 makro + 3 RS) vs +FF (48 fitur), lipatan sama.

Output: data/metrics_ff_experiment.json
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
FF_CSV = os.path.join(BASE, "data", "foreign_flow.csv")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
RS_COLS = ["sector_ret_1", "rs_sector_1", "rs_sector_5"]
FF_COLS = ["ff_buy_ratio", "ff_sell_ratio", "ff_net_ratio"]

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

    # makro (asof mundur)
    import predict_daily as pd_
    macro = pd_.load_macro()
    if macro is not None:
        feat = feat.sort_values("tanggal")
        macro_s = macro[["tanggal"] + MACRO_COLS].sort_values("tanggal")
        feat["tanggal"] = feat["tanggal"].astype("datetime64[us]")
        macro_s["tanggal"] = macro_s["tanggal"].astype("datetime64[us]")
        feat = pd.merge_asof(feat, macro_s, on="tanggal", direction="backward")

    # RS sektor (sama seperti produksi)
    meta = pd.read_csv(LIST_FILE)
    sector_map = dict(zip(meta["name"].astype(str), meta["sector"].astype(str)))
    feat["sector"] = feat["ticker"].map(sector_map).fillna("UNKNOWN")
    feat = feat.sort_values("tanggal")
    g = feat.groupby(["tanggal", "sector"])
    feat["sector_ret_1"] = g["ret_1"].transform("median")
    feat["rs_sector_1"] = feat["ret_1"] - feat["sector_ret_1"]
    feat["rs_sector_5"] = feat["ret_5"] - g["ret_5"].transform("median")

    # foreign flow (exact merge + ffill per saham ≈ asof mundur, tanpa lookahead)
    ff = pd.read_csv(FF_CSV)
    ff["tanggal"] = pd.to_datetime(ff["tanggal"]).astype("datetime64[us]")
    feat["tanggal"] = feat["tanggal"].astype("datetime64[us]")
    feat = feat.sort_values(["ticker", "tanggal"])
    ff = ff.sort_values(["ticker", "tanggal"])
    feat = feat.merge(ff[["tanggal", "ticker", "foreign_buy", "foreign_sell",
                          "foreign_net_vol", "foreign_net_value"]],
                      on=["ticker", "tanggal"], how="left")
    ffill_cols = ["foreign_buy", "foreign_sell", "foreign_net_vol", "foreign_net_value"]
    feat[ffill_cols] = feat.groupby("ticker")[ffill_cols].ffill()
    vol = feat["volume"].replace(0, np.nan)
    feat["ff_buy_ratio"] = feat["foreign_buy"] / vol
    feat["ff_sell_ratio"] = feat["foreign_sell"] / vol
    feat["ff_net_ratio"] = feat["foreign_net_vol"] / vol

    # target terakhir per saham = NaN
    last_idx = feat.groupby("ticker")["tanggal"].idxmax()
    feat.loc[last_idx, "target"] = np.nan
    return feat


def walkforward(feat, cols, label):
    dates = np.sort(feat["tanggal"].unique())
    test_start = pd.Timestamp(dates[int(len(dates) * 0.80)])
    test = feat[feat["tanggal"] > test_start].copy()
    test = test.dropna(subset=["target"])
    test["bulan"] = test["tanggal"].dt.to_period("M")
    months = sorted(test["bulan"].unique())

    monthly = []
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
    print("FF coverage:", feat["ff_net_ratio"].notna().mean() * 100, "% baris")

    base_cols = FEATURE_COLS + MACRO_COLS + RS_COLS
    ff_cols = base_cols + FF_COLS

    print(f"\n=== WALK-FORWARD BASELINE ({len(base_cols)} fitur) ===")
    m_base = walkforward(feat, base_cols, "baseline")
    print(f"\n=== WALK-FORWARD + FOREIGN FLOW ({len(ff_cols)} fitur) ===")
    m_ff = walkforward(feat, ff_cols, "ff")

    y_base, p_base = agg(m_base)
    y_ff, p_ff = agg(m_ff)
    assert len(y_base) == len(y_ff) and (y_base == y_ff).all()

    print("\n=== PERBANDINGAN (OOS, lipatan sama) ===")
    print(f"{'bulan':10s} {'AUC base':>9s} {'AUC +FF':>9s} {'delta':>7s}")
    for mb, mr in zip(m_base, m_ff):
        d = mr["auc"] - mb["auc"]
        print(f"{mb['bulan']:10s} {mb['auc']:9.4f} {mr['auc']:9.4f} {d:+7.4f}")

    ab, ar = roc_auc_score(y_base, p_base), roc_auc_score(y_ff, p_ff)
    print("-" * 38)
    print(f"{'AGREGAT':10s} {ab:9.4f} {ar:9.4f} {ar-ab:+7.4f}")

    verdict = ("FF MEMBANTU" if ar - ab >= 0.003
               else "FF NETRAL" if ar - ab > -0.003
               else "FF MERUGIKAN")
    print(f"\nKesimpulan: {verdict} (delta {ar-ab:+.4f})")

    with open(os.path.join(BASE, "data", "metrics_ff_experiment.json"), "w") as f:
        json.dump({
            "n": int(len(y_base)),
            "auc_baseline": round(ab, 4),
            "auc_with_ff": round(ar, 4),
            "delta": round(ar - ab, 4),
            "verdict": verdict,
            "ff_cols": FF_COLS,
            "monthly": [{"bulan": mb["bulan"], "base": mb["auc"], "ff": mr["auc"]}
                        for mb, mr in zip(m_base, m_ff)],
        }, f, indent=2)
    print("Hasil: data/metrics_ff_experiment.json")


if __name__ == "__main__":
    main()
