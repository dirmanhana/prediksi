"""
predict_daily.py
================
PIPELINE PREDIKSI HARIAN (produksi)
Model terbaik dari semua eksperimen: XGBoost gabungan (cross-sectional, semua saham),
dengan fitur teknikal + makro. Retrain otomatis setiap dijalankan.

Alur:
  1. (opsional) refresh data historis dari Yahoo (incremental, hanya hari yang kurang)
  2. Bangun ulang dataset gabungan dari data/history/*.csv
  3. Feature engineering (34 teknikal + 8 makro)
  4. Train XGBoost gabungan (validasi = 15% tanggal terakhir, early stopping)
  5. Prediksi arah BESOK untuk SEMUA saham
  6. Output ranking + simpan file

Cara pakai:
  python predict_daily.py            # pakai data yang sudah ada
  python predict_daily.py --refresh  # update data dulu (incremental, ~15 menit)

Output:
  - data/predictions_tomorrow.csv / .json
  - data/model_daily.json
  - data/prediction_log.csv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from train_xgb import FEATURE_COLS, add_features  # reuse feature engineering

BASE = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE, "data", "history")
PQ_PATH = os.path.join(BASE, "data", "history_id_5y.parquet")
LIST_FILE = os.path.join(BASE, "data", "stocks_id.csv")
MACRO_CSV = os.path.join(BASE, "data", "macro_id.csv")
OUT_CSV = os.path.join(BASE, "data", "predictions_tomorrow.csv")
OUT_JSON = os.path.join(BASE, "data", "predictions_tomorrow.json")
OUT_MODEL = os.path.join(BASE, "data", "model_daily.json")
LOG_CSV = os.path.join(BASE, "data", "prediction_log.csv")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
YEARS = 5
RANDOM_STATE = 42

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


# ---------------------------------------------------------------- yahoo
def yahoo_session():
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://fc.yahoo.com", timeout=30)
    crumb = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=30).text.strip()
    return s, crumb


def fetch_range(s, crumb, symbol, t1, t2, is_macro=False):
    url_symbol = symbol if is_macro else f"{symbol}.JK"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{url_symbol}"
    for attempt in range(4):
        try:
            r = s.get(url, params={"period1": t1, "period2": t2, "interval": "1d",
                                   "events": "history", "crumb": crumb}, timeout=30)
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            if r.status_code == 401:
                s, crumb = yahoo_session()
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            res = r.json().get("chart", {}).get("result")
            if not res or not res[0].get("timestamp"):
                return None
            ts = res[0]["timestamp"]
            q = res[0]["indicators"]["quote"][0]
            adj = res[0]["indicators"].get("adjclose", [{}])[0].get("adjclose")
            return pd.DataFrame({
                "tanggal": [datetime.fromtimestamp(x).date() for x in ts],
                "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                "close": q.get("close"), "volume": q.get("volume"),
                "adj_close": adj if adj else q.get("close"),
            }).dropna(subset=["close"])
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None


# ---------------------------------------------------------------- data
def refresh_history(tickers):
    """Update incremental: fetch hanya hari yang belum ada utk tiap saham."""
    s, crumb = yahoo_session()
    os.makedirs(HIST_DIR, exist_ok=True)
    now = int(time.time())
    ok = 0
    for i, tk in enumerate(tickers, 1):
        path = os.path.join(HIST_DIR, f"{tk}.csv")
        if os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["tanggal"])
            last = old["tanggal"].max()
            t1 = int(last.timestamp()) - 5 * 86400  # 5 hari buffer (adjustment)
        else:
            old = None
            t1 = now - int(YEARS * 365.25 * 86400)

        df = fetch_range(s, crumb, tk, t1, now)
        if df is None or df.empty:
            if old is None:
                print(f"  [{i}/{len(tickers)}] {tk} - tidak ditemukan")
            continue
        df.insert(0, "ticker", tk)
        if old is not None and not old.empty:
            old = old.drop(columns=["ticker"]) if "ticker" in old.columns else old
            merged = pd.concat([old, df]).drop_duplicates(subset=["tanggal"], keep="last")
            merged = merged.sort_values("tanggal").reset_index(drop=True)
        else:
            merged = df
        merged.to_csv(path, index=False)
        ok += 1
        if i % 100 == 0:
            print(f"  ...refresh {i}/{len(tickers)}")
        time.sleep(0.4)
    print(f"Refresh selesai: {ok}/{len(tickers)} saham diperbarui.")


def rebuild_parquet(tickers):
    """Gabung semua file per saham jadi dataset tunggal (parquet)."""
    frames = []
    for tk in tickers:
        p = os.path.join(HIST_DIR, f"{tk}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            frames.append(pd.read_csv(p, parse_dates=["tanggal"]))
    if not frames:
        sys.exit("Tidak ada data di data/history/. Jalankan: python get_history_id.py 5")
    big = pd.concat(frames, ignore_index=True)
    big["ticker"] = big["ticker"].astype(str)
    big["return_1d"] = big.groupby("ticker")["close"].pct_change()
    big["log_return"] = np.log(big["close"] / big.groupby("ticker")["close"].shift(1))
    big = big.sort_values(["ticker", "tanggal"]).reset_index(drop=True)
    big.to_parquet(PQ_PATH, index=False)
    return big


def load_macro():
    """Ambil fitur makro: pakai cache kalau fresh, else fetch dari Yahoo."""
    try:
        s, crumb = yahoo_session()
        now = int(time.time())
        t1 = now - int(YEARS * 365.25 * 86400)
        ihsg = fetch_range(s, crumb, "^JKSE", t1, now, is_macro=True)
        usd = fetch_range(s, crumb, "IDR=X", t1, now, is_macro=True)
        if ihsg is None or usd is None:
            raise RuntimeError("gagal fetch makro")
        macro = pd.DataFrame({
            "tanggal": ihsg["tanggal"], "ihsg": ihsg["close"],
        }).merge(
            pd.DataFrame({"tanggal": usd["tanggal"], "usd_idr": usd["close"]}),
            on="tanggal", how="inner",
        ).sort_values("tanggal")
        macro["tanggal"] = pd.to_datetime(macro["tanggal"])
        macro["ihsg_ret_1"] = macro["ihsg"].pct_change()
        macro["ihsg_ret_5"] = macro["ihsg"].pct_change(5)
        macro["ihsg_ret_20"] = macro["ihsg"].pct_change(20)
        macro["ihsg_sma20"] = macro["ihsg"] / macro["ihsg"].rolling(20, min_periods=10).mean()
        macro["usd_ret_1"] = macro["usd_idr"].pct_change()
        macro["usd_ret_5"] = macro["usd_idr"].pct_change(5)
        macro["usd_ret_20"] = macro["usd_idr"].pct_change(20)
        macro["usd_sma20"] = macro["usd_idr"] / macro["usd_idr"].rolling(20, min_periods=10).mean()
        macro = macro.replace([np.inf, -np.inf], np.nan)
        macro.to_csv(MACRO_CSV, index=False)
        print(f"Makro di-fetch dari Yahoo ({len(macro):,} hari)")
        return macro
    except Exception as e:
        print(f"Fetch makro gagal ({e}); coba cache {MACRO_CSV}...")
        if os.path.exists(MACRO_CSV):
            return pd.read_csv(MACRO_CSV, parse_dates=["tanggal"])
        return None


# ---------------------------------------------------------------- train & predict
def train_and_predict(df, macro_df, meta):
    full_cols = FEATURE_COLS + (MACRO_COLS if macro_df is not None else [])
    print(f"Fitur: {len(FEATURE_COLS)} teknikal + "
          f"{len(MACRO_COLS) if macro_df is not None else 0} makro = {len(full_cols)}")

    feat = add_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    if macro_df is not None:
        feat = feat.merge(macro_df[["tanggal"] + MACRO_COLS], on="tanggal", how="left")

    # target terakhir per saham = NaN (belum ada "besok") -> tidak dipakai training
    last_idx = feat.groupby("ticker")["tanggal"].idxmax()
    feat.loc[last_idx, "target"] = np.nan

    # buang baris fitur NaN (awal riwayat) & target NaN (untuk training)
    train_df = feat.dropna(subset=full_cols + ["target"])
    # baris terakhir per saham utk prediksi besok
    pred_df = feat.dropna(subset=full_cols).groupby("ticker").tail(1)

    print(f"Training : {len(train_df):,} baris | Prediksi: {len(pred_df):,} saham")

    # split kronologis utk early stopping
    dates = np.sort(train_df["tanggal"].unique())
    vcut = dates[int(len(dates) * 0.85)]
    tr = train_df[train_df["tanggal"] <= vcut]
    va = train_df[train_df["tanggal"] > vcut]

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
    t0 = time.time()
    model.fit(tr[full_cols], tr["target"],
              eval_set=[(va[full_cols], va["target"])], verbose=False)
    print(f"Training selesai ({time.time() - t0:.0f}s), "
          f"iterasi={model.best_iteration}, valid_auc={model.best_score:.4f}")

    X_pred = pred_df[full_cols].astype(float)
    prob = model.predict_proba(X_pred)[:, 1]

    # tuning threshold optimal (max F1) di data validasi
    vprob = model.predict_proba(va[full_cols].astype(float))[:, 1]
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(va["target"], (vprob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    print(f"Threshold optimal (max F1): {best_t:.2f} (F1={best_f1:.4f})")

    out = pred_df[["ticker", "tanggal", "close"]].copy()
    out["prob_up"] = prob
    out = out.merge(meta[["name", "description", "sector", "industry"]],
                    left_on="ticker", right_on="name", how="left")
    out["signal"] = np.where(out["prob_up"] >= best_t, "NAIK ▲", "TURUN ▼")
    out["confidence"] = np.where(out["prob_up"] >= best_t, out["prob_up"], 1 - out["prob_up"])
    out = out.sort_values("prob_up", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out, model, full_cols, float(model.best_score), float(best_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="update data historis dulu")
    args = ap.parse_args()

    meta = pd.read_csv(LIST_FILE)
    tickers = sorted(meta["name"].astype(str).tolist())
    print(f"Saham terdaftar: {len(tickers)}")

    if args.refresh:
        print("Refresh data historis (incremental)...")
        refresh_history(tickers)

    # cek kefresh-an data
    last = None
    for tk in tickers:
        p = os.path.join(HIST_DIR, f"{tk}.csv")
        if os.path.exists(p):
            last = pd.read_csv(p, parse_dates=["tanggal"])["tanggal"].max()
            break
    if last is not None and last.date() < datetime.now().date() - timedelta(days=4):
        print(f"⚠  Data terakhir: {last.date()} (mungkin stale). "
              f"Gunakan --refresh untuk update.")

    print("Membangun ulang dataset gabungan...")
    df = rebuild_parquet(tickers)
    df = df[(df["close"] > 0) & (df["open"] > 0)].copy()
    print(f"Dataset: {len(df):,} baris, {df['ticker'].nunique()} saham, "
          f"{df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")

    macro_df = load_macro()

    print("\nFeature engineering + training + prediksi...")
    out, model, cols, valid_auc, best_t = train_and_predict(df, macro_df, meta)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "tanggal_prediksi": (out["tanggal"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "data_sampai": out["tanggal"].max().strftime("%Y-%m-%d"),
        "jumlah_saham": int(len(out)),
        "model": "XGBoost gabungan (cross-sectional)",
        "fitur": len(cols),
        "valid_auc": round(valid_auc, 4),
        "threshold_optimal": round(best_t, 2),
        "saham": out[["rank", "ticker", "description", "sector",
                      "close", "prob_up", "signal", "confidence"]].to_dict(orient="records"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # simpan model + metadata
    model.save_model(os.path.join(BASE, "data", "model_daily.ubj"))
    with open(OUT_MODEL, "w") as f:
        json.dump({"feature_cols": cols, "trained_at": datetime.now().isoformat(),
                   "valid_auc": round(valid_auc, 4)}, f, indent=2)

    # log riwayat
    log = pd.DataFrame([{
        "tanggal_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_sampai": out["tanggal"].max().strftime("%Y-%m-%d"),
        "n_saham": len(out), "valid_auc": round(valid_auc, 4),
        "threshold": round(best_t, 2),
        "n_naik": int((out["prob_up"] >= best_t).sum()),
    }])
    if os.path.exists(LOG_CSV):
        log.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        log.to_csv(LOG_CSV, index=False)

    # ---------------------------------------------------------- tampilkan
    print("\n" + "=" * 72)
    print("REKOMENDASI BESOK — 15 SAHAM PALING BULLISH")
    print("=" * 72)
    top = out.head(15)
    for _, r in top.iterrows():
        print(f"  {r['rank']:>3d}. {r['ticker']:6s} {r['prob_up']:6.3f}  "
              f"{str(r['description'])[:45]:45s} {r['sector']}")
    print("\n" + "=" * 72)
    print("15 SAHAM PALING BEARISH")
    print("=" * 72)
    for _, r in out.tail(15).iloc[::-1].iterrows():
        print(f"  {r['rank']:>3d}. {r['ticker']:6s} {r['prob_up']:6.3f}  "
              f"{str(r['description'])[:45]:45s} {r['sector']}")

    naik = int((out["prob_up"] >= best_t).sum())
    print("\n" + "=" * 72)
    print(f"RINGKASAN: {naik}/{len(out)} saham diprediksi NAIK besok "
          f"(prob >= {best_t:.2f}) | valid_auc={valid_auc:.4f}")
    print(f"File: {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
