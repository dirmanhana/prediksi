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
import subprocess
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
ARCHIVE = os.path.join(BASE, "data", "prediction_archive.csv")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]
# Fitur relative strength vs sektor (hasil eksperimen: experiment_sector_rs.py)
USE_SECTOR_RS = True
RS_COLS = ["sector_ret_1", "rs_sector_1", "rs_sector_5"]
# Fitur foreign flow (net asing) dari idx.co.id — data/foreign_flow.csv
# HASIL EKSPERIMEN (walk-forward, 205.561 baris OOS): delta AUC -0.0035
# => SEBAGAI FITUR MODEL MERUGIKAN. Data tetap dipakai utk INFO/tampilan bot
# (rekap: Net Asing pasar; cek KODE: asing per saham), bukan fitur prediksi.
USE_FOREIGN_FLOW = False
FF_COLS = ["ff_buy_ratio", "ff_sell_ratio", "ff_net_ratio"]
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
            # tanggal dinormalisasi ke 00:00 (tanggal saja, konsisten antar
            # sumber & dgn CSV lama) — cegah TypeError dan mismatch merge
            return pd.DataFrame({
                "tanggal": pd.to_datetime(ts, unit="s").normalize(),
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
            merged = pd.concat([old, df])
            # normalisasi ulang tanggal (defensif: CSV lama vs fetch baru)
            merged["tanggal"] = pd.to_datetime(merged["tanggal"])
            merged = merged.drop_duplicates(subset=["tanggal"], keep="last")
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
def train_and_predict(df, macro_df, meta, foreign_df=None):
    feat = add_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    if macro_df is not None:
        # merge_asof MUNDUR: kalau fitur makro utk tanggal terakhir belum tersedia
        # (mis. indeks IHSG di Yahoo tertinggal 1 hari), pakai nilai makro hari
        # tersedia SEBELUMNYA. Ini mencegah baris tanggal terakhir gugur karena
        # NaN makro (yang membuat data_sampai & prediksi mundur 1 hari).
        feat = feat.sort_values("tanggal")
        macro_s = macro_df[["tanggal"] + MACRO_COLS].sort_values("tanggal")
        # samakan unit datetime (feat dari parquet = [us], makro bisa [s])
        feat["tanggal"] = feat["tanggal"].astype("datetime64[us]")
        macro_s["tanggal"] = macro_s["tanggal"].astype("datetime64[us]")
        feat = pd.merge_asof(feat, macro_s, on="tanggal", direction="backward")

    if USE_SECTOR_RS:
        # ---- relative strength vs sektor (cross-sectional, tanpa lookahead) ----
        # ret_1/ret_5 saham vs median sesektor pada TANGGAL yang sama (semua
        # nilai diketahui saat penutupan hari tsb).
        sector_map = dict(zip(meta["name"].astype(str), meta["sector"].astype(str)))
        feat["sector"] = feat["ticker"].map(sector_map).fillna("UNKNOWN")
        feat = feat.sort_values("tanggal")
        gs = feat.groupby(["tanggal", "sector"])
        feat["sector_ret_1"] = gs["ret_1"].transform("median")
        feat["rs_sector_1"] = feat["ret_1"] - feat["sector_ret_1"]
        feat["rs_sector_5"] = feat["ret_5"] - gs["ret_5"].transform("median")

    # ---- data foreign flow (net asing) dari idx.co.id ----
    # Exact merge + ffill per saham ≈ asof mundur (nilai foreign utk tanggal T
    # diketahui saat penutupan T — tanpa lookahead). Workaround: merge_asof
    # dgn by= bermasalah di pandas 3.0.
    # Dipakai utk INFO (tampilan bot) SELALU; sebagai FITUR MODEL hanya kalau
    # USE_FOREIGN_FLOW aktif (hasil eksperimen: merugikan -> nonaktif).
    use_ff_data = foreign_df is not None and not foreign_df.empty
    if use_ff_data:
        ff = foreign_df[["tanggal", "ticker", "foreign_buy", "foreign_sell",
                         "foreign_net_vol", "foreign_net_value"]].copy()
        ff["tanggal"] = pd.to_datetime(ff["tanggal"]).astype("datetime64[us]")
        feat["tanggal"] = feat["tanggal"].astype("datetime64[us]")
        feat = feat.sort_values(["ticker", "tanggal"])
        ff = ff.sort_values(["ticker", "tanggal"])
        feat = feat.merge(ff, on=["ticker", "tanggal"], how="left")
        ffill_cols = ["foreign_buy", "foreign_sell", "foreign_net_vol",
                      "foreign_net_value"]
        feat[ffill_cols] = feat.groupby("ticker")[ffill_cols].ffill()
    use_ff_feat = USE_FOREIGN_FLOW and use_ff_data
    if use_ff_feat:
        vol = feat["volume"].replace(0, np.nan)
        feat["ff_buy_ratio"] = feat["foreign_buy"] / vol
        feat["ff_sell_ratio"] = feat["foreign_sell"] / vol
        feat["ff_net_ratio"] = feat["foreign_net_vol"] / vol

    full_cols = FEATURE_COLS + (MACRO_COLS if macro_df is not None else [])
    if USE_SECTOR_RS:
        full_cols = full_cols + RS_COLS
    if use_ff_feat:
        full_cols = full_cols + FF_COLS
    print(f"Fitur: {len(FEATURE_COLS)} teknikal + "
          f"{len(MACRO_COLS) if macro_df is not None else 0} makro + "
          f"{len(RS_COLS) if USE_SECTOR_RS else 0} RS-sektor + "
          f"{len(FF_COLS) if use_ff_feat else 0} foreign = {len(full_cols)}")

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

    # ---- model regresi: perkiraan return besok (info + sizing) ----
    rt = df[["ticker", "tanggal"]].copy()
    rt["ret_tomorrow"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    feat = feat.merge(rt, on=["ticker", "tanggal"], how="left")
    reg_df = feat.dropna(subset=full_cols + ["ret_tomorrow"])
    rdates = np.sort(reg_df["tanggal"].unique())
    rvcut = rdates[int(len(rdates) * 0.85)]
    tr_r = reg_df[reg_df["tanggal"] <= rvcut]
    va_r = reg_df[reg_df["tanggal"] > rvcut]
    reg_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 100,
    }
    reg = xgb.XGBRegressor(**reg_params)
    reg.fit(tr_r[full_cols], tr_r["ret_tomorrow"],
            eval_set=[(va_r[full_cols], va_r["ret_tomorrow"])], verbose=False)
    pred_ret = reg.predict(pred_df[full_cols].astype(float))
    print(f"Regresi return: iterasi={reg.best_iteration} "
          f"(rmse valid={reg.best_score:.5f})")

    cols_out = ["ticker", "tanggal", "close", "volume", "ret_1"]
    if use_ff_data:
        cols_out += ["foreign_buy", "foreign_sell", "foreign_net_vol",
                     "foreign_net_value"]
    out = pred_df[cols_out].copy()
    out["prob_up"] = prob
    out["pred_ret"] = pred_ret
    out["value_traded"] = out["close"] * out["volume"]  # Rp, hari data terakhir
    out = out.merge(meta[["name", "description", "sector", "industry"]],
                    left_on="ticker", right_on="name", how="left")
    out["signal"] = np.where(out["prob_up"] >= best_t, "NAIK ▲", "TURUN ▼")
    out["confidence"] = np.where(out["prob_up"] >= best_t, out["prob_up"], 1 - out["prob_up"])
    out["sizing"] = np.where(
        out["prob_up"] >= best_t,
        np.select([out["prob_up"] >= 0.60, out["prob_up"] >= 0.50],
                  ["BESAR", "SEDANG"], default="KECIL"),
        "—")
    out = out.sort_values("prob_up", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    # simpan baris fitur terakhir per saham (utk fitur "kenapa KODE" di bot)
    feat_last = pred_df[["ticker"] + full_cols].copy()
    feat_last.to_parquet(os.path.join(BASE, "data", "last_features.parquet"), index=False)
    return out, model, full_cols, float(model.best_score), float(best_t)


try:
    import holidays
    _IDX_HOL = holidays.Indonesia(years=list(range(2020, 2031)))
except Exception:
    _IDX_HOL = None  # paket holidays tidak terpasang -> hanya lewati akhir pekan


def next_trading_day(d):
    """Hari perdagangan berikutnya setelah tanggal d.
    Lewati Sabtu/Minggu + hari libur nasional Indonesia (Idul Fitri,
    17 Agustus, dll) jika paket holidays tersedia."""
    nxt = d + pd.Timedelta(days=1)
    while nxt.weekday() >= 5 or (_IDX_HOL is not None and nxt in _IDX_HOL):
        nxt += pd.Timedelta(days=1)
    return nxt


def _atomic_write(df_or_str, path):
    """Tulis file secara atomik (temp + rename) supaya pembaca tidak
    pernah melihat file setengah jadi (penting saat bot membaca bersamaan)."""
    tmp = path + ".tmp"
    if hasattr(df_or_str, "to_csv"):
        df_or_str.to_csv(tmp, index=False)
    else:
        with open(tmp, "w") as f:
            f.write(df_or_str)
    os.replace(tmp, path)


def _append_log(df, path):
    """Append baris log riwayat (buat file baru kalau belum ada)."""
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def notify_admin(subject, detail):
    """Kirim notifikasi error ke nomor admin via API chatetin (best-effort,
    jangan sampai menggagalkan pipeline)."""
    try:
        from wa_bot import load_env, load_admins, ChatetinClient
        env = load_env()
        client = ChatetinClient(env.get("CHATETIN_BASE_URL", "https://wa.chatetin.com"),
                                env.get("CHATETIN_USERNAME", ""),
                                env.get("CHATETIN_PASSWORD", ""))
        client.login()
        for a in load_admins(env):
            client.send_message(a, f"⚠️ *{subject}*\n{detail[:1500]}")
            print(f"Notifikasi dikirim ke admin {a}")
    except Exception as e:
        print(f"⚠️ Gagal kirim notifikasi admin: {e}")


def main():
    # lock anti-bentrok: jangan biarkan predict_daily berjalan ganda
    # (mis. cron & bot --refresh bersamaan)
    try:
        import fcntl
        lock_fd = open(os.path.join(BASE, "data", ".predict_daily.lock"), "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError):
        lock_fd = None  # tanpa lock (Windows / tidak bisa)
    except BlockingIOError:
        sys.exit("predict_daily.py sudah berjalan — tunggu selesai.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="update data historis dulu")
    args = ap.parse_args()

    meta = pd.read_csv(LIST_FILE)
    tickers = sorted(meta["name"].astype(str).tolist())
    print(f"Saham terdaftar: {len(tickers)}")

    if args.refresh:
        print("Refresh data historis (incremental)...")
        refresh_history(tickers)
        # update foreign flow hari terakhir (best-effort, IDX publish setelah tutup)
        if os.path.exists(os.path.join(BASE, "data", "foreign_flow.csv")):
            print("Ambil foreign flow hari terakhir (idx.co.id, best-effort)...")
            try:
                subprocess.run([sys.executable, os.path.join(BASE, "scrape_foreign_flow.py"),
                                "--latest"], cwd=BASE, capture_output=True,
                               text=True, timeout=180)
            except Exception as e:
                print(f"⚠️ Scrape foreign flow gagal (dilanjutkan): {e}")

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

    # foreign flow (net asing) dari idx.co.id — kalau ada file hasil scrape
    foreign_df = None
    ff_path = os.path.join(BASE, "data", "foreign_flow.csv")
    if os.path.exists(ff_path):
        try:
            foreign_df = pd.read_csv(ff_path, parse_dates=["tanggal"])
            print(f"Foreign flow: {len(foreign_df):,} baris, "
                  f"{foreign_df['tanggal'].dt.date.nunique()} hari "
                  f"(s/d {foreign_df['tanggal'].dt.date.max()})")
        except Exception as e:
            print(f"⚠️ Gagal baca foreign_flow.csv: {e}")

    print("\nFeature engineering + training + prediksi...")
    out, model, cols, valid_auc, best_t = train_and_predict(df, macro_df, meta, foreign_df)

    # tulis JSON dulu (atomik), lalu CSV — pembaca (bot) membaca JSON
    has_ff = "foreign_net_value" in out.columns
    rec_cols = ["rank", "ticker", "description", "sector",
                "close", "volume", "value_traded", "ret_1",
                "prob_up", "pred_ret", "sizing",
                "signal", "confidence"]
    if has_ff:
        rec_cols += ["foreign_buy", "foreign_sell", "foreign_net_value"]
    payload = {
        "tanggal_prediksi": next_trading_day(out["tanggal"].max()).strftime("%Y-%m-%d"),
        "data_sampai": out["tanggal"].max().strftime("%Y-%m-%d"),
        "jumlah_saham": int(len(out)),
        "model": "XGBoost gabungan (cross-sectional)",
        "fitur": len(cols),
        "valid_auc": round(valid_auc, 4),
        "threshold_optimal": round(best_t, 2),
        "saham": out[rec_cols].to_dict(orient="records"),
    }
    if has_ff:
        payload["net_foreign"] = {
            "tanggal": out["tanggal"].max().strftime("%Y-%m-%d"),
            "net_value": round(float(out["foreign_net_value"].sum()), 0),
            "net_vol": round(float(out["foreign_net_vol"].sum()), 0),
        }
    _atomic_write(json.dumps(payload, ensure_ascii=False, indent=2), OUT_JSON)
    _atomic_write(out, OUT_CSV)
    print(f"Output ditulis atomik: {OUT_CSV} / {OUT_JSON}")

    # simpan model + metadata
    model.save_model(os.path.join(BASE, "data", "model_daily.ubj"))
    with open(OUT_MODEL, "w") as f:
        json.dump({"feature_cols": cols, "trained_at": datetime.now().isoformat(),
                   "valid_auc": round(valid_auc, 4)}, f, indent=2)

    # ---- arsip prediksi utk verifikasi harian (top-20 naik/turun LIKUID) ----
    liq = out[(out["value_traded"] >= 1_000_000_000) & (out["close"] >= 200)]
    pred_date = next_trading_day(out["tanggal"].max()).strftime("%Y-%m-%d")
    data_until = out["tanggal"].max().strftime("%Y-%m-%d")
    arch = pd.concat([
        liq.head(20)[["ticker", "prob_up"]].assign(arah=1),
        liq.tail(20)[["ticker", "prob_up"]].assign(arah=0),
    ])
    arch.insert(0, "tanggal_prediksi", pred_date)
    arch.insert(1, "data_sampai", data_until)
    if os.path.exists(ARCHIVE):
        old = pd.read_csv(ARCHIVE, dtype={"ticker": str})
        arch = pd.concat([old, arch]).drop_duplicates(
            subset=["tanggal_prediksi", "ticker"], keep="last")
    _atomic_write(arch, ARCHIVE)
    print(f"Arsip prediksi: {len(arch):,} baris -> {ARCHIVE}")

    # log riwayat
    log = pd.DataFrame([{
        "tanggal_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_sampai": out["tanggal"].max().strftime("%Y-%m-%d"),
        "n_saham": len(out), "valid_auc": round(valid_auc, 4),
        "threshold": round(best_t, 2),
        "n_naik": int((out["prob_up"] >= best_t).sum()),
    }])
    if os.path.exists(LOG_CSV):
        _append_log(log, LOG_CSV)
    else:
        _append_log(log, LOG_CSV)

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
    try:
        main()
    except SystemExit:
        raise  # lock/argumen — bukan error pipeline
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err, flush=True)
        try:
            notify_admin("predict_daily.py GAGAL",
                         f"{type(e).__name__}: {e}\n{err[-900:]}")
        except Exception:
            pass
        sys.exit(1)
