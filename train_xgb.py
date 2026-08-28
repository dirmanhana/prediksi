"""
train_xgb.py
============
Model XGBoost untuk memprediksi ARAH PERGERAKAN BESOK (naik/turun)
berdasarkan data historis 5 tahun semua saham IDX.

Pipeline:
  1. Load data historis (data/history_id_5y.parquet)
  2. Feature engineering per saham (indikator teknikal, tanpa lookahead)
  3. Simpan fitur ke database SQLite (data/features.db)
  4. Target : 1 jika close besok > close hari ini, selain itu 0
  5. Split BERBASIS WAKTU (train -> valid -> test) agar tidak bocor ke masa depan
  6. Train XGBoost dengan early stopping
  7. Evaluasi & simpan model

Cara pakai:
  python train_xgb.py
"""

import json
import os
import sqlite3

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PQ = os.path.join(BASE, "data", "history_id_5y.parquet")
DB_PATH = os.path.join(BASE, "data", "features.db")
OUT_MODEL = os.path.join(BASE, "data", "model_xgb_direction.json")
OUT_METRICS = os.path.join(BASE, "data", "metrics.json")
OUT_FI = os.path.join(BASE, "data", "feature_importance.csv")
OUT_FI_PNG = os.path.join(BASE, "data", "feature_importance.png")

RANDOM_STATE = 42

FEATURE_COLS = [
    # return / momentum
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "log_ret_1", "log_ret_5",
    # moving average ratio (harga relatif thd MA)
    "close_sma5", "close_sma10", "close_sma20", "close_sma50", "close_sma100",
    "sma20_sma50",
    "close_ema12", "close_ema26",
    # MACD
    "macd_c", "macd_signal_c", "macd_hist_c",
    # RSI & Bollinger
    "rsi_14", "bb_pct_b", "bb_width",
    # volatilitas & ATR
    "vol_5", "vol_20", "atr_ratio",
    # volume
    "vol_ratio_5", "vol_ratio_20", "vol_change",
    # harga intraday & gap
    "hl_range", "gap",
    # posisi harga 52 minggu
    "dist_52w_high", "dist_52w_low",
    # kalender
    "dayofweek", "month",
]


# ---------------------------------------------------------------- data
def load_data():
    df = pd.read_parquet(DATA_PQ)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    # bersihkan harga tidak valid
    df = df[(df["close"] > 0) & (df["open"] > 0)].copy()
    df = df.sort_values(["ticker", "tanggal"]).reset_index(drop=True)
    return df


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)  # tidak ada loss -> RSI = 100
    return out


# ---------------------------------------------------------------- fitur
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung fitur per saham. Semua fitur hanya memakai data s/d hari ini (no lookahead)."""
    df = df.copy()
    g = df.groupby("ticker", sort=False)

    close = df["close"]
    high, low = df["high"], df["low"]
    volume = df["volume"].fillna(0)

    # --- return & momentum (lag) ---
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}"] = g["close"].pct_change(n)
    df["log_ret_1"] = np.log(close / g["close"].shift(1))
    df["log_ret_5"] = np.log(close / g["close"].shift(5))

    # --- moving averages ---
    for w in [5, 10, 20, 50, 100]:
        df[f"close_sma{w}"] = close / close.groupby(df["ticker"]).transform(
            lambda x: x.rolling(w, min_periods=max(5, w // 5)).mean()
        )
    sma20 = g["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    sma50 = g["close"].transform(lambda x: x.rolling(50, min_periods=20).mean())
    df["sma20_sma50"] = sma20 / sma50

    # --- EMA & MACD ---
    ema12 = close.groupby(df["ticker"]).transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = close.groupby(df["ticker"]).transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["close_ema12"] = close / ema12
    df["close_ema26"] = close / ema26
    macd = ema12 - ema26
    macd_signal = macd.groupby(df["ticker"]).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_c"] = macd / close
    df["macd_signal_c"] = macd_signal / close
    df["macd_hist_c"] = (macd - macd_signal) / close

    # --- RSI & Bollinger ---
    df["rsi_14"] = close.groupby(df["ticker"]).transform(rsi)
    sma20b = g["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    std20 = g["close"].transform(lambda x: x.rolling(20, min_periods=10).std())
    upper, lower = sma20b + 2 * std20, sma20b - 2 * std20
    df["bb_pct_b"] = (close - lower) / (upper - lower).replace(0, np.nan)
    df["bb_width"] = (upper - lower) / sma20b

    # --- volatilitas & ATR ---
    ret = g["close"].pct_change()
    df["vol_5"] = ret.groupby(df["ticker"]).transform(lambda x: x.rolling(5, min_periods=3).std())
    df["vol_20"] = ret.groupby(df["ticker"]).transform(lambda x: x.rolling(20, min_periods=10).std())
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.groupby(df["ticker"]).transform(lambda x: x.ewm(alpha=1 / 14, adjust=False).mean())
    df["atr_ratio"] = atr / close

    # --- volume ---
    vma5 = volume.groupby(df["ticker"]).transform(lambda x: x.rolling(5, min_periods=3).mean())
    vma20 = volume.groupby(df["ticker"]).transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["vol_ratio_5"] = volume / vma5.replace(0, np.nan)
    df["vol_ratio_20"] = volume / vma20.replace(0, np.nan)
    df["vol_change"] = g["volume"].pct_change()

    # --- intraday & gap ---
    df["hl_range"] = (high - low) / close
    df["gap"] = (df["open"] - prev_close) / prev_close

    # --- posisi 52 minggu ---
    roll_max = g["close"].transform(lambda x: x.rolling(252, min_periods=60).max())
    roll_min = g["close"].transform(lambda x: x.rolling(252, min_periods=60).min())
    df["dist_52w_high"] = close / roll_max - 1
    df["dist_52w_low"] = close / roll_min - 1

    # --- kalender ---
    df["dayofweek"] = df["tanggal"].dt.dayofweek
    df["month"] = df["tanggal"].dt.month

    # --- TARGET: arah pergerakan besok ---
    # 1 jika close besok > close hari ini, 0 jika turun / flat
    df["target"] = (g["close"].shift(-1) > close).astype(int)

    return df


# ---------------------------------------------------------------- database
def save_to_sqlite(df: pd.DataFrame):
    """Simpan fitur ke SQLite (data/features.db)."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON features(ticker, tanggal)")
    conn.commit()
    conn.close()
    print(f"Fitur disimpan ke SQLite: {DB_PATH}")


def load_from_sqlite() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features ORDER BY ticker, tanggal", conn)
    conn.close()
    return df


# ---------------------------------------------------------------- split
def time_split(df: pd.DataFrame):
    """Split berbasis waktu: 68% train / 12% valid / 20% test berdasarkan tanggal."""
    dates = np.sort(df["tanggal"].unique())
    n = len(dates)
    cut_train = dates[int(n * 0.68)]
    cut_valid = dates[int(n * 0.80)]
    train = df[df["tanggal"] <= cut_train]
    valid = df[(df["tanggal"] > cut_train) & (df["tanggal"] <= cut_valid)]
    test = df[df["tanggal"] > cut_valid]
    print(f"Train : {train['tanggal'].min().date()} s/d {train['tanggal'].max().date()}  ({len(train):,} baris)")
    print(f"Valid : {valid['tanggal'].min().date()} s/d {valid['tanggal'].max().date()}  ({len(valid):,} baris)")
    print(f"Test  : {test['tanggal'].min().date()} s/d {test['tanggal'].max().date()}  ({len(test):,} baris)")
    return train, valid, test


# ---------------------------------------------------------------- evaluasi
def best_threshold(y_true, y_prob):
    """Cari threshold probabilitas optimal (max F1) pada data validasi."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def evaluate(y_true, y_prob, label="test", threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    base_rate = y_true.mean()
    baseline_acc = max(base_rate, 1 - base_rate)  # selalu prediksi kelas mayoritas
    metrics = {
        "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "n": int(len(y_true)),
        "base_rate_naik": round(base_rate, 4),
        "baseline_accuracy": round(baseline_acc, 4),
    }
    print(f"\n--- Evaluasi {label} (threshold={threshold:.2f}) ---")
    print(f"  Akurasi model : {metrics['accuracy']:.4f}   (baseline mayoritas: {baseline_acc:.4f})")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1            : {metrics['f1']:.4f}")
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (prediksi \\ aktual):")
    print(f"    [[TN={cm[0,0]:6,}  FP={cm[0,1]:6,}]")
    print(f"     [FN={cm[1,0]:6,}  TP={cm[1,1]:6,}]]")
    return metrics


# ---------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("STEP 1: Load & feature engineering")
    print("=" * 60)
    df = load_data()
    print(f"Data mentah: {len(df):,} baris, {df['ticker'].nunique()} saham")

    df = add_features(df)

    # ganti inf (mis. volume=0) dengan NaN, lalu buang baris NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    print(f"Setelah feature engineering & drop NaN: {len(df):,} baris, "
          f"{df['ticker'].nunique()} saham")
    print(f"Distribusi target (naik besok): {df['target'].mean():.4f}")

    print("\n" + "=" * 60)
    print("STEP 2: Simpan fitur ke database SQLite")
    print("=" * 60)
    save_to_sqlite(df)

    print("\n" + "=" * 60)
    print("STEP 3: Split berbasis waktu & training XGBoost")
    print("=" * 60)
    train, valid, test = time_split(df)

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_valid, y_valid = valid[FEATURE_COLS], valid["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

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
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"Early stopping di iterasi ke-{best_iter}")
    print(f"  valid logloss: {model.best_score:.5f}")

    print("\n" + "=" * 60)
    print("STEP 4: Evaluasi")
    print("=" * 60)
    y_prob_valid = model.predict_proba(X_valid)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]

    # tuning threshold di validasi, lalu evaluasi test dengan threshold tsb
    th = best_threshold(y_valid, y_prob_valid)
    print(f"\nThreshold optimal dari validasi (max F1): {th:.2f}")

    metrics_test = evaluate(y_test, y_prob_test, label="TEST (unseen)", threshold=th)
    metrics_test_50 = evaluate(y_test, y_prob_test, label="TEST (threshold 0.50)", threshold=0.5)
    metrics_train = evaluate(y_train, y_prob_train, label="TRAIN", threshold=th)

    print("\n" + "=" * 60)
    print("STEP 5: Feature importance")
    print("=" * 60)
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp["rank"] = range(1, len(imp) + 1)
    print(imp.head(20).to_string(index=False))
    imp.to_csv(OUT_FI, index=False)

    # plot top 20
    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(9, 8))
    plt.barh(top["feature"], top["gain"], color="#1f77b4")
    plt.xlabel("Gain (feature importance)")
    plt.title("Top 20 Fitur - XGBoost Prediksi Arah Besok")
    plt.tight_layout()
    plt.savefig(OUT_FI_PNG, dpi=130)
    print(f"Plot tersimpan: {OUT_FI_PNG}")

    print("\n" + "=" * 60)
    print("STEP 6: Simpan model & metrik")
    print("=" * 60)
    model.save_model(OUT_MODEL)
    joblib.dump({"feature_cols": FEATURE_COLS}, os.path.join(BASE, "data", "model_meta.pkl"))
    with open(OUT_METRICS, "w") as f:
        json.dump({
            "train": metrics_train,
            "test": metrics_test,
            "test_threshold_50": metrics_test_50,
            "optimal_threshold": th,
            "best_iteration": best_iter,
            "params": {k: v for k, v in params.items() if k != "eval_metric"},
        }, f, indent=2)
    print(f"Model   : {OUT_MODEL}")
    print(f"Metrik  : {OUT_METRICS}")
    print(f"SQLite  : {DB_PATH}")

    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO: Prediksi arah besok untuk beberapa saham likuid")
    print("=" * 60)
    demo_tickers = ["BBRI", "TLKM", "BBCA", "BMRI", "BREN"]
    latest = df.groupby("ticker").tail(1).set_index("ticker")
    for tk in demo_tickers:
        if tk in latest.index:
            row = latest.loc[tk]
            X_demo = row[FEATURE_COLS].astype(float).to_frame().T
            prob = model.predict_proba(X_demo)[0, 1]
            arrow = "NAIK ▲" if prob >= th else "TURUN ▼"
            print(f"  {tk:5s} ({row['tanggal'].date()})  prob naik besok = {prob:.3f}  -> {arrow}")


if __name__ == "__main__":
    main()
