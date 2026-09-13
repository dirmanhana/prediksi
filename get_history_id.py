"""
get_history_id.py
=================
Mengambil data historis 5 tahun (daily OHLCV) untuk SEMUA saham IDX
dari Yahoo Finance (ticker format: BBRI.JK), siap untuk modeling (XGBoost).

Fitur:
  - Resume/checkpoint: setiap saham disimpan langsung ke data/history/{kode}.csv
  - Rate-limit handling: sleep adaptif + retry dengan backoff
  - Merge semua saham menjadi satu dataset besar (CSV + Parquet)

Cara pakai:
  python get_history_id.py [tahun] [kode1 kode2 ...]
  contoh:
    python get_history_id.py 5            # 5 tahun, semua saham
    python get_history_id.py 5 BBRI TLKM  # hanya saham tertentu

Output:
  - data/history/{kode}.csv        (per saham)
  - data/history_id_5y.csv         (gabungan, CSV)
  - data/history_id_5y.parquet     (gabungan, Parquet - compact & cepat)
"""

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from waktu import WIB, now_wib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIST_FILE = os.path.join(BASE_DIR, "data", "stocks_id.csv")
HIST_DIR = os.path.join(BASE_DIR, "data", "history")

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def get_session_with_crumb():
    """Buat session Yahoo + ambil crumb (diperlukan untuk chart API)."""
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://fc.yahoo.com", timeout=30)
    crumb = s.get(
        "https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=30
    ).text.strip()
    if not crumb:
        raise RuntimeError("Gagal mendapatkan crumb Yahoo Finance")
    return s, crumb


def fetch_history(s, crumb, symbol, years):
    """Ambil daily OHLCV `years` tahun untuk satu ticker. Return DataFrame atau None."""
    t2 = int(time.time())
    t1 = int(time.time()) - int(years * 365.25 * 24 * 3600)

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.JK"
    params = {
        "period1": t1,
        "period2": t2,
        "interval": "1d",
        "events": "history",
        "crumb": crumb,
    }

    for attempt in range(6):
        try:
            r = s.get(url, params=params, timeout=30)

            if r.status_code == 429:  # rate limited
                wait = 10 * (attempt + 1)
                print(f"    [429] rate limit, tunggu {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401:  # crumb expired
                s, crumb = get_session_with_crumb()
                params["crumb"] = crumb
                time.sleep(2)
                continue

            if r.status_code == 404:  # tidak ada di Yahoo
                return None

            r.raise_for_status()
            res = r.json().get("chart", {}).get("result")
            if not res:
                return None

            ts = res[0].get("timestamp") or []
            if not ts:
                return None

            q = res[0]["indicators"]["quote"][0]
            adj = res[0]["indicators"].get("adjclose", [{}])[0].get("adjclose")

            df = pd.DataFrame(
                {
                    "tanggal": [datetime.fromtimestamp(x, WIB).date() for x in ts],
                    "open": q.get("open"),
                    "high": q.get("high"),
                    "low": q.get("low"),
                    "close": q.get("close"),
                    "volume": q.get("volume"),
                    "adj_close": adj if adj else q.get("close"),
                }
            )
            df = df.dropna(subset=["close"]).reset_index(drop=True)
            return df

        except requests.exceptions.RequestException as e:
            print(f"    [err] {e} (coba {attempt + 1}/6)")
            time.sleep(5 * (attempt + 1))

    return None


def load_tickers(years, only=None):
    """Ambil daftar kode saham dari data/stocks_id.csv."""
    df = pd.read_csv(LIST_FILE)
    tickers = sorted(df["name"].astype(str).tolist())
    if only:
        tickers = [t for t in tickers if t in only]
    return tickers


def main():
    years = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    only = set(sys.argv[2:]) if len(sys.argv) > 2 else None

    os.makedirs(HIST_DIR, exist_ok=True)
    tickers = load_tickers(years, only)
    print(f"Jumlah saham yang akan diambil: {len(tickers)}")
    print(f"Periode: {years} tahun ke belakang ({now_wib().year - int(years)} - sekarang)")

    s, crumb = get_session_with_crumb()
    ok, failed, skipped = [], [], []

    for i, tk in enumerate(tickers, 1):
        path = os.path.join(HIST_DIR, f"{tk}.csv")

        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped.append(tk)
            print(f"[{i}/{len(tickers)}] {tk} - sudah ada, dilewati")
            continue

        df = fetch_history(s, crumb, tk, years)

        if df is None or df.empty:
            failed.append(tk)
            print(f"[{i}/{len(tickers)}] {tk} - GAGAL/tidak ditemukan")
        else:
            df.insert(0, "ticker", tk)
            df.to_csv(path, index=False)
            ok.append(tk)
            print(
                f"[{i}/{len(tickers)}] {tk} - OK "
                f"({len(df)} baris, {df['tanggal'].iloc[0]} s/d {df['tanggal'].iloc[-1]})"
            )

        # sleep adaptif: lebih cepat di awal, jeda ekstra tiap 50 saham
        time.sleep(0.6 if i % 50 else 3.0)

    print(f"\n=== RINGKASAN ===")
    print(f"Berhasil : {len(ok)}")
    print(f"Gagal    : {len(failed)}  {failed[:20]}")
    print(f"Skip     : {len(skipped)}")

    # gabungkan semua menjadi satu dataset
    if ok or skipped:
        merge_and_save(ok + skipped, years)


def merge_and_save(tickers, years):
    """Gabungkan semua file per saham menjadi dataset tunggal."""
    print("\nMenggabungkan semua saham menjadi satu dataset...")
    frames = []
    for tk in tickers:
        path = os.path.join(HIST_DIR, f"{tk}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, parse_dates=["tanggal"]))

    if not frames:
        print("Tidak ada data untuk digabung.")
        return

    big = pd.concat(frames, ignore_index=True)

    # kolom turunan untuk modeling
    big = big.sort_values(["ticker", "tanggal"]).reset_index(drop=True)
    big["ticker"] = big["ticker"].astype(str)
    g = big.groupby("ticker")["close"]
    big["return_1d"] = g.pct_change()
    big["log_return"] = np.log(big["close"] / big.groupby("ticker")["close"].shift(1))

    csv_path = os.path.join(BASE_DIR, "data", f"history_id_{int(years)}y.csv")
    pq_path = os.path.join(BASE_DIR, "data", f"history_id_{int(years)}y.parquet")

    big.to_csv(csv_path, index=False, encoding="utf-8-sig")
    big.to_parquet(pq_path, index=False)

    print(f"Dataset gabungan: {len(big):,} baris x {big.shape[1]} kolom, "
          f"{big['ticker'].nunique()} saham")
    print(f"  - {csv_path}")
    print(f"  - {pq_path}")
    print(f"Rentang tanggal: {big['tanggal'].min().date()} s/d {big['tanggal'].max().date()}")
    print(f"Ukuran parquet: {os.path.getsize(pq_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
