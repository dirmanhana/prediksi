#!/usr/bin/env python3
"""
scrape_foreign_flow.py
======================
Ambil data FOREIGN FLOW (net asing) per saham dari sumber RESMI idx.co.id.

Bagaimana:
  1. Cloudflare memblokir requests biasa (403). Headless Chrome (google-chrome)
     bisa menembus challenge-nya.
  2. Setelah halaman IDX terbuka, panggil API resmi
     /primary/TradingSummary/GetStockSummary?date=YYYY-MM-DD via fetch() DI DALAM
     halaman (sesi Cloudflare yang sama) → JSON berisi ForeignBuy/ForeignSell
     per saham (dalam lembar saham).

Output: data/foreign_flow.csv  (tanggal, ticker, close, volume, value,
        foreign_buy, foreign_sell, foreign_net_vol, foreign_net_value)

Cara pakai:
  python scrape_foreign_flow.py            # backfill semua hari perdagangan dari history
  python scrape_foreign_flow.py --latest   # hanya hari perdagangan terakhir (incremental)
  python scrape_foreign_flow.py --days 30  # backfill 30 hari ke belakang saja

Catatan:
  - Butuh google-chrome terinstall (sudah ada di VPS).
  - Resume-able: tanggal yang sudah ada di CSV dilewati.
  - Data per hari ~963 saham; 5 tahun ≈ 1.200 hari ≈ ±25-40 menit (sekali saja).
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE, "data", "history")
OUT_CSV = os.path.join(BASE, "data", "foreign_flow.csv")

CHROME = "/usr/bin/google-chrome"
PAGE_URL = "https://www.idx.co.id/id/data-pasar"
API_URL = "/primary/TradingSummary/GetStockSummary"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SLEEP = 0.4          # jeda antar request (hindari rate-limit)
BATCH = 40           # simpan CSV tiap N tanggal (atomic, resume-able)


def trading_dates(limit=None):
    """Semua tanggal perdagangan dari data history (gabungan semua saham)."""
    frames = []
    import glob
    for f in sorted(glob.glob(os.path.join(HIST_DIR, "*.csv")))[:200]:
        try:
            frames.append(pd.read_csv(f, usecols=["tanggal"], parse_dates=["tanggal"]))
        except Exception:
            pass
    if not frames:
        sys.exit("Tidak ada data history — jalankan get_history_id.py dulu.")
    dates = pd.concat(frames, ignore_index=True)["tanggal"].dt.date.unique()
    dates = sorted(set(dates))
    if limit:
        dates = dates[-limit:]
    return [d.isoformat() for d in dates]


def load_done():
    """Tanggal yang sudah berhasil diambil (dari CSV)."""
    if not os.path.exists(OUT_CSV):
        return set()
    try:
        df = pd.read_csv(OUT_CSV, usecols=["tanggal"])
        return set(df["tanggal"].unique())
    except Exception:
        return set()


def fetch_day(drv, date_str):
    """Panggil API IDX utk satu tanggal. Return DataFrame atau None."""
    js = (f"return fetch('{API_URL}?date={date_str}&length=5000')"
          f".then(r=>r.json())")
    try:
        d = drv.execute_script(js)
    except Exception:
        return None
    if not isinstance(d, dict) or not d.get("data"):
        return None
    rows = d["data"]
    if not rows:
        return None
    # verifikasi tanggal respon == tanggal diminta (hindari data salah tanggal)
    resp_date = str(rows[0].get("Date", ""))[:10]
    if resp_date != date_str:
        return None
    df = pd.DataFrame(rows)
    df = df.rename(columns={"StockCode": "ticker", "Date": "tanggal"})
    df["tanggal"] = df["tanggal"].astype(str).str[:10]
    df["close"] = pd.to_numeric(df.get("Close"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")
    df["value"] = pd.to_numeric(df.get("Value"), errors="coerce")
    df["foreign_buy"] = pd.to_numeric(df.get("ForeignBuy"), errors="coerce")
    df["foreign_sell"] = pd.to_numeric(df.get("ForeignSell"), errors="coerce")
    df = df[["tanggal", "ticker", "close", "volume", "value",
             "foreign_buy", "foreign_sell"]].dropna(subset=["ticker"])
    df["foreign_net_vol"] = df["foreign_buy"].fillna(0) - df["foreign_sell"].fillna(0)
    df["foreign_net_value"] = df["foreign_net_vol"] * df["close"].fillna(0)
    return df


def save_atomic(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true", help="hanya hari terakhir")
    ap.add_argument("--days", type=int, default=None, help="N hari ke belakang")
    ap.add_argument("--date", type=str, default=None, help="satu tanggal YYYY-MM-DD")
    args = ap.parse_args()

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    if args.date:
        dates = [args.date]
    elif args.latest:
        dates = trading_dates(limit=1)
    else:
        dates = trading_dates(limit=args.days)

    done = load_done()
    todo = [d for d in dates if d not in done]
    print(f"Total tanggal: {len(dates)} | sudah ada: {len(done)} | "
          f"akan diambil: {len(todo)}")

    if not todo:
        print("Semua tanggal sudah ada di CSV. Selesai.")
        return

    opts = Options()
    opts.binary_location = CHROME
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(f"--user-agent={UA}")
    opts.add_argument("--lang=id-ID")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])

    drv = webdriver.Chrome(options=opts)
    drv.set_page_load_timeout(60)
    drv.get(PAGE_URL)
    time.sleep(12)  # biarkan Cloudflare challenge & Nuxt selesai

    new_frames, ok, fail = [], 0, 0
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        df = fetch_day(drv, d)
        if df is not None and not df.empty:
            new_frames.append(df)
            ok += 1
            if ok == 1:
                print(f"  contoh {d}: {df['ticker'].iloc[0]} "
                      f"buy={df['foreign_buy'].iloc[0]:,.0f} "
                      f"sell={df['foreign_sell'].iloc[0]:,.0f}")
        else:
            fail += 1
            print(f"  [{i}/{len(todo)}] {d} TIDAK ADA (non-trading/libur/belum publish)")

        if i % 50 == 0:
            el = (time.time() - t0) / 60
            print(f"  ...{i}/{len(todo)} | ok={ok} fail={fail} ({el:.1f} menit)")
        if len(new_frames) >= BATCH:
            all_df = pd.concat(new_frames, ignore_index=True)
            if os.path.exists(OUT_CSV):
                old = pd.read_csv(OUT_CSV, dtype={"ticker": str})
                all_df = pd.concat([old, all_df]).drop_duplicates(
                    subset=["tanggal", "ticker"], keep="last")
            save_atomic(all_df, OUT_CSV)
            print(f"  ✔ checkpoint tersimpan ({len(all_df):,} baris)")
            new_frames = []
        time.sleep(SLEEP)

    drv.quit()

    if new_frames:
        all_df = pd.concat(new_frames, ignore_index=True)
        if os.path.exists(OUT_CSV):
            old = pd.read_csv(OUT_CSV, dtype={"ticker": str})
            all_df = pd.concat([old, all_df]).drop_duplicates(
                subset=["tanggal", "ticker"], keep="last")
        save_atomic(all_df, OUT_CSV)

    final = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    print(f"\nSELESAI: ok={ok} gagal={fail} | total CSV: {len(final):,} baris, "
          f"{final['tanggal'].nunique() if not final.empty else 0} hari, "
          f"{final['ticker'].nunique() if not final.empty else 0} saham")
    if not final.empty:
        print("Rentang:", final["tanggal"].min(), "s/d", final["tanggal"].max())


if __name__ == "__main__":
    main()
