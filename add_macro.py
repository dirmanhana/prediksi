"""
add_macro.py
============
Menambahkan fitur makro (IHSG & kurs USD/IDR) ke database fitur.

Sumber: Yahoo Finance
  - ^JKSE  = Indeks Harga Saham Gabungan (IHSG)
  - IDR=X  = kurs USD/IDR

Output:
  - data/macro_id.csv            (data makro mentah + turunan)
  - data/features_macro.db       (SQLite: tabel features_macro = fitur lama + fitur makro)

Cara pakai:
  python add_macro.py
"""

import os
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DB_IN = os.path.join(BASE, "data", "features.db")
DB_OUT = os.path.join(BASE, "data", "features_macro.db")
MACRO_CSV = os.path.join(BASE, "data", "macro_id.csv")

MACRO_COLS = [
    "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
    "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
]

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def yahoo_session():
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://fc.yahoo.com", timeout=30)
    crumb = s.get(
        "https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=30
    ).text.strip()
    return s, crumb


def fetch_close(s, crumb, symbol, years=5):
    """Ambil close harian satu simbol dari Yahoo."""
    t2 = int(time.time())
    t1 = int(time.time()) - int(years * 365.25 * 24 * 3600)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = s.get(
        url,
        params={"period1": t1, "period2": t2, "interval": "1d",
                "events": "history", "crumb": crumb},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    close = res["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({
        "tanggal": [datetime.fromtimestamp(x).date() for x in ts],
        "close": close,
    }).dropna()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df.set_index("tanggal")["close"]


def main():
    print("Mengambil data makro dari Yahoo Finance...")
    s, crumb = yahoo_session()

    ihsg = fetch_close(s, crumb, "^JKSE")
    print(f"IHSG : {len(ihsg):,} hari, {ihsg.index.min().date()} s/d {ihsg.index.max().date()}")
    usd = fetch_close(s, crumb, "IDR=X")
    print(f"USD/IDR: {len(usd):,} hari, {usd.index.min().date()} s/d {usd.index.max().date()}")

    macro = pd.DataFrame({"ihsg": ihsg, "usd_idr": usd}).sort_index()
    macro = macro.dropna()

    # --- fitur turunan ---
    macro["ihsg_ret_1"] = macro["ihsg"].pct_change()
    macro["ihsg_ret_5"] = macro["ihsg"].pct_change(5)
    macro["ihsg_ret_20"] = macro["ihsg"].pct_change(20)
    macro["ihsg_sma20"] = macro["ihsg"] / macro["ihsg"].rolling(20, min_periods=10).mean()

    macro["usd_ret_1"] = macro["usd_idr"].pct_change()
    macro["usd_ret_5"] = macro["usd_idr"].pct_change(5)
    macro["usd_ret_20"] = macro["usd_idr"].pct_change(20)
    macro["usd_sma20"] = macro["usd_idr"] / macro["usd_idr"].rolling(20, min_periods=10).mean()

    macro = macro.replace([np.inf, -np.inf], np.nan)
    macro.to_csv(MACRO_CSV)
    print(f"Data makro disimpan: {MACRO_CSV} ({len(macro):,} baris)")

    # --- gabung dengan fitur saham ---
    conn = sqlite3.connect(DB_IN)
    feat = pd.read_sql("SELECT * FROM features", conn)
    conn.close()
    feat["tanggal"] = pd.to_datetime(feat["tanggal"])
    print(f"Fitur lama: {len(feat):,} baris x {feat.shape[1]} kolom")

    merged = feat.merge(
        macro[["ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
               "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20"]],
        left_on="tanggal", right_index=True, how="left",
    )
    merged = merged.dropna(subset=MACRO_COLS).reset_index(drop=True)

    if os.path.exists(DB_OUT):
        os.remove(DB_OUT)
    conn = sqlite3.connect(DB_OUT)
    merged.to_sql("features_macro", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_ticker_date ON features_macro(ticker, tanggal)")
    conn.commit()
    conn.close()

    print(f"Fitur makro: {len(merged):,} baris x {merged.shape[1]} kolom")
    print(f"  fitur makro baru: {MACRO_COLS}")
    print(f"Disimpan ke: {DB_OUT}")


if __name__ == "__main__":
    main()
