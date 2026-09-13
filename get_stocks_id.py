"""
get_stocks_id.py
================
Mengambil daftar SEMUA saham yang terdaftar di Bursa Efek Indonesia (IDX/BEI)
menggunakan TradingView Scanner (sumber data live).

Output:
  - data/stocks_id.csv      (CSV, UTF-8 with BOM agar mudah dibuka di Excel)
  - data/stocks_id.xlsx     (Excel)
  - data/stocks_id.json     (JSON)

Cara pakai:
  python get_stocks_id.py
"""

import json
import time

import pandas as pd
import requests

from waktu import now_wib

# Kolom yang diambil dari TradingView
COLUMNS = [
    "name",            # kode saham (ticker), mis. BBRI
    "description",     # nama perusahaan, mis. PT Bank Rakyat Indonesia (Persero) Tbk
    "sector",          # sektor (Finance, Energy, dll)
    "industry",        # industri (Regional Banks, dll)
    "close",           # harga terakhir (IDR)
    "open",            # harga pembukaan
    "high",            # harga tertinggi
    "low",             # harga terendah
    "change",          # perubahan harga (%)
    "volume",          # volume perdagangan (lembar)
    "total_shares_outstanding",  # jumlah saham beredar
    "average_volume_10d_calc",   # rata-rata volume 10 hari
    "currency",        # mata uang (IDR)
    "exchange",        # bursa (IDX)
]

SCANNER_URL = "https://scanner.tradingview.com/indonesia/scan"
CHUNK = 500  # jumlah baris per permintaan


def fetch_all_stocks():
    """Ambil semua saham IDX dengan pagination."""
    all_rows = []
    start = 0
    total = None

    while True:
        body = {
            "symbols": {"tickers": [], "query": {"types": []}},
            "columns": COLUMNS,
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [start, start + CHUNK],
        }
        resp = requests.post(
            SCANNER_URL,
            json=body,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        if total is None:
            total = data.get("totalCount", 0)
            print(f"Total saham di IDX: {total}")

        rows = data.get("data", [])
        if not rows:
            break

        for r in rows:
            all_rows.append(dict(zip(COLUMNS, r["d"])))

        start += len(rows)
        print(f"  ...diambil {len(all_rows)}/{total}")
        time.sleep(0.5)  # jeda agar tidak kena rate-limit

        if start >= total or len(rows) < CHUNK:
            break

    return all_rows


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom turunan: market cap, tanggal data, dsb."""
    df = df.copy()

    # market cap = harga terakhir x jumlah saham beredar
    df["market_cap"] = df["close"] * df["total_shares_outstanding"]

    # bersihkan kolom numerik
    for col in ["close", "open", "high", "low", "change", "volume",
                "total_shares_outstanding", "average_volume_10d_calc", "market_cap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_cap_miliar_idr"] = (df["market_cap"] / 1e9).round(2)

    # urutkan: kode saham A-Z
    df = df.sort_values("name").reset_index(drop=True)

    df.insert(0, "no", range(1, len(df) + 1))
    df["tanggal_data"] = now_wib().strftime("%Y-%m-%d %H:%M:%S")
    return df


def main():
    print("Mengambil daftar saham IDX dari TradingView...")
    rows = fetch_all_stocks()

    df = enrich(pd.DataFrame(rows))
    print(f"\nTotal saham: {len(df)}")

    # statistik ringkas
    print("\nJumlah saham per sektor:")
    print(df["sector"].value_counts().to_string())

    print("\n10 saham dengan market cap terbesar:")
    top = df.nlargest(10, "market_cap")[["name", "description", "close", "market_cap_miliar_idr"]]
    print(top.to_string(index=False))

    # simpan ke file
    import os
    os.makedirs("data", exist_ok=True)
    csv_path = "data/stocks_id.csv"
    xlsx_path = "data/stocks_id.xlsx"
    json_path = "data/stocks_id.json"

    # CSV dengan BOM (UTF-8-sig) agar terbaca rapi di Excel Windows
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    df.to_excel(xlsx_path, index=False, sheet_name="Saham IDX")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\nFile tersimpan:")
    print(f"  - {csv_path}")
    print(f"  - {xlsx_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    main()
