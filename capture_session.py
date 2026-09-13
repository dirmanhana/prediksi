#!/usr/bin/env python3
"""
capture_session.py
==================
Ambil data penutupan SESI 1 & SESI 2 untuk saham yang DIPREDIKSI
(top-20 NAIK + top-20 TURUN) dari Yahoo Finance (interval 15m), lalu simpan
ke data/session_bars.csv.

Kenapa perlu dijalankan rutin?
  Yahoo hanya menyimpan interval 15m untuk ~60 hari terakhir. Kalau tidak
  disimpan, data sesi akan hilang dan tidak bisa di-backfill lagi.

Cara pakai:
  python capture_session.py              # incremental (hanya yg belum lengkap)
  python capture_session.py --days 55    # jendela hari (maks 59)
  python capture_session.py --all        # paksa ambil semua ticker dlm jendela
  python capture_session.py --pause 0.5  # jeda antar request (detik)

Output: data/session_bars.csv
"""

import argparse
import os
import sys

import pandas as pd

import session_data as sd

BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(BASE, "data", "prediction_archive.csv")


def load_archive():
    if not os.path.exists(ARCHIVE):
        return None
    return pd.read_csv(ARCHIVE, dtype={"ticker": str})


def main():
    ap = argparse.ArgumentParser(description="Capture sesi 1 & 2 saham prediksi")
    ap.add_argument("--days", type=int, default=sd.DEFAULT_DAYS,
                    help=f"jendela hari ke belakang (maks {sd.MAX_DAYS})")
    ap.add_argument("--all", action="store_true",
                    help="paksa ambil semua ticker dalam jendela (bukan hanya yg kosong)")
    ap.add_argument("--pause", type=float, default=0.35,
                    help="jeda antar request Yahoo (detik)")
    args = ap.parse_args()

    arch = load_archive()
    if arch is None or arch.empty:
        sys.exit(f"Arsip prediksi belum ada/kosong: {ARCHIVE}")

    if args.all:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=args.days)
        a = arch.copy()
        a["tanggal_prediksi"] = pd.to_datetime(a["tanggal_prediksi"])
        a = a[(a["tanggal_prediksi"] >= cutoff) &
              (a["tanggal_prediksi"] <= pd.Timestamp.now().normalize())]
        tickers = sorted(a["ticker"].astype(str).unique())
    else:
        tickers = sd.tickers_needing_capture(arch, max_age_days=args.days)

    if not tickers:
        print("Semua prediksi dalam jendela sudah punya data sesi 1 & 2. Tidak ada yg perlu diambil.")
        return

    print(f"Akan mengambil sesi untuk {len(tickers)} saham "
          f"(jendela {args.days} hari)...")
    n_bar, n_ok, n_fail = sd.capture(tickers, days=args.days, pause=args.pause)

    alls = sd.load_sessions()
    if len(alls):
        alls["tanggal"] = pd.to_datetime(alls["tanggal"])
        n_tgl = alls["tanggal"].nunique()
        n_tk = alls["ticker"].nunique()
        print(f"\nSelesai: {n_ok} saham OK, {n_fail} gagal, {n_bar} bar sesi baru.")
        print(f"Total tersimpan: {len(alls):,} bar | {n_tk} saham | {n_tgl} hari "
              f"({alls['tanggal'].min().date()} s/d {alls['tanggal'].max().date()})")
        print(f"-> {sd.SESSION_CSV}")
    else:
        print("Tidak ada data tersimpan.")


if __name__ == "__main__":
    main()
