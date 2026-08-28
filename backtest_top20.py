#!/usr/bin/env python3
"""
backtest_top20.py
=================
Backtest JUJUR strategi "long Top-20 NAIK" dari model gabungan XGBoost,
menggunakan prediksi walk-forward out-of-sample (data/walkforward_preds.csv).

Yang diukur (dengan & tanpa filter likuiditas + biaya transaksi):
  - Hit rate (berapa % rekomendasi yang benar-benar naik besok)
  - Return rata-rata per hari (winsorized utk tahan outlier)
  - t-statistik & Sharpe (apakah edge signifikan?)
  - Perbandingan vs buy-and-hold semua saham

Cara pakai:  python backtest_top20.py
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

MIN_VALUE = 1_000_000_000   # Rp1 miliar/hari (sama dgn filter bot)
MIN_PRICE = 200
COST = 0.003                # biaya round-trip 0.3% (komisi + slippage minimal)


def main():
    preds = pd.read_csv(os.path.join(BASE, "data", "walkforward_preds.csv"),
                        parse_dates=["tanggal"])
    hist = pd.read_parquet(os.path.join(BASE, "data", "history_id_5y.parquet"))
    hist = hist[["ticker", "tanggal", "close", "volume"]].copy()
    hist = hist[hist["close"] > 0].sort_values(["ticker", "tanggal"])
    hist["ret_next"] = hist.groupby("ticker")["close"].shift(-1) / hist["close"] - 1
    hist["up_next"] = (hist["ret_next"] > 0).astype(int)
    hist["value"] = hist["close"] * hist["volume"]

    m = preds.merge(hist, on=["ticker", "tanggal"], how="left")
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_next", "value"])
    m["liquid"] = (m["value"] >= MIN_VALUE) & (m["close"] >= MIN_PRICE)
    print(f"Data OOS: {len(m):,} prediksi, {m['tanggal'].nunique()} hari, "
          f"{m['ticker'].nunique()} saham\n")

    print("=" * 78)
    print("STRATEGI: long Top-20 NAIK (equal weight, tahan 1 hari)")
    print("=" * 78)
    rows = []
    for label, use_liq in [("SEMUA saham", False), ("HANYA likuid", True)]:
        ml = m[m["liquid"]] if use_liq else m
        daily = []
        for d, g in ml.groupby("tanggal"):
            top = g.sort_values("prob", ascending=False).head(20)
            if len(top) < 20:
                continue
            daily.append({
                "hit": top["up_next"].mean(),
                "ret": top["ret_next"].clip(-0.1, 0.1).mean(),
                "ret_net": top["ret_next"].clip(-0.1, 0.1).mean() - COST,
            })
        r = pd.DataFrame(daily)
        base = ml["up_next"].mean()
        hit = r["hit"].mean()
        ret = r["ret"].mean()
        ret_net = r["ret_net"].mean()
        se = r["ret_net"].std() / np.sqrt(len(r))
        tstat = ret_net / se if se > 0 else 0
        sharpe = r["ret_net"].mean() / r["ret_net"].std() * np.sqrt(252) \
            if r["ret_net"].std() > 0 else 0
        rows.append((label, use_liq, hit, base, ret, ret_net, tstat, sharpe, len(r)))
        print(f"\n[{label}]  ({ml['ticker'].nunique()} saham, {len(r)} hari)")
        print(f"  Hit rate top-20      : {hit*100:5.1f}%   (base rate {base*100:.1f}%)")
        print(f"  Return bruto/hari    : {ret*100:+.3f}%")
        print(f"  Return NETO/hari     : {ret_net*100:+.3f}%  (biaya 0.3% round-trip)")
        print(f"  t-statistik          : {tstat:5.1f}   (|t|>2 = signifikan)")
        print(f"  Sharpe tahunan       : {sharpe:5.2f}")

    print("\n" + "=" * 78)
    print("KESIMPULAN")
    print("=" * 78)
    sem, liq = rows[0], rows[1]
    print(f"- Filter likuiditas menurunkan hit rate "
          f"{sem[2]*100:.1f}% -> {liq[2]*100:.1f}% (edge tipis di saham layak jual)")
    print(f"- Return neto: {sem[6]:.1f} t-stat (semua) vs {liq[6]:.1f} t-stat (likuid)")
    verdict = ("SIGNIFIKAN secara statistik" if abs(liq[6]) >= 2
               else "BELUM signifikan secara statistik (t < 2)")
    print(f"- Strategi top-20 likuid setelah biaya: {verdict}")
    print("\nCatatan: belum termasuk spread bid-ask, slippage eksekusi, lot 100,")
    print("auto-reject 7/10%, suspensi, dan pajak. Realisasi pasti lebih rendah.")


if __name__ == "__main__":
    main()
