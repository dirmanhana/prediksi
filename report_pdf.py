"""
report_pdf.py
=============
Laporan PDF "DATA BOT TRADING" — mirip spreadsheet tim:
untuk tiap tanggal prediksi, daftar Top-10 saham yang diprediksi NAIK
beserta pergerakan penutupan SESI 1 & SESI 2.

Aturan (diverifikasi sama dgn spreadsheet tim):
  - Return SESI 1 = close_sesi1 / close_hari_sebelumnya - 1
  - Return SESI 2 = close_sesi2 / close_hari_sebelumnya - 1
  - "SUSPEND" bila tidak ada transaksi (tak ada bar / volume 0)

Output: data/reports/laporan_bot_YYYYMMDD.pdf

Dipakai oleh perintah `monev` di wa_bot.py.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from waktu import now_wib

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_CSV = os.path.join(BASE, "data", "eval", "prediction_eval.csv")
SESSION_CSV = os.path.join(BASE, "data", "session_bars.csv")
OUT_DIR = os.path.join(BASE, "data", "reports")

TOP_N = 10           # jumlah saham per tanggal (top-10 naik)
BANDS_PER_PAGE = 2   # tiap band = 2 blok tanggal berdampingan
DEFAULT_DAYS = 14    # "2 minggu terakhir"


# ---------------------------------------------------------------- data
def _fmt_pct(v):
    """Format persen gaya Indonesia (koma) + tanda, mis. +9,60% / -5,93%."""
    return f"{v * 100:+.2f}%".replace(".", ",")


def _session_df():
    if not os.path.exists(SESSION_CSV):
        return None
    try:
        s = pd.read_csv(SESSION_CSV, dtype={"ticker": str})
    except Exception:
        return None
    if s.empty:
        return None
    s["tanggal"] = pd.to_datetime(s["tanggal"]).dt.normalize()
    s["sesi"] = s["sesi"].astype(int)
    s["volume"] = pd.to_numeric(s["volume"], errors="coerce").fillna(0)
    return s


def build_blocks(days=DEFAULT_DAYS, top_n=TOP_N):
    """Bikin daftar blok (per tanggal prediksi) utk laporan PDF.

    Return list of dict: {tanggal: Timestamp, rows: [{ticker, s1, s2, s1v, s2v}]}
    """
    if not os.path.exists(EVAL_CSV):
        return []
    d = pd.read_csv(EVAL_CSV, dtype={"ticker": str})
    if d.empty:
        return []
    d["tanggal_prediksi"] = pd.to_datetime(d["tanggal_prediksi"]).dt.normalize()

    s = _session_df()
    if s is None:
        return []
    max_date = s["tanggal"].max()
    cutoff = max_date - pd.Timedelta(days=days - 1)
    d = d[(d["tanggal_prediksi"] >= cutoff) & (d["tanggal_prediksi"] <= max_date)]
    if d.empty:
        return []

    idx = {}
    for tk, tg, sesi, c, vol in zip(s["ticker"], s["tanggal"], s["sesi"],
                                    s["close"], s["volume"]):
        idx[(tk, pd.Timestamp(tg), int(sesi))] = (c, vol)

    def cell(tk, tgl, sesi, basis):
        v = idx.get((tk, tgl, sesi))
        if v is None:
            return "SUSPEND", None
        c, vol = v
        if not vol or pd.isna(c) or pd.isna(basis) or not basis:
            return "SUSPEND", None
        val = float(c) / float(basis) - 1
        return _fmt_pct(val), val

    blocks = []
    for tgl, sub in d.groupby("tanggal_prediksi"):
        naik = sub[sub["arah"] == 1].sort_values("prob_up", ascending=False).head(top_n)
        rows = []
        for r in naik.itertuples():
            s1, s1v = cell(r.ticker, tgl, 1, r.basis_close)
            s2, s2v = cell(r.ticker, tgl, 2, r.basis_close)
            rows.append({"ticker": r.ticker, "s1": s1, "s2": s2,
                         "s1v": s1v, "s2v": s2v})
        if rows:
            blocks.append({"tanggal": pd.Timestamp(tgl), "rows": rows})
    blocks.sort(key=lambda b: b["tanggal"])
    return blocks


# ---------------------------------------------------------------- render
def _draw_block(ax, bbox, block):
    rows = (block["rows"] if block else [])[:TOP_N]
    tanggal = block["tanggal"].strftime("%d-%m-%Y") if block else ""
    col_labels = ["Tanggal", "Top 10 BOT naik", "SESI 1", "SESI 2"]
    cell_text = []
    for i in range(TOP_N):
        if i < len(rows):
            r = rows[i]
            cell_text.append([tanggal if i == 0 else "", r["ticker"], r["s1"], r["s2"]])
        else:
            cell_text.append(["", "", "", ""])
    t = ax.table(cellText=cell_text, colLabels=col_labels, bbox=bbox,
                 cellLoc="center", colWidths=[0.20, 0.30, 0.25, 0.25])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    for (ri, ci), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if ri == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        txt = cell.get_text().get_text()
        if ci in (2, 3):
            if txt.startswith("+"):
                cell.set_facecolor("#e8f5e9")
            elif txt.startswith("-"):
                cell.set_facecolor("#ffebee")
            elif txt == "SUSPEND":
                cell.set_facecolor("#eeeeee")
                cell.set_text_props(color="#888888")
        elif ci == 0 and txt:
            cell.set_text_props(fontweight="bold")
        elif ci == 1 and txt:
            cell.set_text_props(fontweight="bold")


def _make_fig(page_bands):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.5, 0.965, "DATA BOT TRADING", ha="center", va="center",
             fontsize=18, fontweight="bold", color="#1a1a1a")
    sub = ("Top 10 BOT naik + pergerakan penutupan SESI 1 & SESI 2  |  "
           "return basis close hari sebelumnya")
    fig.text(0.5, 0.925, sub, ha="center", va="center", fontsize=9.5, color="#555555")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    top, bottom = 0.90, 0.06
    band_h = (top - bottom) / BANDS_PER_PAGE
    for bi, band in enumerate(page_bands):
        y = top - (bi + 1) * band_h
        _draw_block(ax, [0.015, y + 0.012, 0.475, band_h - 0.03], band[0])
        if len(band) > 1:
            _draw_block(ax, [0.51, y + 0.012, 0.475, band_h - 0.03], band[1])

    fig.text(0.5, 0.02,
             f"Dibuat {now_wib():%d-%m-%Y %H:%M} WIB  |  SUSPEND = tidak ada transaksi  |  "
             "hijau = naik, merah = turun  |  bukan saran investasi",
             ha="center", va="center", fontsize=7, color="#888888")
    return fig


def render_pdf(blocks, path=None, bands_per_page=BANDS_PER_PAGE):
    if not blocks:
        return None
    path = path or os.path.join(OUT_DIR, f"laporan_bot_{now_wib():%Y%m%d}.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bands = [blocks[i:i + 2] for i in range(0, len(blocks), 2)]
    pages = [bands[i:i + bands_per_page] for i in range(0, len(bands), bands_per_page)]
    with PdfPages(path) as pdf:
        for page_bands in pages:
            fig = _make_fig(page_bands)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def render_png(blocks, path, bands_per_page=BANDS_PER_PAGE, dpi=110):
    """Render halaman pertama sbg PNG (utk pratinjau)."""
    if not blocks:
        return None
    bands = [blocks[i:i + 2] for i in range(0, len(blocks), 2)]
    fig = _make_fig(bands[:bands_per_page])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def build_pdf(days=DEFAULT_DAYS, path=None, top_n=TOP_N):
    """Bikin PDF laporan; return path atau None bila tak ada data."""
    blocks = build_blocks(days=days, top_n=top_n)
    if not blocks:
        return None
    return render_pdf(blocks, path=path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Laporan PDF DATA BOT TRADING")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--png", action="store_true", help="render halaman 1 ke PNG")
    args = ap.parse_args()
    blocks = build_blocks(days=args.days)
    print(f"blok tanggal: {len(blocks)}")
    for b in blocks:
        print(" ", b["tanggal"].date(), "->", len(b["rows"]), "saham:",
              ", ".join(r["ticker"] for r in b["rows"]))
    out = render_pdf(blocks)
    print("PDF:", out)
    if args.png:
        png = os.path.join(OUT_DIR, "pratinjau.png")
        render_png(blocks, png)
        print("PNG:", png)
