"""
eval_report.py
==============
Bangun laporan MONEV (monitoring & evaluasi): hasil prediksi vs AKTUAL,
termasuk penutupan SESI 1 & SESI 2.

Definisi kolom evaluasi:
  - basis_close    : close hari T (data_sampai)  -> dasar prediksi
  - open_t1        : open hari T+1 (tanggal_prediksi) -> entry realistis
  - close_sesi1    : penutupan sesi 1 hari T+1 (12:00 / Jumat 11:30 WIB)
  - close_sesi2    : penutupan sesi 2 hari T+1 (16:00 WIB) == close harian
  - status_model   : 1 bila close(T+1) > close(T)   -> sesuai target training
  - status_sesi1   : 1 bila close_sesi1 > open(T+1) -> entry pagi, exit siang
  - status_eksekusi: 1 bila close_sesi2 > open(T+1) -> entry pagi, tahan sampai tutup
  - benar_*        : 1 bila status_* cocok dgn arah prediksi (1=naik, 0=turun)

Output:
  - data/eval/prediction_eval.csv   (baris = satu prediksi)
  - data/eval/monev_summary.json    (ringkasan agregat)

Cara pakai:
  python eval_report.py                 # bangun + tampilkan ringkasan (7 hari)
  python eval_report.py --days 30       # ringkasan 30 hari
  python eval_report.py --capture       # sekaligus capture sesi dulu
  python eval_report.py --no-push       # jangan auto-commit + push ke git
"""

import argparse
import json
import os
import subprocess
from datetime import datetime

import pandas as pd

import session_data as sd
from waktu import WIB

BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(BASE, "data", "prediction_archive.csv")
PARQUET = os.path.join(BASE, "data", "history_id_5y.parquet")
EVAL_DIR = os.path.join(BASE, "data", "eval")
EVAL_CSV = os.path.join(EVAL_DIR, "prediction_eval.csv")
SUMMARY_JSON = os.path.join(EVAL_DIR, "monev_summary.json")

DEFAULT_CAPTURE_DAYS = 55

EVAL_COLS = [
    "tanggal_prediksi", "data_sampai", "ticker", "arah", "prob_up",
    "basis_close", "open_t1", "close_t1", "close_sesi1", "close_sesi2",
    "ret_sesi1", "ret_sesi2",
    "status_model", "status_sesi1", "status_eksekusi",
    "benar_model", "benar_sesi1", "benar_eksekusi",
]


# ---------------------------------------------------------------- util
def _pct(a, b):
    """Return persen (a/b - 1); None bila data tidak lengkap."""
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return float(a) / float(b) - 1


def _status(a, b):
    """1 bila a > b, 0 bila tidak; None bila data tidak lengkap."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return 1 if a > b else 0


def _daily_lookup(arch):
    """Dict (ticker, tanggal) -> (open, close) dari parquet harian."""
    daily = pd.read_parquet(PARQUET, columns=["ticker", "tanggal", "open", "close"])
    daily["tanggal"] = pd.to_datetime(daily["tanggal"]).dt.normalize()
    daily["ticker"] = daily["ticker"].astype(str)
    daily = daily[daily["ticker"].isin(set(arch["ticker"].astype(str)))]
    return {(tk, tgl): (o, c) for tk, tgl, o, c in
            zip(daily["ticker"], daily["tanggal"], daily["open"], daily["close"])}


def _session_lookup():
    """Dict (ticker, tanggal, sesi) -> close dari session_bars.csv."""
    s = sd.load_sessions()
    out = {}
    if s is None or s.empty:
        return out
    s = s.copy()
    s["tanggal"] = pd.to_datetime(s["tanggal"]).dt.normalize()
    for tk, tgl, sesi, c in zip(s["ticker"].astype(str), s["tanggal"],
                                s["sesi"], s["close"]):
        out[(tk, pd.Timestamp(tgl), int(sesi))] = c
    return out


# ---------------------------------------------------------------- build
def build(save=True):
    """Bangun DataFrame evaluasi prediksi vs aktual (dan simpan bila save=True)."""
    if not os.path.exists(ARCHIVE):
        return pd.DataFrame(columns=EVAL_COLS)
    arch = pd.read_csv(ARCHIVE, dtype={"ticker": str})
    if arch.empty:
        return pd.DataFrame(columns=EVAL_COLS)
    arch["tanggal_prediksi"] = pd.to_datetime(arch["tanggal_prediksi"]).dt.normalize()
    arch["data_sampai"] = pd.to_datetime(arch["data_sampai"]).dt.normalize()

    d_idx = _daily_lookup(arch)
    s_idx = _session_lookup()

    rows = []
    for r in arch.itertuples(index=False):
        tk = str(r.ticker)
        t1 = pd.Timestamp(r.tanggal_prediksi)
        t0 = pd.Timestamp(r.data_sampai)
        _, basis = d_idx.get((tk, t0), (None, None))
        o1, c1 = d_idx.get((tk, t1), (None, None))
        cs1 = s_idx.get((tk, t1, 1))
        cs2 = s_idx.get((tk, t1, 2))
        if cs2 is None:
            cs2 = c1  # sesi 2 == close harian (identik)
        arah = int(r.arah)

        status_model = _status(c1, basis)
        status_sesi1 = _status(cs1, o1)
        status_eksekusi = _status(cs2, o1)
        rows.append({
            "tanggal_prediksi": t1,
            "data_sampai": t0,
            "ticker": tk,
            "arah": arah,
            "prob_up": float(r.prob_up),
            "basis_close": basis,
            "open_t1": o1,
            "close_t1": c1,
            "close_sesi1": cs1,
            "close_sesi2": cs2,
            "ret_sesi1": _pct(cs1, o1),
            "ret_sesi2": _pct(cs2, o1),
            "status_model": status_model,
            "status_sesi1": status_sesi1,
            "status_eksekusi": status_eksekusi,
            "benar_model": None if status_model is None else int(status_model == arah),
            "benar_sesi1": None if status_sesi1 is None else int(status_sesi1 == arah),
            "benar_eksekusi": None if status_eksekusi is None else int(status_eksekusi == arah),
        })

    df = pd.DataFrame(rows, columns=EVAL_COLS)
    if not df.empty:
        df = df.sort_values(["tanggal_prediksi", "ticker"]).reset_index(drop=True)
    if save:
        os.makedirs(EVAL_DIR, exist_ok=True)
        tmp = EVAL_CSV + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, EVAL_CSV)
    return df


# ---------------------------------------------------------------- ringkasan
def _window(df, days=None):
    if df is None or df.empty or not days:
        return df if df is not None else pd.DataFrame(columns=EVAL_COLS)
    mx = df["tanggal_prediksi"].max()
    cutoff = mx - pd.Timedelta(days=int(days) - 1)
    return df[df["tanggal_prediksi"] >= cutoff]


def _hit(sub, col):
    s = sub[col].dropna()
    return float(s.mean()) if len(s) else None


def _avg(sub, col):
    s = sub[col].dropna()
    return float(s.mean()) if len(s) else None


def summarize(df, days=None):
    """Ringkasan agregat (dict) utk `days` hari terakhir (None = semua)."""
    d = _window(df, days) if df is not None else pd.DataFrame(columns=EVAL_COLS)
    out = {
        "n": int(len(d)),
        "hari": int(d["tanggal_prediksi"].nunique()) if len(d) else 0,
        "periode": ([str(d["tanggal_prediksi"].min().date()),
                     str(d["tanggal_prediksi"].max().date())] if len(d) else None),
    }
    for label in ("naik", "turun"):
        sub = d[d["arah"] == (1 if label == "naik" else 0)] if len(d) else d
        out[label] = {
            "n": int(len(sub)),
            "model": _hit(sub, "benar_model"),
            "sesi1": _hit(sub, "benar_sesi1"),
            "eksekusi": _hit(sub, "benar_eksekusi"),
            "ret_sesi1": _avg(sub, "ret_sesi1"),
            "ret_sesi2": _avg(sub, "ret_sesi2"),
        }
    out["model_overall"] = _hit(d, "benar_model")
    out["eksekusi_overall"] = _hit(d, "benar_eksekusi")
    if len(d):
        up1 = d[d["status_sesi1"] == 1]
        out["persist"] = (float((up1["status_eksekusi"] == 1).mean())
                          if len(up1) else None)
    else:
        out["persist"] = None
    return out


def _fp(x):
    return f"{x * 100:.0f}%" if x is not None else "–"


def _fr(x):
    return f"{x * 100:+.2f}%" if x is not None else "–"


def summary_text(df, days=7):
    """Teks laporan monev utk WhatsApp."""
    d = _window(df, days)
    out = summarize(df, days)
    if out["n"] == 0:
        return ("📊 *MONEV PREDIKSI*\n\nBelum ada prediksi yang bisa dievaluasi.\n"
                "Jalankan `capture_session.py` bila data sesi belum ada.")
    lines = [
        "📊 *MONEV — HASIL PREDIKSI vs AKTUAL*",
        f"📅 {days} hari terakhir: {out['periode'][0]} → {out['periode'][1]}",
        f"🧾 {out['n']} prediksi | {out['hari']} hari",
        "————————————————",
    ]
    for label, emoji in (("naik", "🟢"), ("turun", "🔴")):
        s = out[label]
        if s["n"] == 0:
            continue
        lines.append(f"{emoji} *PREDIKSI {label.upper()}* (n={s['n']})")
        if s["model"] is not None:
            lines.append(f"  • Model close→close: {_fp(s['model'])} benar")
        if s["sesi1"] is not None:
            lines.append(f"  • Sesi 1 (open→12:00): {_fp(s['sesi1'])} benar "
                         f"| rata2 {_fr(s['ret_sesi1'])}")
        if s["eksekusi"] is not None:
            lines.append(f"  • Sampai tutup (open→16:00): {_fp(s['eksekusi'])} benar "
                         f"| rata2 {_fr(s['ret_sesi2'])}")
        lines.append("")
    if out["persist"] is not None:
        lines.append(f"📈 Dari yg naik di sesi 1, {_fp(out['persist'])} tetap naik saat tutup")
    if out["model_overall"] is not None:
        lines.append(f"🧮 Akurasi model keseluruhan: {_fp(out['model_overall'])}")
    if len(d):
        lines.append("————————————————")
        lines.append("📅 *Per hari (5 terakhir):*")
        grp = d.groupby(d["tanggal_prediksi"].dt.date)
        for tgl, sub in list(grp)[-5:]:
            hm = _hit(sub[sub["arah"] == 1], "benar_model")
            ht = _hit(sub[sub["arah"] == 0], "benar_model")
            lines.append(f"  {tgl.strftime('%d-%m')}: naik {_fp(hm)} | turun {_fp(ht)}")
    lines.append("————————————————")
    lines.append("ℹ️ Model = close besok vs close hari ini (target training).")
    lines.append("ℹ️ Sesi 1/tutup = entry di open, exit di penutupan sesi.")
    lines.append("⚠️ Bukan saran investasi.")
    return "\n".join(lines)


def save_summary(days=7):
    os.makedirs(EVAL_DIR, exist_ok=True)
    df = build(save=False)
    data = summarize(df, days)
    data["generated_at"] = datetime.now(WIB).strftime("%Y-%m-%d %H:%M WIB")
    tmp = SUMMARY_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SUMMARY_JSON)
    return data


# ---------------------------------------------------------------- git
def git_autocommit_push(message, paths=("data/eval", "data/reports"), timeout=120):
    """Commit + push hasil evaluasi ke origin. Return status string."""
    def run(args):
        return subprocess.run(["git", "-C", BASE, *args],
                              capture_output=True, text=True, timeout=timeout)

    run(["add", "--", *paths])
    if run(["diff", "--cached", "--quiet"]).returncode == 0:
        return "nochange"
    c = run(["commit", "-m", message])
    if c.returncode != 0:
        return f"commit-gagal: {c.stderr.strip()[:200]}"
    p = run(["push", "origin", "HEAD"])
    if p.returncode != 0:
        return f"push-gagal: {p.stderr.strip()[:200]}"
    return "pushed"


# ---------------------------------------------------------------- harian
def run_daily(days=DEFAULT_CAPTURE_DAYS, auto_push=True, verbose=True):
    """Rutin harian: capture sesi (incremental) -> build eval -> simpan -> push."""
    result = {"capture_tickers": 0, "capture_ok": 0, "capture_fail": 0,
              "bars": 0, "eval_rows": 0, "pdf": None, "push": "off"}
    arch = None
    if os.path.exists(ARCHIVE):
        arch = pd.read_csv(ARCHIVE, dtype={"ticker": str})
    if arch is not None and not arch.empty:
        todo = sd.tickers_needing_capture(arch, max_age_days=days)
        if todo:
            bars, ok, fail = sd.capture(todo, days=days, verbose=verbose)
            result.update(capture_tickers=len(todo), capture_ok=ok,
                          capture_fail=fail, bars=bars)
    # backup harian data sesi (tidak di-commit, tidak bisa diregenerate >60 hari)
    sd.backup_sessions(verbose=verbose)
    df = build(save=True)
    result["eval_rows"] = int(len(df))
    save_summary(days=min(int(days), 7))
    # PDF "DATA BOT TRADING" dibuat SEBELUM push supaya ikut ter-commit
    try:
        import report_pdf as _rp
        result["pdf"] = _rp.build_pdf(days=14)
    except Exception as e:
        if verbose:
            print(f"⚠️ Gagal bikin PDF: {type(e).__name__}: {e}")
    if auto_push and len(df):
        msg = ("monev: evaluasi prediksi vs aktual s/d "
               f"{datetime.now(WIB).strftime('%Y-%m-%d')}")
        result["push"] = git_autocommit_push(msg)
    return result


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Bangun laporan monev prediksi")
    ap.add_argument("--days", type=int, default=7, help="jendela ringkasan (hari)")
    ap.add_argument("--capture", action="store_true", help="capture sesi dulu")
    ap.add_argument("--capture-days", type=int, default=DEFAULT_CAPTURE_DAYS)
    ap.add_argument("--no-push", action="store_true", help="jangan commit+push")
    args = ap.parse_args()

    if args.capture:
        r = run_daily(days=args.capture_days, auto_push=not args.no_push)
        print(f"Capture: {r['capture_ok']}/{r['capture_tickers']} saham OK, "
              f"{r['bars']} bar baru | eval rows: {r['eval_rows']} | push: {r['push']}")
    else:
        df = build(save=True)
        save_summary(args.days)
        if not args.no_push and len(df):
            msg = ("monev: evaluasi prediksi vs aktual s/d "
                   f"{datetime.now(WIB).strftime('%Y-%m-%d')}")
            print("push:", git_autocommit_push(msg))
    df = build(save=False)
    print()
    print(summary_text(df, args.days))


if __name__ == "__main__":
    main()
