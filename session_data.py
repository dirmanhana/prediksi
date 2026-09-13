"""
session_data.py
===============
Ambil & olah data intraday 15-menit (Yahoo Finance) menjadi bar per SESI
perdagangan IDX.

Definisi sesi (WIB = UTC+7):
  Senin-Kamis : Sesi 1 = 09:00-12:00 | Sesi 2 = 13:30-16:00
  Jumat       : Sesi 1 = 09:00-11:30 | Sesi 2 = 14:00-16:00

Catatan penting:
  - Bar 16:00 WIB = penutupan resmi IDX. Nilainya SAMA dengan close harian
    di data/history/*.csv (sudah diverifikasi), jadi close_sesi2 == close harian.
  - Yahoo hanya menyediakan interval 15-menit untuk ~60 hari terakhir. Karena
    itu data sesi WAJIB disimpan sendiri (data/session_bars.csv) — tidak bisa
    di-backfill lagi setelah keluar jendela 60 hari.

Output: data/session_bars.csv
  kolom: tanggal, ticker, sesi, open, high, low, close, volume, source, captured_at
"""

import os
import shutil
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from waktu import WIB

BASE = os.path.dirname(os.path.abspath(__file__))
SESSION_CSV = os.path.join(BASE, "data", "session_bars.csv")
PARQUET = os.path.join(BASE, "data", "history_id_5y.parquet")
MAX_DAYS = 59          # batas keras Yahoo utk interval 15m ("within the last 60 days")
DEFAULT_DAYS = 55
BACKUP_DIR = os.environ.get("SESSION_BACKUP_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "data", "backups"))
BACKUP_KEEP = 14       # jumlah file backup harian yang disimpan

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

SESSION_COLS = ["tanggal", "ticker", "sesi", "open", "high", "low",
                "close", "volume", "source", "captured_at"]


# ---------------------------------------------------------------- yahoo
def yahoo_session():
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://fc.yahoo.com", timeout=30)
    crumb = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb",
                  timeout=30).text.strip()
    if not crumb:
        raise RuntimeError("Gagal mendapatkan crumb Yahoo Finance")
    return s, crumb


class YahooIntraday:
    """Pembungkus session Yahoo + crumb (crumb bisa expire -> login ulang)."""

    def __init__(self):
        self.s, self.crumb = yahoo_session()

    def fetch(self, ticker, days=DEFAULT_DAYS):
        """Ambil bar 15-menit `days` hari terakhir utk satu ticker.
        Return DataFrame (dt_utc, open, high, low, close, volume) atau None."""
        days = max(1, min(int(days), MAX_DAYS))
        t2 = int(time.time())
        t1 = t2 - days * 86400
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.JK"
        for attempt in range(5):
            try:
                r = self.s.get(url, params={
                    "period1": t1, "period2": t2, "interval": "15m",
                    "events": "history", "crumb": self.crumb,
                }, timeout=30)
                if r.status_code == 429:          # rate limited
                    time.sleep(6 * (attempt + 1))
                    continue
                if r.status_code == 401:          # crumb expired
                    self.s, self.crumb = yahoo_session()
                    continue
                if r.status_code == 404:          # ticker tak ada di Yahoo
                    return None
                r.raise_for_status()
                res = r.json().get("chart", {}).get("result")
                if not res or not res[0].get("timestamp"):
                    return None
                q = res[0]["indicators"]["quote"][0]
                df = pd.DataFrame({
                    "dt_utc": pd.to_datetime(res[0]["timestamp"], unit="s", utc=True),
                    "open": q.get("open"),
                    "high": q.get("high"),
                    "low": q.get("low"),
                    "close": q.get("close"),
                    "volume": q.get("volume"),
                })
                return df.dropna(subset=["close"]).reset_index(drop=True)
            except requests.RequestException:
                time.sleep(4 * (attempt + 1))
        return None


# ---------------------------------------------------------------- sesi
def _sesi_of(dt_wib):
    """1 = sesi 1, 2 = sesi 2, 0 = jeda (bukan jam perdagangan).

    Bar Yahoo diberi label waktu MULAI bar:
      - Senin-Kamis: sesi 1 bar terakhir 11:45, sesi 2 mulai 13:30
      - Jumat      : sesi 1 bar terakhir 11:15, sesi 2 mulai 14:00
    """
    hm = dt_wib.hour * 60 + dt_wib.minute
    friday = dt_wib.weekday() == 4
    s1_end = 11 * 60 + 30 if friday else 12 * 60
    s2_start = 14 * 60 if friday else 13 * 60 + 30
    if hm < s1_end:
        return 1
    if hm >= s2_start:
        return 2
    return 0


def _first_valid(series):
    s = series.dropna()
    return s.iloc[0] if len(s) else float("nan")


def derive_sessions(df, ticker):
    """Ubah bar 15-menit menjadi bar per sesi (satu baris per tanggal x sesi)."""
    if df is None or df.empty:
        return None
    d = df.copy()
    d["dt_wib"] = d["dt_utc"].dt.tz_convert(WIB)
    d["tanggal"] = d["dt_wib"].dt.date
    d["sesi"] = d["dt_wib"].apply(_sesi_of)
    d = d[d["sesi"] > 0]
    if d.empty:
        return None
    d = d.sort_values("dt_wib")
    agg = (d.groupby(["tanggal", "sesi"])
             .agg(open=("open", _first_valid),
                  high=("high", "max"),
                  low=("low", "min"),
                  close=("close", "last"),
                  volume=("volume", "sum"))
             .reset_index())
    if agg.empty:
        return None
    agg.insert(1, "ticker", ticker)
    agg["tanggal"] = pd.to_datetime(agg["tanggal"])
    agg["sesi"] = agg["sesi"].astype(int)
    return agg


# ---------------------------------------------------------------- simpan
def load_sessions():
    """Baca data/session_bars.csv (DataFrame kosong bila belum ada)."""
    if not os.path.exists(SESSION_CSV):
        return pd.DataFrame(columns=SESSION_COLS)
    try:
        df = pd.read_csv(SESSION_CSV, dtype={"ticker": str})
    except Exception:
        return pd.DataFrame(columns=SESSION_COLS)
    if df.empty:
        return pd.DataFrame(columns=SESSION_COLS)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["sesi"] = df["sesi"].astype(int)
    return df


def merge_sessions(new_df):
    """Gabung bar baru ke session_bars.csv (dedup: tanggal+ticker+sesi, keep last)."""
    if new_df is None or new_df.empty:
        return load_sessions()
    new = new_df.copy()
    new["tanggal"] = pd.to_datetime(new["tanggal"])
    new["sesi"] = new["sesi"].astype(int)
    old = load_sessions()
    if old.empty:
        out = new
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = (out.drop_duplicates(subset=["tanggal", "ticker", "sesi"], keep="last")
              .sort_values(["tanggal", "ticker", "sesi"])
              .reset_index(drop=True))
    for col in SESSION_COLS:
        if col not in out.columns:
            out[col] = None
    tmp = SESSION_CSV + ".tmp"
    out[SESSION_COLS].to_csv(tmp, index=False)
    os.replace(tmp, SESSION_CSV)
    return out


# ---------------------------------------------------------------- kebutuhan
def _traded_pairs():
    """Set (ticker, tanggal) yang BENAR-BENAR ada di data harian = hari bursa.

    Dipakai utk menyaring `tickers_needing_capture`: pasangan yang tidak ada di
    data harian (akhir pekan, libur bursa, saham suspend, atau tanggal arsip
    yang salah) tidak akan pernah punya bar sesi -> jangan di-fetch berulang.
    Return None bila parquet tidak tersedia (fallback: tanpa filter).
    """
    if not os.path.exists(PARQUET):
        return None
    try:
        d = pd.read_parquet(PARQUET, columns=["ticker", "tanggal"])
    except Exception:
        return None
    d["tanggal"] = pd.to_datetime(d["tanggal"]).dt.normalize()
    return set(zip(d["ticker"].astype(str), d["tanggal"]))


def tickers_needing_capture(archive, sessions=None, max_age_days=DEFAULT_DAYS, now=None):
    """Daftar ticker yang prediksinya BELUM punya sesi 1 & 2 lengkap.

    `archive` = DataFrame prediction_archive.csv (kolom ticker, tanggal_prediksi).
    Hanya tanggal_prediksi dalam jendela `max_age_days` yang dihitung (di luar
    itu memang sudah tidak bisa diambil lagi dari Yahoo). Pasangan yang bukan
    hari bursa (tidak ada di data harian) dilewati.
    """
    if archive is None or len(archive) == 0:
        return []
    now = now or datetime.now(WIB)
    cutoff = pd.Timestamp(now.date()) - pd.Timedelta(days=max_age_days)
    today = pd.Timestamp(now.date())

    a = archive.copy()
    a["tanggal_prediksi"] = pd.to_datetime(a["tanggal_prediksi"]).dt.normalize()
    a = a[(a["tanggal_prediksi"] >= cutoff) & (a["tanggal_prediksi"] <= today)]
    if a.empty:
        return []

    sessions = load_sessions() if sessions is None else sessions
    have = set()
    if sessions is not None and len(sessions) > 0:
        s = sessions.copy()
        s["tanggal"] = pd.to_datetime(s["tanggal"]).dt.normalize()
        g = s.groupby(["ticker", "tanggal"])["sesi"].nunique()
        have = set(g[g >= 2].index)

    traded = _traded_pairs()
    need = set()
    for tk, tgl in zip(a["ticker"].astype(str), a["tanggal_prediksi"]):
        key = (tk, pd.Timestamp(tgl))
        if traded is not None and key not in traded:
            continue  # bukan hari bursa utk saham ini -> tak ada sesi
        if key not in have:
            need.add(tk)
    return sorted(need)


def capture(tickers, days=DEFAULT_DAYS, pause=0.35, verbose=True):
    """Ambil sesi utk daftar ticker lalu simpan. Return (n_bar, n_ok, n_gagal)."""
    tickers = sorted({str(t).upper() for t in tickers if str(t).strip()})
    if not tickers:
        return 0, 0, 0
    y = YahooIntraday()
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frames = []
    n_ok = n_fail = 0
    for i, tk in enumerate(tickers, 1):
        df = y.fetch(tk, days)
        sess = derive_sessions(df, tk) if df is not None else None
        if sess is not None and not sess.empty:
            sess["source"] = "yahoo_15m"
            sess["captured_at"] = captured_at
            frames.append(sess)
            n_ok += 1
        else:
            n_fail += 1
        if verbose and (i % 25 == 0 or i == len(tickers)):
            print(f"  ...capture {i}/{len(tickers)} (ok={n_ok}, gagal={n_fail})", flush=True)
        time.sleep(pause)
    if not frames:
        return 0, n_ok, n_fail
    new = pd.concat(frames, ignore_index=True)
    merge_sessions(new)
    return len(new), n_ok, n_fail


# ---------------------------------------------------------------- backup
def backup_sessions(keep=BACKUP_KEEP, verbose=True):
    """Salin session_bars.csv ke backup harian (rotasi `keep` file terakhir).

    File ini tidak di-commit ke git dan tidak bisa diregenerate setelah keluar
    jendela 60 hari Yahoo, jadi perlu backup terpisah.
    Lokasi bisa diubah via env SESSION_BACKUP_DIR.
    """
    if not os.path.exists(SESSION_CSV):
        return None
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        stamp = datetime.now(WIB).strftime("%Y%m%d")
        dst = os.path.join(BACKUP_DIR, f"session_bars_{stamp}.csv")
        if not os.path.exists(dst):
            shutil.copy2(SESSION_CSV, dst)
        files = sorted(f for f in os.listdir(BACKUP_DIR)
                       if f.startswith("session_bars_") and f.endswith(".csv"))
        for old in files[:-keep] if keep > 0 else []:
            try:
                os.remove(os.path.join(BACKUP_DIR, old))
            except OSError:
                pass
        if verbose:
            print(f"Backup sesi -> {dst} ({min(len(files), keep) if keep > 0 else len(files)} file disimpan)")
        return dst
    except OSError as e:
        if verbose:
            print(f"⚠️ Backup sesi gagal: {e}")
        return None
