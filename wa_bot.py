#!/usr/bin/env python3
"""
wa_bot.py
=========
Bot WhatsApp untuk prediksi saham IDX — via API wa.chatetin.com.

Alur:
  - Poll chat yang masuk (tiap N detik) dari nomor yang diizinkan (ALLOWED_NUMBERS)
  - Perintah "prediksi"/"top 20" -> kirim Top 20 saham berpotensi NAIK & TURUN besok
  - Balasan SELALU ke pengirim pesan (tidak pernah broadcast / kirim ke nomor random)

Cara pakai:
  1. cp .env.example .env  lalu isi kredensial chatetin + nomor yang diizinkan
  2. python wa_bot.py                # pakai prediksi yang sudah ada
     python wa_bot.py --refresh      # jalankan predict_daily.py dulu (retrain + prediksi baru)

Perintah di WhatsApp:
  - "prediksi" / "top 20"  -> Top 20 NAIK & Top 20 TURUN besok
  - "cek BBRI"             -> detail prediksi 1 saham
  - "help"                 -> menu bantuan

Keamanan:
  - Hanya merespon nomor di ALLOWED_NUMBERS (default: 6285720300059 utk test)
  - Semua aksi dicatat di data/wa_bot.log & state pesan di data/wa_bot_state.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

VERSION = "v0.10.0"

BASE = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(BASE, "data", "predictions_tomorrow.json")
STATE_FILE = os.path.join(BASE, "data", "wa_bot_state.json")
LOG_FILE = os.path.join(BASE, "data", "wa_bot.log")
ENV_FILE = os.path.join(BASE, ".env")

DEFAULT_ALLOWED = "6285720300059"
DEFAULT_INTERVAL = 3


# ---------------------------------------------------------------- util
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_env():
    """Parser .env sederhana (tanpa dependency eksternal)."""
    env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if k.startswith("CHATETIN_") or k in ("ALLOWED_NUMBERS", "POLL_INTERVAL"):
            env[k] = v
    return env


# ---------------------------------------------------------------- client
class ChatetinClient:
    """Klien minimal API wa.chatetin.com (login + kirim + baca pesan)."""

    def __init__(self, base_url, username, password):
        self.base = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token = None
        self.device_id = None
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "wa-bot-xgbooxt/1.0"})

    # -- auth ----------------------------------------------------------
    def login(self, force=False):
        if self.token and not force:
            return self.token
        r = self.s.post(f"{self.base}/auth/login", json={
            "username": self.username, "password": self.password,
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "SUCCESS":
            raise RuntimeError(f"login gagal: {data.get('message')}")
        self.token = data["results"]["token"]
        self.s.headers.update({"Authorization": f"Bearer {self.token}"})
        log("Login chatetin berhasil")
        return self.token

    def _request(self, method, path, **kw):
        """Request dengan retry + backoff exponensial pada 429/5xx."""
        attempts = 0
        while True:
            try:
                r = self.s.request(method, f"{self.base}{path}", timeout=30, **kw)
            except requests.RequestException as e:
                # token mungkin expire -> coba login ulang sekali
                attempts += 1
                log(f"Request error {path}: {e}; coba login ulang (attempt {attempts})")
                self.login(force=True)
                r = self.s.request(method, f"{self.base}{path}", timeout=30, **kw)
            if r.status_code == 401 and attempts < 2:
                attempts += 1
                self.login(force=True)
                continue
            if r.status_code in (429, 500, 502, 503) and attempts < 4:
                attempts += 1
                wait = 2 ** attempts + 1
                log(f"⚠️ {path}: HTTP {r.status_code} — backoff {wait}s "
                    f"(attempt {attempts}/4)")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()

    # -- devices -------------------------------------------------------
    def get_device(self):
        """Ambil device pertama yang sudah logged_in."""
        data = self._request("GET", "/devices")
        devices = data.get("results", [])
        for d in devices:
            if d.get("state") == "logged_in":
                self.device_id = d["id"]
                return d
        raise RuntimeError("Tidak ada device yang logged_in di akun chatetin")

    def register_webhook(self, url, events="message", secret=""):
        """Daftarkan webhook ke chatetin utk event tertentu."""
        if not self.device_id:
            self.get_device()
        data = self._request("PATCH", f"/devices/{self.device_id}/webhook",
                             json={"webhook_url": url, "webhook_events": events,
                                   "webhook_secret": secret,
                                   "webhook_insecure_skip_verify": False})
        return data

    # -- send ----------------------------------------------------------
    def send_message(self, phone, message):
        """phone: nomor atau jid penuh (62857...@s.whatsapp.net)."""
        if not phone.endswith("@s.whatsapp.net"):
            phone = f"{phone}@s.whatsapp.net"
        data = self._request("POST", "/send/message",
                             json={"phone": phone, "message": message})
        if data.get("code") != "SUCCESS":
            raise RuntimeError(f"kirim pesan gagal: {data.get('message')}")
        return data

    def send_typing(self, phone, action="start"):
        """Indikator mengetik (start/stop) — biar balasan terasa cepat.
        Opsional: kalau gagal, abaikan (tidak menggagalkan balasan)."""
        if not phone.endswith("@s.whatsapp.net"):
            phone = f"{phone}@s.whatsapp.net"
        try:
            self._request("POST", "/send/chat-presence",
                          json={"phone": phone, "action": action})
        except Exception:
            pass

    # -- read ----------------------------------------------------------
    def list_chats(self, limit=50):
        data = self._request("GET", "/chats", params={"limit": limit})
        return data.get("results", {}).get("data", [])

    def chat_messages(self, jid, limit=20):
        from urllib.parse import quote
        data = self._request("GET", f"/chat/{quote(jid, safe='')}/messages",
                             params={"limit": limit})
        return data.get("results", {}).get("data", [])


# ---------------------------------------------------------------- formatter
import re


# Batas likuiditas minimal agar saham layak direkomendasikan
MIN_VALUE_TRADED = 1_000_000_000   # Rp1 miliar / hari (nilai transaksi)
MIN_PRICE = 200                    # Rp200 (hindari penny stock ekstrem)

TRACK_FILE = os.path.join(BASE, "data", "bot_track_record.csv")
HIST_DIR = os.path.join(BASE, "data", "history")
MACRO_CSV = os.path.join(BASE, "data", "macro_id.csv")


def load_predictions():
    """Baca data prediksi terbaru. Return (payload, error)."""
    if not os.path.exists(PRED_FILE):
        return None, ("Belum ada file prediksi. Jalankan dulu: "
                      "`python predict_daily.py` lalu jalankan bot lagi.")
    try:
        with open(PRED_FILE) as f:
            return json.load(f), None
    except (json.JSONDecodeError, OSError) as e:
        return None, f"Gagal membaca prediksi: {e}"


def is_liquid(item, min_value=MIN_VALUE_TRADED, min_price=MIN_PRICE):
    """Saham layak rekomendasi: nilai transaksi & harga di atas ambang."""
    val = item.get("value_traded") or 0
    price = item.get("close") or 0
    return val >= min_value and price >= min_price


def liquid_saham(payload, min_value=MIN_VALUE_TRADED, min_price=MIN_PRICE):
    """Daftar saham yang lolos filter likuiditas (urutan prob turun)."""
    return [x for x in payload["saham"] if is_liquid(x, min_value, min_price)]


def format_short(item, arrow=True):
    """Satu baris ringkas untuk daftar."""
    prob = item["prob_up"] * 100
    name = item.get("description") or item["ticker"]
    if len(name) > 42:
        name = name[:40] + ".."
    return f"{item['rank']:>3d}. {item['ticker']:<6s} {prob:5.1f}%  {name}"


def format_top(payload, n=20):
    """Top N NAIK + Top N TURUN — HANYA saham likuid, dalam SATU pesan."""
    saham = payload["saham"]
    liq = liquid_saham(payload)
    n = max(1, min(int(n), len(liq)))
    tanggal = payload.get("tanggal_prediksi", "?")
    auc = payload.get("valid_auc", 0)
    jumlah = payload.get("jumlah_saham", len(saham))

    top = liq[:n]
    bottom = list(reversed(liq[-n:]))
    filtered = len(saham) - len(liq)

    lines = [
        "📊 *PREDIKSI SAHAM IDX — BESOK*",
        f"📅 Prediksi untuk: {tanggal}",
        f"🤖 XGBoost gabungan | {len(liq)} saham likuid (dari {jumlah}) | AUC {auc:.3f}",
        "————————————————",
        f"🟢 *TOP {n} POTENSI NAIK ▲ (likuid)*",
    ] + [format_short(x) for x in top] + [
        "————————————————",
        f"🔴 *TOP {n} POTENSI TURUN ▼ (likuid)*",
    ] + [format_short(x) for x in bottom] + [
        "————————————————",
    ]
    if filtered:
        lines.append(f"🔒 {filtered} saham illikuid/penny dikeluarkan (nilai < Rp1M atau harga < Rp200)")
    lines.append("⚠️ Bukan saran investasi. AUC ~0.58 = sinyal lemah, gunakan bijak.")
    return "\n".join(lines)


def format_single(payload, ticker):
    """Detail prediksi satu saham."""
    ticker = ticker.upper().replace(".JK", "")
    for x in payload["saham"]:
        if x["ticker"].upper() == ticker:
            p = x["prob_up"] * 100
            conf = (x.get("confidence") or 0) * 100 or max(p, 100 - p)
            ind = x.get("industry") or ""
            sektor = x.get("sector") or "-"
            if ind:
                sektor = f"{sektor} | {ind}"
            val = (x.get("value_traded") or 0) / 1e9
            liq_note = (f"💧 Nilai transaksi: Rp {val:,.2f} M/hari"
                        f"\n🔒 *LIKUID* — layak diperdagangkan" if is_liquid(x)
                        else f"💧 Nilai transaksi: Rp {val:,.3f} M/hari"
                        f"\n⚠️ *ILLIKUID/penny* — sulit dieksekusi, hati-hati")
            # foreign flow (net asing) dari idx.co.id
            fnv = x.get("foreign_net_value")
            if fnv is not None:
                fb = x.get("foreign_buy") or 0
                fs = x.get("foreign_sell") or 0
                fnv_m = fnv / 1e9
                emoji = "🟢" if fnv_m >= 0 else "🔴"
                asing_line = (f"\n🌍 Asing: beli {fb:,.0f} / jual {fs:,.0f} lembar "
                              f"| net {fnv_m:+,.1f} M {emoji}")
            else:
                asing_line = ""
            # perkiraan return + ukuran saran (dari model regresi)
            pr = x.get("pred_ret")
            ret_line = f"\n📈 Perkiraan return besok: {pr * 100:+.2f}%" if pr is not None else ""
            size = x.get("sizing")
            size_line = f"\n⚖️ Ukuran saran (long): *{size}*" if size and size != "—" else ""
            # rekam jejak prediksi NAIK utk saham ini
            hit, total = ticker_track(x["ticker"])
            if total >= 3:
                track_line = f"\n🎯 Rekam jejak NAIK: {hit}/{total} benar ({hit / total * 100:.0f}%)"
            elif total > 0:
                track_line = f"\n🎯 Rekam jejak NAIK: {hit}/{total} (data kurang)"
            else:
                track_line = ""
            return (
                f"📊 *{x['ticker']}* — {x.get('description', '-')}\n"
                f"🏭 Sektor: {sektor}\n"
                f"💵 Close terakhir: Rp {x['close']:,.0f}\n"
                f"{liq_note}{asing_line}\n"
                f"🎯 Probabilitas NAIK besok: {p:.1f}%\n"
                f"📈 Sinyal: {x['signal']}{ret_line}{size_line}\n"
                f"🎚 Confidence: {conf:.1f}%{track_line}\n"
                f"📅 Prediksi: {payload.get('tanggal_prediksi', '?')}\n"
                f"💡 Ketik `kenapa {x['ticker']}` utk penjelasan sinyal.\n"
                f"⚠️ Bukan saran investasi."
            ), None
    return None, f"❌ Kode *{ticker}* tidak ditemukan. Contoh: `cek BBRI`"


# ---------------------------------------------------------------- fitur baru
FEATURE_LABELS = {
    "ret_1": "return 1 hari", "ret_2": "return 2 hari", "ret_3": "return 3 hari",
    "ret_5": "return 5 hari", "ret_10": "return 10 hari", "ret_20": "return 20 hari",
    "log_ret_1": "log-return 1 hari", "log_ret_5": "log-return 5 hari",
    "close_sma5": "harga vs SMA5", "close_sma10": "harga vs SMA10",
    "close_sma20": "harga vs SMA20", "close_sma50": "harga vs SMA50",
    "close_sma100": "harga vs SMA100", "sma20_sma50": "SMA20 vs SMA50",
    "close_ema12": "harga vs EMA12", "close_ema26": "harga vs EMA26",
    "macd_c": "MACD (ternormalisasi)", "macd_signal_c": "MACD signal",
    "macd_hist_c": "MACD histogram",
    "rsi_14": "RSI 14", "bb_pct_b": "Bollinger %B", "bb_width": "lebar Bollinger",
    "vol_5": "volatilitas 5 hari", "vol_20": "volatilitas 20 hari",
    "atr_ratio": "ATR / harga",
    "vol_ratio_5": "volume vs rata2 5 hari", "vol_ratio_20": "volume vs rata2 20 hari",
    "vol_change": "perubahan volume",
    "hl_range": "range high-low", "gap": "gap pembukaan",
    "dist_52w_high": "jarak dari 52w high", "dist_52w_low": "jarak dari 52w low",
    "dayofweek": "hari dalam pekan", "month": "bulan",
    "ihsg_ret_1": "IHSG 1 hari", "ihsg_ret_5": "IHSG 5 hari", "ihsg_ret_20": "IHSG 20 hari",
    "ihsg_sma20": "IHSG vs SMA20",
    "usd_ret_1": "USD/IDR 1 hari", "usd_ret_5": "USD/IDR 5 hari",
    "usd_ret_20": "USD/IDR 20 hari", "usd_sma20": "USD/IDR vs SMA20",
    "sector_ret_1": "return sektor (median)", "rs_sector_1": "outperform sektor 1 hari",
    "rs_sector_5": "outperform sektor 5 hari",
    "ff_buy_ratio": "beli asing / volume", "ff_sell_ratio": "jual asing / volume",
    "ff_net_ratio": "net asing / volume",
}


def format_kenapa(payload, ticker):
    """Jelaskan SINYAL satu saham via SHAP: fitur apa yang mendorong probabilitas."""
    ticker = ticker.upper().replace(".JK", "")
    item = next((x for x in payload["saham"] if x["ticker"].upper() == ticker), None)
    if not item:
        return None, f"❌ Kode *{ticker}* tidak ditemukan. Contoh: `kenapa BBRI`"
    try:
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(BASE, "data", "model_daily.ubj"))
        cols = json.load(open(os.path.join(BASE, "data", "model_daily.json")))["feature_cols"]
        feat = pd.read_parquet(os.path.join(BASE, "data", "last_features.parquet"))
        row = feat[feat["ticker"] == ticker]
        if row.empty:
            return None, f"❌ Data fitur *{ticker}* tidak ditemukan."
        X = row[cols].astype(float)
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
        total = contrib.sum()
        # kontribusi tiap fitur ke PROBABILITAS (gaya force plot)
        vals = {}
        for i, c in enumerate(contrib[:-1]):
            without = total - contrib[i]
            vals[cols[i]] = 1 / (1 + np.exp(-total)) - 1 / (1 + np.exp(-without))
        pos = sorted(((v, k) for k, v in vals.items()), reverse=True)[:3]
        neg = sorted(((v, k) for k, v in vals.items()))[:3]

        def fmt(v, k):
            label = FEATURE_LABELS.get(k, k)
            return f"  {label}: {v * 100:+.1f}%"

        lines = [
            f"🔍 *KENAPA {ticker} → {item['signal']}*",
            f"🎯 Probabilitas model: {item['prob_up'] * 100:.1f}%",
            "————————————————",
            "🔼 *Mendorong NAIK:*",
        ] + [fmt(v, k) for v, k in pos if abs(v) > 0.001] + [
            "————————————————",
            "🔽 *Mendorong TURUN:*",
        ] + [fmt(v, k) for v, k in neg if abs(v) > 0.001] + [
            "————————————————",
            "💡 Kontribusi = perubahan probabilitas bila fitur itu dihilangkan.",
            "⚠️ Bukan saran investasi.",
        ]
        return "\n".join(lines), None
    except Exception as e:
        return None, f"⚠️ Gagal menghitung penjelasan: {e}"


def format_rekap(payload):
    """Rekap pasar: IHSG, breadth, top gainers/losers (hanya saham likuid)."""
    import pandas as pd
    macro = None
    if os.path.exists(MACRO_CSV):
        try:
            macro = pd.read_csv(MACRO_CSV, parse_dates=["tanggal"])
        except Exception:
            macro = None
    liq = liquid_saham(payload)
    up = sum(1 for x in liq if (x.get("ret_1") or 0) > 0)
    dn = len(liq) - up
    val = sum((x.get("value_traded") or 0) for x in liq)

    lines = [
        "📊 *REKAP PASAR IDX*",
        f"📅 Data s/d {payload.get('data_sampai', '?')} | "
        f"Prediksi {payload.get('tanggal_prediksi', '?')}",
        "———————————————",
    ]
    if macro is not None and not macro.empty:
        m = macro.iloc[-1]
        ihsg = m.get("ihsg")
        if ihsg and not pd.isna(ihsg):
            r1 = (m.get("ihsg_ret_1") or 0) * 100
            r5 = (m.get("ihsg_ret_5") or 0) * 100
            emoji = "🟢" if r1 >= 0 else "🔴"
            lines.append(f"📈 IHSG: {ihsg:,.0f} {emoji} ({r1:+.2f}% 1h | {r5:+.2f}% 5h)")
        usd = m.get("usd_idr")
        if usd and not pd.isna(usd):
            lines.append(f"💵 USD/IDR: {usd:,.0f}")
    nf = payload.get("net_foreign") or {}
    if nf.get("net_value") is not None:
        nv = nf["net_value"] / 1e9
        emoji = "🟢" if nv >= 0 else "🔴"
        lines.append(f"🌍 *Net Asing*: {nv:+,.0f} M {emoji} "
                     f"(tanggal {nf.get('tanggal', '?')})")
    lines.append(f"🏦 Breadth (saham likuid): {up} naik 🟢 / {dn} turun 🔴")
    lines.append(f"💵 Nilai transaksi likuid: Rp {val / 1e12:,.2f} T")
    lines.append("———————————————")

    g = sorted((x for x in liq if (x.get("ret_1") or 0) != 0),
               key=lambda x: -(x.get("ret_1") or 0))[:5]
    l = sorted((x for x in liq if (x.get("ret_1") or 0) != 0),
               key=lambda x: (x.get("ret_1") or 0))[:5]
    lines.append("🟢 *Top gainers (likuid):*")
    lines += [f"  {x['ticker']:<6s} {(x.get('ret_1') or 0) * 100:+6.2f}%  "
              f"{(x.get('description') or '')[:35]}" for x in g]
    lines.append("🔴 *Top losers (likuid):*")
    lines += [f"  {x['ticker']:<6s} {(x.get('ret_1') or 0) * 100:+6.2f}%  "
              f"{(x.get('description') or '')[:35]}" for x in l]
    lines += ["———————————————",
              "📌 Prediksi besok: `prediksi` | penjelasan: `kenapa KODE`",
              "⚠️ Bukan saran investasi."]
    return "\n".join(lines), None


def format_riwayat(payload, ticker):
    """Riwayat prediksi satu saham (dari rekam jejak terverifikasi)."""
    import pandas as pd
    ticker = ticker.upper().replace(".JK", "")
    if not os.path.exists(TRACK_FILE):
        return None, "📭 Belum ada rekam jejak terverifikasi (butuh beberapa hari berjalan)."
    try:
        df = pd.read_csv(TRACK_FILE)
    except Exception:
        return None, "⚠️ Gagal membaca rekam jejak."
    sub = df[df["ticker"] == ticker]
    if sub.empty:
        return None, f"❌ Belum ada riwayat utk *{ticker}*."
    n_total = len(df[df["ticker"] == ticker])
    sub = sub.sort_values("tanggal_prediksi", ascending=False).head(10)
    lines = [f"📜 *RIWAYAT {ticker}*",
             f"({n_total} prediksi tercatat, 10 terakhir)"]
    for _, r in sub.iterrows():
        arah = "NAIK ▲" if int(r["arah"]) == 1 else "TURUN ▼"
        akt = "naik" if int(r["aktual_naik"]) == 1 else "turun"
        ok = (int(r["arah"]) == 1 and int(r["aktual_naik"]) == 1) or \
             (int(r["arah"]) == 0 and int(r["aktual_naik"]) == 0)
        mark = "✅" if ok else "❌"
        lines.append(
            f"  {r['tanggal_prediksi']} {arah} (prob {r['prob_up']:.2f})\n"
            f"     aktual {akt} {mark}")
    hit = (sub["aktual_naik"].astype(int) == sub["arah"].astype(int)).mean()
    lines.append(f"  Akurasi {len(sub)} prediksi terakhir: {hit * 100:.0f}%")
    lines.append("⚠️ Bukan saran investasi.")
    return "\n".join(lines), None


# ---------------------------------------------------------------- verifikasi harian
ARCHIVE = os.path.join(BASE, "data", "prediction_archive.csv")

# ---------------------------------------------------------------- allowed numbers (admin)
ALLOWED_FILE = os.path.join(BASE, "data", "allowed_numbers.json")
BOT_ADMINS = "6285720300059,6285780535433"   # bisa diubah via .env BOT_ADMINS


def load_admins(env=None):
    """Nomor admin yang boleh mengelola daftar user (dari .env BOT_ADMINS)."""
    raw = (env or {}).get("BOT_ADMINS", "") or BOT_ADMINS
    return {n.strip() for n in raw.split(",") if n.strip()}


def load_allowed_numbers():
    """Nomor yang diizinkan chat ke bot (file JSON dinamis).
    Kalau file belum ada -> seed dari .env ALLOWED_NUMBERS."""
    if not os.path.exists(ALLOWED_FILE):
        # seed sekali dari env (utk kompatibilitas konfigurasi lama)
        env = load_env()
        raw = env.get("ALLOWED_NUMBERS", DEFAULT_ALLOWED)
        nums = {n.strip() for n in raw.split(",") if n.strip()}
        save_allowed_numbers(nums)
        return nums | load_admins(env)
    try:
        with open(ALLOWED_FILE) as f:
            return set(json.load(f).get("allowed", [])) | load_admins()
    except (OSError, json.JSONDecodeError):
        return {DEFAULT_ALLOWED} | load_admins()


def save_allowed_numbers(nums):
    with open(ALLOWED_FILE, "w") as f:
        json.dump({"allowed": sorted(set(nums))}, f, indent=1)


def _validate_phone(num):
    """Validasi nomor WA Indonesia: 62 + 8-12 digit."""
    num = num.strip().replace(" ", "").replace("-", "")
    if num.startswith("+"):
        num = num[1:]
    if num.startswith("08"):
        num = "62" + num[1:]
    if num.startswith("8"):
        num = "62" + num
    return num if re.fullmatch(r"62\d{8,12}", num) else None


# ---------------------------------------------------------------- watchlist
WATCH_FILE = os.path.join(BASE, "data", "watchlists.json")
WATCH_REPORT_TIME = "18:00"   # default push otomatis (bisa diubah via .env)


def _watch_num(jid):
    """jid/phone -> nomor user (62xxxxxxxxxx)."""
    return (jid or "").split("@")[0]


def load_watchlists():
    """Semua watchlist per-user: {nomor: [ticker, ...]}"""
    try:
        with open(WATCH_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_watchlist(number):
    return sorted(load_watchlists().get(_watch_num(number), []))


def save_watchlist(number, tickers):
    allw = load_watchlists()
    allw[_watch_num(number)] = sorted(set(tickers))
    with open(WATCH_FILE, "w") as f:
        json.dump(allw, f, indent=1)


def _parse_tickers(text):
    """Ubah 'TLKM,BBRI, ASII' -> ['TLKM','BBRI','ASII'] (valid saja)."""
    out = []
    for t in re.split(r"[,;\s]+", text):
        t = t.strip().upper().replace(".JK", "")
        if t and re.fullmatch(r"[A-Z0-9.]{2,6}", t) and t not in out:
            out.append(t)
    return out


def ticker_track(ticker):
    """Rekam jejak per saham utk prediksi NAIK: (benar, total)."""
    if not os.path.exists(TRACK_FILE):
        return 0, 0
    try:
        import pandas as pd
        df = pd.read_csv(TRACK_FILE)
        sub = df[(df["ticker"] == ticker.upper()) & (df["arah"] == 1)]
        if sub.empty:
            return 0, 0
        return int((sub["aktual_naik"] == 1).sum()), len(sub)
    except Exception:
        return 0, 0


def format_watch_report(payload, number):
    """Laporan status watchlist milik satu user -> (teks, error)."""
    watch = load_watchlist(number)
    if not watch:
        return None, ("📭 Watchlist masih kosong. Ketik: `watch TLKM,BBRI,ASII` "
                      "untuk menetapkan.")
    saham = {x["ticker"].upper(): x for x in payload["saham"]}
    lines = [
        "📊 *LAPORAN WATCHLIST*",
        f"📅 Data s/d {payload.get('data_sampai', '?')} | "
        f"Prediksi {payload.get('tanggal_prediksi', '?')}",
        "———————————————",
    ]
    for tk in watch:
        x = saham.get(tk)
        if not x:
            lines.append(f"⚪ *{tk}* — tidak ada di data prediksi")
            continue
        p = x["prob_up"] * 100
        emoji = "🟢" if str(x["signal"]).startswith("NAIK") else "🔴"
        liq = "LIKUID" if is_liquid(x) else "ILLIKUID ⚠️"
        hit, total = ticker_track(tk)
        track = f"{hit}/{total} benar" if total >= 3 else "data kurang"
        lines.append(
            f"{emoji} *{x['ticker']}* — Rp {x['close']:,.0f}\n"
            f"   Besok: {x['signal']} {p:.1f}% | {liq}\n"
            f"   Rekam: {track}"
        )
    lines += ["———————————————", "⚠️ Bukan saran investasi."]
    return "\n".join(lines), None


def report_worker(client, report_time, recap_time):
    """Thread: kirim laporan watchlist & rekap pasar otomatis harian.
    Setiap user menerima laporan watchlist-nya MASING-MASING; rekap pasar
    dikirim ke SEMUA user diizinkan (daftar dibaca dinamis)."""
    last_watch = ""
    last_recap = ""
    while True:
        try:
            now = datetime.now().strftime("%H:%M")
            today = datetime.now().strftime("%Y-%m-%d")
            if report_time and now >= report_time and last_watch != today:
                payload, err = load_predictions()
                if not err:
                    for num in load_allowed_numbers():
                        if load_watchlist(num):
                            text, _ = format_watch_report(payload, num)
                            if text:
                                client.send_message(f"{num}@s.whatsapp.net", text)
                                log(f"Laporan watchlist ({report_time}) -> {num}")
                last_watch = today
            if recap_time and now >= recap_time and last_recap != today:
                payload, err = load_predictions()
                if not err:
                    text, _ = format_rekap(payload)
                    if text:
                        for num in load_allowed_numbers():
                            client.send_message(f"{num}@s.whatsapp.net", text)
                        log(f"Rekap pasar otomatis ({recap_time}) -> semua user")
                last_recap = today
        except Exception as e:
            log(f"⚠️ Laporan otomatis gagal: {e}")
        time.sleep(45)


def _read_history_for(tickers):
    """Baca close utk ticker tertentu — prioritas parquet (cepat), fallback CSV."""
    try:
        import pandas as pd
        pq = os.path.join(BASE, "data", "history_id_5y.parquet")
        if os.path.exists(pq):
            h = pd.read_parquet(pq, columns=["ticker", "tanggal", "close"])
            h = h[h["ticker"].isin(tickers)]
            return h
    except Exception:
        pass
    frames = []
    for tk in tickers:
        path = os.path.join(HIST_DIR, f"{tk}.csv")
        if os.path.exists(path):
            try:
                frames.append(pd.read_csv(path, usecols=["ticker", "tanggal", "close"],
                                          parse_dates=["tanggal"]))
            except Exception:
                pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def verify_and_record(payload=None):
    """Verifikasi prediksi lama (dari arsip) vs aktual, catat ke TRACK_FILE.
    Prediksi terverifikasi: tanggal_prediksi sudah lewat & ada close aktual."""
    import pandas as pd
    if not os.path.exists(ARCHIVE):
        return summarize_track()
    try:
        arch = pd.read_csv(ARCHIVE, dtype={"ticker": str})
        if arch.empty:
            return summarize_track()
        arch["tanggal_prediksi"] = pd.to_datetime(arch["tanggal_prediksi"])
        arch["data_sampai"] = pd.to_datetime(arch["data_sampai"])
    except Exception as e:
        log(f"Baca arsip gagal: {e}")
        return summarize_track()

    # sudah terverifikasi sebelumnya?
    done = set()
    if os.path.exists(TRACK_FILE):
        try:
            old = pd.read_csv(TRACK_FILE)
            done = set(zip(old["tanggal_prediksi"], old["ticker"]))
        except Exception:
            done = set()

    # baris yang belum diverifikasi & tanggalnya sudah lewat
    todo = arch[~arch.apply(lambda r: (str(r.tanggal_prediksi.date()), r.ticker) in done,
                            axis=1)].copy()
    if todo.empty:
        return summarize_track()

    hist = _read_history_for(todo["ticker"].unique().tolist())
    if hist is None or hist.empty:
        return summarize_track()
    hist = hist.sort_values(["ticker", "tanggal"])

    new_rows = []
    for r in todo.itertuples():
        h = hist[hist["ticker"] == r.ticker]
        if h.empty:
            continue
        # close di hari data (c0) dan hari trading BERIKUTNYA (c1)
        hh = h[h["tanggal"] <= r.data_sampai]
        if hh.empty:
            continue
        c0_date, c0 = hh.iloc[-1]["tanggal"], hh.iloc[-1]["close"]
        nxt = h[h["tanggal"] > c0_date]
        if nxt.empty:
            continue  # belum ada hari berikutnya (prediksi belum jatuh tempo)
        c1_date, c1 = nxt.iloc[0]["tanggal"], nxt.iloc[0]["close"]
        # hanya verifikasi kalau hari berikutnya sudah lewat (data nyata)
        if c1_date > pd.Timestamp.today():
            continue
        aktual = 1 if c1 > c0 else 0
        new_rows.append({
            "tanggal_prediksi": str(r.tanggal_prediksi.date()),
            "data_sampai": str(r.data_sampai.date()),
            "ticker": r.ticker, "arah": int(r.arah),
            "prob_up": round(float(r.prob_up), 4),
            "aktual_naik": aktual,
        })

    if new_rows:
        df = pd.DataFrame(new_rows)
        if os.path.exists(TRACK_FILE):
            old = pd.read_csv(TRACK_FILE)
            df = pd.concat([old, df]).drop_duplicates(
                subset=["tanggal_prediksi", "ticker"], keep="last")
        df.to_csv(TRACK_FILE, index=False)
        log(f"Verifikasi: {len(new_rows)} prediksi baru tercatat (total {len(df)})")
    return summarize_track()


def summarize_track():
    """Rekam jejak: berapa % rekomendasi top-20 NAIK yang benar-benar naik."""
    if not os.path.exists(TRACK_FILE):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(TRACK_FILE)
        if df.empty:
            return None
        up = df[(df["arah"] == 1) & (df["aktual_naik"].notna())]
        dn = df[(df["arah"] == 0) & (df["aktual_naik"].notna())]
        if up.empty and dn.empty:
            return None
        days = df["tanggal_prediksi"].nunique()
        s = ["🎯 *Rekam jejak bot*"]
        if not up.empty:
            s.append(f"  NAIK ▲: {(up['aktual_naik']==1).sum()}/{len(up)} benar ({up['aktual_naik'].mean()*100:.0f}%)")
        if not dn.empty:
            s.append(f"  TURUN ▼: {((dn['aktual_naik']==0)).sum()}/{len(dn)} benar ({((dn['aktual_naik']==0)).mean()*100:.0f}%)")
        s.append(f"  ({days} hari terverifikasi, s/d {df['tanggal_prediksi'].max()})")
        return "\n".join(s)
    except Exception:
        return None


def check_freshness(payload):
    """Warnai kalau data prediksi sudah lama."""
    try:
        t = datetime.strptime(payload["tanggal_prediksi"], "%Y-%m-%d")
        if t.date() < datetime.now().date() - timedelta(days=2):
            return (f"\n\n⚠️ *Data prediksi mungkin sudah usang* "
                    f"(untuk {payload.get('tanggal_prediksi')}). "
                    f"Jalankan `python predict_daily.py --refresh` lalu restart bot.")
    except (KeyError, ValueError):
        pass
    return ""


# ---------------------------------------------------------------- handler
def parse_command(content):
    """Parse perintah -> (cmd, arg). cmd: top|cek|help, atau None kalau bukan perintah.
    Didukung: prediksi / top 20 / top N / prediksi top N / cek KODE / prediksi KODE / help."""
    low = (content or "").strip().lower()
    if not low:
        return None
    # daftar penuh
    if low in ("prediksi", "prediksi harian", "prediksi hari ini",
               "rekomendasi", "rekomendasi harian", "saham", "list",
               "top", "top 20", "top20"):
        return ("top", 20)
    # top N / prediksi top N
    m = re.match(r"^(?:prediksi|rekomendasi)\s+top\s*(\d+)$", low)
    if m:
        return ("top", int(m.group(1)))
    m = re.match(r"^top\s*(\d+)$", low)
    if m:
        return ("top", int(m.group(1)))
    # cek KODE / prediksi KODE / rekomendasi KODE
    m = re.match(r"^(?:cek|/cek|cari|prediksi|rekomendasi)\s+([a-z0-9.]+)$", low)
    if m:
        return ("cek", m.group(1))
    # admin: kelola daftar user yang diizinkan (hanya utk nomor admin)
    m = re.match(r"^(?:add user|tambah user|izin|tambah nomor|daftarkan)\s+(.+)$", low)
    if m:
        return ("admin_add", m.group(1))
    m = re.match(r"^(?:hapus user|remove user|cabut|hapus nomor)\s+(.+)$", low)
    if m:
        return ("admin_del", m.group(1))
    if low in ("daftar user", "list user", "users", "daftar nomor",
               "list nomor", "daftar allowed"):
        return ("admin_list", None)
    # watchlist: watch TLKM,BBRI / tambah / hapus / lapor
    m = re.match(r"^(?:watch|pantau|set watch|set pantau)\s+(.+)$", low)
    if m:
        return ("watch", m.group(1))
    if low in ("watch", "pantau", "watchlist"):
        return ("watch_show", None)
    m = re.match(r"^(?:tambah|add|tambah watch)\s+(.+)$", low)
    if m:
        return ("watch_add", m.group(1))
    m = re.match(r"^(?:hapus|remove|unwatch|delete)\s+(.+)$", low)
    if m:
        return ("watch_del", m.group(1))
    if low in ("lapor", "report", "laporan"):
        return ("lapor", None)
    # update data: update / refresh / tarik data / ambil data
    if low in ("update", "update data", "refresh", "refresh data",
               "update data saham", "tarik data", "ambil data",
               "perbarui", "perbarui data", "update prediksi",
               "refresh prediksi"):
        return ("refresh", None)
    # kenapa KODE / why KODE / alasan KODE — penjelasan sinyal (SHAP)
    m = re.match(r"^(?:kenapa|why|alasan|analisa|analisis)\s+([a-z0-9.]+)$", low)
    if m:
        return ("kenapa", m.group(1))
    # riwayat KODE / track KODE — rekam jejak historis satu saham
    m = re.match(r"^(?:riwayat|track|history|histori|rekam)\s+([a-z0-9.]+)$", low)
    if m:
        return ("riwayat", m.group(1))
    # rekap pasar
    if low in ("rekap", "rekap pasar", "ringkasan", "ringkasan pasar",
               "market recap", "recap", "rekap harian"):
        return ("rekap", None)
    if low in ("help", "bantuan", "menu"):
        return ("help", None)
    if low in ("versi", "version", "versi bot", "cek versi"):
        return ("versi", None)
    return None


HELP_TEXT = (
    "🤖 *Bot Prediksi Saham IDX*\n"
    "Perintah yang tersedia:\n"
    "• `prediksi` / `top 20` — Top 20 potensi NAIK & TURUN besok (saham LIKUID saja)\n"
    "• `top 5` / `top 10` — Top N sesuai angka\n"
    "• `cek BBRI` — detail 1 saham (likuiditas + probabilitas + ukuran saran)\n"
    "• `kenapa BBRI` — penjelasan sinyal (fitur apa yg mendorong naik/turun)\n"
    "• `rekap` — ringkasan pasar (IHSG, Net Asing, gainers/losers, breadth)\n"
    "• `riwayat BBRI` — rekam jejak prediksi historis saham itu\n"
    "• `watch TLKM,BBRI` — set watchlist saham yang dipantau\n"
    "• `tambah TLKM` / `hapus TLKM` — ubah watchlist\n"
    "• `lapor` — laporan status semua saham watchlist\n"
    "• `update` / `refresh` — ambil data terbaru + retrain (jika data basi)\n"
    "• `versi` — info versi bot & model\n"
    "• `help` — menu ini\n\n"
    "👑 *Perintah admin* (hanya nomor admin):\n"
    "• `tambah user 628xxxx` / `hapus user 628xxxx` / `daftar user`\n\n"
    "🔔 Laporan watchlist otomatis dikirim tiap hari (lihat WATCH_REPORT_TIME di .env).\n"
    "🔒 Saham illikuid/penny dikeluarkan otomatis dari daftar.\n"
    "⚠️ Hasil bukan saran investasi."
)


def format_version(payload):
    """Info versi bot + model — utk verifikasi cepat apakah sudah update."""
    git = ""
    try:
        r = subprocess.run(["git", "-C", BASE, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            git = f" ({r.stdout.strip()})"
    except Exception:
        pass
    lines = [
        f"🤖 *Bot Prediksi Saham IDX* — {VERSION}{git}",
        f"🧠 Model: {payload.get('model', '?')} | {payload.get('fitur', '?')} fitur | "
        f"AUC {payload.get('valid_auc', 0):.4f}",
        f"📅 Prediksi utk: {payload.get('tanggal_prediksi', '?')} | "
        f"Data s/d: {payload.get('data_sampai', '?')}",
        f"📊 {payload.get('jumlah_saham', '?')} saham | threshold {payload.get('threshold_optimal', 0.3)}",
        "⚠️ Bukan saran investasi.",
    ]
    return "\n".join(lines)


def handle_watch(client, cmd, jid):
    """Kelola watchlist per-user: watch / watch_show / watch_add / watch_del."""
    kind, arg = cmd
    if kind == "watch":
        tks = _parse_tickers(arg)
        if not tks:
            client.send_message(jid, "Format: `watch TLKM,BBRI,ASII` "
                                      "(kode saham dipisah koma)")
        else:
            save_watchlist(jid, tks)
            client.send_message(jid,
                f"📌 Watchlist disetel: {', '.join(tks)}\n"
                f"Ketik `lapor` utk laporan status, atau `tambah`/`hapus` utk ubah.")
            log(f"-> {jid}: watch set {tks}")
    elif kind == "watch_show":
        w = load_watchlist(jid)
        if w:
            client.send_message(jid,
                f"📌 Watchlist kamu: {', '.join(w)}\n"
                f"(`watch A,B,C` utk ganti total, `lapor` utk laporan)")
        else:
            client.send_message(jid,
                "📭 Watchlist kosong. Ketik `watch TLKM,BBRI` untuk menetapkan.")
        log(f"-> {jid}: watch_show")
    elif kind == "watch_add":
        tks = _parse_tickers(arg)
        if not tks:
            client.send_message(jid, "Format: `tambah TLKM,BBRI`")
            return
        new = sorted(set(load_watchlist(jid) + tks))
        save_watchlist(jid, new)
        client.send_message(jid,
            f"➕ Ditambahkan: {', '.join(tks)}\n"
            f"📌 Watchlist kamu: {', '.join(new)}")
        log(f"-> {jid}: watch add {tks}")
    elif kind == "watch_del":
        tks = _parse_tickers(arg)
        cur = load_watchlist(jid)
        removed = [t for t in tks if t in cur]
        new = [t for t in cur if t not in tks]
        save_watchlist(jid, new)
        client.send_message(jid,
            f"➖ Dihapus: {', '.join(removed) if removed else 'tidak ada'}\n"
            f"📌 Watchlist kamu: {', '.join(new) if new else '(kosong)'}")
        log(f"-> {jid}: watch del {removed}")


def handle_command(client, msg, env):
    """Proses satu pesan masuk -> kirim balasan ke pengirim."""
    import time as _t
    t0 = _t.time()
    content = (msg.get("content") or "").strip()
    jid = msg.get("chat_jid") or msg.get("sender_jid")
    if not content or not jid:
        return
    cmd = parse_command(content)
    if not cmd:
        return

    # perintah watchlist tidak butuh file prediksi
    if cmd[0] in ("watch", "watch_show", "watch_add", "watch_del"):
        handle_watch(client, cmd, jid)
        return

    payload, err = load_predictions()
    if err:
        client.send_message(jid, f"⚠️ {err}")
        log(f"-> {jid}: prediksi file error")
        return
    stale = check_freshness(payload)

    if cmd[0] == "lapor":
        text, err = format_watch_report(payload, jid)
        if err:
            client.send_message(jid, err)
        else:
            client.send_message(jid, text + stale)
        log(f"-> {jid}: lapor ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "top":
        n = cmd[1]
        text = format_top(payload, n) + stale
        track = summarize_track()
        if track:
            text += "\n\n" + track
        client.send_message(jid, text)
        log(f"-> {jid}: top {n} dikirim ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "cek":
        text, err = format_single(payload, cmd[1])
        if err:
            client.send_message(jid, err)
        else:
            client.send_message(jid, text + stale)
        log(f"-> {jid}: cek {cmd[1]} ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "kenapa":
        text, err = format_kenapa(payload, cmd[1])
        if err:
            client.send_message(jid, err)
        else:
            client.send_message(jid, text)
        log(f"-> {jid}: kenapa {cmd[1]} ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "rekap":
        text, err = format_rekap(payload)
        if err:
            client.send_message(jid, err)
        else:
            client.send_message(jid, text + stale)
        log(f"-> {jid}: rekap ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "riwayat":
        text, err = format_riwayat(payload, cmd[1])
        if err:
            client.send_message(jid, err)
        else:
            client.send_message(jid, text)
        log(f"-> {jid}: riwayat {cmd[1]} ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "help":
        client.send_message(jid, HELP_TEXT)
        log(f"-> {jid}: help dikirim ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "versi":
        client.send_message(jid, format_version(payload))
        log(f"-> {jid}: versi dikirim ({_t.time()-t0:.1f}s)")

    elif cmd[0] == "refresh":
        handle_refresh(client, jid, payload)
        log(f"-> {jid}: update data diminta ({_t.time()-t0:.1f}s)")

    elif cmd[0] in ("admin_add", "admin_del", "admin_list"):
        handle_admin(client, cmd, jid, env)
        log(f"-> {jid}: admin {cmd[0]} ({_t.time()-t0:.1f}s)")


# ---------------------------------------------------------------- admin whitelist
def handle_admin(client, cmd, jid, env):
    """Kelola daftar user yang diizinkan — HANYA utk nomor admin."""
    sender = (jid or "").split("@")[0]
    admins = load_admins(env)
    if sender not in admins:
        client.send_message(jid,
            "⛔ Akses ditolak — hanya nomor admin yang bisa kelola daftar user.")
        log(f"-> {sender}: coba akses admin DITOLAK")
        return
    kind, arg = cmd

    if kind == "admin_add":
        nums = [_validate_phone(n) for n in arg.split(",")]
        nums = [n for n in nums if n]
        if not nums:
            client.send_message(jid,
                "Format: `tambah user 6281234567890` (boleh beberapa, pisah koma)")
            return
        cur = load_allowed_numbers()
        new = sorted(cur | set(nums))
        save_allowed_numbers(new)
        client.send_message(jid,
            f"✅ Nomor ditambahkan: {', '.join(nums)}\n"
            f"📋 User diizinkan ({len(new)}): {', '.join(sorted(new))}")
        log(f"Admin {sender} tambah user: {nums}")

    elif kind == "admin_del":
        nums = [_validate_phone(n) for n in arg.split(",")]
        nums = [n for n in nums if n]
        cur = load_allowed_numbers()
        admins = load_admins(env)
        removed = [n for n in nums if n in cur]
        kept = [n for n in cur if n not in nums or n in admins]  # admin tak bisa dihapus
        save_allowed_numbers(kept)
        msg = f"➖ Dihapus: {', '.join(removed) if removed else 'tidak ada'}"
        blocked = [n for n in nums if n in admins]
        if blocked:
            msg += f"\n⛔ Nomor admin tidak bisa dihapus: {', '.join(blocked)}"
        msg += f"\n📋 User diizinkan ({len(kept)}): {', '.join(sorted(kept))}"
        client.send_message(jid, msg)
        log(f"Admin {sender} hapus user: {removed}")

    elif kind == "admin_list":
        cur = load_allowed_numbers()
        admins = load_admins(env)
        regular = sorted(cur - admins)
        client.send_message(jid,
            f"📋 *DAFTAR USER YANG DIIZINKAN* ({len(cur)})\n"
            f"👑 Admin: {', '.join(sorted(admins))}\n"
            f"👤 User: {', '.join(regular) if regular else '(belum ada)'}")
        log(f"Admin {sender} lihat daftar user")


# ---------------------------------------------------------------- update data
def data_is_current(payload, max_gap_days=3):
    """True kalau data prediksi masih terkini (gap ≤ max_gap_days hari).
    Aturan: Senin→Jumat(-3), Selasa→Senin(-1), dst — aman utk akhir pekan."""
    dstr = (payload or {}).get("data_sampai")
    if not dstr:
        return False, None
    try:
        d = datetime.strptime(dstr, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False, None
    gap = (datetime.now().date() - d).days
    return gap <= max_gap_days, d


def handle_refresh(client, jid, payload):
    """Perintah update data: kalau sudah terkini -> bilang terkini;
    kalau basi -> jalankan predict_daily.py --refresh di thread, lalu
    balas 'pembaharuan data selesai'."""
    current, d = data_is_current(payload)
    if current:
        client.send_message(jid,
            f"✅ Data sudah terkini (s/d {d}).\n"
            f"Tidak perlu update. Ketik `prediksi` utk daftar terbaru.")
        return

    # data basi -> jalankan update di thread (biar bot tetap responsif)
    def work():
        try:
            log(f"Update data dimulai utk {jid} (data s/d {d})")
            client.send_message(jid,
                "⏳ Memulai pembaharuan data + retrain model...\n"
                "Bisa memakan ±10-20 menit. Saya kabari kalau selesai.")
            proc = subprocess.run(
                [sys.executable, os.path.join(BASE, "predict_daily.py"), "--refresh"],
                cwd=BASE, capture_output=True, text=True, timeout=2400)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                msg = err[-1] if err else "gagal tidak diketahui"
                client.send_message(jid, f"⚠️ Pembaharuan data gagal: {msg}")
                log(f"Update data GAGAL utk {jid}: {msg}")
                return
            p2, perr = load_predictions()
            if perr:
                client.send_message(jid,
                    f"✅ Pembaharuan data selesai! (tapi gagal baca hasil: {perr})")
            else:
                nd = p2.get("data_sampai", "?")
                auc = p2.get("valid_auc", 0)
                n = p2.get("jumlah_saham", 0)
                client.send_message(jid,
                    f"✅ Pembaharuan data selesai!\n"
                    f"📅 Data terbaru s/d {nd}\n"
                    f"🤖 Model {n} saham | AUC {auc:.3f}\n"
                    f"Ketik `prediksi` utk daftar terbaru.")
                log(f"Update data SELESAI utk {jid} (s/d {nd})")
        except subprocess.TimeoutExpired:
            client.send_message(jid,
                "⚠️ Pembaharuan data timeout (>40 menit). Coba lagi nanti.")
        except Exception as e:
            client.send_message(jid, f"⚠️ Error saat update: {e}")
            log(f"Update data error: {e}")

    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------- pesan masuk
class MessageProcessor:
    """Proses satu pesan masuk — dipakai bersama oleh polling & webhook."""

    def __init__(self, client, env, seen, cutoff):
        self.client = client
        self.env = env
        self.seen = seen
        self.cutoff = cutoff

    def handle(self, m, jid=None):
        mid = m.get("id")
        if not mid or mid in self.seen or m.get("is_from_me"):
            return False
        sender = (m.get("sender_jid") or jid or "").split("@")[0]
        if sender not in load_allowed_numbers():  # dinamis (bisa ditambah via admin)
            return False
        # lewati pesan lama (sebelum bot mulai / run sebelumnya)
        try:
            ts = datetime.fromisoformat(
                (m.get("timestamp") or "").replace("Z", "+00:00"))
            if ts < self.cutoff:
                self.seen.add(mid)
                return False
        except (ValueError, TypeError):
            pass
        self.seen.add(mid)
        content = (m.get("content") or "").strip()
        log(f"Pesan baru dari {sender}: {content[:60]!r}")
        if parse_command(content):
            self.client.send_typing(jid or f"{sender}@s.whatsapp.net", "start")
            try:
                handle_command(self.client, m, self.env)
            finally:
                self.client.send_typing(jid or f"{sender}@s.whatsapp.net", "stop")
        else:
            log(f"-> {sender}: bukan perintah, diabaikan (tanpa balasan)")
        return True


# ---------------------------------------------------------------- webhook
def _deep_get(d, *paths):
    """Ambil nilai dari dict lewat beberapa jalur alternatif (defensive)."""
    for path in paths:
        node = d
        ok = True
        for k in path:
            if isinstance(node, dict) and k in node and node[k] is not None:
                node = node[k]
            else:
                ok = False
                break
        if ok and node is not None:
            return node
    return None


def parse_webhook_message(raw):
    """Ekstrak pesan dari payload webhook chatetin (format fleksibel).
    Normalisasi ke bentuk yg sama dgn pesan polling:
    {id, chat_jid, sender_jid, content, timestamp, is_from_me}"""
    if not isinstance(raw, dict):
        return None
    event = _deep_get(raw, ("event",), ("type",), ("event_type",))
    if event and not any(x in str(event).lower()
                         for x in ("message", "chat")):
        return None  # event presence/dll -> abaikan
    m = {}
    m["id"] = _deep_get(raw, ("data", "id"), ("message", "id"), ("id",),
                         ("data", "message", "id"), ("message_id",))
    m["content"] = _deep_get(raw, ("data", "content"), ("message", "content"),
                              ("content",), ("data", "message", "text"),
                              ("data", "text"), ("text",),
                              ("message", "body"), ("body",))
    m["chat_jid"] = _deep_get(raw, ("data", "chat_jid"), ("message", "chat_jid"),
                               ("chat_jid",), ("data", "chatId"), ("chatId",),
                               ("data", "from"), ("from",), ("jid",))
    m["sender_jid"] = _deep_get(raw, ("data", "sender_jid"),
                                 ("message", "sender_jid"), ("sender_jid",),
                                 ("data", "sender"), ("sender",),
                                 ("data", "senderId"), ("senderId",),
                                 ("data", "author"), ("author",))
    m["timestamp"] = _deep_get(raw, ("data", "timestamp"),
                                ("message", "timestamp"), ("timestamp",),
                                ("data", "ts"), ("ts",))
    is_me = _deep_get(raw, ("data", "is_from_me"), ("message", "is_from_me"),
                      ("is_from_me",), ("data", "fromMe"), ("fromMe",))
    m["is_from_me"] = bool(is_me)
    if not m["content"] and not m["id"]:
        return None
    return m


def start_webhook_server(port, processor, secret):
    """HTTP server kecil: terima POST webhook dari chatetin."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                hdr = (self.headers.get("X-Webhook-Secret")
                       or self.headers.get("X-Secret") or "")
                if secret and hdr != secret:
                    self.send_response(401)
                    self.end_headers()
                    return
                raw = json.loads(body or b"{}")
                log(f"Webhook POST: {str(raw)[:200]}")
                m = parse_webhook_message(raw)
                if m:
                    processor.handle(m)
                self.send_response(200)
            except Exception as e:
                log(f"Webhook error: {e}")
                self.send_response(400)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    log(f"Webhook server aktif di port {port} (path POST bebas, /health utk cek)")
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="jalankan predict_daily.py dulu sebelum bot jalan")
    ap.add_argument("--once", action="store_true",
                    help="proses pesan sekali lalu keluar (untuk test)")
    args = ap.parse_args()

    env = load_env()
    base = env.get("CHATETIN_BASE_URL", "https://wa.chatetin.com")
    user = env.get("CHATETIN_USERNAME", "")
    pwd = env.get("CHATETIN_PASSWORD", "")
    allowed_raw = env.get("ALLOWED_NUMBERS", DEFAULT_ALLOWED)
    interval = float(env.get("POLL_INTERVAL", DEFAULT_INTERVAL))
    allowed = {n.strip() for n in allowed_raw.split(",") if n.strip()}

    # ambang likuiditas (bisa diubah via .env)
    global MIN_VALUE_TRADED, MIN_PRICE
    MIN_VALUE_TRADED = float(env.get("MIN_VALUE_TRADED", MIN_VALUE_TRADED))
    MIN_PRICE = float(env.get("MIN_PRICE", MIN_PRICE))

    if not user or not pwd:
        sys.exit("Isi CHATETIN_USERNAME & CHATETIN_PASSWORD di file .env "
                 "(lihat .env.example)")

    if args.refresh:
        log("Menjalankan predict_daily.py (retrain + prediksi baru)...")
        os.system(f"cd {BASE} && python3 predict_daily.py")
        log("predict_daily.py selesai.")

    client = ChatetinClient(base, user, pwd)
    client.login()
    dev = client.get_device()
    log(f"Device terhubung: {dev.get('display_name')} ({dev.get('jid')})")

    # verifikasi prediksi kemarin vs aktual + rekam jejak (tiap start bot)
    track = verify_and_record()
    if track:
        log(track)

    # state pesan yang sudah diproses (anti double-reply)
    seen = set()
    cutoff = datetime.now(timezone.utc)  # hanya pesan BARU setelah bot start
    if os.path.exists(STATE_FILE):
        try:
            st = json.load(open(STATE_FILE))
            seen = set(st.get("processed", []))
            last = st.get("last_processed_ts")
            if last:
                cutoff = datetime.fromisoformat(last.replace("Z", "+00:00"))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    log(f"Pemantauan dimulai | interval {interval:.0f}s | "
        f"nomor diizinkan: {sorted(load_allowed_numbers())} | "
        f"pesan lama sebelum {cutoff.isoformat()} diabaikan")

    # thread laporan watchlist + rekap pasar otomatis harian
    report_time = env.get("WATCH_REPORT_TIME", WATCH_REPORT_TIME)
    recap_time = env.get("RECAP_PUSH_TIME", "").strip()
    import threading
    threading.Thread(target=report_worker,
                     args=(client, report_time, recap_time),
                     daemon=True).start()
    log(f"Laporan watchlist otomatis: setiap hari {report_time} WIB"
        + (f" | Rekap pasar: {recap_time} WIB" if recap_time else ""))

    processor = MessageProcessor(client, env, seen, cutoff)
    webhook_url = env.get("WEBHOOK_URL", "").strip()

    if webhook_url:
        # ---------- MODE WEBHOOK (skala besar: hemat API, tanpa polling) ----------
        webhook_port = int(env.get("WEBHOOK_PORT", "8080"))
        webhook_secret = env.get("WEBHOOK_SECRET", "").strip()
        try:
            client.register_webhook(webhook_url, events="message",
                                    secret=webhook_secret)
            log(f"Webhook terdaftar ke chatetin: {webhook_url}")
        except Exception as e:
            log(f"⚠️ Gagal daftar webhook: {e}")
        threading.Thread(target=start_webhook_server,
                         args=(webhook_port, processor, webhook_secret),
                         daemon=True).start()
        log(f"Mode WEBHOOK aktif (port {webhook_port}) — polling dimatikan "
            f"(hemat API utk skala banyak user)")
        if args.once:
            time.sleep(3)
            return
        # loop utama: jaga proses hidup + verifikasi rekam jejak berkala
        while True:
            time.sleep(3600)
            try:
                track = verify_and_record()
                if track:
                    log(track)
            except Exception as e:
                log(f"⚠️ Verifikasi berkala gagal: {e}")
        # ----------------------------------------------------------------

    last_state_write = 0.0
    while True:
        try:
            # polling langsung ke jid nomor yang diizinkan (dinamis, tanpa list semua chat)
            for number in load_allowed_numbers():
                jid = f"{number}@s.whatsapp.net"
                try:
                    msgs = client.chat_messages(jid, limit=10)
                except requests.HTTPError:
                    continue  # chat belum ada -> belum ada pesan
                for m in msgs:
                    processor.handle(m, jid)
            # simpan state bila ada perubahan / tiap 60 detik
            now_ts = time.time()
            if seen or now_ts - last_state_write >= 60:
                with open(STATE_FILE, "w") as f:
                    json.dump({
                        "processed": sorted(seen)[-2000:],
                        "last_processed_ts": datetime.now(timezone.utc).isoformat(),
                    }, f)
                last_state_write = now_ts
        except requests.RequestException as e:
            log(f"⚠️ Error koneksi: {e}")
        except Exception as e:
            log(f"⚠️ Error: {type(e).__name__}: {e}")

        if args.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
