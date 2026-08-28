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
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

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
        try:
            r = self.s.request(method, f"{self.base}{path}", timeout=30, **kw)
        except requests.RequestException as e:
            # token mungkin expire -> coba login ulang sekali
            log(f"Request error {path}: {e}; coba login ulang...")
            self.login(force=True)
            r = self.s.request(method, f"{self.base}{path}", timeout=30, **kw)
        if r.status_code == 401:
            self.login(force=True)
            r = self.s.request(method, f"{self.base}{path}", timeout=30, **kw)
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
            return (
                f"📊 *{x['ticker']}* — {x.get('description', '-')}\n"
                f"🏭 Sektor: {sektor}\n"
                f"💵 Close terakhir: Rp {x['close']:,.0f}\n"
                f"{liq_note}\n"
                f"🎯 Probabilitas NAIK besok: {p:.1f}%\n"
                f"📈 Sinyal: {x['signal']}\n"
                f"🎚 Confidence: {conf:.1f}%\n"
                f"📅 Prediksi: {payload.get('tanggal_prediksi', '?')}\n"
                f"⚠️ Bukan saran investasi."
            ), None
    return None, f"❌ Kode *{ticker}* tidak ditemukan. Contoh: `cek BBRI`"


# ---------------------------------------------------------------- verifikasi harian
ARCHIVE = os.path.join(BASE, "data", "prediction_archive.csv")


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
    if low in ("help", "bantuan", "menu"):
        return ("help", None)
    return None


HELP_TEXT = (
    "🤖 *Bot Prediksi Saham IDX*\n"
    "Perintah yang tersedia:\n"
    "• `prediksi` / `top 20` — Top 20 potensi NAIK & TURUN besok (saham LIKUID saja)\n"
    "• `top 5` / `top 10` — Top N sesuai angka\n"
    "• `cek BBRI` — detail 1 saham (likuiditas + probabilitas)\n"
    "• `help` — menu ini\n\n"
    "🔒 Saham illikuid/penny dikeluarkan otomatis dari daftar.\n"
    "⚠️ Hasil bukan saran investasi."
)


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

    payload, err = load_predictions()
    if err:
        client.send_message(jid, f"⚠️ {err}")
        log(f"-> {jid}: prediksi file error")
        return
    stale = check_freshness(payload)

    if cmd[0] == "top":
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

    elif cmd[0] == "help":
        client.send_message(jid, HELP_TEXT)
        log(f"-> {jid}: help dikirim ({_t.time()-t0:.1f}s)")


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
        f"nomor diizinkan: {sorted(allowed)} | pesan lama sebelum {cutoff.isoformat()} diabaikan")

    last_state_write = 0.0
    while True:
        try:
            # polling langsung ke jid nomor yang diizinkan (tanpa list semua chat)
            for number in allowed:
                jid = f"{number}@s.whatsapp.net"
                try:
                    msgs = client.chat_messages(jid, limit=10)
                except requests.HTTPError:
                    continue  # chat belum ada -> belum ada pesan
                for m in msgs:
                    mid = m.get("id")
                    if mid in seen or m.get("is_from_me"):
                        continue
                    sender = (m.get("sender_jid") or "").split("@")[0]
                    if sender not in allowed:
                        continue
                    # lewati pesan lama (sebelum bot mulai / run sebelumnya)
                    try:
                        ts = datetime.fromisoformat(
                            (m.get("timestamp") or "").replace("Z", "+00:00"))
                        if ts < cutoff:
                            seen.add(mid)
                            continue
                    except ValueError:
                        pass
                    seen.add(mid)
                    content = (m.get("content") or "").strip()
                    log(f"Pesan baru dari {sender}: {content[:60]!r}")
                    if parse_command(content):
                        client.send_typing(jid, "start")   # indikator mengetik
                        try:
                            handle_command(client, m, env)
                        finally:
                            client.send_typing(jid, "stop")
                    else:
                        log(f"-> {sender}: bukan perintah, diabaikan (tanpa balasan)")
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
