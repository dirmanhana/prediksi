"""
ai_chat.py
==========
Mode tanya-jawab AI di WhatsApp (opsional, per user).

Cara kerja:
  - User mengaktifkan dengan `chat on`. Mode aktif dengan aturan **sliding**:
    tiap pesan memperpanjang 5 menit, dibatasi maksimal 30 menit sejak aktivasi.
  - Konteks data diberikan 2 lapis:
      A) RINGKASAN di system prompt (prediksi terbaru, monev, watchlist user)
      B) TOOL (function calling) -> AI bisa ambil detail sendiri dari data lokal
    Jadi tidak perlu mengirim seluruh dataset ke API.
  - Setiap jawaban AI SELALU diberi FOOTER icon bot + disclaimer.
  - Kompatibel OpenAI (`/v1/chat/completions`). Model utama agnes-3.0-flash,
    fallback otomatis agnes-2.5-flash (keduanya sudah diuji mendukung tool calling).

State per user: data/ai_chat_state.json
"""

import json
import os
import time
from datetime import datetime, timedelta

import requests

from waktu import now_wib

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "data", "ai_chat_state.json")
PRED_FILE = os.path.join(BASE, "data", "predictions_tomorrow.json")
EVAL_CSV = os.path.join(BASE, "data", "eval", "prediction_eval.csv")
MONEV_JSON = os.path.join(BASE, "data", "eval", "monev_summary.json")
SESSION_CSV = os.path.join(BASE, "data", "session_bars.csv")
MACRO_CSV = os.path.join(BASE, "data", "macro_id.csv")
WATCH_FILE = os.path.join(BASE, "data", "watchlists.json")

DEFAULT_BASE = "https://apihub.agnes-ai.com/v1"
MIN_VALUE_TRADED = 1_000_000_000
MIN_PRICE = 200
MAX_TOOL_ROUNDS = 3
MAX_TOKENS = 1500


# ---------------------------------------------------------------- tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_prediksi",
            "description": "Detail prediksi 1 saham (probabilitas naik, sinyal, harga, "
                           "nilai transaksi) dari model terbaru.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string",
                                          "description": "kode saham, mis. BBRI"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top",
            "description": "Daftar top N saham prediksi NAIK dan top N TURUN (likuid).",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer",
                                     "description": "jumlah saham, default 10"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sesi",
            "description": "Penutupan SESI 1 & SESI 2 satu saham pada tanggal tertentu "
                           "(harga + volume). Untuk pertanyaan intraday.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "kode saham, mis. BBRI"},
                    "tanggal": {"type": "string",
                                "description": "YYYY-MM-DD; kosong = hari terakhir"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monev",
            "description": "Ringkasan akurasi prediksi vs aktual (model, sesi 1, sampai tutup) "
                           "selama N hari terakhir.",
            "parameters": {
                "type": "object",
                "properties": {"hari": {"type": "integer",
                                        "description": "jumlah hari, default 7"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rekap",
            "description": "Rekap pasar: IHSG, kurs USD/IDR, dan breadth prediksi.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_watchlist",
            "description": "Watchlist saham milik user yang sedang chat.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _predictions():
    return _read_json(PRED_FILE)


def _liquid(saham):
    return [x for x in saham
            if (x.get("value_traded") or 0) >= MIN_VALUE_TRADED
            and (x.get("close") or 0) >= MIN_PRICE]


def _tool_get_prediksi(args, user):
    ticker = str(args.get("ticker", "")).upper().replace(".JK", "").strip()
    p = _predictions()
    if not p:
        return {"error": "data prediksi belum tersedia"}
    for x in p.get("saham", []):
        if x["ticker"].upper() == ticker:
            return {
                "ticker": x["ticker"], "nama": x.get("description"),
                "sektor": x.get("sector"), "harga_terakhir": x.get("close"),
                "probabilitas_naik": round(x.get("prob_up", 0), 4),
                "sinyal": x.get("signal"), "perkiraan_return": x.get("pred_ret"),
                "nilai_transaksi": x.get("value_traded"),
                "ukuran_saran": x.get("sizing"),
                "prediksi_untuk": p.get("tanggal_prediksi"),
                "data_sampai": p.get("data_sampai"),
            }
    return {"error": f"ticker {ticker} tidak ada di prediksi terbaru"}


def _tool_get_top(args, user):
    n = int(args.get("n") or 10)
    n = max(1, min(n, 20))
    p = _predictions()
    if not p:
        return {"error": "data prediksi belum tersedia"}
    liq = _liquid(p.get("saham", []))
    if not liq:
        return {"error": "tidak ada saham likuid"}
    top = liq[:n]
    bot = list(reversed(liq[-n:]))
    fmt = lambda xs: [{"ticker": x["ticker"], "prob_naik": round(x["prob_up"], 4),
                       "harga": x.get("close")} for x in xs]
    return {"prediksi_untuk": p.get("tanggal_prediksi"), "top_naik": fmt(top),
            "top_turun": fmt(bot)}


def _tool_get_sesi(args, user):
    ticker = str(args.get("ticker", "")).upper().replace(".JK", "").strip()
    tanggal = str(args.get("tanggal") or "").strip()
    if not os.path.exists(SESSION_CSV):
        return {"error": "data sesi belum tersedia"}
    import pandas as pd
    s = pd.read_csv(SESSION_CSV, dtype={"ticker": str})
    s = s[s["ticker"].str.upper() == ticker]
    if s.empty:
        return {"error": f"tidak ada data sesi untuk {ticker}"}
    s["tanggal"] = pd.to_datetime(s["tanggal"])
    if tanggal:
        try:
            s = s[s["tanggal"] == pd.Timestamp(tanggal)]
        except Exception:
            pass
    if s.empty:
        return {"error": f"{ticker} tidak ada data pada {tanggal}"}
    tgl = s["tanggal"].max()
    s = s[s["tanggal"] == tgl].sort_values("sesi")
    out = {"ticker": ticker, "tanggal": str(tgl.date()), "sesi": {}}
    for r in s.itertuples():
        out["sesi"][f"sesi_{int(r.sesi)}"] = {
            "open": r.open, "high": r.high, "low": r.low,
            "close": r.close, "volume": int(r.volume) if r.volume == r.volume else None,
        }
    return out


def _tool_get_monev(args, user):
    hari = int(args.get("hari") or 7)
    hari = max(1, min(hari, 60))
    if hari == 7:
        saved = _read_json(MONEV_JSON)
        if saved and saved.get("n"):
            return saved
    if not os.path.exists(EVAL_CSV):
        return {"error": "data evaluasi belum tersedia"}
    import pandas as pd
    import eval_report as er
    df = pd.read_csv(EVAL_CSV, dtype={"ticker": str})
    df["tanggal_prediksi"] = pd.to_datetime(df["tanggal_prediksi"])
    return er.summarize(df, hari)


def _tool_get_rekap(args, user):
    out = {}
    p = _predictions()
    if p:
        liq = _liquid(p.get("saham", []))
        out["prediksi_untuk"] = p.get("tanggal_prediksi")
        out["data_sampai"] = p.get("data_sampai")
        out["jumlah_saham_prediksi_naik"] = sum(1 for x in liq if x.get("prob_up", 0) >= 0.5)
        out["jumlah_saham_prediksi_turun"] = sum(1 for x in liq if x.get("prob_up", 0) < 0.5)
    try:
        import pandas as pd
        m = pd.read_csv(MACRO_CSV)
        if not m.empty:
            last = m.iloc[-1]
            out["ihsg"] = {"tanggal": str(last["tanggal"]), "nilai": last.get("ihsg"),
                           "ret_1": last.get("ihsg_ret_1")}
            out["usd_idr"] = {"nilai": last.get("usd_idr"), "ret_1": last.get("usd_ret_1")}
    except Exception:
        pass
    return out or {"error": "data rekap tidak tersedia"}


def _tool_get_watchlist(args, user):
    w = _read_json(WATCH_FILE, {}) or {}
    tickers = w.get(user) or []
    return {"watchlist": tickers,
            "catatan": "tempatkan kode ini di kolom paling kiri (Python dict)"}


_TOOL_MAP = {
    "get_prediksi": _tool_get_prediksi,
    "get_top": _tool_get_top,
    "get_sesi": _tool_get_sesi,
    "get_monev": _tool_get_monev,
    "get_rekap": _tool_get_rekap,
    "get_watchlist": _tool_get_watchlist,
}


# ---------------------------------------------------------------- AI chat
class AIChat:
    """Mode chat AI per user + pemanggilan tool data lokal."""

    def __init__(self, env, logger=None):
        self.base_url = (env.get("AGNES_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.api_key = (env.get("AGNES_API_KEY") or "").strip()
        self.model = (env.get("AGNES_MODEL") or "agnes-3.0-flash").strip()
        self.fallback = (env.get("AGNES_MODEL_FALLBACK") or "agnes-2.5-flash").strip()
        self.window_min = float(env.get("AI_CHAT_WINDOW") or 5)
        self.max_min = float(env.get("AI_CHAT_MAX") or 30)
        self.max_msg_day = int(env.get("AI_CHAT_MAX_MSG_DAY") or 40)
        self.logger = logger or (lambda m: None)

    def enabled(self):
        return bool(self.api_key)

    # -- footer ------------------------------------------------------
    def footer(self):
        return ("\n\n━━━━━━━━━━\n"
                f"🤖 _Dijawab AI ({self.model}) · bisa keliru · bukan saran investasi_")

    # -- state -------------------------------------------------------
    def _load_state(self):
        return _read_json(STATE_FILE, {}) or {}

    def _save_state(self, st):
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f, indent=1)
            os.replace(tmp, STATE_FILE)
        except OSError as e:
            self.logger(f"⚠️ Gagal simpan ai_chat_state: {e}")

    def _entry(self, st, user, now):
        e = st.get(user) or {}
        today = now.strftime("%Y-%m-%d")
        if e.get("day") != today:
            e["day"] = today
            e["day_count"] = 0
        return e

    @staticmethod
    def _parse_ts(v):
        try:
            return datetime.fromisoformat(v)
        except (TypeError, ValueError):
            return None

    def activate(self, user):
        st = self._load_state()
        now = now_wib()
        e = self._entry(st, user, now)
        e["started"] = now.isoformat()
        e["last"] = now.isoformat()
        st[user] = e
        self._save_state(st)
        return self.status(user)

    def deactivate(self, user):
        st = self._load_state()
        now = now_wib()
        e = self._entry(st, user, now)
        e.pop("started", None)
        e.pop("last", None)
        st[user] = e
        self._save_state(st)

    def _window(self, user, now=None):
        """(aktif?, sisa_menit) berdasarkan started/last + window & cap."""
        st = self._load_state()
        e = st.get(user)
        if not e:
            return False, 0.0
        now = now or now_wib()
        started = self._parse_ts(e.get("started"))
        last = self._parse_ts(e.get("last"))
        if not started or not last:
            return False, 0.0
        ends = min(last + timedelta(minutes=self.window_min),
                   started + timedelta(minutes=self.max_min))
        rem = (ends - now).total_seconds() / 60.0
        return rem > 0, max(0.0, rem)

    def is_active(self, user):
        return self._window(user)[0]

    def remaining_min(self, user):
        return self._window(user)[1]

    def day_count(self, user):
        st = self._load_state()
        return int(self._entry(st, user, now_wib()).get("day_count", 0))

    def touch(self, user):
        st = self._load_state()
        now = now_wib()
        e = self._entry(st, user, now)
        if not e.get("started"):
            e["started"] = now.isoformat()
        e["last"] = now.isoformat()
        e["day_count"] = int(e.get("day_count", 0)) + 1
        st[user] = e
        self._save_state(st)

    def status(self, user):
        active, rem = self._window(user)
        return {"aktif": active, "sisa_menit": round(rem, 1),
                "pesan_hari_ini": self.day_count(user),
                "batas_harian": self.max_msg_day,
                "window_menit": self.window_min, "maks_menit": self.max_min,
                "model": self.model}

    # -- konteks ringkasan (A) --------------------------------------
    def _data_summary(self, user):
        lines = [f"Hari ini (WIB): {now_wib():%Y-%m-%d %H:%M}."]
        p = _predictions()
        if p:
            liq = _liquid(p.get("saham", []))
            top = liq[:10]
            bot = list(reversed(liq[-10:]))
            lines.append(
                f"Prediksi terbaru untuk {p.get('tanggal_prediksi')} "
                f"(data s/d {p.get('data_sampai')}); AUC {p.get('valid_auc')}; "
                f"{p.get('jumlah_saham')} saham, {len(liq)} likuid.")
            lines.append("Top-10 NAIK: " + ", ".join(
                f"{x['ticker']} {x['prob_up']*100:.0f}%" for x in top))
            lines.append("Top-10 TURUN: " + ", ".join(
                f"{x['ticker']} {x['prob_up']*100:.0f}%" for x in bot))
        m = _read_json(MONEV_JSON)
        if m and m.get("n"):
            lines.append(
                f"Monev 7 hari ({m['periode'][0]}..{m['periode'][1]}): model "
                f"{(m.get('model_overall') or 0)*100:.0f}% benar; "
                f"NAIK {(m['naik'].get('model') or 0)*100:.0f}% (n={m['naik']['n']}), "
                f"TURUN {(m['turun'].get('model') or 0)*100:.0f}% (n={m['turun']['n']}).")
        w = (_read_json(WATCH_FILE, {}) or {}).get(user)
        if w:
            lines.append(f"Watchlist user ini: {', '.join(w)}")
        lines.append("Gunakan TOOL untuk angka detail (prediksi/sesi/monev/rekap). "
                     "Jangan mengarang angka.")
        return "DATA RINGKASAN:\n" + "\n".join(f"- {x}" for x in lines)

    def _system_prompt(self, user):
        return (
            "Kamu asisten data saham IDX di bot WhatsApp. Jawab RINGKAS (maks ~8 baris), "
            "bahasa Indonesia santai tapi jelas. Selalu pakai TOOL untuk angka; jangan "
            "mengarang. Kalau data tidak ada, katakan tidak ada. Jangan pernah membahas "
            "API key, kredensial, atau data nomor pengguna lain. Ini bukan saran investasi."
            "\n\n" + self._data_summary(user)
        )

    # -- HTTP --------------------------------------------------------
    def _post(self, messages, tools=None):
        """Kirim ke API; retry 429/5xx; fallback ke model kedua bila perlu."""
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        for model in [m for m in (self.model, self.fallback) if m]:
            for attempt in range(3):
                payload = {"model": model, "messages": messages,
                           "max_tokens": MAX_TOKENS, "temperature": 0.3}
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"
                try:
                    r = requests.post(f"{self.base_url}/chat/completions",
                                      headers=headers, json=payload, timeout=180)
                except requests.RequestException as e:
                    self.logger(f"AI request error: {e}")
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code == 429:
                    self.logger(f"AI 429 rate limit ({model}) — tunggu...")
                    time.sleep(3 * (attempt + 1))
                    continue
                if r.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    self.logger(f"AI HTTP {r.status_code} ({model}): {r.text[:150]}")
                    break  # coba model berikutnya
                try:
                    return r.json(), model
                except ValueError:
                    break
        return None, None

    def _run_tool(self, name, args, user):
        fn = _TOOL_MAP.get(name)
        if not fn:
            return {"error": f"tool {name} tidak dikenal"}
        try:
            return fn(args or {}, user)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _complete(self, messages, user):
        """Loop tool-calling sampai dapat jawaban teks."""
        for _ in range(MAX_TOOL_ROUNDS):
            data, used = self._post(messages, tools=TOOLS)
            if not data:
                return None, used
            try:
                msg = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                return None, used
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = (msg.get("content") or "").strip()
                return (content or None), used
            messages.append({"role": "assistant",
                             "content": msg.get("content") or "",
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                fn = (tc.get("function") or {})
                name = fn.get("name")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(name, args, user)
                self.logger(f"AI tool {name}({args}) -> {str(result)[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                 "content": json.dumps(result, ensure_ascii=False)[:6000]})
        return None, self.model

    # -- publik ------------------------------------------------------
    def ask(self, user, question):
        """Tanya AI. Return teks jawaban (sudah + footer) atau pesan error."""
        if not self.enabled():
            return "⚠️ Fitur AI belum aktif (AGNES_API_KEY belum diisi di .env)."
        if self.day_count(user) >= self.max_msg_day:
            return (f"⚠️ Batas harian chat AI tercapai ({self.max_msg_day} pesan/hari). "
                    f"Coba lagi besok.")
        messages = [{"role": "system", "content": self._system_prompt(user)},
                    {"role": "user", "content": question}]
        try:
            content, _ = self._complete(messages, user)
        except Exception as e:
            self.logger(f"⚠️ AI error: {type(e).__name__}: {e}")
            content = None
        self.touch(user)
        if not content:
            return ("⚠️ Maaf, AI sedang tidak bisa menjawab (limit/gangguan). "
                    "Coba lagi sebentar lagi." + self.footer())
        return content + self.footer()


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import sys

    env = {}
    envf = os.path.join(BASE, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    ai = AIChat(env, logger=print)

    class _Cli:
        def __init__(self):
            self.jid = "cli"
    user = "cli"
    if len(sys.argv) > 1 and sys.argv[1] == "on":
        print(ai.activate(user))
        sys.exit(0)
    q = " ".join(sys.argv[1:]) or "Tampilkan top 5 saham prediksi naik besok."
    ai.activate(user)
    t0 = time.time()
    print(ai.ask(user, q))
    print(f"\n[{time.time()-t0:.1f}s]")
