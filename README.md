# XGBoost IDX — Prediksi Arah Saham Indonesia

Pipeline lengkap: ambil data **semua saham Bursa Efek Indonesia (IDX)** →
feature engineering → training XGBoost → **prediksi arah pergerakan besok**
(naik/turun) untuk semua saham.

> 📖 Baca `CONVERSATION.md` untuk catatan lengkap seluruh proses & eksperimen,
> dan `CHANGELOG.md` untuk riwayat versi.

## 🚀 Cara pakai cepat

```bash
# 1. Ambil daftar semua saham IDX (887 saham)
python get_stocks_id.py

# 2. Ambil data historis 5 tahun (daily OHLCV) — ~15-20 menit
python get_history_id.py 5

# 3. Prediksi arah besok (retrain otomatis + rekomendasi ranking)
python predict_daily.py

# 3b. Update data dulu kalau sudah beberapa hari tidak dijalankan
python predict_daily.py --refresh
```

**Output prediksi**: `data/predictions_tomorrow.csv` & `.json`
(ranking, probabilitas naik, sinyal ▲▼, confidence) + `data/model_daily.ubj`.

### Otomatisasi harian (crontab, setiap hari kerja 17:30 WIB)
```bash
30 17 * * 1-5 cd /home/dirman/workspace/xgbooxt && python3 predict_daily.py --refresh >> data/cron.log 2>&1
```

## 🤖 Bot WhatsApp (wa.chatetin.com)

Bot merespon perintah di WhatsApp. **Nomor bot otomatis mengikuti akun
wa.chatetin.com** — diambil dari device yang `logged_in` tiap kali bot start
(tidak di-hardcode). Kalau ganti nomor WhatsApp di chatetin, restart service
untuk memakai nomor baru.

| Perintah | Balasan |
|---|---|
| `prediksi` / `top 20` | Top 20 potensi NAIK ▲ & Top 20 potensi TURUN ▼ besok (saham **likuid** saja) |
| `top 5` / `top 10` | Top N sesuai angka |
| `cek BBRI` | Detail 1 saham (probabilitas, sektor, **likuiditas**, perkiraan return, ukuran saran, rekam jejak) |
| `kenapa BBRI` | **Penjelasan sinyal** — fitur apa yang mendorong NAIK/TURUN (SHAP) |
| `rekap` | **Rekap pasar**: IHSG, USD/IDR, breadth, top gainers/losers, nilai transaksi |
| `riwayat BBRI` | **Rekam jejak historis** prediksi saham itu (terverifikasi) |
| `monev` / `monev 30` | **Hasil prediksi vs AKTUAL** — hit-rate model + penutupan **sesi 1** & **sesi 2**, rata-rata return (7 hari terakhir, atau N hari) + **kirim PDF DATA BOT TRADING** |
| `chat on` / `chat off` / `chat` | **Mode tanya-jawab AI** soal semua data (sliding 5 menit, maks 30 menit). Setiap jawaban diberi footer icon bot + disclaimer |
| `watch TLKM,BBRI` / `tambah` / `hapus` | Atur **watchlist** saham yang dipantau |
| `lapor` | Laporan status semua saham watchlist (harga, prediksi, rekam jejak) |
| `update` / `refresh` | Ambil data terbaru + retrain (jika data basi) |
| `update paksa` | Pastikan semuanya siap: refresh hanya jika data basi / file model kurang |
| `versi` | Info versi bot & model (cek apakah sudah update) |
| `help` | Menu bantuan |

**Keamanan**: bot hanya merespon nomor di daftar diizinkan (`ALLOWED_NUMBERS`
di `.env` atau `data/allowed_numbers.json`) dan hanya membalas pesan yang
berupa perintah — chat biasa diabaikan, tidak pernah broadcast ke nomor random.

**Kelola user dari HP (admin whitelist)**: nomor admin (`BOT_ADMINS` di `.env`,
default 6285720300059 & 6285780535433) bisa menambah/menghapus nomor yang
boleh chat ke bot **langsung dari WhatsApp**, tanpa edit file/restart:

| Perintah admin | Fungsi |
|---|---|
| `tambah user 6281234567890` | Izinkan nomor baru (langsung aktif) |
| `hapus user 6281234567890` | Cabut izin (nomor admin tak bisa dihapus) |
| `daftar user` | Lihat semua nomor yang diizinkan |
| `ai config` | Lihat konfigurasi AI (base URL, API key ter-mask, model) |
| `ai set base <url>` / `ai set key <apikey>` | Ganti base URL / API key AI (langsung aktif) |
| `ai set model <nama>` / `ai set fallback <nama>` | Ganti model utama / cadangan |
| `ai test` / `ai reset` | Cek koneksi & model / kembalikan ke nilai `.env` |

Nomor non-admin yang mencoba perintah ini ditolak (⛔).

> 💡 **Tidak perlu restart** saat menambah/menghapus user — daftar diizinkan
dibaca dari `data/allowed_numbers.json` setiap kali ada pesan masuk / tiap
siklus polling. Restart hanya diperlukan jika mengubah `.env` (mis.
`BOT_ADMINS`, `WATCH_REPORT_TIME`, `WEBHOOK_URL`).

**Filter likuiditas**: saham dengan nilai transaksi < Rp1 miliar/hari atau harga
< Rp200 otomatis **dikeluarkan** dari daftar (penny stock illikuid tidak bisa
dieksekusi). Ambang bisa diubah via `MIN_VALUE_TRADED` & `MIN_PRICE` di `.env`.

**Verifikasi harian**: tiap start, bot membandingkan prediksi lama vs harga
aktual (dari history) dan menampilkan **rekam jejak** di balasan — berapa %
rekomendasi NAIK yang benar-benar naik. Data di `data/bot_track_record.csv`
dan arsip prediksi di `data/prediction_archive.csv`.

**Watchlist & laporan otomatis**: set daftar saham favorit dengan `watch`,
ketik `lapor` kapan saja, atau biarkan bot **push laporan otomatis tiap hari**
(default 18:00 WIB, ubah via `WATCH_REPORT_TIME` di `.env`; kosongkan utk
nonaktif). Laporan berisi harga terakhir, prediksi besok, likuiditas, dan
rekam jejak per saham. Catatan: data **harian**, bukan harga real-time.
Watchlist **per-user** (tiap nomor punya daftar sendiri).

**Update data via WhatsApp**: perintah `update` / `refresh` — kalau data
prediksi masih terkini (gap ≤ 3 hari), bot membalas *"✅ Data sudah terkini"*;
kalau basi, bot menjalankan `predict_daily.py --refresh` di background
(±10-20 menit) lalu membalas *"✅ Pembaharuan data selesai"* + tanggal data
dan AUC model terbaru.

- **`update paksa`** (alias `paksa update` / `force refresh`) — "pastikan
  semuanya siap": kalau data sudah terkini DAN file model lengkap, balas
  *"✅ Data sudah terkini + model siap"* (tanpa kerja berat); baru refresh
  kalau data basi atau ada file kurang (mis. habis `git pull` — file model
  tidak ikut di-commit). `update` biasa juga kini jalan walau file prediksi
  belum ada.
- Update otomatis ikut mengambil **foreign flow hari terakhir** (Net Asing di
  `rekap` / `cek KODE`) via `scrape_foreign_flow.py --latest`.
- Setelah update, kirim **`versi`** utk konfirmasi versi bot (commit hash),
  tanggal prediksi, dan AUC model.
- Jika update gagal, **log lengkap** tersimpan di `data/predict_daily.log`
  (bot hanya membalas baris error terakhir) — cek file itu utk diagnosa.

**Rekap pasar otomatis**: selain laporan watchlist, bot bisa push **rekap
pasar** (IHSG, breadth, gainers/losers, **Net Asing**) ke semua user tiap
hari — isi `RECAP_PUSH_TIME` di `.env` (mis. `18:05`); kosongkan utk
nonaktif.

**Foreign flow (net asing)**: diambil dari sumber resmi idx.co.id via
headless Chrome (Cloudflare) — `scrape_foreign_flow.py`. Data 5 tahun di
`data/foreign_flow.csv`; `rekap` & `cek KODE` menampilkannya. ⚠️ Hasil
walk-forward: sebagai fitur model **merugikan** (delta AUC -0.0035), jadi
dipakai hanya sebagai info, bukan input prediksi.

```bash
# Setup (sekali)
cp .env.example .env   # isi kredensial chatetin + nomor yang diizinkan

# Jalankan manual / test
python wa_bot.py              # pakai prediksi yang sudah ada
python wa_bot.py --refresh    # retrain + prediksi baru dulu
python wa_bot.py --once       # proses pesan sekali lalu keluar (test)

# Jalankan sebagai service (auto-restart)
systemctl --user enable --now wa_bot.service
journalctl --user -u wa_bot.service -f   # lihat log
```

File terkait: `wa_bot.py`, `.env.example`, `run_bot.sh`, `wa_bot.service`,
log di `data/wa_bot.log`.

## 🖥️ Deploy ke VPS (skala banyak user)

```bash
bash install.sh   # install deps (tanpa torch), .env, systemd, cron
```

**Dua mode pesan masuk** (pilih di `.env`):

| Mode | Cara kerja | Cocok utk |
|---|---|---|
| **Polling** (default) | Bot cek pesan tiap `POLL_INTERVAL` detik | 1–10 user |
| **Webhook** | chatetin POST event pesan ke `WEBHOOK_URL` bot | puluhan–ratusan user |

Mode webhook: isi `WEBHOOK_URL` (alamat publik VPS, port `WEBHOOK_PORT`,
default 8080) → bot daftarkan webhook ke chatetin & matikan polling.
Endpoint `/health` utk cek server. Opsional `WEBHOOK_SECRET` utk verifikasi
header `X-Webhook-Secret`.

**Catatan skala**: model cross-sectional = retrain 1×/hari utk SEMUA user
(peak RAM ±2 GB, muat di VPS 8 GB). Beban per-user hanya pesan masuk/balasan.
`predict_daily.py` memakai lock anti-bentrok + tulis file atomik, dan
`_request` punya backoff exponensial utk rate-limit (429/5xx).

## 📁 Struktur proyek

| Script | Fungsi |
|---|---|
| `get_stocks_id.py` | Daftar 887 saham IDX (via TradingView scanner) → CSV/Excel/JSON |
| `get_history_id.py` | Download 5 tahun OHLCV per saham (Yahoo, resume-able) |
| `train_xgb.py` | Feature engineering + XGBoost baseline (34 fitur teknikal) → SQLite |
| `add_macro.py` | Fitur makro: IHSG & kurs USD/IDR (Yahoo) |
| `train_xgb_macro.py` | XGBoost + fitur makro |
| `walk_forward.py` | Walk-forward validation (retrain bulanan, evaluasi realistis) |
| `train_lstm.py` | LSTM (PyTorch) pembanding deep learning |
| `ensemble.py` | Ensemble XGBoost + LSTM |
| `per_stock_models.py` | Model per-saham utk 20 saham likuid + backtest trading |
| `predict_daily.py` | **Pipeline produksi**: retrain + prediksi besok semua saham |
| `session_data.py` | Ambil & olah bar 15m Yahoo jadi **penutupan sesi 1 & 2** per saham |
| `capture_session.py` | Simpan penutupan sesi 1 & 2 saham prediksi ke `data/session_bars.csv` |
| `eval_report.py` | **Monev**: evaluasi prediksi vs aktual (→ `data/eval/`) + auto-push |
| `report_pdf.py` | Laporan PDF **DATA BOT TRADING** (top-10 naik, sesi 1 & 2) |
| `ai_chat.py` | **Mode chat AI** (`chat on`): ringkasan data + tool calling + footer bot |
| `waktu.py` | Satu sumber kebenaran waktu **WIB** (`now_wib`, `today_wib`), dipakai semua script |
| `scrape_foreign_flow.py` | Ambil **foreign flow (net asing)** per saham dari idx.co.id (headless Chrome + API resmi IDX) |
| `wa_bot.py` | **Bot WhatsApp**: balas `prediksi`/`top N`/`cek KODE` (filter likuid + rekam jejak) |
| `backtest_top20.py` | Backtest jujur strategi top-20 (likuiditas + biaya) |
| `install.sh` | Skrip deploy VPS (deps, .env, systemd, cron) |
| `CONVERSATION.md` | Catatan percakapan & seluruh eksperimen |
| `CHANGELOG.md` | Riwayat versi |

## 📊 Hasil eksperimen (data TEST out-of-sample)

| Model | ROC-AUC | Kesimpulan |
|---|---|---|
| **XGBoost gabungan + walk-forward** | **0.576** | 🏆 terbaik & konsisten (0.55–0.62 per bulan) |
| XGBoost gabungan (single split) | 0.598* | optimistis (1x train) |
| Ensemble XGB + LSTM | 0.575 | setara, F1 sedikit naik |
| LSTM | 0.528 | kalah — data tabular terbatas |
| Per-saham (20 likuid) | 0.506 | ❌ data per saham terlalu sedikit |

\* Angka jujur (walk-forward) = **0.576**.

**Kesimpulan**: pendekatan gabungan (semua saham, cross-sectional) menang
karena berbagi kekuatan statistik antar saham. Prediksi arah harian secara
intrinsik sulit — AUC ~0.58 sudah di atas random dan stabil antar bulan.

> Fitur relative-strength vs sektor diuji walk-forward (12 bulan OOS): delta
> AUC **+0.0008 (netral)** — disertakan di produksi utk konteks penjelasan
> sinyal (`kenapa KODE`), bukan penambah akurasi.

### ⚠️ Hasil backtest jujur strategi Top-20 (walk-forward OOS, 224 hari)

| Skenario | Hit rate | Return NETO/hari | t-stat |
|---|---|---|---|
| Top-20 semua saham (+biaya 0.3%) | 58.2% (base 38.6%) | +1.49% | 9.5 |
| **Top-20 hanya LIKUID** (+biaya 0.3%) | **49.6%** (base 41.4%) | **+0.24%** | **1.9** |

**Kesimpulan penting**: return +1.8%/hari itu *mirage* penny stock — 57% picks
harga < Rp500. Setelah filter likuiditas (nilai ≥ Rp1 M/hari, harga ≥ Rp200),
edge hampir hilang dan **belum signifikan statistik** (t=1.9). Bot sudah
menerapkan filter ini; jangan berharap profit besar dari daftar ini.

### Fitur paling penting
`vol_5` (volatilitas 5 hari) → `ret_1`, `log_ret_1`, `hl_range`, `ret_2`,
`close_sma5` — volatilitas & momentum jangka pendek paling menentukan.

## 🗄️ Data & database

| File | Isi |
|---|---|
| `data/history/{kode}.csv` | OHLCV per saham (884 file) |
| `data/history_id_5y.parquet` / `.csv` | 948.713 baris gabungan (2021-08 s/d 2026-08) |
| `data/features.db` | Fitur teknikal (SQLite, 833k baris × 34 fitur) |
| `data/features_macro.db` | Fitur + makro IHSG/USD-IDR (SQLite, 830k baris) |
| `data/stocks_id.csv` / `.xlsx` / `.json` | Daftar 887 saham + profil |
| `data/macro_id.csv` | IHSG & USD/IDR harian 5 tahun |
| `data/model_daily.ubj` + `.json` | Model produksi + metadata |
| `data/predictions_tomorrow.csv` / `.json` | **Prediksi besok** (ranking + likuiditas) |
| `data/prediction_log.csv` | Riwayat run harian (AUC, jumlah naik/turun) |
| `data/foreign_flow.csv` | **Net asing per saham** dari idx.co.id (1.065.585 baris, 5 tahun) |
| `data/prediction_archive.csv` | Arsip top-20 naik/turun harian (untuk verifikasi) |
| `data/bot_track_record.csv` | Rekam jejak: hasil aktual vs prediksi bot |
| `data/wa_watchlist.json` | Watchlist saham yang dipantau user |

## 🐍 Kebutuhan

Python 3.10+, paket: `pandas`, `numpy`, `requests`, `xgboost`, `scikit-learn`,
`openpyxl`, `pyarrow`, `torch`, `matplotlib`.

```bash
pip install --break-system-packages pandas numpy requests xgboost scikit-learn \
    openpyxl pyarrow torch matplotlib
```

## ⚠️ Catatan penting

1. **Bukan saran investasi** — AUC ~0.58 berarti sinyal lemah; jangan trading
   penuh hanya dari model ini.
2. Data resmi IDX (`idx.co.id`) diblokir Cloudflare → pakai **TradingView**
   (daftar saham) & **Yahoo Finance** (historis).
3. Fitur makro tidak menaikkan AUC di pendekatan gabungan, tapi tetap
   disertakan di pipeline produksi.
4. Threshold sinyal dituning otomatis (max F1) di data validasi — bukan 0.5.
5. Sinyal di atas adalah *probabilitas arah*, bukan jaminan hasil.
