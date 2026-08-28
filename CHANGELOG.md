# Changelog

Semua perubahan penting pada proyek XGBoost IDX.

## [v0.8.1] — 2026-08-28
### Added
- **Admin whitelist dari WhatsApp**: nomor admin (`BOT_ADMINS` di `.env`,
default 6285720300059 & 6285780535433) bisa menambah/menghapus nomor yang
  diizinkan langsung dari chat — tanpa edit file/restart:
  - `tambah user 628xxx` / `hapus user 628xxx` / `daftar user`
  - User baru langsung aktif; nomor admin tidak bisa dihapus.
- Daftar diizinkan kini dinamis (`data/allowed_numbers.json`, seed dari
  `.env`); dipakai polling, webhook, laporan otomatis, dan verifikasi.
- Validasi format nomor Indonesia (62/08/+62/8xxxx → 62xxxxxxxxxx).

### Security
- Perintah admin hanya berlaku utk nomor di `BOT_ADMINS`; selain itu ditolak.

## [v0.8.0] — 2026-08-28
### Added
- **Perintah `update` / `refresh` di bot**: ambil data terbaru + retrain dari
  WhatsApp. Kalau data masih terkini (gap ≤ 3 hari) → balas *"Data sudah
  terkini"*; kalau basi → jalankan `predict_daily.py --refresh` di thread
  background (bot tetap responsif) lalu balas *"Pembaharuan data selesai"*
  dengan tanggal data & AUC terbaru.
- Alias: `update data`, `refresh data`, `tarik data`, `ambil data`,
  `perbarui data`.

### Note
- Update bisa memakan ±10-20 menit (incremental 887 saham + retrain).
- Aman bentrok dgn cron 17:30 (lock flock di predict_daily.py).

## [v0.7.0] — 2026-08-28
### Added
- **Mode webhook**: `WEBHOOK_URL` / `WEBHOOK_PORT` / `WEBHOOK_SECRET` di `.env`.
  chatetin POST event pesan ke server bot → hemat API utk skala banyak user
  (polling otomatis dimatikan saat webhook aktif).
- **Parser webhook defensif** (`parse_webhook_message`) + server HTTP bawaan
  (`/health`, verifikasi `X-Webhook-Secret`).
- **Watchlist per-user**: tiap nomor punya daftar sendiri
  (`data/watchlists.json`); laporan otomatis harian dikirim per-user.
- **`install.sh`**: deploy VPS sekali jalan — deps (tanpa torch), `.env`,
  systemd service, cron harian.
- **Lock anti-bentrok** di `predict_daily.py` (flock) + **tulis file atomik**
  (temp+rename) supaya bot tidak membaca file setengah jadi.
- **Backoff exponensial** di `_request` utk rate-limit (429/5xx).

### Refactor
- Logika pesan masuk dipindah ke `MessageProcessor` — dipakai bersama oleh
  polling & webhook (anti duplikasi balasan).

## [v0.6.0] — 2026-08-28
### Added
- **Fitur watchlist**: `watch TLKM,BBRI`, `tambah TLKM`, `hapus TLKM`, `lapor`.
- **Laporan otomatis harian**: bot push status watchlist tiap hari (default
  18:00 WIB, atur via `WATCH_REPORT_TIME` di `.env`; kosongkan utk nonaktif).
- Laporan per saham: harga terakhir, prediksi besok, sinyal, likuiditas, dan
  rekam jejak per saham (`ticker_track`).
- `data/wa_watchlist.json` untuk penyimpanan watchlist.

### Catatan
- Data bersifat **harian** (update 1×/hari via cron `predict_daily.py --refresh`),
  bukan harga real-time.

## [v0.5.0] — 2026-08-28
### Added
- **Filter likuiditas di bot**: saham nilai transaksi < Rp1 M/hari atau harga
  < Rp200 otomatis dikeluarkan dari daftar top N / prediksi (penny stock
  illikuid tidak bisa dieksekusi).
- **Verifikasi harian + rekam jejak**: bot membandingkan prediksi lama vs harga
  aktual tiap start, menampilkan hit rate NAIK/TURUN di balasan.
- **Arsip prediksi harian** (`data/prediction_archive.csv`) di `predict_daily.py`.
- **`backtest_top20.py`**: backtest jujur strategi top-20 dengan filter
  likuiditas & biaya transaksi 0.3%.
- Konfigurasi ambang likuiditas via `.env` (`MIN_VALUE_TRADED`, `MIN_PRICE`).
- `data/predictions_tomorrow.{csv,json}` kini memuat `volume` & `value_traded`.

### Changed
- `cek KODE` menampilkan likuiditas + label LIKUID/ILLIKUID.

### Fixed
- **Temuan evaluasi**: return +1.8%/hari sebelumnya adalah artefak penny stock
  illikuid — setelah filter likuiditas, edge nyaris hilang (t=1.9, belum
  signifikan). Hasil dipublikasikan di README.

## [v0.4.0] — 2026-08-28
### Added
- Dukungan perintah `top N` (top 3 / top 5 / top 10 / top 20 / `prediksi top N`).
- `prediksi KODE` setara `cek KODE`.

### Fixed
- "top 3" sebelumnya diabaikan (hanya `top 20` yang dikenali).
- Balasan top list digabung jadi **1 pesan** (sebelumnya 2 pesan + jeda 1s).

## [v0.3.0] — 2026-08-28
### Changed
- **Percepatan balasan bot**: interval polling 10s → 3s.
- Polling langsung ke JID nomor diizinkan (tanpa list semua chat).
- Indikator *typing* (chat-presence) sebelum balas.
- Penulisan state file hanya saat ada perubahan / tiap 60s.

## [v0.2.0] — 2026-08-28
### Added
- **Bot WhatsApp** (`wa_bot.py`) via API wa.chatetin.com:
  - Perintah `prediksi` / `top 20` / `cek KODE` / `help`.
  - Whitelist `ALLOWED_NUMBERS` (bot hanya balas nomor terdaftar).
  - Hanya membalas pesan yang berupa perintah (chat biasa diabaikan).
  - Pesan lama sebelum bot start diabaikan (anti balas chat lama).
- `wa_bot.service` (systemd user, auto-restart), `run_bot.sh`, `.env.example`.

### Security
- Bot tidak pernah broadcast ke nomor random; balasan selalu ke pengirim.

## [v0.1.0] — 2026-08-28
### Added
- `get_stocks_id.py`: 887 saham IDX via TradingView scanner.
- `get_history_id.py`: download 5 tahun OHLCV dari Yahoo (resume-able).
- `train_xgb.py`: feature engineering 34 fitur teknikal + XGBoost baseline.
- `add_macro.py` / `train_xgb_macro.py`: fitur makro IHSG & USD/IDR.
- `walk_forward.py`: validasi walk-forward bulanan (AUC OOS 0.576).
- `train_lstm.py` / `ensemble.py`: pembanding deep learning.
- `per_stock_models.py`: model per-saham 20 likuid + backtest.
- `predict_daily.py`: pipeline produksi harian (retrain + prediksi besok).

### Hasil eksperimen
- XGBoost gabungan + walk-forward: **AUC OOS 0.576** (terbaik).
- LSTM & per-saham kalah; ensemble setara.
