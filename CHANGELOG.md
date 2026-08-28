# Changelog

Semua perubahan penting pada proyek XGBoost IDX.

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
