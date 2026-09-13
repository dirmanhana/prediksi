# Changelog

Semua perubahan penting pada proyek XGBoost IDX.

## [v0.12.2] — 2026-09-13
### Fixed
- **Laporan watchlist/rekap terkirim ulang setiap restart bot** setelah jam
  laporan. Dulu `last_watch`/`last_recap` hanya di memori (reset saat start),
  jadi restart sore/malam mengirim laporan lagi. Sekarang tanggal terakhir
  kirim disimpan ke `data/report_state.json`.
- **Capture sesi mengulang ticker yang tidak mungkin ada datanya.** Arsip
  memuat tanggal non-bursa (mis. `2026-08-29` = Sabtu) dan beberapa ticker
  tanpa bar 15m. `tickers_needing_capture()` kini menyaring pasangan lewat data
  harian (parquet) -> hanya hari bursa yang benar-benar ditransaksikan yang
  di-fetch, tidak diulang terus tiap hari.
- **Monev bisa dievaluasi sebelum data harian siap.** `MONEV_TIME` default
  dipindah ke **18:30 WIB** (setelah cron `predict_daily` 17:30 WIB) supaya
  close harian hari itu sudah ada. Kalau proses gagal, `monev_worker` akan
  mencoba lagi (dulu langsung ditandai selesai).

### Added
- **Backup harian `data/session_bars.csv`** ke `data/backups/` (rotasi 14 file
  terakhir). Data sesi tidak di-commit & tidak bisa diregenerate setelah
  jendela 60 hari Yahoo, jadi perlu backup terpisah. Lokasi bisa diubah via
  env `SESSION_BACKUP_DIR`.

### Changed
- Jadwal monev harian: `MONEV_TIME=18:30` WIB (dulu 16:10).

## [v0.12.1] — 2026-09-13
### Changed
- **Seluruh sistem memakai WIB (Asia/Jakarta).** Timezone server diubah dari
  `Etc/UTC` → `Asia/Jakarta`; crontab & laporan otomatis kini berjalan sesuai
  jam WIB (sebelumnya `WATCH_REPORT_TIME=18:00` efektif terkirim ~01:00 WIB).
- **Modul baru `waktu.py`** — satu sumber kebenaran waktu (`now_wib`,
  `today_wib`, `WIB`). Semua script (`wa_bot`, `predict_daily`,
  `get_history_id`, `add_macro`, `get_stocks_id`, `session_data`,
  `eval_report`) memakai modul ini, jadi perilaku tidak lagi bergantung pada
  timezone server (aman kalau VPS baru masih UTC).
- Normalisasi tanggal dari Yahoo (daily & intraday) kini eksplisit WIB.

### Fixed
- **`report_worker` (laporan watchlist & rekap pasar) meleset 7 jam** — dulu
  memakai waktu lokal server yang UTC. Sekarang berbasis WIB.

## [v0.12.0] — 2026-09-13
### Added
- **Perintah `monev`** (alias `evaluasi`) — laporan hasil prediksi vs AKTUAL,
  memuat penutupan **sesi 1** (12:00 / Jumat 11:30 WIB) dan **sesi 2**
  (16:00 WIB = penutupan resmi). Menampilkan 3 tolok ukur: model close→close,
  entry pagi → exit sesi 1, dan entry pagi → exit penutupan — plus rata-rata
  return & hit-rate per hari.
- **`session_data.py` + `capture_session.py`** — ambil bar 15-menit Yahoo untuk
  saham yang diprediksi (top-20 NAIK + top-20 TURUN), lalu simpan penutupan per
  sesi ke `data/session_bars.csv`. Yahoo hanya menyediakan interval 15m untuk
  ~60 hari terakhir, jadi data harus disimpan agar tidak hilang.
- **`eval_report.py`** — bangun `data/eval/prediction_eval.csv` +
  `data/eval/monev_summary.json`, lalu auto-commit + push ke origin.
- **Monev harian otomatis** (thread `monev_worker`) setelah `MONEV_TIME`
  (default 16:10 WIB) + notifikasi ke admin.

### Fixed
- **Verifikasi rekam jejak mandek di mode POLLING** — `verify_and_record()`
  dulu hanya dijalankan sekali saat start, karena loop periodiknya cuma ada di
  mode webhook. Sekarang dijalankan tiap 10 menit dari `monev_worker` (berlaku
  di kedua mode), sehingga `bot_track_record.csv` tidak lagi basi.

### Catatan
- Server berjalan di UTC; jadwal monev dihitung dari WIB (UTC+7).
- `data/session_bars.csv` TIDAK di-commit; hanya `data/eval/` yang di-commit.

## [v0.11.3] — 2026-08-29
### Fixed
- **Diagnostik jelas saat data training kosong**: error `IndexError ... size 0`
  di `train_and_predict` (sering: history VPS kosong/parsial/schema beda) kini
  menampilkan info lengkap — jumlah baris fitur, target terisi, kolom dengan
  >90% NaN, dan ringkasan dataset mentah.
- **Log lengkap `predict_daily`**: saat `update`/`update paksa` via bot, output
  penuh subproses disimpan ke `data/predict_daily.log` (bot hanya membalas
  baris terakhir) — memudahkan diagnosa di VPS.

## [v0.11.2] — 2026-08-29
### Changed
- **`update paksa` lebih pintar**: kalau data sudah terkini DAN semua file
  pendukung model ada (prediksi, model .ubj, metadata, last_features) →
  balas "✅ Data sudah terkini + model siap" tanpa refresh. Tetap refresh
  kalau ada file kurang (mis. habis git pull) atau data basi.

## [v0.11.1] — 2026-08-29
### Fixed
- **Crash `predict_daily.py --refresh` di VPS** (TypeError: Cannot compare
  Timestamp with datetime.date): kolom `tanggal` hasil fetch Yahoo berupa
  `datetime.date`, sedangkan CSV lama di-parse sebagai Timestamp → gagal saat
  `sort_values` di pandas 3.x. Sekarang `fetch_range` mengembalikan Timestamp
  ternormalisasi 00:00 (`.normalize()`) — konsisten dgn CSV & antar sumber
  (IHSG vs USD/IDR punya jam berbeda → sebelumnya merge makro bisa kosong).
- Normalisasi tanggal juga diterapkan defensif sebelum `drop_duplicates`/`sort`
  di `refresh_history`.

## [v0.11.0] — 2026-08-29
### Added
- **Perintah `versi`** — info versi bot (commit hash), model, fitur, AUC, dan
  tanggal prediksi/data — utk verifikasi cepat apakah VPS sudah update.
- **`update paksa`** (alias `paksa update` / `force refresh`) — jalankan
  `predict_daily.py --refresh` SELALU, tanpa cek kefresh-an data. Berguna utk
  setup VPS baru / memaksa retrain walau data dianggap terkini.
- **`update`/`refresh` bisa jalan tanpa file prediksi** — sebelumnya ditolak
  ("Belum ada file prediksi") di VPS fresh; sekarang `update` otomatis jalan
  utk setup pertama (download history 5 tahun + retrain + prediksi).
- Perintah `update` sekarang juga menjawab utk memastikan foreign flow hari
  terakhir ikut ter-update (via `scrape_foreign_flow.py --latest`).

## [v0.10.0] — 2026-08-29
### Added
- **Foreign flow (net asing) dari idx.co.id** — scraping via headless Chrome
  (menembus Cloudflare) + API resmi `GetStockSummary` (kolom ForeignBuy /
  ForeignSell per saham). Backfill 5 tahun selesai: `data/foreign_flow.csv`
  (1.065.585 baris, 1.199 hari, 2021-08 s/d 2026-08).
- `scrape_foreign_flow.py` — backfill penuh / incremental (`--latest`);
  dipanggil otomatis saat `predict_daily.py --refresh` (best-effort).
- **Bot**: `rekap` menampilkan **Net Asing pasar**; `cek KODE` menampilkan
  beli/jual/net asing per saham.
### Notes
- **Sebagai FITUR MODEL, foreign flow MERUGIKAN**: eksperimen walk-forward
  (12 bulan, 205.561 baris OOS) → delta AUC **-0.0035** (0.5973 → 0.5938,
  kalah 7/12 bulan). Fitur dinonaktifkan (`USE_FOREIGN_FLOW=False`); data
  tetap dipakai utk info/tampilan bot.
- Workaround: `merge_asof` dengan `by=` bermasalah di pandas 3.0.3 → pakai
  exact merge + ffill per saham (setara asof mundur, tanpa lookahead).

## [v0.9.0] — 2026-08-29
### Added
- **`kenapa KODE`**: penjelasan sinyal per saham via SHAP (xgboost `pred_contribs`,
  tanpa dependency baru) — fitur apa yg mendorong NAIK/TURUN. Data fitur terakhir
  per saham disimpan di `data/last_features.parquet`.
- **`rekap`**: ringkasan pasar (IHSG, USD/IDR, breadth saham likuid, top
  gainers/losers, nilai transaksi). Bisa di-push otomatis harian via
  `RECAP_PUSH_TIME` di `.env` (kosongkan utk nonaktif).
- **`riwayat KODE`**: rekam jejak prediksi historis per saham (terverifikasi).
- **`cek KODE` diperkaya**: perkiraan return besok (`pred_ret`, model regresi
  XGBoost baru) + ukuran saran (KECIL/SEDANG/BESAR) + rekam jejak NAIK saham itu.
- **Fitur relative strength vs sektor** (`sector_ret_1`, `rs_sector_1`,
  `rs_sector_5`) di pipeline produksi. Hasil eksperimen walk-forward (12 bulan,
  205.561 baris OOS): delta AUC **+0.0008 → netral** (tidak merugikan, 7/12 bulan
  menang). Tetap disertakan utk konteks penjelasan sinyal.
- **Kalender libur IDX**: `next_trading_day()` juga melewati hari libur nasional
  Indonesia (paket `holidays`, ditambahkan ke install.sh).
- **Notifikasi error ke admin**: kalau `predict_daily.py` gagal (cron/bot),
  admin mendapat WA berisi ringkasan error (best-effort).

### Notes
- **Aliran asing (foreign flow) TIDAK diimplementasikan** — tidak tersedia dari
  sumber gratis (TradingView tidak punya kolomnya, RTI berbayar, bursa.go.id
  diblokir). Diverifikasi langsung ke scanner API.

## [v0.8.2] — 2026-08-29
### Fixed
- **Label tanggal prediksi akhir pekan**: `tanggal_prediksi` sebelumnya dihitung
  `data_sampai + 1 hari` — kalau data berakhir Jumat, label jadi Sabtu (bursa
  tutup). Sekarang pakai `next_trading_day()` yang melewati Sabtu/Minggu
  (mis. data s/d Jumat → label prediksi Senin). Berlaku utk JSON & arsip.
- **Data makro tertinggal memotong prediksi**: fitur makro (IHSG/USD-IDR)
  digabung dgn `merge_asof` mundur — kalau Yahoo belum punya nilai indeks utk
  tanggal terakhir (mis. IHSG tertinggal 1 hari), dipakai nilai hari tersedia
  sebelumnya. Sebelumnya baris tanggal terakhir gugur (NaN makro) sehingga
  `data_sampai` & prediksi mundur 1 hari (prediksi jadi utk hari yang sudah
  lewat).
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
- Tambah/hapus user **tidak perlu restart** — daftar dibaca dinamis dari
  `data/allowed_numbers.json` (polling & webhook). Restart hanya utk
  perubahan `.env`.

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
