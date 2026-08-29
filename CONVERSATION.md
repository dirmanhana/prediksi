# 📝 Catatan Percakapan — Proyek XGBoost IDX

> Proyek: Analisis & prediksi saham Bursa Efek Indonesia (IDX) dengan Python + XGBoost
> Tanggal: 2026-08-28
> Lokasi: `/home/dirman/workspace/xgbooxt`

---

## 1. Permintaan awal: daftar semua saham Indonesia

**User**: "halo, gunakan python dan buat data list semua saham yang ada di indonesia"

**Yang dikerjakan:**
- Mencoba endpoint resmi IDX (`idx.co.id`) → **diblokir Cloudflare (403)**.
- Mencoba alternatif: r.jina.ai proxy (gagal, CAPTCHA), ICDX.io (domain mati), Yahoo screener (crumb butuh auth, region tidak memfilter).
- **Solusi yang berhasil**: TradingView Scanner `scanner.tradingview.com/indonesia/scan` → 887 saham IDX.
- Membuat `get_stocks_id.py`: mengambil 887 saham + nama perusahaan, sektor, industri, harga, volume, market cap (dihitung dari close × saham beredar).
- Output: `data/stocks_id.csv`, `data/stocks_id.xlsx`, `data/stocks_id.json`.

**Hasil**: 887 saham, 20 sektor (Finance 179, Process Industries 93, dst). Top market cap: BBCA (Rp 801T), BYAN, DCII, BBRI, BREN.

---

## 2. Data historis 5 tahun untuk modeling

**User**: "ambil data historis 5 tahun untuk modeling"

**Yang dikerjakan:**
- Membuat `get_history_id.py` — downloader daily OHLCV 5 tahun dari Yahoo Finance (format `BBRI.JK`).
- Fitur: resume/checkpoint per saham, rate-limit handling (429 backoff), crumb refresh (401), simpan langsung per file `data/history/{kode}.csv`.
- Menjalankan semua 887 saham di background (~15 menit).

**Hasil**: 948.713 baris, 887 saham, periode 2021-08-30 s/d 2026-08-28.
- `data/history_id_5y.parquet` (18 MB) dan `.csv` (86 MB)
- Kolom: ticker, tanggal, OHLCV, adj_close, return_1d, log_return
- Catatan: saham IPO baru punya riwayat lebih pendek (AADI Des 2024, dll).

---

## 3. Model XGBoost: prediksi arah besok

**User**: "ya buat arah pergerakan besok, untuk data base bebas, boleh gunakan apa saja kalau di butuhkan"

**Yang dikerjakan:**
- `train_xgb.py`:
  - **Feature engineering** (34 fitur per saham, tanpa lookahead): return lags (1,2,3,5,10,20), log return, rasio harga vs SMA (5,10,20,50,100), EMA, MACD, RSI-14 (Wilder), Bollinger %B & width, volatilitas, ATR, volume ratio, gap, range high-low, jarak 52-minggu, dayofweek, month.
  - **Target**: 1 jika close besok > close hari ini (flat → 0).
  - **Database**: fitur disimpan ke SQLite `data/features.db` (846.643 baris).
  - **Split berbasis waktu** (anti-lookahead): 68% train / 12% valid / 20% test berdasarkan tanggal.
  - XGBoost (depth 6, lr 0.05, early stopping) + tuning threshold optimal (max F1).
- Kendala yang diatasi: XGBoost 3.x memindahkan `early_stopping_rounds` ke constructor; data `inf` dari volume=0 → ganti NaN; demo prediksi dtype object → cast float.

**Hasil** (data TEST unseen Sep 2025–Agu 2026, 187.855 baris):
- ROC-AUC **0.598**, akurasi 62.8% vs baseline 62.1% (threshold 0.5)
- Threshold optimal 0.30 → F1 0.555, recall 86%, precision 41%
- Fitur terpenting: **vol_5** (volatilitas), ret_1, log_ret_1, hl_range, ret_2, close_sma5.
- Model: `data/model_xgb_direction.json`, plot feature importance, `data/metrics.json`.

---

## 4. DeepSeek / deep learning?

**User**: "kalau gunakan model seperti deepseek?"

**Klarifikasi**: dua makna — (1) DeepSeek LLM: butuh API key (tidak ada), tidak cocok untuk data OHLCV; berguna untuk sentimen berita. (2) Deep learning: diuji dengan LSTM.

**Yang dikerjakan:**
- `train_lstm.py` — LSTM 2 layer (hidden 48, ~36.5K param) PyTorch CPU.
- Window 10 hari × 34 fitur, 824.426 sample, split waktu sama, early stopping pada valid AUC.

**Hasil perbandingan** (data TEST sama):

| Model | ROC-AUC | F1 |
|---|---|---|
| XGBoost | **0.598** | 0.554 |
| LSTM | 0.536 | 0.547 |

XGBoost menang — wajar untuk data tabular. Deep learning butuh data jauh lebih banyak, fitur mentah, dan GPU.

---

## 5. Fitur makro + walk-forward + ensemble

**User**: "ya lanjut, kecuali fitur sentimen berita"

**Yang dikerjakan:**
1. `add_macro.py` — fetch IHSG (^JKSE) & USD/IDR (IDR=X) dari Yahoo → `data/macro_id.csv`, `data/features_macro.db` (830.508 baris × 53 kolom, 8 fitur makro baru).
2. `train_xgb_macro.py` — XGBoost + makro: AUC 0.581 (TURUN dari 0.598) → makro tidak membantu di pendekatan gabungan.
3. `walk_forward.py` — retrain bulanan (expanding window), 12 langkah:

   **Hasil**: AUC OOS **0.576**, konsisten per bulan (0.55–0.62, tidak pernah random). Ini angka jujur/realistis.
4. `train_lstm.py` (dimodifikasi) — LSTM + makro: AUC 0.528.
5. `ensemble.py` — rata-rata proba XGBoost(walk-forward) + LSTM:

   | Model | AUC | F1 |
   |---|---|---|
   | XGBoost walk-forward | 0.576 | 0.555 |
   | LSTM | 0.528 | 0.554 |
   | Ensemble 50/50 | 0.567 | 0.558 |

**Kesimpulan**: ensemble F1 naik tipis, AUC tidak; LSTM terlalu lemah.

---

## 6. Model per-saham (rekomendasi diuji — ternyata kalah)

**User**: "ya, terapkan rekomendasi mu"

**Yang dikerjakan:** `per_stock_models.py` — 20 saham paling likuid (nilai transaksi 12 bln terakhir), masing-masing:
- Klasifikasi walk-forward bulanan → AUC per saham.
- Regresi return besok (XGBRegressor) + backtest strategi "long jika pred > 0" (biaya 0.2% round-trip) vs Buy & Hold.

**Hasil**:
- Rata-rata AUC/saham: **0.506** (≈ random!) — hanya 4/19 saham AUC > 0.55 (BBCA 0.60, ASII 0.58, DSSA 0.56, CUAN 0.55).
- Strategi menang vs Buy&Hold: 9/19 (lempar koin).
- **Kesimpulan penting**: model per-saham kalah dari model gabungan — data per saham (~700–1.100 baris) terlalu sedikit untuk XGBoost 42 fitur. Model gabungan 875 saham lebih kuat karena berbagi kekuatan statistik.

**Peringkat akhir semua pendekatan**:

| # | Pendekatan | AUC OOS | Status |
|---|---|---|---|
| 1 | XGBoost gabungan + walk-forward | 0.576 | 🏆 Terbaik |
| 2 | XGBoost gabungan (single split) | 0.598* | optimistis |
| 3 | Ensemble XGB+LSTM | 0.575 | setara |
| 4 | XGBoost per-saham | 0.506 | ❌ |
| 5 | LSTM | 0.528 | ❌ |

\*0.598 optimistis (1x train); angka jujur = 0.576.

---

## 7. Pipeline produksi (pengemasan)

**User**: "ya kemas"

**Yang dikerjakan:** `predict_daily.py` — pipeline prediksi harian produksi:
1. (opsional `--refresh`) update incremental data historis dari Yahoo.
2. Rebuild dataset gabungan dari `data/history/*.csv`.
3. Feature engineering (34 teknikal + 8 makro).
4. Retrain XGBoost gabungan (validasi = 15% tanggal terakhir, early stopping) — valid_auc 0.579, ±17 detik.
5. Tuning threshold optimal (0.30, max F1).
6. Prediksi besok untuk 875 saham → `data/predictions_tomorrow.csv` & `.json`, `data/model_daily.ubj`, `data/prediction_log.csv`.
7. Perbaikan: bug `^JKSE.JK` (suffix .JK salah untuk makro), sinyal pakai threshold optimal bukan 0.5.

**Contoh rekomendasi (28 Agu 2026)**: bullish PPGL 0.61, MGLV 0.60, BOLT 0.60; bearish BABP 0.01, MAXI 0.01, GOTO 0.02. 676/875 diprediksi naik.

Crontab harian: `30 17 * * 1-5 cd ... && python3 predict_daily.py --refresh`

---

## 8. Penutup (file final)

- `README.md` — dokumentasi proyek.
- `CONVERSATION.md` — catatan percakapan ini.

## ⚠️ Disclaimer

Semua hasil riset ini **bukan saran investasi**. AUC ~0.58 = sinyal lemah;
jangan digunakan untuk trading penuh tanpa validasi lebih lanjut.

## 📁 Daftar script proyek

| Script | Fungsi |
|---|---|
| `get_stocks_id.py` | Daftar 887 saham IDX (TradingView scanner) |
| `get_history_id.py` | Download 5 tahun OHLCV per saham (Yahoo, resume) |
| `train_xgb.py` | Feature engineering + XGBoost baseline → SQLite |
| `add_macro.py` | Fitur makro IHSG & USD/IDR |
| `train_xgb_macro.py` | XGBoost + makro |
| `walk_forward.py` | Walk-forward validation (retrain bulanan) |
| `train_lstm.py` | LSTM (PyTorch) pembanding |
| `ensemble.py` | Ensemble XGBoost + LSTM |
| `per_stock_models.py` | Model per-saham (20 likuid) + backtest |
| `predict_daily.py` | **Pipeline produksi harian** |

---

## 9. Perbaikan label akhir pekan + fitur baru (v0.8.2–v0.9.0)

**User**: "prediksi untuk besok padahal kemarin Jumat sekarang Sabtu — periksa logika"

**Bug 1 (label)**: `tanggal_prediksi = data_sampai + 1 hari` tanpa akhir pekan
→ data s/d Jumat dilabeli Sabtu. Fix: `next_trading_day()` (lewati Sabtu/Minggu).

**Bug 2 (tersembunyi)**: makro (IHSG) dari Yahoo tertinggal 1 hari → baris tanggal
terakhir gugur karena NaN makro → `data_sampai` & prediksi mundur 1 hari (prediksi
utk hari yang sudah lewat). Fix: merge makro `merge_asof` mundur.

**Fitur baru (v0.9.0)**: `kenapa KODE` (SHAP), `rekap` (rekap pasar, bisa
auto-push via RECAP_PUSH_TIME), `riwayat KODE`, `cek KODE` diperkaya (perkiraan
return + ukuran saran KECIL/SEDANG/BESAR), fitur RS-sektor, kalender libur IDX
(paket holidays), notifikasi error ke admin.

**Eksperimen RS-sektor** (walk-forward 12 bulan, 205.561 baris OOS):
baseline 0.5910 vs +RS 0.5918 → delta **+0.0008 = netral**. Tetap disertakan
utk konteks penjelasan.

**Foreign flow**: tidak tersedia di sumber gratis (TradingView scanner tidak punya
kolomnya) → tidak diimplementasikan.

**Status**: prediksi utk Senin 31-08-2026 (data s/d Jumat 28-08), valid_auc 0.5811,
45 fitur. Bot siap diuji.

---

## 10. Foreign flow dari idx.co.id (v0.10.0)

**User**: "untuk foreign flow bisakah ambil data scrape dari https://www.idx.co.id saja?"

**Investigasi**: idx.co.id diblokir Cloudflare (403) utk requests biasa —
cloudscraper juga gagal. r.jina.ai kena CAPTCHA. Tapi **headless Chrome
(google-chrome, sudah terinstall) MENEMBUS Cloudflare** → API resmi
`/primary/TradingSummary/GetStockSummary?date=YYYY-MM-DD` bisa dipanggil
via fetch() di dalam halaman (sesi CF sama), mengembalikan **ForeignBuy &
ForeignSell per saham** (963 saham/hari).

**Backfill**: `scrape_foreign_flow.py` — 1.199 hari perdagangan (2021-08 s/d
2026-08), 1.065.585 baris, 0 gagal, ±16 menit. Resume-able + checkpoint
tiap 40 hari; `--latest` utk incremental harian (dipanggil otomatis saat
`predict_daily.py --refresh`).

**Eksperimen fitur** (walk-forward 12 bulan, 205.561 baris OOS):
baseline 0.5973 vs +FF 0.5938 → delta **-0.0035 = MERUGIKAN** (kalah 7/12
bulan). Net asing per-saham ternyata tidak menambah sinyal model gabungan.

**Keputusan**: `USE_FOREIGN_FLOW=False` (bukan fitur model), tapi data
tetap ditampilkan di bot — `rekap` (Net Asing pasar: 28 Agu = -Rp 409 M,
net jual) & `cek KODE` (beli/jual/net per saham, BBRI net +Rp 10 M).

**Workaround teknis**: `merge_asof` dgn `by=` rusak di pandas 3.0.3 →
pakai exact merge + ffill per saham (setara asof mundur, tanpa lookahead).

**Status**: prediksi utk Senin 31-08-2026 (45 fitur, valid_auc 0.5819).
