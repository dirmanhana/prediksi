# XGBoost IDX — Prediksi Arah Saham Indonesia

Pipeline lengkap: ambil data **semua saham Bursa Efek Indonesia (IDX)** →
feature engineering → training XGBoost → **prediksi arah pergerakan besok**
(naik/turun) untuk semua saham.

> 📖 Baca `CONVERSATION.md` untuk catatan lengkap seluruh proses & eksperimen.

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
| `CONVERSATION.md` | Catatan percakapan & seluruh eksperimen |

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
| `data/predictions_tomorrow.csv` / `.json` | **Prediksi besok** (ranking) |
| `data/prediction_log.csv` | Riwayat run harian (AUC, jumlah naik/turun) |

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
