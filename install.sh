#!/usr/bin/env bash
# ============================================================
# install.sh — Deploy Bot Prediksi Saham IDX ke VPS (Ubuntu/Debian)
#
# Cara pakai:
#   1. Salin folder proyek ini ke VPS (scp/git clone)
#   2. cd xgbooxt && bash install.sh
#   3. Ikuti petunjuk: isi .env, lalu download data (opsional)
#
# Catatan: torch TIDAK diinstall (hanya utk eksperimen LSTM, tidak
# dipakai pipeline produksi) — hemat ~750MB.
# ============================================================
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
APP="xgbooxt"
PY="python3"

echo "=== [1/6] Cek Python & pip ==="
$PY --version || { echo "Install python3 dulu: sudo apt install python3 python3-pip"; exit 1; }

echo "=== [2/6] Install dependensi Python (tanpa torch) ==="
$PY -m pip install --break-system-packages \
    pandas numpy requests xgboost scikit-learn openpyxl pyarrow \
    2>&1 | tail -2

echo "=== [3/6] Siapkan .env ==="
if [ ! -f "$BASE/.env" ]; then
    cp "$BASE/.env.example" "$BASE/.env"
    echo "  .env dibuat dari .env.example — SILAKAN EDIT:"
    echo "    nano $BASE/.env"
    echo "  (wajib: CHATETIN_USERNAME, CHATETIN_PASSWORD, ALLOWED_NUMBERS,"
    echo "   WEBHOOK_URL kalau pakai mode webhook)"
    read -r -p "  Tekan Enter setelah selesai mengedit .env..." _ || true
else
    echo "  .env sudah ada — dilewati."
fi

echo "=== [4/6] Data saham & historis ==="
if [ ! -f "$BASE/data/stocks_id.csv" ]; then
    echo "  Mengambil daftar saham IDX (887 saham)..."
    (cd "$BASE" && $PY get_stocks_id.py)
fi
if [ ! -d "$BASE/data/history" ] || [ -z "$(ls -A "$BASE/data/history" 2>/dev/null)" ]; then
    echo "  ⏳ Download 5 tahun data historis (~15-20 menit)..."
    echo "  Bisa di-skip dulu lalu jalankan manual: python3 get_history_id.py 5"
    read -r -p "  Download sekarang? [y/N] " yn || yn=n
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        (cd "$BASE" && $PY get_history_id.py 5)
    else
        echo "  ⏭ Skip — jalankan nanti: cd $BASE && python3 get_history_id.py 5"
    fi
else
    echo "  Data historis sudah ada."
fi

echo "=== [5/6] Service systemd (auto-restart) ==="
SERVICE="/etc/systemd/system/wa_bot.service"
if [ ! -f "$SERVICE" ]; then
    cat > "$SERVICE" <<EOF
[Unit]
Description=WhatsApp Bot Prediksi Saham IDX
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$BASE
ExecStart=$PY $BASE/wa_bot.py
Restart=always
RestartSec=15
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable wa_bot.service
    echo "  Service dibuat: $SERVICE"
else
    echo "  Service sudah ada."
fi

echo "=== [6/6] Cron harian (update data + retrain 17:30 WIB) ==="
CRON_LINE="30 17 * * 1-5 cd $BASE && $PY predict_daily.py --refresh >> $BASE/data/cron.log 2>&1"
( crontab -l 2>/dev/null | grep -v "predict_daily.py --refresh" ; echo "$CRON_LINE" ) | crontab -
echo "  Cron terpasang: 30 17 * * 1-5 (hari kerja)"

echo ""
echo "============================================================"
echo "✅ INSTALL SELESAI"
echo "============================================================"
echo "Langkah berikutnya:"
echo "  1. Uji prediksi dulu (sekali):  cd $BASE && python3 predict_daily.py"
echo "  2. Jalankan bot:                systemctl start wa_bot.service"
echo "     Log:                         journalctl -u wa_bot.service -f"
echo "  3. Cek kesehatan webhook:       curl http://localhost:8080/health"
echo ""
echo "Mode pesan masuk:"
echo "  - Tanpa WEBHOOK_URL di .env  -> polling (cocok utk 1-10 user)"
echo "  - Dengan WEBHOOK_URL         -> webhook (skala besar, hemat API)"
echo "    (arahkan WEBHOOK_URL ke alamat publik VPS, buka port 8080)"
echo "============================================================"
