#!/usr/bin/env bash
# Jalankan bot WhatsApp prediksi saham IDX (pakai data prediksi yang ada)
cd "$(dirname "$0")"
exec python3 wa_bot.py "$@"
