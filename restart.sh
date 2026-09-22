#!/usr/bin/env bash
# ============================================================
# restart.sh — Restart bot WhatsApp Prediksi Saham IDX
# Jalan utama: systemd (wa_bot.service).
# Fallback   : stop + start via nohup (wa_bot.pid).
# ============================================================
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
SERVICE="wa_bot"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

if command -v systemctl >/dev/null 2>&1 && [ -f "/etc/systemd/system/$SERVICE.service" ]; then
    echo "🔄 Restart service $SERVICE..."
    $SUDO systemctl restart "$SERVICE"
    sleep 2
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        echo "✅ Bot berjalan. Log: journalctl -u $SERVICE -f"
    else
        echo "⚠️  Service gagal jalan — cek: journalctl -u $SERVICE -n 50"
        exit 1
    fi
    exit 0
fi

# ---- Fallback: stop lalu start nohup ------------------------
echo "🔄 Restart bot (nohup)..."
"$BASE/stop.sh"
"$BASE/start.sh"