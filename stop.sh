#!/usr/bin/env bash
# ============================================================
# stop.sh — Hentikan bot WhatsApp Prediksi Saham IDX
# Jalan utama: systemd (wa_bot.service).
# Fallback   : matikan proses wa_bot.py via PID file.
# ============================================================
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
SERVICE="wa_bot"
PIDFILE="$BASE/wa_bot.pid"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

if command -v systemctl >/dev/null 2>&1 && [ -f "/etc/systemd/system/$SERVICE.service" ]; then
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        echo "🛑 Menghentikan service $SERVICE..."
        $SUDO systemctl stop "$SERVICE"
        echo "✅ Bot berhenti."
    else
        echo "ℹ️  Service $SERVICE sudah berhenti / tidak aktif."
    fi
    exit 0
fi

# ---- Fallback: kill via PID file ----------------------------
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "🛑 Menghentikan bot (PID $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 2
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
fi

if pgrep -f "wa_bot.py" >/dev/null 2>&1; then
    echo "🛑 Masih ada proses wa_bot.py — mematikan..."
    pkill -f "wa_bot.py" 2>/dev/null || true
fi
echo "✅ Bot berhenti."