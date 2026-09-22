#!/usr/bin/env bash
# ============================================================
# start.sh — Mulai bot WhatsApp Prediksi Saham IDX
# Jalan utama: systemd (wa_bot.service) — auto-restart & ikut boot.
# Fallback   : nohup background (kalau systemd tidak tersedia).
# ============================================================
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
SERVICE="wa_bot"
PY="$BASE/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
LOG="$BASE/wa_bot.log"
PIDFILE="$BASE/wa_bot.pid"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

[ -f "$BASE/.env" ] || { echo "❌ .env belum ada. Salin dulu: cp .env.example .env"; exit 1; }

if command -v systemctl >/dev/null 2>&1 && [ -f "/etc/systemd/system/$SERVICE.service" ]; then
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        echo "✅ Bot $SERVICE sudah berjalan (systemd)."
        exit 0
    fi
    echo "🚀 Menjalankan bot via systemd ($SERVICE)..."
    $SUDO systemctl start "$SERVICE"
    sleep 2
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        echo "✅ Bot berjalan. Log: journalctl -u $SERVICE -f"
    else
        echo "⚠️  Service gagal start — cek: journalctl -u $SERVICE -n 50"
        exit 1
    fi
    exit 0
fi

# ---- Fallback nohup (tanpa systemd) -------------------------
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "✅ Bot sudah berjalan (PID $(cat "$PIDFILE"))."
    exit 0
fi

echo "🚀 Menjalankan bot di background (nohup)..."
cd "$BASE"
nohup "$PY" "$BASE/wa_bot.py" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 2
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "✅ Bot berjalan (PID $(cat "$PIDFILE")). Log: tail -f $LOG"
else
    echo "⚠️  Bot langsung mati — cek: tail -50 $LOG"
    rm -f "$PIDFILE"
    exit 1
fi