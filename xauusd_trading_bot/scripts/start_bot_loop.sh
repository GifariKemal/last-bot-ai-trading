#!/bin/bash
# Auto-restart loop untuk XAUUSD Trading Bot.
# Handles IC Markets demo server maintenance window (~01:00-04:30 UTC).
# Bot exit bersih → tunggu 30 detik → restart otomatis.
#
# Usage:
#   bash start_bot_loop.sh
#   bash start_bot_loop.sh 2>logs/bot_live_err.log &   # background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Auto-restart loop started"

while true; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting bot..."
    python main.py --mode live -y

    EXIT_CODE=$?
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Bot exited (code $EXIT_CODE). Restarting in 30 seconds..."
    sleep 30
done
