#!/bin/bash
# Stop Bot Script
# Location: xauusd_trading_bot/stop_bot.sh

PID_FILE="/tmp/xauusd_bot.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Bot is not running (no PID file found)"
    exit 0
fi

BOT_PID=$(cat "$PID_FILE")

if ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo "Stopping bot runner (PID: $BOT_PID)..."

    # Remove PID file first to signal stop
    rm -f "$PID_FILE"

    # Send SIGTERM to runner
    kill "$BOT_PID" 2>/dev/null

    # Wait for graceful shutdown (max 10 seconds)
    for i in {1..10}; do
        if ! ps -p "$BOT_PID" > /dev/null 2>&1; then
            echo "Bot stopped successfully"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Force stopping bot..."
    kill -9 "$BOT_PID" 2>/dev/null
    echo "Bot force stopped"
else
    echo "Bot process not found, cleaning up PID file"
    rm -f "$PID_FILE"
fi

# Also kill any python main.py processes
echo "Cleaning up any remaining bot processes..."
pkill -f "python.*main.py --mode live" 2>/dev/null && echo "Killed remaining processes" || echo "No remaining processes"

echo "Done"
