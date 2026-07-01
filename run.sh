#!/usr/bin/env bash
# Start (or stop) the stock-plan dashboard without needing Claude.
# Usage:
#   ./run.sh          start the app (or just open it if already running)
#   ./run.sh stop      stop the running app

set -euo pipefail
cd "$(dirname "$0")"

PIXI="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"
PORT=8501
URL="http://localhost:$PORT"
LOG="stock-plan.log"

if [[ "${1:-}" == "stop" ]]; then
    PIDS=$(lsof -ti "tcp:$PORT" || true)
    if [[ -n "$PIDS" ]]; then
        echo "$PIDS" | xargs kill
        echo "Stopped stock-plan (port $PORT)."
    else
        echo "Nothing running on port $PORT."
    fi
    exit 0
fi

if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
    echo "Already running at $URL"
    open "$URL"
    exit 0
fi

echo "Starting stock-plan..."
nohup "$PIXI" run streamlit run app.py --server.headless true --server.port "$PORT" > "$LOG" 2>&1 &
disown

for _ in $(seq 1 30); do
    if curl -s -o /dev/null "$URL"; then
        open "$URL"
        echo "Running at $URL  (stop anytime with: ./run.sh stop)"
        exit 0
    fi
    sleep 0.5
done

echo "Still starting up — check $URL in a moment, or see $LOG if it doesn't come up."
