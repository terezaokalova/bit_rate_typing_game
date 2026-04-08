#!/usr/bin/env bash
set -e
PORT=8080

echo ""
echo "  Bit rate game"
echo ""
echo "  starting server at http://localhost:$PORT"
echo "  open that URL in your browser to play"
echo "  press Ctrl+C to stop"
echo ""

if command -v python3 &> /dev/null; then
    python3 -m http.server "$PORT" --bind 127.0.0.1
elif command -v python &> /dev/null; then
    python -m http.server "$PORT" --bind 127.0.0.1
else
    echo "error: python not found"
    exit 1
fi
