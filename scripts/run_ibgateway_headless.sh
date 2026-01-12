#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:1
export HOME=/home/aleks

# Start Xvfb if not running
if ! pgrep -x Xvfb >/dev/null; then
  /usr/bin/Xvfb :1 -screen 0 1280x720x16 &
fi

# Wait until Xvfb is actually ready
for i in {1..100}; do
  if DISPLAY=:1 xdpyinfo >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

exec /home/aleks/Jts/ibgateway/1042/ibgateway
