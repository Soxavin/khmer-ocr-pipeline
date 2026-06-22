#!/usr/bin/env bash
# Tear down the Surya llamacpp backend (llama-server) started for OCR.
# Surya keeps the server resident (SURYA_INFERENCE_KEEP_ALIVE=true); on a crash
# or unclean exit it can be orphaned, holding unified memory and a port. Run this
# to stop it cleanly:
#   bash stop-metal-macos.sh
#
# Single-user scope: this matches any process named "llama-server". If you run a
# separate llama-server for other purposes, stop that yourself instead.

pids=$(pgrep -f llama-server)

if [ -z "$pids" ]; then
  echo "No llama-server process found — nothing to stop."
  exit 0
fi

echo "Stopping llama-server process(es): $pids"
# Graceful first, then force any that survive.
kill $pids 2>/dev/null
sleep 2
remaining=$(pgrep -f llama-server)
if [ -n "$remaining" ]; then
  echo "Force-killing: $remaining"
  kill -9 $remaining 2>/dev/null
fi

if pgrep -f llama-server >/dev/null; then
  echo "WARNING: some llama-server process(es) could not be stopped."
  exit 1
fi
echo "llama-server stopped."
