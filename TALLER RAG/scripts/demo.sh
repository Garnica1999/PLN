#!/usr/bin/env bash
set -uo pipefail

API="${API_URL:-http://localhost:8010}"
QUESTION="${QUESTION:-¿En qué ciudad se celebrará la COP16?}"

echo "== health =="
curl -s --max-time 15 "$API/health" | python3 -m json.tool

echo "== retrieve =="
curl -s --max-time 60 "$API/retrieve" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"$QUESTION\",\"k\":5}" | python3 -m json.tool | head -60

echo "== ask =="
curl -s --max-time 240 "$API/ask" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"$QUESTION\",\"k\":5,\"ambiguity_policy\":\"off\"}" | python3 -m json.tool | head -80
