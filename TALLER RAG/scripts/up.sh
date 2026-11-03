#!/usr/bin/env bash
# Arranca la capa de datos en el servidor y la aplicacion en la estacion.
# Uso: scripts/up.sh [up|down|status]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-carlos@192.168.1.100}"
CONTEXT="${CONTEXT:-server}"
DATA_COMPOSE=(docker --context "$CONTEXT" compose -f docker-compose.data.yml)
APP_COMPOSE=(docker compose)
LOCAL_SERVICES=(api frontend ollama)
HEALTH_URL="${HEALTH_URL:-http://localhost:8010/health}"

log() { printf '[stack] %s\n' "$*"; }
die() { printf '[stack] error: %s\n' "$*" >&2; exit 1; }

ensure_context() {
  if ! docker context inspect "$CONTEXT" >/dev/null 2>&1; then
    log "creando contexto Docker $CONTEXT -> $SERVER"
    docker context create "$CONTEXT" --docker "host=ssh://$SERVER" >/dev/null
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$SERVER" true \
    || die "no hay clave SSH para $SERVER; ejecuta ssh-copy-id $SERVER"
}

ensure_tunnel() {
  if ss -ltn | grep -q ':19530 '; then
    log "tunel 19530 ya activo"
    return
  fi
  log "abriendo tunel 19530 -> $SERVER"
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 19530:localhost:19530 "$SERVER"
  for _ in $(seq 1 15); do
    ss -ltn | grep -q ':19530 ' && return
    sleep 1
  done
  die "el tunel no abrio el puerto 19530"
}

wait_remote() {
  log "esperando etcd, MinIO y Milvus"
  for _ in $(seq 1 90); do
    state="$("${DATA_COMPOSE[@]}" ps --format '{{.Service}} {{.State}}' 2>/dev/null || true)"
    if grep -q '^etcd running' <<<"$state" \
       && grep -q '^minio running' <<<"$state" \
       && grep -q '^milvus running' <<<"$state"; then
      return
    fi
    sleep 2
  done
  die "la capa de datos remota no quedo lista"
}

wait_api() {
  log "esperando la API"
  for _ in $(seq 1 90); do
    if curl -fsS "$HEALTH_URL" 2>/dev/null | grep -q '"milvus":true'; then
      return
    fi
    sleep 2
  done
  die "la API no ve Milvus; revisa el tunel y los logs"
}

up() {
  ensure_context
  ensure_tunnel
  log "levantando la capa de datos remota"
  "${DATA_COMPOSE[@]}" up -d
  wait_remote
  log "levantando API, frontend y Ollama"
  "${APP_COMPOSE[@]}" up -d --no-deps "${LOCAL_SERVICES[@]}"
  wait_api
  log "listo: http://localhost:8501  API http://localhost:8010/health"
}

down() {
  log "deteniendo la app local"
  "${APP_COMPOSE[@]}" stop "${LOCAL_SERVICES[@]}"
  log "deteniendo la capa de datos remota"
  "${DATA_COMPOSE[@]}" stop
  log "detenido (el tunel sigue abierto)"
}

status() {
  echo "== servidor =="
  "${DATA_COMPOSE[@]}" ps
  echo "== estacion =="
  "${APP_COMPOSE[@]}" ps
  echo "== salud =="
  curl -s "$HEALTH_URL" || true
  echo
}

case "${1:-up}" in
  up) up ;;
  down) down ;;
  status|ps) status ;;
  *) echo "uso: $0 [up|down|status]" >&2; exit 1 ;;
esac
