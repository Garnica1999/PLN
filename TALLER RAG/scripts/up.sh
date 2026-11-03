#!/usr/bin/env bash
# Capa de datos en Docker Swarm (servidor) + aplicacion con GPU en la estacion.
# Uso: scripts/up.sh [up|down|status|logs|prewarm|health]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STACK="${STACK:-taller-rag}"
CONTEXT="${CONTEXT:-server}"
FILE="${FILE:-docker-stack.yml}"
STACK_CMD=(docker --context "$CONTEXT" stack)
APP_SERVICES=(api frontend ollama)
HEALTH_URL="${HEALTH_URL:-http://localhost:8010/health}"

log() { printf '[stack] %s\n' "$*"; }
die() { printf '[stack] error: %s\n' "$*" >&2; exit 1; }

ensure_swarm() {
  state="$(docker --context "$CONTEXT" info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null || true)"
  [ "$state" = "active" ] || die "el nodo $CONTEXT no esta en un swarm; ejecuta docker swarm init"
}

wait_data() {
  log "esperando etcd, MinIO y Milvus"
  for _ in $(seq 1 120); do
    if "${STACK_CMD[@]}" ps "$STACK" --filter desired-state=running \
        --format '{{.CurrentState}}' 2>/dev/null | grep -c '^Running' | grep -q '^3$'; then
      return
    fi
    sleep 2
  done
  die "la capa de datos no quedo lista; revisa: $0 status"
}

wait_api() {
  log "esperando la API"
  for _ in $(seq 1 150); do
    if curl -fsS "$HEALTH_URL" 2>/dev/null | grep -q '"milvus":true'; then
      return
    fi
    sleep 2
  done
  die "la API no ve Milvus; revisa: $0 status"
}

up() {
  ensure_swarm
  log "desplegando la capa de datos en $CONTEXT"
  # Tras un stack rm la red overlay tarda en desaparecer y el deploy puede
  # fallar con "network not found"; se reintenta.
  for attempt in 1 2 3; do
    if "${STACK_CMD[@]}" deploy -c "$FILE" "$STACK"; then
      break
    fi
    [ "$attempt" -eq 3 ] && die "no se pudo desplegar $STACK"
    log "reintentando el despliegue en 5 s"
    sleep 5
  done
  wait_data
  log "levantando API, frontend y Ollama con GPU"
  docker compose up -d --no-deps "${APP_SERVICES[@]}"
  wait_api
  log "listo: http://localhost:8501  API http://localhost:8010/health"
}

down() {
  log "deteniendo la aplicacion local"
  docker compose stop "${APP_SERVICES[@]}"
  log "eliminando los servicios del stack (los volumenes se conservan)"
  "${STACK_CMD[@]}" rm "$STACK"
}

status() {
  echo "== servicios del stack =="
  "${STACK_CMD[@]}" services "$STACK"
  echo "== tareas del stack =="
  "${STACK_CMD[@]}" ps "$STACK"
  echo "== aplicacion local =="
  docker compose ps
  echo "== salud =="
  curl -s "$HEALTH_URL" || true
  echo
}

logs() {
  local service="${1:-milvus}"
  case "$service" in
    api|frontend|ollama|prewarm) docker compose logs --tail=80 "$service" ;;
    *) docker --context "$CONTEXT" service logs --tail=80 "${STACK}_${service}" ;;
  esac
}

prewarm() { docker compose run --rm prewarm; }

health() { curl -s "$HEALTH_URL"; echo; }

case "${1:-up}" in
  up) up ;;
  down) down ;;
  status|ps) status ;;
  logs) shift; logs "$@" ;;
  prewarm) prewarm ;;
  health) health ;;
  *) echo "uso: $0 [up|down|status|logs <servicio>|prewarm|health]" >&2; exit 1 ;;
esac
