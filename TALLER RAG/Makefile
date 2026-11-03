SHELL := /bin/bash
SERVER := carlos@192.168.1.100
DATA_COMPOSE := docker --context server compose -f docker-compose.data.yml
APP_COMPOSE := docker compose

.PHONY: up down ps logs tunnel health

up: tunnel
	$(DATA_COMPOSE) up -d
	$(APP_COMPOSE) up -d --no-deps api frontend
	@echo "esperando a que la API vea Milvus..."
	@until curl -fsS http://localhost:8010/health 2>/dev/null | grep -q '"milvus":true'; do sleep 2; done
	@echo "stack listo"

down:
	$(APP_COMPOSE) stop api frontend
	$(DATA_COMPOSE) stop

ps:
	@echo "== servidor =="
	@$(DATA_COMPOSE) ps
	@echo "== estacion =="
	@$(APP_COMPOSE) ps

logs:
	$(DATA_COMPOSE) logs --tail=80

tunnel:
	@ss -ltn | grep -q ':19530 ' || ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 19530:localhost:19530 $(SERVER)
	@echo "tunel 19530 activo"

health:
	@curl -s http://localhost:8010/health; echo
