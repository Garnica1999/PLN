SHELL := /bin/bash
STACK := taller-rag
CONTEXT := server

.PHONY: up down ps logs health prewarm

up:
	./scripts/up.sh up

down:
	./scripts/up.sh down

ps:
	./scripts/up.sh status

logs:
	./scripts/up.sh logs $(service)

health:
	./scripts/up.sh health

prewarm:
	./scripts/up.sh prewarm
