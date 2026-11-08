SHELL := /bin/bash
STACK := taller-rag
CONTEXT := server
COMPOSE_ALL := docker compose -f docker-compose.all.yml
SAMPLE ?= eval/all_docs.parquet
COLLECTION ?= rag_full_bge_m3_d1024_v1
SPARSE ?= rag_full_e5base_d768_v2_bm25_v1

.PHONY: up down ps logs health prewarm all-up all-down

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

all-up:
	$(COMPOSE_ALL) up -d --build
	@echo "esperando la API..."
	@for i in $$(seq 1 90); do \
		if curl -sf http://localhost:8010/health >/dev/null; then break; fi; \
		sleep 5; \
	done
	$(COMPOSE_ALL) exec -T -e OLLAMA_HOST=127.0.0.1:11435 ollama ollama pull gemma4:e4b
	$(COMPOSE_ALL) exec -T -e OLLAMA_HOST=127.0.0.1:11435 ollama ollama pull gemma2:2b
	$(COMPOSE_ALL) --profile rocm run --rm -T ingest-rocm python -m src.scripts.ingest_v2 --sample $(SAMPLE) --collection $(COLLECTION) --drop --batch-size 128
	$(COMPOSE_ALL) exec -T api python -m src.scripts.build_sparse_collection --source $(COLLECTION) --target $(SPARSE)
	$(COMPOSE_ALL) exec -T api python -m src.scripts.index_solr --solr http://localhost:8983/solr/noticias
	$(COMPOSE_ALL) exec -T api python -m src.scripts.index_solr_vectors --solr http://localhost:8983/solr/noticias
	$(COMPOSE_ALL) exec -T api python -m src.scripts.prewarm
	@echo "listo: http://localhost:8501"

all-down:
	$(COMPOSE_ALL) down
