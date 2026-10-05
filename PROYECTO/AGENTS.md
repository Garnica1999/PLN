# AGENTS.md — PROYECTO (Juris-PLN)

Sistema de PLN sobre 23.742 sentencias de la Corte Constitucional de Colombia (1992-2021).
Documentos de contexto: `objetivos.md` (qué busca el proyecto) y `arquitectura.md` (cómo está
construido y con qué modelos).

## Estructura y estado
- `PRIMERA ENTREGA/`: EDA del corpus, informe y figuras (base congelada).
- `SEGUNDA ENTREGA/`: sistema completo (scripts 00-23, `lib/`, modelos, resultados, QA).
- Estado (2026-10-04/05): v1 y v2 cerrados (QA GO); ciclo v3: Sprint A sin adopciones
  (ADR-17), **Sprint B completado con adopción v2.2** (B2 retriever + B1 reranker, ADR-18; QA
  GO; sin deriva temporal en 2022-2026), C3 (corpus 2022-2026) completado, C2 (LLM ALIA
  zero-shot) negativo documentado. Auditoría Fase 1: ruido del silver M4 ~32 % (cleanlab),
  H1/H2/H2b sin adopción, protocolo nuevo (split temporal + holdout sellado) y packs de
  anotación O1/B3 con pre-anotaciones LLM (proxies, NO humanas).
- Antes de continuar, lee en `SEGUNDA ENTREGA/plan/`: `PLAN.md`,
  `PLAN_v3_recuperacion_robusta.md`, `ESTADO_Y_PENDIENTES.md`, `REGISTRO_EXPERIMENTOS.md` y
  las últimas entradas de `contexto.md`.

## Entorno
- Intérprete obligatorio: `/home/carlos/venv/bin/python` (Python 3.14.7); no uses el del sistema.
- Spark solo en el script 00: `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk` (Java 26 rompe
  PySpark 4.2; ADR-7).
- GPU AMD RX 9070 XT (ROCm 7.2) para BETO, embeddings, rerank y fine-tuning; el producto corre
  en CPU. Verifica VRAM libre antes de usarla (el guard de `lib/reranker_bge.py` ya espera).
- LLM local disponible vía ollama (puerto 11434) para pre-anotaciones; nunca presentar salidas
  automáticas como anotación humana.

## Comandos (desde `SEGUNDA ENTREGA/`)
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
source /home/carlos/venv/bin/activate

python scripts/00_csv_a_parquet.py                  # pipeline 00→12 en orden (documento_e2/README_ENTREGA.md)
python scripts/10_evaluar_sentencia.py --id 2195 --salida /tmp/demo.json
python scripts/checks_qa_final.py                   # QA de regresión v2 (v1: checks_qa_final_v1.py)
python scripts/checks_perf_producto.py              # perfilado del producto
python scripts/21_pack_validacion_m4.py --resumen   # pack O1 de validación humana
```
- Jobs largos con `nohup`/`setsid` y reanudación: `05_lematizacion.py` (~2 h),
  `07_transformer.py` (~6,7 h GPU), `23_b1_pares_reranker.py` (~2 h CPU), entrenamientos GPU.

## Reglas del proyecto
- Resultados congelados: no sobrescribas `resultados/*.json` de v1/v2 ni regeneres
  `data/splits_m4.parquet` (ADR-11/16); los experimentos nuevos usan sufijo `_vN`/`c3_*` o
  nombre nuevo.
- Protocolo M4 vigente: splits canónicos + `splits_m4_temporal.parquet` + **holdout sellado**
  (`holdout_m4_bloqueado.parquet`); no entrenar ni evaluar con él hasta el gate final de F2.
- Anti-leakage: test intocable; en los entrenamientos de dominio (B1/B2) excluir las 90
  sentencias del benchmark M7.
- Terminología: la referencia actual es **silver** (bloque mayoritario de citas NER, sin
  verificación humana); "gold" solo para anotación humana (pack O1, mini-gold/gold M7). No
  presentes etiquetas automáticas como humanas.
- Un script completo por experimento; los negativos se conservan como evidencia. Gates de
  adopción: IC 95 % bootstrap + consistencia A/B; los smokes no deciden.
- Registra cada experimento en `plan/REGISTRO_EXPERIMENTOS.md` y `resultados/metricas.json`;
  las decisiones costosas de revertir en `plan/decisiones/ADR-*.md`.
- No hagas commit, push ni PR sin pedido explícito de Carlos. `data/`, `modelos/` y artefactos
  pesados están en `.gitignore`: no los fuerces.
- codebase-memory: índice del repo = proyecto `home-carlos-Git-PLN`.
