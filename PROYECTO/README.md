# PROYECTO — Juris-PLN

Sistema de PLN sobre sentencias de la Corte Constitucional de Colombia (1992-2021, con
extensión 2022-2026). Proyecto del curso de Procesamiento de Lenguaje Natural (Maestría en
Inteligencia Artificial, Pontificia Universidad Javeriana).

## Documentación de contexto

- `AGENTS.md` — operación del proyecto (entorno, comandos, reglas).
- `objetivos.md` — objetivo doble B+A (bloque predominante silver + temática validada por
  expertos) y objetivos específicos O1-O4.
- `arquitectura.md` — arquitectura real (M0-M7, pipeline, producto v2.2, protocolo, límites).

## Entregas

- `PRIMERA ENTREGA/` — E1: EDA y caracterización del corpus (textometría, figuras, informe).
- `SEGUNDA ENTREGA/` — E2: sistema completo. Planes en `SEGUNDA ENTREGA/plan/`, producto CLI en
  `SEGUNDA ENTREGA/scripts/10_evaluar_sentencia.py`, informes en `SEGUNDA ENTREGA/documento_e2/`.

## Estado (2026-10-05)

- Ciclo v3 "recuperación robusta": Sprint A sin adopciones; **Sprint B adoptado como producto
  v2.2** (retriever denso B2 + reranker de dominio B1 + hubs α=1,5; ADR-18) — nDCG@10 0,693
  (silver) / 0,783 (gold-LLM), sin deriva temporal en 2022-2026 (0,747).
- C3: corpus 2022-2026 descargado y preparado (2.330 docs). C2 (LLM ALIA zero-shot): negativo
  documentado. Packs de validación humana anotados: **B3 (M7) confirma v2.2** (nDCG@10 0,785) y
  **O1 (M4): el silver no es fiable** (acuerdo humano 23,5 %; ADR-19).
- Pendientes y futuro: `SEGUNDA ENTREGA/plan/PENDIENTES_Y_FUTURO.md`.

## Reproducción y entorno

- Guía completa: `SEGUNDA ENTREGA/documento_e2/README_ENTREGA.md` e
  `informe_final_ciclo_v3.md`.
- Entorno: `/home/carlos/venv` (Python 3.14); `JAVA_HOME=/usr/lib/jvm/java-17-openjdk` para el
  script 00 (PySpark); GPU AMD ROCm para entrenamientos (el producto corre en CPU por defecto).
