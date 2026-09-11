# VEREDICTO QA v2 — Ciclo 2 "Juris-PLN" (Fase X)

Fecha: 2026-09-11 · Ejecutado por: ml-qa-engineer (orquestador completó verificaciones finales)
Artefactos: `resultados/qa_final_v2/` (10 JSON + logs + entradas) · script: `scripts/checks_qa_final.py` (v2; v1 preservado en `checks_qa_final_v1.py`)

## Veredicto: **GO**

10/10 casos E2E aprobados + integridad de artefactos v2 verificada (exit 0).
Control negativo: el script v2 FALLA (82 hallazgos, exit 1) contra los JSON v1 —
los checks discriminan correctamente ambos formatos.

## Casos E2E (10/10 OK)

| Caso | Modo | Validación | Área top-1 (prob) | Recuperados | α hubs | t (s) |
|---|---|---|---|---|---|---|
| pos_2337 | default | True | derechos_fundamentales (0.9515) | 10 | 0.75 | 4.0 |
| pos_2728 | default | True | regimen_economico (0.9098) | 10 | 0.75 | 5.9 |
| pos_6251 | rerank | True | derechos_fundamentales (0.635) | 10 | 1.5 | 14.4 |
| pos_6968 | default | True | derechos_fundamentales (0.6462) | 10 | 0.75 | 4.2 |
| pos_13015 | default | True | principios_fundamentales (0.9134) | 10 | 0.75 | 5.8 |
| pos_13494 | default | True | principios_fundamentales (0.8728) | 10 | 0.75 | 4.1 |
| pos_19890 | default | True | derechos_fundamentales (0.785) | 10 | 0.75 | 7.0 |
| pos_20178 | rerank | True | principios_fundamentales (0.9101) | 10 | 1.5 | 18.1 |
| neg_gaitan | — | False (0.0) | — | 0 | — | 0.9 |
| neg_petro | — | False (0.0) | — | 0 | — | 0.9 |

Stratificación: 8 providencias positivas (2-3 por periodo 1992-2001/2002-2011/2012-2021)
+ 2 negativos (discursos Gaitán y Petro). Dos casos corren el pipeline rerank v2 completo
(E5 en cuda + bge-reranker-v2-m3 fp16 + prior α=1.5); el resto el default (BM25+hubs α=0.75).

## Drift v1 → v2

- **Clasificador/validación: ESTABLES** — 8/8 casos con misma área top-1 y misma
  probabilidad (≡ 1e-6) que el QA v1; negativos idénticos (False, score 0.0).
  Correcto: la Fase X NO cambió ClasificadorArea (gates U/V no cumplidos; LogReg v1).
- **Retrieval: cambios esperados por diseño v2** (no es drift): recuperados 5→10
  (k=10), `hubs_usado=True` con α=0.75/1.5 según modo, ruta rerank = híbrido top-50
  (antes bm25 top-20). Tiempos: 4-7 s default, 14-18 s rerank (una vez cargados E5+reranker).
- **Integridad**: md5 canónicos intactos (m4_baseline 60f8fd…, m4_jerarquico 94425f…,
  m4_eurovoc 64496c…, v8 746621…, prior_hubs 80ef66…); `resultados/qa_final/` (v1) sin
  cambios; `metricas.json` sin cambios (3f684b8d…) durante la QA.

## Riesgos residuales (documentados)

1. **Caso 6251**: los arts 21/209 (gold NER) siguen fuera del top-10 incluso en modo
   rerank (posición 220/305 en BM25 de 1ª etapa; el híbrido no los sube al top-50).
   Limitación de recall de primera etapa documentada en v6/v8; el reranker solo
   reordena candidatos existentes. Candidato a iteración futura (gold estándar).
2. Fallback E5→BM25 documentado y probado (warning en log); sin evidencia de fallo real.
3. Cache E5 compartida entre modos de índice: se invalida/reconstruye por `n_pasajes`
   (sin resultado erróneo; sugerencia cosmética de sufijo `_full`).
4. Latencia rerank (14-18 s) carga E5+reranker en la 1ª llamada; en CPU sería mayor
   (documentado; el producto usa cuda si está disponible).

## Rollback

Revertir en caso de problema: `componentes_producto.py` y `10_evaluar_sentencia.py`
a la versión v1 (zip v1 `PROYECTO/juris_pln_e2_entrega.zip`); artefactos de datos v2
no afectan a v1 (caches separadas; resultados v1 intactos).

## Checklist DoD v2

- [x] 10/10 casos E2E (validación, área, arts, citas, resumen)
- [x] Integridad de artefactos v2 + md5 canónicos
- [x] Control negativo discrimina v1/v2
- [x] Drift de clasificador/validación = 0
- [x] Cambios de retrieval documentados (5→10, hubs, rerank v2)
- [x] Rollback identificado
- [x] `plan/contexto.md` actualizado
