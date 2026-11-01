# Fusion tardia y techo de reordenamiento (M7 v13)

Fecha: 2026-11-01T01:29:46-05:00. Casos: 90. Fuente: `PROYECTO/SEGUNDA ENTREGA/resultados/vinculo_cp_benchmark_v13.json`.

## nDCG@10 por ruta fija

| Ruta | nDCG@10 |
|---|---:|
| hibrido_b2_rerank_chunks_top50_b1 (v2.2) | 0.6926 |
| hibrido_rerank_chunks_top50 | 0.5125 |
| bm25_rerank_chunks_top50 | 0.4798 |
| bm25_rerank_chunks | 0.4382 |
| hibrido | 0.4200 |
| bm25 | 0.3798 |
| bm25_rerank | 0.2962 |
| e5 | 0.2646 |

Oráculo de selección por caso: **0.6996**.
Techo de reordenar el top-10 actual: **0.7359**.
Relevantes en el top-10 (media): **4.80** de 11.44 del gold.

## Combinaciones sin GPU

| Combinación | nDCG@10 | Delta vs v2.2 | IC 95 % | P(mejora) |
|---|---:|---:|---|---:|
| v2.2 + consenso de 3 rutas | 0.6926 | +0.0000 | [+0.0000; +0.0000] | 0.000 |
| borda | 0.5497 | -0.1430 | [-0.1694; -0.1183] | 0.000 |
| conteo de rutas | 0.5431 | -0.1495 | [-0.1778; -0.1237] | 0.000 |

## Lectura

El top-10 actual contiene menos de la mitad del gold, así que reordenarlo
tiene un techo bajo (+0,043). La mejora real exige promover candidatos de
las posiciones 11 a 50, es decir, un mejor reranker, no otra regla de fusión.
