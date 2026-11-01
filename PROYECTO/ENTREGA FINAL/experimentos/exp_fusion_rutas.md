# Fusion tardia de rutas (M7 v13)

Fecha: 2026-11-01T00:41:30-05:00. Casos: 90. Fuente: `PROYECTO/SEGUNDA ENTREGA/resultados/vinculo_cp_benchmark_v13.json`.

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

Oraculo de seleccion por caso: **0.6996**.
Cobertura de la union de los top-10 de todas las rutas: **0.5732**.

## Fusiones

| Fusion | nDCG@10 | Delta vs v2.2 | IC 95 % | P(mejora) |
|---|---:|---:|---|---:|
| v2.2 + todas las rutas (0.25) | 0.6350 | -0.0577 | [-0.0769; -0.0379] | 0.000 |

## Lectura

Una fusion se adopta solo si el delta pareado tiene el IC 95 % por encima de cero
y la mejora es consistente. El oraculo marca el techo de elegir ruta por caso.
