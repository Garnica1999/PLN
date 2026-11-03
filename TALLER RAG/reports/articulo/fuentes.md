# Fuentes de las cifras del artículo

Cada tabla del artículo se respalda en un artefacto. El verificador debe comparar el valor del `.tex` con el artefacto y reportar solo discrepancias.

| Tabla o afirmación | Artefacto |
|---|---|
| Composición del corpus, idiomas, duplicados, longitud, ruido | `docs/EXPERIMENTOS.md` §7 y §10; `reports/informe/metricas_nueva_configuracion.md` |
| Textometría (Tabla II) | `docs/EXPERIMENTOS.md` §7 |
| Limpieza (Tabla III) | `docs/EXPERIMENTOS.md` §8.2 y §8.3 |
| E1 y E2 | `docs/EXPERIMENTOS.md` §9.1 y §9.2 |
| E6 (piloto contra corpus) | `docs/EXPERIMENTOS.md` §13; `reports/tablas/bench_bge_m3_full.csv` |
| E10 (e5, BM25, bge-m3) | `docs/EXPERIMENTOS.md` §18 |
| E11 y E12 (ef, fusión, gold ampliado) | `docs/EXPERIMENTOS.md` §19 y §20 |
| Reranking final y fases 6 y 7 | `reports/tablas/mejoras_recuperacion.csv`, `reports/tablas/mejoras_diversidad.csv`, `reports/tablas/metricas_v4_sin.csv` |
| E4 (modelos y kappa) | `docs/EXPERIMENTOS.md` §11; `reports/tablas/e4_kappa.csv`, `e4_h1.csv` |
| E5 (inyección) | `docs/EXPERIMENTOS.md` §15 |
| Compuerta (Tabla) | `reports/tablas/gate_reranker_metrics.csv`, `gate_reranker_reservado.csv` y `gate_reranker_cv.csv`; modelo `modelos/domain_classifier_v2.json`. Los artefactos anteriores (`domain_classifier_metrics.csv`, `domain_classifier_predictions.csv`) quedan como historia |
| RAGAS (Tabla) | `reports/tablas/ragas_gemma4-e4b_bge-m3.csv` (media y n por métrica) |
| MTP (Tabla) | `docs/ARQUITECTURA.md` §4.7 |
| Respuesta final (37/38, 25/37, F1, latencia) | `reports/tablas/metricas_final_sin.csv` y `metricas_final_con.csv` del commit `125bbc7`: 7.938/6.834 ms, 25/37 de match estricto. La corrida limpia de 6.788/6.482 ms no conserva CSV y no se usa como referencia |
| Conjunto reservado híbrido (recall, respuesta, compuerta, latencia) | `reports/tablas/reservado_hibrida_resumen.csv`, `reservado_hibrida_por_pregunta.csv`; `reports/corridas/reservado_hibrida_raw.jsonl` |
| Capa de datos remota | `docker-compose.data.yml`; el API se conecta por un túnel SSH a `localhost:19530` |
| Profundidad de la unión | `reports/tablas/reservado_union_depth.csv`, `reservado_union_depth_resumen.csv` |
| Ambigüedad (detección y políticas) | `eval/gold/gold_ambiguedad.jsonl`; `reports/tablas/ambiguedad_resumen.csv`, `ambiguedad_por_pregunta.csv` |
| Modelos y versiones | `docker-compose.yml`, manifiestos de Ollama y `docs/ARQUITECTURA.md` §3; tabla XVIII del artículo |
| Reranker p50 | `reports/tablas/mejoras_recuperacion.csv` (rerank5 1.297 ms, rerank20 5.460 ms) |
| Chat (11/14, 14/14) | `reports/tablas/chat_eval_final_*.csv` |
| Gold (43, 19/24, 38/5, kappa) | `eval/gold/README.md`, `docs/EXPERIMENTOS.md` §10 |
| Configuración final | `docs/ARQUITECTURA.md` §3 |
