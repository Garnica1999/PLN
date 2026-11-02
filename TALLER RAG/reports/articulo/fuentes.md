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
| Compuerta (Tabla) | `reports/tablas/domain_classifier_metrics.csv`, `domain_classifier_predictions.csv` |
| RAGAS (Tabla) | `reports/tablas/ragas_gemma4-e4b_bge-m3.csv` (media y n por métrica) |
| MTP (Tabla) | `docs/ARQUITECTURA.md` §4.7 |
| Respuesta final (37/38, 25/37, F1, latencia) | `reports/tablas/metricas_final_sin.csv` y `metricas_final_con.csv` del commit `125bbc7`: 7.938/6.834 ms, 25/37 de match estricto. La corrida limpia de 6.788/6.482 ms no conserva CSV y no se usa como referencia |
| Modelos y versiones | `docker-compose.yml`, manifiestos de Ollama y `docs/ARQUITECTURA.md` §3; tabla XVIII del artículo |
| Reranker p50 | `reports/tablas/mejoras_recuperacion.csv` (rerank5 1.297 ms, rerank20 5.460 ms) |
| Chat (11/14, 14/14) | `reports/tablas/chat_eval_final_*.csv` |
| Gold (43, 19/24, 38/5, kappa) | `eval/gold/README.md`, `docs/EXPERIMENTOS.md` §10 |
| Configuración final | `docs/ARQUITECTURA.md` §3 |
