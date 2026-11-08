# Reportes

Índice de entregables. Cada archivo se genera con un script y se puede reproducir.

| Archivo | Qué contiene | Script |
|---|---|---|
| `informe/informe_final.pdf` | Informe de investigación | LaTeX en `informe/` |
| `figuras/perfil_*.png` | Perfilado del corpus | `src/scripts/profile_corpus.py` |
| `figuras/metricas_*.png` | Comparación de configuraciones | por definir en F7 |
| `tablas/perfilado.json` | Integridad y distribuciones | `src/scripts/profile_corpus.py` |
| `tablas/metricas.csv` | Resumen de métricas por corrida | `src/scripts/metrics.py` |
| `tablas/metricas.md` | Tabla legible de métricas | `src/scripts/metrics.py` |
| `tablas/resultados_detalle.csv` | Métricas por pregunta | `src/scripts/metrics.py` |
| `tablas/ingesta.jsonl` | Resumen de cada ingesta | `src/scripts/ingest.py` |
| `corridas/*.jsonl` | Respuestas crudas por corrida | `src/scripts/evaluate.py` |
| `tablas/vecinos.json` | Vecinos semánticos de ejemplo | `src/scripts/neighbors.py` |

Reglas:

- Ningún número del informe se escribe a mano. Sale de `corridas/` y `tablas/`.
- Las corridas se conservan aunque fallen; el campo `error` explica por qué.
- El Parquet y los pesos no se versionan.
