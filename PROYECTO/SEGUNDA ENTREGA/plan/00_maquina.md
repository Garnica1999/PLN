# Perfil de máquina E2 (Fase 00, 2026-09-08)

| Recurso | Valor detectado |
|---|---|
| Python | `/home/carlos/venv/bin/python` (3.14.7) — venv canónico |
| CPU | AMD Ryzen 9 5950X — 16 núcleos físicos / 32 hilos |
| RAM | 31 GB total (~18 GB disponibles) |
| GPU | AMD Radeon RX 9070 XT (17.1 GB VRAM, ROCm; `torch.cuda` OK, torch 2.14) — para transformers/embeddings |
| Java | OpenJDK 26.0.2 (`/usr/bin/java`) — PySpark sin instalar aún |
| Disco | 519 GB libres en /home |
| Parquet | pyarrow 24.0.0 ya instalado |
| Pendiente | `pip install pyspark transformers datasets networkx sentence-transformers` en el venv |

## Decisiones de recursos derivadas
- Migración CSV 1.67 GB → **Parquet particionado tipo/año** (ADR-7, PySpark; verificar compatibilidad Java 26, fallback openjdk-21 + `JAVA_HOME` puntual).
- Lematización/POS/stemming corpus completo (~261 M palabras): **background con checkpoints por chunks** (spaCy `n_process` alto, reanudable).
- Fine-tuning (BETO/LEGAL-BERT) y embeddings: GPU ROCm, batch según VRAM, AMP.
- Splits train/val/test 70/15/15 una sola vez, guardados en `data/`.
