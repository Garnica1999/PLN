# 00 — Perfil de máquina (TALLER-4 IR CEV)

- **Fecha**: 2026-09-03. **Python**: `/home/carlos/venv/bin/python` (3.14.7, venv canónico).
- **CPU**: AMD Ryzen 9 5950X — 16 núcleos físicos / 32 hilos. `n_jobs=-1`, 32 hilos disponibles.
- **RAM**: 31 GB (con 15 GB libres al inicio; usar carga en streaming si es necesario).
- **GPU**: AMD Radeon RX 9070 XT (dedicada, 16 GB VRAM, ROCm `torch.cuda=True`). **No requerida**: el pipeline del taller (extracción PDF, spaCy CPU, TF-IDF) es CPU-bound.
- **Disco**: 529 GB libres.
- **Paquetes clave ya instalados**: spacy 3.8.13 + `es_core_news_md` (lematizador español OK), scikit-learn 1.8.0, pandas 2.3.3, pymupdf 1.27.2.3, wordcloud 1.9.6, nbclient 0.10.4, rdflib (tesauro), matplotlib/seaborn.
- **Herramienta externa**: `pdftotext` (poppler-utils) disponible y validado sobre los PDFs CEV.

## Recursos derivados
- Preprocesado spaCy: `n_process=8-16` (CPU bound, 32 hilos).
- PDFs: `pdftotext -layout` página a página en paralelo (32 hilos).
- Carga JSON 231 MB: `json.load` OK con RAM disponible (~2-3 GB pico); guardar corpus limpio en disco.
- Todos los entrenamientos/experimentos con el venv canónico `/home/carlos/venv/bin/python`.
