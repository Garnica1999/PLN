# Perfil de máquina (Fase 00)

| Recurso | Valor detectado |
|---|---|
| Python | `/home/carlos/venv/bin/python` (3.14.7) — venv canónico con pandas 2.3.3, nltk 3.9.4, spacy 3.8.13, wordcloud, nbclient, ruff |
| CPU | AMD Ryzen 9 5950X — 16 núcleos físicos / 32 hilos |
| RAM | 31 GB total (~18 GB disponibles) |
| GPU | AMD Radeon RX 9070 (16 GB VRAM, ROCm). No usada: tarea NLP en CPU |
| Disco | 526 GB libres |

## Decisiones de recursos derivadas
- Procesamiento del corpus completo (23.750 filas / ~1.68 GB CSV / ~272 M palabras) **en chunks y con streaming** para respetar la RAM.
- `nltk` + regex para tokenización/limpieza (rápido en CPU, 32 hilos si aplica `multiprocessing`).
- spaCy (`es_core_news_md`) disponible; lematización completa del corpus NO es viable en tiempos cortos de CPU → se declara como trabajo futuro de la entrega (abstract corregido).
- NLTK data y modelo spaCy ya descargados en el venv (`/home/carlos/nltk_data`).

## Hallazgos del dataset (verificación previa, escaneo completo 23.750 filas)
- Columnas reales: `['', 'Texto']` (sin columna ID; el ID se parsea del inicio del texto).
- Prefijo parseable con regex `^(sentencia\s+)?([cts])(u)?\s*(\d{1,4})\s+(\d{2,4})`: **23.742 / 23.750 (99.97%)**.
  - Tipos: T 17.099, C 6.246, SU 397. Todos con número de 3 dígitos.
  - Rango de años real: **1992–2021** (2021 solo 20 docs; el archivo se llama 1992_2022 pero no hay sentencias 2022).
- 8 filas NO son sentencias: 6 son **autos** (A-xxx/yy) y 2 son **basura HTML/JS de scraping** (rows 16200, 16268) → candidatas a exclusión documentada.
- Licencia dataset: CC-BY-4.0.
