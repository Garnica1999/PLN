# %% [markdown]
"""# Sec 05 — Entregables y reproduccion del taller

## Entregables (rutas dentro de TALLER-4)

| Entregable | Ruta |
|---|---|
| Tomos PDF crudos (10 PDF + zip original) | `data/raw/books/`, `data/raw/books-*.zip` |
| Entrevistas crudas (2.486, JSON unico) | `data/raw/entrevistas_all_2023-03-21_14-24_05.json` |
| Corpus de libros: segmentos (917) y por tomo | `data/corpus/segmentos_libros.jsonl`, `data/corpus/segmentos/<TOMO>.jsonl` |
| Corpus de entrevistas normalizado | `data/corpus/entrevistas_normalizadas.jsonl` |
| Corpus preprocesado (lemas, tokens, metadatos) | `data/corpus/preprocesado/` (`libros_*` 917 docs, `entrevistas_*` 2.486 docs) |
| Config congelada del modelo | `config/` (`tfidf_config.json`, `stopwords_habla.txt`, `lemas_correcciones.json`) |
| Codigo de extraccion, preprocesado, EDA y evaluacion | `scripts/01_extraer_libros.py`, `02_preprocesar.py`, `03_eda.py`, `04_evaluar_muestra.py` |
| Modelo IR TF-IDF (base y variantes A/B/C) | `experimentos/experimento_1_tfidf_base.py` |
| Figuras del EDA (14) | `imagenes/eda_*.png` |
| Ranking, reportes, muestra etiquetada y metricas | `resultados/` (`ranking_top5*.csv`, `ir_tfidf_report.json`, `evaluacion_metricas.json`, `metricas.json`, `eda/`) |
| Codigo fuente de este notebook | `notebook_cells/sec00_setup.py` ... `sec05_entrega.py`, `scripts/ensamblar_notebook_t4.py` |
| Notebook entregable (ejecutado) | `notebooks/Taller_4_IR_CEV.ipynb` |
| Tests | `tests/` (`test_extraccion.py`, `test_preprocesado.py`, `test_ir_tfidf.py`) |

## Como reproducir

**Requisitos** (venv de Carlos, Python 3.14):

```bash
/home/carlos/venv/bin/pip install pandas scikit-learn spacy matplotlib seaborn wordcloud
/home/carlos/venv/bin/python -m spacy download es_core_news_md   # modelo de la Fase 4
```

- `poppler-utils` (binario `pdftotext`) solo se necesita para REGENERAR la
  extraccion de los libros desde los PDF (Fase 3); el corpus ya extraido no
  lo requiere.
- Para re-ejecutar este notebook se adicionan `nbformat` y `nbclient`
  (solo herramienta de ejecucion, no del pipeline).

**Pipeline completo, en orden** (desde `TALLER-4/`):

```bash
# Fase 3: extraccion y segmentacion de los 10 tomos (917 segmentos)
/home/carlos/venv/bin/python scripts/01_extraer_libros.py

# Fase 4: normalizacion estructural + spaCy (lematizacion; ~9 min, 8 hilos)
/home/carlos/venv/bin/python scripts/02_preprocesar.py

# Fase 5: EDA, figuras y congelacion de config/tfidf_config.json
/home/carlos/venv/bin/python scripts/03_eda.py

# Fase 6: experimento IR base (ranking_top5.csv y reportes)
/home/carlos/venv/bin/python experimentos/experimento_1_tfidf_base.py
# Variantes comparativas (no tocan los archivos del base):
/home/carlos/venv/bin/python experimentos/experimento_1_tfidf_base.py --variante A
/home/carlos/venv/bin/python experimentos/experimento_1_tfidf_base.py --variante B
/home/carlos/venv/bin/python experimentos/experimento_1_tfidf_base.py --variante C

# Evaluacion de la muestra de 30 pares (etiquetado proxy + metricas P@1/P@3)
/home/carlos/venv/bin/python scripts/04_evaluar_muestra.py

# Tests (pipeline sintetico, sin corpus completo)
/home/carlos/venv/bin/python -m pytest tests/ -q
```

**Re-ejecucion de este notebook** (ensamblado de celdas + ejecucion):

```bash
/home/carlos/venv/bin/python scripts/ensamblar_notebook_t4.py
/home/carlos/venv/bin/python -m nbclient ...
```

La seccion 00 del notebook detecta la raiz del proyecto automaticamente
(`buscar_raiz()`: ancestros, subcarpetas de hasta 3 niveles - cubre
`/content/PLN-main/TALLER-4` al descomprimir el zip del repo en /content -,
rutas tipicas de Colab y `TALLER4_ROOT`; opcionalmente auto-descarga el
repositorio con `AUTO_DESCARGAR_REPO = True`), por lo que puede abrirse en
Jupyter desde `notebooks/`, desde la raiz de TALLER-4 o en Colab con la
carpeta TALLER-4 completa subida (celda de diagnostico con checklist de
prerrequisitos en la sec 00). Las figuras del EDA se generan DENTRO del
notebook (Sec 02, outputs PNG embebidos, visibles en GitHub/Colab); las tres
figuras oficiales que requieren datos no versionados (chars y paginas de
entrevistas, near-duplicates) se muestran desde `imagenes/` con el helper
`mostrar_png()`.

**Regeneracion opcional dentro del notebook** (Sec 0b): el codigo canonico
de scripts/01-03 viaja embebido por el ensamblador (fuente unica en build) y
solo se ejecuta si faltan archivos y activas los flags de la celda de
configuracion (`REGEN_AUTOMATICO`, `FUERZAR_REGENERACION`,
`DESCARGAR_FUENTES`, `RUTA_FUENTES`). Por defecto todo apagado: la seccion
queda inerte porque el repositorio ya trae los derivados; la primera
regeneracion completa toma 15-60 min en Colab y nunca reescribe `config/`.
"""
