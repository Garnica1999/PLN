# TALLER-4 — Recuperación de Información sobre corpus de la Comisión de la Verdad (CEV)

Taller del curso **Procesamiento de Lenguaje Natural** (Pontificia Universidad
Javeriana). Proceso de IR (puntos 1-3 del enunciado) que relaciona **2.486
entrevistas** de víctimas (CEV/JEP, formato JSON) con **segmentos de los 10
tomos** del Informe Final de la Comisión de la Verdad (2022).

## Entregables
- `notebooks/Taller_4_IR_CEV.ipynb` — notebook único (Colab-compatible),
  ejecutado de extremo a extremo (0 errores): corpus → EDA → modelo IR TF-IDF →
  evaluación cualitativa → observaciones.
- `informe/informe_taller4.md` — informe técnico (3.276 palabras, 5 tablas,
  14 figuras).
- `data/corpus/` — corpus extraído y preprocesado: `segmentos_libros.jsonl`
  (917 segmentos), `entrevistas_normalizadas.jsonl` (2.486), `preprocesado/`
  (lemas y tokens por corpus).
- `imagenes/eda_*.png` (14 figuras del EDA) + `resultados/` (18 CSVs EDA,
  rankings TF-IDF, evaluación etiquetada, `metricas.json` por fase).

## Resultados principales
- Extracción: 10/10 tomos, 917 segmentos (TOC niveles 1-2), mapeo 91-100%,
  cobertura 100%.
- EDA: TTR_m1000 libros 0,609 vs entrevistas 0,514; Yule K 31,8 vs 38,7;
  solapamiento Jaccard 0,221; 345 pares near-duplicados (>0,85) documentados.
- IR TF-IDF: 118.748 features; cobertura 99,92%; p@1 estricto 0,533 / aceptable
  1,0 y p@3 estricto 0,767 sobre 30 pares etiquetados; sesgo top-1 TESTIMONIAL
  (76,8%) explicado como efecto de dominio (registro oral).

## Estructura
```
TALLER-4/
├── config/          hardware.json, toc_tomos.json, stopwords_habla.txt,
│                    lemas_correcciones.json, tfidf_config.json
├── data/raw/        entrevistas JSON, zip de libros, tesauro RDF (sin tocar)
├── data/corpus/     segmentos, entrevistas normalizadas, preprocesado spaCy
├── docs/            PDFs del taller y rúbrica
├── notebooks/       Taller_4_IR_CEV.ipynb (ENTREGABLE)
├── notebook_cells/  celdas fuente por sección (sec00-sec05)
├── scripts/         01_extraer_libros.py, 02_preprocesar.py, 03_eda.py,
│                    ensamblar_notebook_t4.py
├── experimentos/    experimento_1_tfidf_base.py (modelo IR)
├── imagenes/        figuras del EDA
├── resultados/      CSVs, reportes JSON, metricas.json, evaluación etiquetada
├── tests/           pytest (43 tests: extracción, preprocesado, IR)
├── informe/         informe_taller4.md (ENTREGABLE)
└── plan/            PLAN.md, arquitectura.md, estado_del_arte.md, ADRs,
                     verificacion_requisitos.md, verificacion_rubrica.md
```

## Ejecución
- Local: `notebooks/Taller_4_IR_CEV.ipynb` con el venv `/home/carlos/venv/bin/python`
  (pandas, scikit-learn, spacy + `es_core_news_md`, matplotlib, wordcloud,
  seaborn, nbclient; poppler-utils solo para regenerar la extracción de PDFs).
- Colab: subir la carpeta `TALLER-4` y abrir el notebook; `buscar_raiz()`
  localiza `data/corpus/` automáticamente. El corpus ya viene procesado;
  la regeneración del preprocesado spaCy es opcional.

Alcance aprobado: puntos 1-3 del taller (Rocchio/BM25, heatmap e informe final
completo quedan como mejoras futuras documentadas en el informe).
