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
- Colab: ver "Ejecución en Colab (paso a paso)" más abajo.

## Ejecución en Colab (paso a paso)

El notebook localiza solo la raíz del proyecto: busca el marcador
`data/corpus/segmentos_libros.jsonl` en ancestros, subcarpetas (hasta 3
niveles, cubre `/content/PLN-main/TALLER-4`), rutas típicas y la variable
`TALLER4_ROOT`; si no lo encuentra lanza un error claro con estas opciones
(la celda de diagnóstico de la Sec 00 imprime un checklist ✓/✗ de archivos):

- **Opción A (subir la carpeta)**: descarga el zip del repositorio
  (GitHub → Code → Download ZIP), súbelo a Colab y descomprímelo en
  `/content` (deja la carpeta en `/content/PLN-main/TALLER-4`, que se
  detecta automáticamente). Alternativa: subir directamente la carpeta
  `TALLER-4` completa a `/content/TALLER-4`.
- **Opción B (auto-descarga)**: abre `notebooks/Taller_4_IR_CEV.ipynb` con
  "Open in Colab" y pon `AUTO_DESCARGAR_REPO = True` en la celda de
  configuración de la Sec 00: si la raíz no aparece, el notebook descarga el
  repositorio automáticamente y reintenta.
- **Opción C (Drive)**: monta Google Drive y deja la carpeta en
  `/content/drive/MyDrive/TALLER-4`, o define la variable de entorno
  `TALLER4_ROOT` con la ruta exacta de la carpeta TALLER-4.

### Regeneración opcional del corpus dentro del notebook (sección "Sec 0b — Regeneración")

Los archivos derivados los producen `scripts/01-03` (producción); su código
canónico viaja **expandido como celdas de código normales** del notebook por
el ensamblador (fuente única en build: si cambian los scripts, se
re-ensambla; el notebook no incrusta código-como-string). En Run all esas
celdas solo definen funciones; las etapas solo se ejecutan si faltan
archivos y activas los flags de su celda de configuración
(`REGEN_AUTOMATICO`, `FUERZAR_REGENERACION`, `DESCARGAR_FUENTES` y
`RUTA_FUENTES` para fuentes del Drive del taller); por defecto todo está en
False y la sección queda inerte (imprime "sin acción"). La primera
regeneración completa (extracción + normalizar + spaCy + EDA) toma 15-60
minutos en Colab y **nunca reescribe `config/`** (la etapa "congelar" del EDA
no se ejecuta desde el notebook).

Alcance aprobado: puntos 1-3 del taller (Rocchio/BM25, heatmap e informe final
completo quedan como mejoras futuras documentadas en el informe).
