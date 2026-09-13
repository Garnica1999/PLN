# TALLER-4 — Recuperación de Información sobre corpus de la Comisión de la Verdad (CEV)

Taller del curso **Procesamiento de Lenguaje Natural** (Pontificia Universidad
Javeriana). Proceso de IR completo (puntos 1-6 del enunciado) que relaciona
**2.486 entrevistas** de víctimas (CEV/JEP, formato JSON) con **segmentos de
los 10 tomos** del Informe Final de la Comisión de la Verdad (2022), con
TF-IDF base, Okapi BM25 y Rocchio implementados a mano dentro del notebook,
y comparación de corpus con heatmap entrevistas ↔ libros.

## Entregables
- `notebooks/Taller_4_IR_CEV.ipynb` — notebook único (Colab-compatible),
  ejecutado de extremo a extremo (0 errores): corpus → EDA → modelo IR TF-IDF
  → evaluación cualitativa → BM25/Rocchio manuales → heatmap y cuadro de
  vínculos entrevistas ↔ libros → observaciones y entregables.
- `informe/informe_taller4.md` — informe técnico integrador (punto 6).
- `data/corpus/` — corpus extraído y preprocesado: `segmentos_libros.jsonl`
  (917 segmentos), `entrevistas_normalizadas.jsonl` (2.486), `preprocesado/`
  (lemas y tokens por corpus).
- `imagenes/eda_*.png` (14 figuras del EDA) + `resultados/` (18 CSVs EDA,
  rankings TF-IDF, evaluación etiquetada, `metricas.json` por fase).
- `resultados/ranking_bm25.csv`, `resultados/ranking_rocchio.csv` —
  rankings del punto 4 (salidas del notebook, 12.420 filas cada uno).
- `imagenes/heatmap_entrevistas_libros.png`,
  `resultados/heatmap_tomos.csv`,
  `resultados/vinculos_entrevistas_libros.csv` — punto 5 (salidas del
  notebook).

## Resultados principales
- Extracción: 10/10 tomos, 917 segmentos (TOC niveles 1-2), mapeo 91-100%,
  cobertura 100%.
- EDA: TTR_m1000 libros 0,609 vs entrevistas 0,514; Yule K 31,8 vs 38,7;
  solapamiento Jaccard 0,221; 345 pares near-duplicados (>0,85) documentados.
- IR TF-IDF: 118.748 features; cobertura 99,92%; p@1 estricto 0,533 / aceptable
  1,0 y p@3 estricto 0,767 sobre 30 pares etiquetados; sesgo top-1 TESTIMONIAL
  (76,8%) explicado como efecto de dominio (registro oral).
- P4 (métricas manuales): Okapi BM25 (k1=1,5, b=0,75; avgdl 1.466 tokens) y
  Rocchio (α=1, β=0,75, γ=0,15, pseudo-relevancia top-5/bottom-5); acuerdo de
  top-1 57,0% base↔BM25 y 70,1% base↔Rocchio; los tres comparten el sesgo
  TESTIMONIAL (76,8 / 75,0 / 78,4%).
- P5: heatmap 10×10 (grupo por top-1 BM25 × tomo de libro) con diagonal media
  0,0725 > fuera de diagonal 0,0608 y cuadro de 30 vínculos (3 por tomo).

## Estructura
```
TALLER-4/
├── config/          hardware.json, toc_tomos.json, stopwords_habla.txt,
│                    lemas_correcciones.json, tfidf_config.json
├── data/raw/        entrevistas JSON, zip de libros, tesauro RDF (sin tocar)
├── data/corpus/     segmentos, entrevistas normalizadas, preprocesado spaCy
├── docs/            PDFs del taller y rúbrica
├── notebooks/       Taller_4_IR_CEV.ipynb (ENTREGABLE)
├── notebook_cells/  celdas fuente por sección (sec00-sec03c, sec04-sec05)
├── scripts/         01_extraer_libros.py, 02_preprocesar.py, 03_eda.py,
│                    ensamblar_notebook_t4.py
├── experimentos/    experimento_1_tfidf_base.py (modelo IR)
├── imagenes/        figuras del EDA + heatmap del punto 5
├── resultados/      CSVs, reportes JSON, metricas.json, evaluación etiquetada
│                    y rankings BM25/Rocchio + heatmap/vínculos (puntos 4-5)
├── tests/           pytest (51 tests: extracción, preprocesado, IR,
│                    independencia del notebook)
├── informe/         informe_taller4.md (ENTREGABLE)
└── plan/            PLAN.md, arquitectura.md, estado_del_arte.md, ADRs,
                     verificacion_requisitos.md, verificacion_rubrica.md
```

## Ejecución
- Local: `notebooks/Taller_4_IR_CEV.ipynb` con el venv `/home/carlos/venv/bin/python`
  (pandas, scikit-learn, spacy + `es_core_news_md`, matplotlib, wordcloud,
  seaborn, nbclient; poppler-utils solo para regenerar la extracción de PDFs).
- Colab: ver "Ejecución en Colab (paso a paso)" más abajo.

## Requisitos (independencia del notebook)

El notebook es **independiente**: para correrlo solo necesitas **3 insumos**
(descarga **manual** desde el Drive del taller a `data/raw/`):

1. **Dataset de entrevistas**: `data/raw/entrevistas_all_2023-03-21_14-24_05.json`
   (~231 MB; archivo de entrevistas del Drive del taller).
2. **Libros**: `data/raw/books-*.zip` (zip del Drive) o la carpeta
   `data/raw/books/` con los PDFs (~300 MB; carpeta de libros del Drive).
3. **Tesauro**: `data/raw/Resource-tesauro.rdf` (solo lo usa el cruce de
   vocabulario del EDA de producción; el resto del notebook no lo requiere).

Todo lo demás (**caches derivados**: segmentos, normalizadas, lemas/tokens,
docs y los CSVs/figuras del EDA) el notebook lo **genera automáticamente**
en la sección "Sec 0b — Regeneración" (`REGEN_AUTOMATICO=True` por defecto)
a partir de esos insumos con el código canónico de `scripts/01-03`
(expandido como celdas de código normales). La config curada del pipeline
(muletillas, correcciones de lemas, ruido tipográfico y parámetros TF-IDF)
viaja **embebida** como constantes literales en la Sec 00; `config/`,
`resultados/` e `imagenes/` son **opcionales** (si existen, el notebook los
compara/verifica y los usa para las figuras oficiales con `mostrar_png`).

- **Primera ejecución sin caches**: 15-60 minutos (libros ~5-15 min; spaCy
  ~10-40 min). Si el insumo necesario no está en `data/raw/`, el notebook
  avisa cómo descargarlo y sigue sin excepción (las secciones muestran
  avisos `[aviso]/[info]` y los cálculos inline).
- **Si descargas el repositorio completo**: los caches ya viajan en
  `data/corpus/` (los 3 insumos >100 MB NO están versionados: descárgalos
  una sola vez del Drive) y el Run all termina en ~1 minuto.
- `config/`, `resultados/` e `imagenes/` pueden borrarse sin romper el
  notebook: se usan solo como verificación opcional o respaldo visual
  (evidencia: `tests/test_notebook_dependencias.py`, test sandbox (g), que
  ejecuta el notebook sin esos directorios con 0 errores).

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
celdas solo definen funciones. La regeneración real la dispara el driver de
la sección con `REGEN_AUTOMATICO = True` (por defecto): recorre las etapas
con caches faltantes y, si los insumos de los 3 permitidos están en
`data/raw/`, ejecuta la etapa (primera ejecución 15-60 min); si falta un
insumo, imprime cómo descargarlo manualmente y sigue sin excepción. Los
flags `FUERZAR_REGENERACION`, `DESCARGAR_FUENTES` y `RUTA_FUENTES` permiten
forzar la re-ejecución o apuntar a una carpeta local con las fuentes.
**Nunca reescribe `config/`** (la etapa "congelar" del EDA no se ejecuta
desde el notebook; la config curada del notebook está embebida en la
Sec 00).

Alcance: los **6 puntos del taller** están cubiertos — P1-P3 (corpus, EDA y
modelo IR TF-IDF con evaluación cualitativa), P4 (Okapi BM25 y Rocchio
implementados manualmente y comparados), P5 (heatmap entrevistas ↔ libros y
cuadro de vínculos) y P6 (informe final integrador en
`informe/informe_taller4.md`). El notebook es independiente: solo requiere
los 3 insumos de `data/raw/` y escribe sus salidas (rankings BM25/Rocchio,
heatmap y vínculos) sin leerlas como requisito.

> **Nota de entrega.** El ZIP final (`entregable/ENTREGABLE_TALLER4.zip`) incluye
> los 3 insumos del notebook (dataset, libros y tesauro) y el informe final en
> PDF; se genera localmente y no se versiona en git por su tamaño (~393 MB).
> Ver `entregable/LEEME.txt`.
