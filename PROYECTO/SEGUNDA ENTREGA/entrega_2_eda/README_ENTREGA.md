# Entrega 2 — EDA Avanzado, Representación y Propuesta Metodológica

**Proyecto:** Procesamiento de Lenguaje Natural — Sentencias de la Corte Constitucional de Colombia (1992–2021)
**Autores:** Erika Valeria Rivera Jiménez · Julián Guillermo Carrillo Meneses · Carlos Andrés Garnica Salazar · Óscar Andrés Prieto Moreno
**Institución:** Pontificia Universidad Javeriana, Bogotá, Colombia
**Fecha:** 2026-10-04

---

## 1. Qué contiene esta entrega

```
entrega_2_eda/
├── informe/          Informe en LaTeX + PDF (28 pp) y bibliografía
├── notebook/         Notebook consolidado, ejecutado de principio a fin (salidas embebidas)
├── presentacion/     Presentación (18 diapositivas .pptx), guion y script de generación
├── datos/            Conjunto de datos medido + muestra + diccionario + lemas por documento
├── figuras/          Todas las figuras del análisis (incluidas las nuevas de la Entrega 2)
├── resultados/       Métricas y derivados (JSON/CSV): EDA extendido, TF-IDF, log-odds, entorno
├── plan/             Plan de trabajo, enunciado convertido y nota de ontologías
├── experimentos/     Log de la ejecución del notebook (con sello sha256)
└── PLAN_ENTREGA_II.md  Plan maestro con la bitácora completa del proceso
```

## 2. Correspondencia con el enunciado (Entrega 2)

| # | Requisito | Dónde se cumple |
|---|---|---|
| 1 | Correcciones y modificaciones respecto a la Entrega 1 | `informe/main.pdf` sección 2 (con razones verificadas) |
| 2 | EDA avanzado: ontologías/tesauros/recursos léxicos; diversidad léxica; coocurrencias; bigramas/trigramas | `informe/main.pdf` sección 4; `notebook/` sección 9 y sección 12; figuras y `resultados/eda_extendido/` |
| 3 | Conjunto de datos medido (corpus preprocesado + métricas, formato legible, adjunto) | `datos/corpus_medido.parquet` (23.742 filas) + `datos/corpus_medido_muestra.csv` + `datos/diccionario_datos.md`; `informe/main.pdf` sección 3.3 |
| 4 | Representación de los textos (selección justificada + ejemplos de vectorización) | `informe/main.pdf` sección 5; `notebook/` sección 11; `figuras/fig_representacion_tfidf.png`, `fig_embeddings_ejemplo.png` |
| 5 | Propuesta metodológica de los modelos | `informe/main.pdf` sección 6 (tarea, particiones, modelos a comparar y criterios) |
| 6 | Presentación 10–15 diapositivas | `presentacion/Entrega_2_Presentacion.pptx` (18 diapositivas, notas del orador) |
| 7 | Formato: informe PDF + carpeta con notebooks y datos | `informe/main.pdf` + esta carpeta completa |

## 3. Cómo reproducir

- **Notebook** (≈12 min; resuelve rutas automáticamente desde esta carpeta o desde `notebook/`; ejecutar con el entorno del proyecto activado — Python con las dependencias de la sección 0 del notebook):
  ```bash
  cd entrega_2_eda
  jupyter nbconvert --to notebook --execute \
    notebook/Entrega_2_EDA_Representacion.ipynb \
    --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=python3
  ```
  El notebook incluye, detrás del flag `RECALCULAR_MORFOLOGIA=False`, el código de la pasada
  morfológica completa (spaCy/POS/MTLD, ≈2 h) que por defecto se lee de los artefactos precomputados.
- **Informe:** instrucciones en `informe/README_compilar.md` (TinyTeX: `pdflatex` ×3 + `bibtex`).
- **Presentación:** `python presentacion/generar_presentacion.py`.

## 4. Conjunto de datos medido

- `datos/corpus_medido.parquet` — **23.742 sentencias** con: `id, tipo, numero, ano, texto, texto_limpio,
  n_chars_texto, n_palabras_raw, n_palabras_clean, n_stopwords, n_lemas, mtld` (ver `datos/diccionario_datos.md`).
- `datos/corpus_medido_muestra.csv` — muestra estratificada (tipo × década) de 100 filas.
- `datos/lemas_por_doc.parquet` — lemas por documento (insumo de la representación TF-IDF).
- **Fuente original:** `Manuel/sentencias-corte-cons-colombia-1992-2021` (Hugging Face Hub, licencia CC-BY-4.0);
  cobertura real 1992–2021 (2021 parcial: 20 documentos).
- **Tamaño:** la carpeta completa pesa ≈2,1 GB, principalmente por los dos Parquet de `datos/`.

## 5. Estado de validación (QA)

- **QA-A (sellado del notebook, dataset y cifras base): GO** — sha256 del notebook ligado al log de
  ejecución; dataset verificado (23.742 filas, sin nulos, sumas exactas); limpieza idéntica a la
  Entrega 1 reproducida documento a documento (23.742/23.742).
- **QA-B (cifras del informe y la presentación): NO-GO inicial → corregido y verificado** — 1 corrección
  crítica de redacción (masa acumulada del TF-IDF), 2 de tiempos y 3 de detalle, aplicadas y recompiladas.
- Alcance: 1992–2021; sin resultados de modelos (la sección 6 describe la metodología de la fase de modelado).
- Actualización (2026-10-05): 8 figuras adicionales (longitudes, tipos por año, ley de Zipf, MTLD vs longitud, espectro de frecuencias, vocabulario por año, longitud vs ventana de 512 tokens y cobertura del vocabulario) integradas en notebook e informe; recorte de redundancias aplicado (prosa y tablas), figuras ampliadas a tamaño legible, regeneradas a mayor resolución (220 dpi, tipografía interna ampliada, sin texto superpuesto) y maquetación ajustada (informe final: 28 pp); trabajo futuro con sustento en el estado del arte y datos del proyecto; notebook con figuras inline; informe sin menciones de archivos/rutas y con revisión editorial (cada decisión con su justificación); presentación ampliada a 18 diapositivas (objetivo, metodología y trabajos futuros con las herramientas por etapa) con el estilo de la Entrega 1; verificación final completada (GO).

## 6. Para el envío

- **Paquete de entrega completo (recomendado):** `Entrega_2_proyecto_completo.zip` — incluye el conjunto de datos medido y los lemas, además de todo lo demás. También puede entregarse la propia carpeta `entrega_2_eda/` completa.
- **`Entrega_2_documentos.zip`** — versión ligera (solo documentos: informe, notebook, presentación, figuras, resultados y la muestra del corpus, sin los datos completos) para revisión rápida.
- Ambos viven en la carpeta superior (`SEGUNDA ENTREGA/`).
