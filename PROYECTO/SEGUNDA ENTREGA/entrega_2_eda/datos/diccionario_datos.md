# Diccionario de datos: corpus_medido.parquet

Generado por el notebook de la Entrega 2 (seccion 10). El parquet tiene 23.742 filas;
la muestra `corpus_medido_muestra.csv` conserva las mismas variables mas `decada` y
`texto_limpio_preview`. Los textos no se modifican respecto del corpus curado.

| variable | tipo | descripcion | ejemplo | fuente |
|---|---|---|---|---|
| `id` | string | Identificador del documento en el corpus (clave estable del join). | `993` | Corpus curado (E2) |
| `tipo` | string | Tipo de providencia: T (tutela), C (constitucionalidad), SU (unificacion). | `T` | Parseo de la clave de relatoria (E1/E2) |
| `numero` | int | Numero de la sentencia (p. ej. 001). | `236` | Parseo de la clave de relatoria (E1/E2) |
| `ano` | int | Anio de emision (1992-2021). | `1993` | Parseo de la clave de relatoria (E1/E2) |
| `texto` | string | Texto crudo completo de la sentencia (promedio ~11.000 palabras). | `T-236-93 Sentencia No Sentencia  No. T-236/93 PERSONA DISMIN...` | Corpus curado (E2) |
| `texto_limpio` | string | Texto normalizado: minusculas, sin diacriticos, solo letras y espacios; tokens de ruido filtrados. | `sentencia no sentencia no persona dismin...` | Pipeline de limpieza (E1/E2) |
| `n_chars_texto` | int | Numero de caracteres del texto crudo. | `37687` | Calculado en el notebook (seccion 10) |
| `n_palabras_raw` | int | Palabras por split() del texto crudo (incluye numeros y claves). | `5984` | Pasada de textometria (seccion 2) |
| `n_palabras_clean` | int | Tokens alfabeticos tras limpieza y filtro de ruido. | `5796` | Pasada de textometria (seccion 2) |
| `n_stopwords` | int | Tokens limpios que son stopwords gramaticales (306 formas normalizadas). | `2789` | Pasada de textometria (seccion 2) |
| `n_lemas` | int | Lemas por documento (spaCy es_core_news_md, pasada de E1). | `5744` | PRIMERA ENTREGA (lemas_por_doc.parquet) |
| `mtld` | float | Diversidad lexica MTLD (McCarthy-Jarvis 2010, umbral 0.72). | `67.07511918192107` | PRIMERA ENTREGA (mtld_por_doc.parquet) |
| `decada` | string | (solo en la muestra CSV) bloque decenal del anio, usado para estratificar. | `1990-1999` | Derivado del anio (seccion 10) |
| `texto_limpio_preview` | string | (solo en la muestra CSV) primeros 300 caracteres del texto limpio. | `sentencia no sentencia no persona disminuida fisicamente ate...` | Derivado de texto_limpio (seccion 10) |

## Notas de trazabilidad

- `n_palabras_raw`, `n_palabras_clean` y `n_stopwords` los calcula este notebook y se
  auditan contra las cifras canonicas de E1 (260.988.316 / 250.012.101 / 121.205.437).
- `n_lemas` y `mtld` provienen de la pasada pesada de morfologia de E1 (spaCy + MTLD).
- El join es por `id`; se valido que los 23.742 ids del corpus, las metricas, los lemas
  y el MTLD coinciden sin faltantes.
