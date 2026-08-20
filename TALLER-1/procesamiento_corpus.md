# Procesamiento del corpus "presidentes" — limpieza avanzada y su impacto

Fecha: 2026-08-20
Corpus: `open_corpus_co_es` → `presidentes` (22 discursos de posesión presidencial, 1942-2026)
Aplicado en: `Taller_Presidentes_final.ipynb`, sección 4.2 (celdas 38-39)
Validación: revisión técnica independiente + auditoría de datos antes/después (cifras reproducidas exactas)

---

## 1. Contexto: artefactos detectados en el corpus crudo

Auditoría empírica sobre `load_corpus("presidentes")` (654.615 caracteres, 106.177 tokens crudos):

| Artefacto | Cantidad | Origen probable |
|---|---|---|
| Palabras partidas por guion de fin de línea (`económi-\nco`) | 552 uniones | Scan con OCR: discursos de López Michelsen (264) y Turbay (283) tienen pies de página y números de página |
| Líneas plantilla de scan | 108 líneas | Metadata del escaneo insertada en el texto |
| Tokens con dígitos en vez de letras (`pa1·a`, `c0lombia`) | 21 tokens | OCR (confusión 1→r/l/i, 0→o) |
| "n" suelta (`n ación`) | 22 menciones | OCR (palabras partidas sin guion) |
| Palabras funcionales duplicadas (`en en`, `de de`) | 11 casos | OCR (repetición de línea) |
| Puntuación tipográfica pegada (`“el`, `paz”`, `–como`) | 232 tokens | Normal en texto tipográfico — SIN acción (ya la maneja el tokenizer de spaCy + is_punct y la regex de NLTK) |

## 2. Procesamientos implementados (pipeline de 6 etapas, celda 39)

Orden de aplicación sobre `df["discurso"]` (el texto original se conserva en `df["discurso_original"]`):

### Etapa 1 — P1: limpieza de líneas plantilla (`limpiar_lineas_plantilla`)
Elimina **solo líneas completas** que son metadata del scan:
- Números de página sueltos (líneas solo con dígitos): **32**
- Pies de fecha "agosto 1974"/"agosto 1978": **17**
- Encabezado de radio "señal memoria / rtvc": **7** (Laureano Gómez 1950 ×4, Pastrana 1970 ×3)
- Líneas decorativas "---": **44** y "*****": **8**
- **Total: 108 líneas** (en 5 discursos; 82 de ellas en C. Lleras 1966, López 1974 y Turbay 1978)

### Etapa 2 — De-hyphenation (`de_hyphenate`, implementado previamente)
Une `palabra-\n` + continuación **solo si la concatenación existe en el léxico** (tokens del corpus ∪ vocabulario vectorial de spaCy):
- **525 de 552 uniones aplicadas** (95,1 %), 27 rechazadas (pseudopalabras como `comisio-laureano` — la unión ciega crea basura: `abatiendolar`)
- Ejemplos: `económi-\nco`→económico, `go-\nbierno`→gobierno, `aspiracio-\nnes`→aspiraciones, `santan-\nder`→santander, `públi-\nca`→pública

### Etapa 3 — P2: corrección OCR dígito→letra (`corregir_ocr_digitos`)
Con validación léxica estricta (candidatos: `1·`→r, `1-`→r, `1`→{r,l,i,b,n}, `0`→o):
- **8 tokens corregidos**: `pa1·a`→para, `pudie1·a`→pudiera, `unive1·sidad`→universidad, `p1·ensa.`→prensa., `luchado1·`→luchador, `fue1-zas`→fuerzas, `c0lombia`→colombia, `1eutral`→neutral
- No corregibles (documentados): `colop1biano` (necesita p→m), `12o/o`, `km2,`, `40%`, `13-a.`, `m-19.`, `humana.209.`

### Etapa 4 — P3: unión de "n" suelta (`unir_n_suelta`)
Iterativo (hasta 3 pasadas), con validación léxica:
- `n ación`→**nación** (7 casos; "nación" pasa de 176 a 183 lemas)
- Casos de 3 trozos: `n u estro`→nuestro, `n un ca`→nunca, `n o rte`→norte
- Conserva "n o he/se/hay/..." cuando la concatenación no es palabra válida

### Etapa 5 — P4: deduplicación de funcionales (`deduplicar_funcionales`)
Whitelist estricta {en, de, el, la, para, por, un, una, con, que, los, las}:
- **11 duplicados eliminados**: "en en" ×5, "por por" ×2, "de de" ×1, "el el" ×1, "para para" ×1, "un un" ×1
- **Conservados**: "no no" (3) y "si si" (2) — posible retórica (cita bíblica "sí, sí; no, no")

## 3. Impacto en los puntos del taller (antes → después)

| Punto | ¿Cambió? | Evidencia |
|---|---|---|
| 1. Exploración | NO | No depende del texto |
| 2. Conteos | SÍ | tokens_nltk del corpus: 53.998 → 53.938 (−60) |
| 3. Frecuencias | SÍ | Turbay: 10 filas del top-20 (salen "n", "agosto", "1978"); López: 5 filas (sale "renta", entra "nación") |
| 4. Nubes | SÍ | 3 filas (corriente) + 4 filas (década) |
| 5. N-gramas/Matcher | SÍ | "fuerza armado" 33→35 |
| 6. Diversidad | SÍ | 4 filas; López TTR 0,655→0,658 |
| 7. Estilometría | SÍ | Flesch: Lleras 16,75→16,64 · López 17,15→17,18 · Turbay 19,15→19,05 |
| 8. Temporal | SÍ | total_palabras: 1966 sin cambio · 1974: 5.500→5.485 · 1978: 7.574→7.543 |
| 9. NER | SÍ | 1.007 → 1.009 entidades (LOC +1, ORG −1, PER +2) |
| 10. Grafos | SÍ | sujeto-verbo: 108→110 aristas ("gobierno→interferir/proteger" nuevas) |

## 4. Validación realizada

- **Revisión técnica** (ml-reviewer): reglas correctas (solo líneas completas, validación léxica en P2/P3, whitelist en P4), de-hyphenation intacto, 0 errores de ejecución en 134 celdas, sin regresión (solo celdas 38-39 y 131 cambiadas), Anexo A coherente.
- **Auditoría de datos** (independiente): las 4 correcciones reproducidas con cifras exactas (108/8/10/11), delta −60 cuadra, consistencia cruzada verificada (Flesch y totales idénticos entre CSVs), 36/36 artefactos regenerados.
- **Riesgo residual documentado**: `colop1biano` y ~5 palabras raras no corregidas; "n"/"1978" residuales en Turbay provienen de otras líneas OCR (pie de página ya eliminado; persisten menciones dentro del texto).

## 5. Referencias

- Luque, Agudelo, Leal & Sosa (2023). *Characterization of the presidential inauguration speech and identification of political communities in Colombia*. arXiv:2306.13989 — paper de los mismos 22 discursos (fuente web de Presidencia; no reporta estos artefactos).
- RAE (2010). *Ortografía de la lengua española* — el guion de fin de línea divide por sílabas: la palabra partida es una sola.
- Literatura de preprocesamiento OCR/histórico: NewsEye (2020), OCR post-correction (arXiv:2108.02899, 2107.00927, 2205.15575, 2310.13213, 2504.00414).
