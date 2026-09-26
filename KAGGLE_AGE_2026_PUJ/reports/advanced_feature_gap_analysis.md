# Análisis de brechas del espacio de features (`advanced_feature_gap_analysis.md`)

Generado por `scripts/advanced_01_freeze_y_gap.py` sobre los artefactos reales del proyecto (nada de este documento es estimación).

## 1. ¿Qué features existen hoy?

- **Densas locales (43 en el catálogo, 35 útiles)**: textometría y estilo, calculadas por `src/textometry.py` (grupos C y D del catálogo).
- **Representaciones dispersas**: TF-IDF `char_wb(3,6)` (300k y 700k features), canal de menciones (word 1-gram de `@cuenta`), canal de hashtags, y word 1-2.
- **Scores de modelos (72 features del meta-modelo)**: 13 miembros del ensamble.
- **Total de features directas del meta-modelo**: 107.

## 2. ¿Qué familias existen y cuáles están subrepresentadas?

| Familia del brief | Estado actual | Cobertura |
|---|---|---|
| A. Estadísticos de distribución | solo medias y algunas desviaciones | **baja** |
| B. Distribución de caracteres | ratios básicos (mayúsculas, dígitos, puntuación) | media |
| C. Distribución de longitud de palabras | media, máximo y desviación | **baja** |
| D. Riqueza léxica | TTR, CTTR, hapax, entropía | media |
| E. Rareza léxica (frecuencia de corpus) | **ausente** | **nula** |
| F. Perfil léxico por clase (log-odds) | solo análisis exploratorio, no features | **nula** |
| G. Distancia a prototipos | Rocchio como miembro del ensamble | media |
| H. Word shape / ortografía | elongaciones y mayúsculas agregadas | baja |
| I. Posición dentro del tweet | solo `starts_mention` | **baja** |
| J. Menciones supervisadas (P(clase|mención)) | TF-IDF de menciones (sin supervisión) | **nula** |
| K. Retuits y cuenta fuente supervisada | `is_rt` y conteo de menciones | **baja** |
| L. Hashtags supervisados | TF-IDF de hashtags | **baja** |
| M. URLs y dominios | conteo de URLs | **baja** |
| N. Estructura del texto | `n_sentences` | baja |
| O. N-gramas estructurados | matriz de 14 configuraciones ya evaluada | alta |
| P. Representaciones alternativas (NB-SVM) | **no evaluado** | **nula** |
| Q. Tokenizador propio de tweets | **no evaluado** | **nula** |
| R. Entropía / concentración | entropía léxica y de caracteres únicos | media |
| S. Compresibilidad | **no evaluado** | **nula** |
| T. Repetición | runs de caracteres y puntuación | baja |
| U. Posición léxica (primera/última palabra) | **ausente** | **nula** |
| V. Morfología / sílabas | **ausente** | **nula** |
| W. POS profundo | no implementado (recurso preentrenado en zona gris, D-001) | nula |
| X. Distancia al léxico por clase | log-odds solo en EDA | **nula** |
| Y. OOV y novedad | **ausente** | **nula** |
| Z. Interacciones de features | ninguna explícita | **nula** |

## 3. ¿Qué familias están altamente correlacionadas?

Pares con mayor redundancia entre features densas (|ρ| de Spearman, Fase 2):

- `E-008` — char_wb 3-6 + canal de menciones: macro-F1 0.5557 (314,420 features)
- `E-003c` — char_wb 3-6: macro-F1 0.5413 (300,000 features)
- `E-003b` — char_wb 3-5: macro-F1 0.5408 (224,628 features)
- `E-004b` — char_wb 3-6 + word 1-gram: macro-F1 0.5354 (339,409 features)
- `E-007a` — union sin IDF (TF puro): macro-F1 0.5339 (336,493 features)
- `E-005c` — solo quitar URLs + union: macro-F1 0.5277 (292,170 features)

Reportado en `reports/tables/representaciones_pareado.csv`: `ttr`~`hapax_ratio` ρ=0,999 · `n_unique_words`~`lexical_entropy` ρ=0,998 · `stopword_ratio`~`content_ratio` ρ=0,997 · `cttr`~`lexical_entropy` ρ=0,997.

## 4. Fenómenos del EDA sin representación

Las features más informativas hoy (Kruskal ε²) son:

- `n_chars`: ε² = 0.1255
- `unique_char_ratio`: ε² = 0.0669
- `cttr`: ε² = 0.0664
- `lexical_entropy`: ε² = 0.0652
- `n_unique_words`: ε² = 0.0645

Y por información mutua: `avg_word_len`, `n_chars`, `std_word_len`, `punct_ratio`.

Fenómenos identificados en el EDA que **no** tienen representación estructurada: rareza léxica de corpus, perfil léxico por clase, identidad supervisada de menciones/hashtags/cuenta fuente, posición de los elementos, forma de palabra, entropía por tipo, compresibilidad y repetición más allá de runs de caracteres.

## 5. Fenómenos solo representados vía TF-IDF (no como features estructuradas)

- **Identidad de menciones**: el canal TF-IDF las usa, pero sin agregar la probabilidad por clase -> no se puede expresar '*esta cuenta mencionada es típica de 13-17*'.
- **Hashtags**: igual, solo bolsa de palabras.
- **Cuenta fuente de retuits** (`RT @cuenta:`): solo como token de mención, sin tratar el evento estructurado.
- **Dominios de URL**: los enlaces del corpus son acortadores `t.co` (verificado en el EDA), por lo que el dominio no aporta; la señal está en la *presencia y posición*.

## 6. Información presente en el texto crudo que se pierde

- **Posición** de menciones, hashtags, URLs y puntuación (solo se cuentan).
- **Distribuciones internas**: percentiles, asimetría, curtosis, MAD y coeficiente de variación de la longitud de palabra y de los huecos entre elementos.
- **Concentración**: qué tan repetitivo es el vocabulario (Simpson, HHI, Yule's K).
- **Rareza**: frecuencia de corpus de cada token (requiere estadística de train, in-fold).
- **Novedad**: proporción de tokens OOV respecto del vocabulario de train.

## 7. Señales fuertes del error analysis sin modelo especializado

- **Tweets cortos**: 52.5% de error en el quintil más corto (ya probado con especialistas por longitud: +0,0000, pero el brief pide revisarlo con features nuevas de la familia E/Y).
- **Fronteras adyacentes**: los pares ya tienen modelos dedicados (E-028a, precisión 0,7428), pero esos modelos **no** usan features supervisadas de entidades.
- **Errores universales**: aún no analizados como intersección entre modelos (§54 del brief).

## 8. Features que podrían duplicar información

Toda la familia A (percentiles de longitud de palabra) correlaciona con `avg_word_len`, `n_words` y `n_chars` ya existentes; la familia D (riqueza) duplica `ttr`/`cttr`; la familia R (entropía) duplica `lexical_entropy` y `unique_char_ratio`. **Criterio**: se medirá información mutua incremental y solo se conservarán las que aporten sobre lo existente, no por existir.

## 9. Qué falta para capturar distribución, posición, concentración y variabilidad

| Dimensión | Qué falta | Familia propuesta |
|---|---|---|
| Distribución | percentiles, asimetría, curtosis, CV, MAD | A, C |
| Posición | primera/última posición normalizada de cada tipo de elemento | I, U |
| Concentración | Simpson, HHI, Yule's K, Gini | R, D |
| Variabilidad | varianza de posición, dispersión de scores, desacuerdo entre modelos | Z, meta |
| Supervisión de entidades | P(clase\|mención), P(clase\|hashtag), P(clase\|fuente RT) | J, K, L, X |

## 10. Conclusión del análisis

El sistema actual explota bien la **forma superficial** del tweet (longitud, ratios, n-gramas de carácter) y la **bolsa de palabras** de sus canales léxicos. Lo que no explota es la **estadística supervisada de las entidades** (menciones, hashtags, cuenta fuente) ni las **distribuciones internas**, la **rareza/novedad léxica** ni la **posición**. Esas son las cuatro líneas con señal ausente demostrable y son las que se abordan primero.