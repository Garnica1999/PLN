# Matriz de cobertura de hipótesis — fase de lexicones (P0)

Generada por `scripts/phase36_auditoria_lexicones.py` **antes** de construir o descargar cualquier recurso (§4: no ejecutar un experimento hasta comprobar que es nuevo).

Experimentos registrados en el historial: **67**. Experimentos que mencionan un canal léxico implementado: **0**.

## 1. Familias de señal léxica: qué existe ya y qué falta

| # | Familia | Estado | Qué hay realmente | Evidencia | Implicación para esta fase |
|---|---|---|---|---|---|
| L1 | Lista léxica cerrada general (stopwords) | **EXISTE (mínima)** | SPANISH_STOPWORDS: ~140 entradas autoelaboradas en el propio proyecto. Alimenta dos features: stopword_ratio y content_ratio. | `src/textometry.py:38-49 (frozenset, comentario 'recurso léxico cerrado, sin parámetros aprendidos'); src/feature_catalog.py:43-44; features/dense_train.parquet (columnas stopword_ratio, content_ratio)` | Precedente aceptado: una lista fija autoelaborada es admisible. Ampliar de 1 lista a 24 subcategorías es genuinamente nuevo, pero el listón de redundancia es real: ~140 palabras funcionales cubren el高频 de cualquier texto español. |
| L2 | Categorías temáticas de dominio | **NUNCA PROBADA como canal de features** | Solo evidencia **observacional**: el Gate A registra que el léxico temático diferencia (fandoms/MTV en 13-17 frente a política/noticias en 65-xx) y un análisis de términos por clase. Adicionalmente el contenido entra por TF-IDF word(1-2) y char_wb(3-5) de los 14 miembros, token a token. | `PROJECT_STATE.md (Gate A, punto 3 'Señal confirmada'); reports/tables/top_terms_by_class.csv; experiments.csv (fase 3, análisis lingüístico)` | La hipótesis fue confirmada por observación pero **nunca explotada** como agregado de baja dimensión. Esa es la brecha exacta que ataca esta fase. |
| L3 | Registro / informalidad / slang | **NUNCA PROBADA como lexicon; parcialmente implícita** | TF-IDF char_wb(3-5) captura la forma ortográfica informal y las abreviaciones; las features de forma ya cubren mayúsculas, alargamientos, emoticonos y signos de exclamación/interrogación. | `features/age_structured_features.parquet (shape_allcaps_ratio, shape_repeated_letter_token_ratio, punct_excl_density, punct_quest_density); features/dense_train.parquet (elongated_count, n_emoticon, n_excl, n_quest)` | Máximo riesgo de redundancia de las tres familias: es la que más solapa con los n-gramas de carácter. Por eso la familia A es la primera en cribarse. |
| L4 | Etapas de vida / contexto social | **NUNCA PROBADA** | Nada. Solo señales de longitud y estructura social indirectas (RT, menciones). | `reports/age_hypothesis_coverage.md (no figura ninguna hipótesis léxica de contexto vital); experiments.csv: 0 experimentos con 'lexicon|wordlist'` | La familia con más superficie nueva: vocabulario de escuela, universidad, empleo, parentalidad, hogar y juez de edad no es un concepto que los n-gramas representen como categoría. |
| L5 | P(clase|palabra) supervisado | **FUERA DE ALCANCE por diseño** | Existe un canal léxico supervisado, pero solo para entidades: P(clase|mención), P(clase|hashtag) y P(clase|fuente de RT), aprendido in-fold con suavizado empírico-bayesiano. | `src/advanced_features.py:326-340 (supervised_entity_features, kind in {mention, hashtag, rt_source})` | No existe ningún canal por palabra libre, así que la fase no repite nada. Construirlo sería señal supervisada de la tarea y §5 lo prohíbe: se documenta como línea distinta, no como parte de esta fase. |
| L6 | Embeddings / transformers / modelos de lenguaje | **PROHIBIDO (y sin uso)** | transformers, sentence-transformers y torch están instalados en el venv y listados en requirements.txt. Uso en el camino de modelado = 0: el único import del proyecto es una detección de capacidad de GPU. | `grep de imports en src/ y scripts/ → único match: src/gpu.py:130 ('import torch' dentro de una comprobación de disponibilidad)` | Auditoría §32: el entorno no está limpio de paquetes, pero el uso es cero. Se documenta explícitamente (D-038) en lugar de ocultarlo. |

## 2. Recursos externos: admisibilidad (§1 y §5)

| Recurso | Veredicto | Motivo |
|---|---|---|
| Lexicones de edad / autor profiling (e.g. listas de 'twitter por edad') | **DESCARTADO** | §5 prohíbe recursos con etiquetas de edad o construidos con ellas: equivaldría a introducir la solución externa de la competencia. |
| NRC Emotion / EmoLex, AFINN | **NO UTILIZADO** | Admisibles en licencia y simbólicos, pero miden afecto, no edad, y su senyal en tweets ASCII sin emoji depende de intensificadores ya cubiertos por char n-grams. No responde a la pregunta de la fase. |
| WordNet español (MCR) | **DESCARTADO por licencia** | Licencia por palabra y mixta (BSD/CC-BY según término): §1 cond. 7 y 9 exigen poder establecer la licencia razonablemente. Además es una traducción del WordNet inglés, de calidad discutible para español. |
| spaCy POS/lookups locales | **YA DESCARTADO en el proyecto** | D-026: el corpus ASCII degrada el etiquetado; variación medida -0,0032. No se reabre. |
| Listas externas de slang de Twitter | **DESCARTADO** | Mayoritariamente en inglés, basadas en emoji (inexistentes en el corpus, D-009) y muchas construidas con datos de plataformas, sin licencia académica clara. |
| Lexicones autoelaborados en el proyecto | **ADMITIDO (elegido)** | Satisfacen §1 completo: no son embeddings, no son modelos, no contienen etiquetas, no hubo corpus externo, no se usan para pseudo-labels ni para modificar el texto, licencia propia y procedencia declarada (P1). |

## 3. Verificación automática de la afirmación "nunca probado"

**Nivel 1 (decisivo)**: términos que solo existirían si hubiera un canal léxico implementado (`age_lexicon`, `lexicon_features`, `entradas_lexicon`, `subcategoria`).

| Familia | Coincidencias en `experiments.csv` | Coincidencias en el código | Veredicto |
|---|---|---|---|
| L2 | 0 | 0 | **nunca evaluada** |
| L3 | 0 | 0 | **nunca evaluada** |
| L4 | 0 | 0 | **nunca evaluada** |

**Nivel 2 (contextual, no decisivo)**: términos Lazarus del español. Se listan para no esconder coincidencias, aunque ninguna constituye un canal de features.

| Familia | Patrón | Dónde aparece (muestra) | Lectura |
|---|---|---|---|
| L2 | `lexicon|lexic[oó]n|categorias?_tematic|tema_dominio|dominio_tem` | `advanced_01_freeze_y_gap.py:336`, `advanced_02_features.py:220` | `lexicón` se menciona al documentar la **decisión** de usar un recurso léxico cerrado, no como canal implementado. |
| L3 | `slang|jerga|registro_informal|abreviacion|muletilla|vulgaridad` | `fase03_linguistic.py:76`, `fase03_linguistic.py:97` | las coincidencias están en `fase03_linguistic.py` y `eda.py`: son la **hipótesis** de que las abreviaciones son señal, analizada por forma del token, no por lista léxica. |
| L4 | `etapa_vida|life_stage|universidad|trabajo|parentalidad|jubilacion` | `experimento_e019_modelos_densos.py:15`, `fase06_07_08_documentacion.py:157` | «trabajo» y «universidad» son palabras_common del español: aparecen en la prosa de los informes, no como variable. |

## 4. Experimentos históricos que podrían solaparse

Ninguno de los 67 experimentos previos construyó un canal léxico agregado. Los que más se acercan y, por tanto, los que hay que **no repetir ni re-etiquetar como nuevo**:

1. **Fase 3, análisis lingüístico** (`03_linguistic_analysis.ipynb`, `reports/tables/top_terms_by_class.csv`): observationalmente ya mostró señal temática. No es una línea de features; su tabla por clase **no** puede reutilizarse como recurso (§13: ajustar categorías con la distribución de etiquetas está prohibido).
2. **Canal supervisado de entidades** (`src/advanced_features.py:326`): único precedente de señal léxica aprendida. Los lexicones de esta fase son su contraparte **no supervisada**, que es justamente lo que no existe todavía.
3. **stopword_ratio / content_ratio** (fase 4, textometría): el precedente directo de recurso léxico cerrado; fija el listón de redundancia contra el que se criba la familia A.
4. **TF-IDF word(1-2) y char_wb(3-5)** de los 14 miembros: no es un experimento que se pueda repetir, pero es **el competidor real**: cualquier ganancia del bloque léxico debe superar el control de permutación (DD-6 del plan).

## 5. Hipótesis que quedan abiertas en esta fase

| # | Hipótesis abierta | Familia | Predicción si es cierta | Cómo se decide |
|---|---|---|---|---|
| H-LEX-A | La pertenencia a categorías de **registro e informalidad** aporta una dimensión no capturada por los n-gramas de carácter | A | mejora el error en los segmentos `solo_url`/`muy_corto` y en las fronteras centrales | P2 con control de redundancia y de permutación |
| H-LEX-B | El vocabulario de **etapas de vida** informa el contexto social y eso se convierte en señal de edad | B | el bloque `Vida` supera al bloque permutado en LOFO | P3 con el mismo control |
| H-LEX-C | El **perfil temático** aporta una señal agregada parcialmente complementaria al token individual | C | el bloque `Temas` supera al permutado y rescata errores reales de V12 | P4 con el mismo control |

## 6. Criterio de cierre de cada familia (§28-29)

- Una familia que **no** supere el gate (Δ ≥ +0,003 con ≥4/5 folds) queda **cerrada**: no se reintenta combinándola con más features para rescatarla.
- Una familia redundante con variables existentes (§12) queda **documentada y descartada** antes de llegar a la fusión.
- El resultado nulo es un cierre legítimo (§29 B, C y D), no un fracaso: la expectativa previa declarada es nula.

## 7. Qué NO se hará en esta fase

- No se descargan lexicones (decisión DD-1: autoelaborados en el proyecto).
- No se construye ningún vectorizador ni representación aprendida.
- No se construye un lexicon supervisado ni se ajustan categorías con las etiquetas.
- No se reutiliza el holdout (agotado) ni se crea otro.
- No se generan submissions ni se toca `age_submission_v12.csv` ni `age_submission_v12_reproducible.csv`.

