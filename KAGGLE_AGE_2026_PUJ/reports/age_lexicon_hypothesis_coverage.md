# Matriz de cobertura de hipótesis — fase de lexicones (P0)

Generada por `scripts/phase36_auditoria_lexicones.py` **antes** de construir cualquier recurso (§4: no ejecutar un experimento hasta comprobar que es nuevo).

Experimentos registrados en el historial: **72**. Experimentos que mencionan un canal léxico implementado: **0**.

## 1. Familias de señal léxica: qué existe ya y qué falta

| # | Familia | Estado | Qué hay realmente | Evidencia | Implicación |
|---|---|---|---|---|---|
| L1 | Lista léxica cerrada general (stopwords) | **EXISTE (mínima)** | SPANISH_STOPWORDS: ~140 entradas autoelaboradas en el propio proyecto. Alimenta dos features: stopword_ratio y content_ratio. | `src/textometry.py:38-49 (frozenset, comentario 'recurso léxico cerrado, sin parámetros aprendidos'); src/feature_catalog.py:43-44` | Precedente aceptado: una lista fija autoelaborada es admisible. Ampliar de 1 lista a 24 subcategorías es nuevo, pero el listón de redundancia es alto. |
| L2 | Categorías temáticas de dominio | **NUNCA PROBADA como canal de features** | Solo evidencia observacional: el Gate A registra que el léxico temático diferencia (fandoms/MTV en 13-17 frente a política/noticias en 65-xx). Adicionalmente el contenido entra token a token por TF-IDF word(1-2) y char_wb(3-5). | `PROJECT_STATE.md (Gate A, punto 3); reports/tables/top_terms_by_class.csv; experiments.csv (fase 3)` | La hipótesis fue confirmada por observación pero nunca explotada como agregado de baja dimensión. Esa es la brecha que ataca esta fase. |
| L3 | Registro / informalidad / slang | **NUNCA PROBADA como lexicon; parcialmente implícita** | TF-IDF char_wb(3-5) captura la forma ortográfica informal; las features de forma ya cubren mayúsculas, alargamientos, emoticonos y signos. | `features/age_structured_features.parquet (shape_allcaps_ratio, punct_excl_density, punct_quest_density); features/dense_train.parquet (elongated_count, n_emoticon)` | Máximo riesgo de redundancia: es la que más solapa con los n-gramas de carácter. |
| L4 | Etapas de vida / contexto social | **NUNCA PROBADA** | Nada. Solo señales de longitud y estructura social indirectas (RT, menciones). | `reports/age_hypothesis_coverage.md (no figura ninguna hipótesis léxica de contexto vital)` | La familia con más superficie nueva. |
| L5 | P(clase|palabra) supervisado | **FUERA DE ALCANCE por diseño** | Existe un canal léxico supervisado, pero solo para entidades: P(clase|mención), P(clase|hashtag) y P(clase|fuente de RT), aprendido in-fold. | `src/advanced_features.py:326-340 (supervised_entity_features, kind in {mention, hashtag, rt_source})` | No existe canal por palabra libre, así que la fase no repite nada. Construirlo sería señal supervisada de la tarea y §5 lo prohíbe. |
| L6 | Embeddings / transformers / modelos de lenguaje | **PROHIBIDO (y sin uso)** | transformers, sentence-transformers y torch están instalados en el venv y listados en requirements.txt. Uso en el camino de modelado = 0: el único import es una detección de GPU. | `grep de imports en src/ y scripts/ → único match: src/gpu.py:130` | Auditoría §32: el entorno no está limpio de paquetes, pero el uso es cero (D-038). |

## 2. Recursos externos: admisibilidad (§1 y §5)

| Recurso | Veredicto | Motivo |
|---|---|---|
| Lexicones de edad / autor profiling | **DESCARTADO** | §5 prohíbe recursos con etiquetas de edad o construidos con ellas. |
| NRC Emotion / EmoLex, AFINN | **NO UTILIZADO** | Admisibles en licencia, pero miden afecto, no edad, y su señal depende de intensificadores ya cubiertos por char n-grams. |
| WordNet español (MCR) | **DESCARTADO por licencia** | Licencia por palabra y mixta: §1 cond. 7 y 9 exigen poder establecer la licencia razonablemente. Además es traducción del WordNet inglés. |
| spaCy POS/lookups locales | **YA DESCARTADO en el proyecto** | D-026: el corpus ASCII degrada el etiquetado; variación medida -0,0032. |
| Listas externas de slang de Twitter | **DESCARTADO** | Mayoritariamente en inglés, basadas en emoji (inexistentes, D-009) y sin licencia académica clara. |
| Lexicones autoelaborados en el proyecto | **ADMITIDO (elegido)** | Satisfacen §1 completo: no son embeddings, no son modelos, no contienen etiquetas, no hubo corpus externo, no se usan para pseudo-labels ni para modificar el texto, licencia propia y procedencia declarada. |

## 3. Verificación automática de la afirmación "nunca probado"

**Nivel 1 (decisivo)**: términos que solo existirían si hubiera un canal léxico implementado (`age_lexicon`, `lexicon_features`, `entradas_lexicon`, `subcategoria`), excluyendo los ficheros de la propia fase.

| Familia | Coincidencias en `experiments.csv` | Coincidencias en el código | Veredicto |
|---|---|---|---|
| L2 | 0 | 0 | **nunca evaluada** |
| L3 | 0 | 0 | **nunca evaluada** |
| L4 | 0 | 0 | **nunca evaluada** |

**Nivel 2 (contextual, no decisivo)**: términos Lazarus del español, listados para no esconder coincidencias.

| Familia | Dónde aparece | Lectura |
|---|---|---|
| L2 | `advanced_01_freeze_y_gap.py:336`, `advanced_02_features.py:220` | «lexicón» aparece al documentar la decisión de usar un recurso léxico cerrado. |
| L3 | `fase03_linguistic.py:76`, `fase03_linguistic.py:97` | las coincidencias están en el análisis lingüístico (fase 3) y en eda.py: es la hipótesis de que las abreviaciones son señal, analizada por forma del token. |
| L4 | `experimento_e019_modelos_densos.py:15`, `fase06_07_08_documentacion.py:157` | «trabajo» y «universidad» son palabras comunes del español: prosa de los informes. |

## 4. Experimentos históricos que podrían solaparse

1. **Fase 3, análisis lingüístico**: mostró señal temática de forma observacional. Su tabla por clase **no** puede reutilizarse como recurso (§13: ajustar categorías con la distribución de etiquetas está prohibido).
2. **Canal supervisado de entidades** (`src/advanced_features.py:326`): único precedente de señal léxica aprendida. Los lexicones son su contraparte **no supervisada**.
3. **stopword_ratio / content_ratio**: el precedente directo de recurso léxico cerrado; fija el listón de redundancia.
4. **TF-IDF word(1-2) y char_wb(3-5)**: no es repetible, pero es **el competidor real**; cualquier ganancia del bloque léxico debe superar el control de permutación (DD-6).

## 5. Hipótesis abiertas en esta fase

| # | Hipótesis | Familia | Decisión |
|---|---|---|---|
| H-LEX-A | La pertenencia a categorías de registro aporta una dimensión no capturada por los n-gramas de carácter | A | P2 con control de redundancia y de permutación |
| H-LEX-B | El vocabulario de etapas de vida informa el contexto social y se convierte en señal de edad | B | P3 con el mismo control |
| H-LEX-C | El perfil temático aporta una señal agregada complementaria al token individual | C | P4 con el mismo control |

## 6. Criterio de cierre por familia (§28-29)

- Una familia que no supere el gate (Δ ≥ +0,003 con ≥4/5 folds) queda **cerrada**.
- Una familia redundante con variables existentes (§12) queda **documentada y descartada** antes de la fusión.
- El resultado nulo es un cierre legítimo (§29 B, C y D): la expectativa previa es nula.

## 7. Qué NO se hizo en esta fase

- No se descargaron lexicones (decisión DD-1: autoelaborados).
- No se construyó ningún vectorizador ni representación aprendida.
- No se construyó lexicones supervisados ni se ajustaron categorías con etiquetas.
- No se reutilizó el holdout (agotado) ni se creó otro.
- No se generaron submissions ni se tocaron los artefactos de V12.

