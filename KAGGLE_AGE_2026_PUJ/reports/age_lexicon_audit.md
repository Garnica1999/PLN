# Auditoría de la fase de lexicones (P8)

**Pregunta**: ¿aporta un pequeño número de lexicones lingüísticos generales una señal de registro, etapa vital o dominio temático que **complemente** a V12?

**Gate pre-registado**: delta >= +0,003 y >= 4/5 folds positivos, Y superar al control de permutacion. **Holdout**: agotado, no se ha reutilizado. **Leaderboard**: no se ha consultado. **Sin** embeddings, transformers, modelos externos, vectorizadores aprendidos nuevos, corpus/labels externos ni etiquetas de test.

## 1. Qué se construyó

- **24 subcategorías** en tres familias: registro (estilo), etapas de vida (contexto vital) y dominios temáticos (contenido).
- 1536 entradas declaradas -> **1328 efectivas** tras el cribado de aplicabilidad al corpus (frecuencia mínima 3 en el train, sin usar etiquetas).
- 1133 tokens únicos; 93 ambiguos descartados (DD-5).
- Cobertura de tokens: registro 5.72%, vida 0.77%, temas 10.01%.
- La familia **vida** apenas aparece en el corpus: es la señal menos promising desde el principio y así lo confirma todo lo que siguió.

## 2. Control lexicon-only: ¿existe la señal? (§14)

Suelo de referencia: azar 0.1667 y clase prior 0,0485 (clases casi balanceadas). V12 alcanza 0.6004.

| Familia | Columnas | LogReg | LinearSVC | Lectura |
|---|---|---|---|---|
| registro | 19 | 0.152 | 0.1483 | por debajo del azar: sin señal medible por sí misma |
| vida | 16 | 0.0731 | 0.0729 | por debajo del azar: sin señal medible por sí misma |
| temas | 37 | 0.2033 | 0.2018 | **supera el azar**: señal real por sí misma |

## 3. Redundancia (§12)

- **3 de 72 columnas** superan |rho| >= 0.8 contra alguna variable existente o contra las 279 features de V12 (correlación máxima 0.966, mediana 0,105).
- Las tres son `abreviaciones_internet`: sus conteos correlacionan 0,93-0,97 con la longitud del tweet, es decir, es un **proxy del tamaño**, no de registro. Se excluyeron de la fusión por §12.
- **El bloque no es una reformulación de lo que V12 ya tiene**: la hipótesis de §12 se responde **sí** a nivel de variable.

## 4. Fusión con V12 y control de permutación (P6, DD-6)

Referencia V12 reproducida en la misma sesión: **0.6004** ± 0.0042 (per fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]), verificada por aserción contra el valor canónico 0,6004.

| Candidato | Columnas | macro-F1 | Δ | Folds+ | Nula permutada (media/máx) | Ventaja | Gate |
|---|---|---|---|---|---|---|---|
| V12 + Registro | 16 | 0.6005 | +0.0001 | 4/5 | 0.6005 / 0.6005 | +0.0000 | no |
| V12 + Vida | 16 | 0.6004 | +0.0000 | 2/5 | 0.5994 / 0.5994 | +0.0010 | no |
| V12 + Temas | 37 | 0.5995 | -0.0009 | 2/5 | 0.5997 / 0.5997 | -0.0002 | no |
| V12 + Registro + Vida + Temas | 69 | 0.5986 | -0.0018 | 2/5 | 0.5965 / 0.5965 | +0.0021 | no |

**El control de permutación es la pieza decisiva.** Dos de los cuatro bloques superan a su propia nula (`Vida` +0,0010 y la unión +0,0021), pero ambas cantidades están **por debajo del gate** y la unión es además **peor que V12** en valor absoluto: gana contra la nula y pierde contra la referencia, que es la definición de señal no accionable.

## 5. Complementariedad y error (§19-§21)

Errores de V12 en el OOF: **67,610**. Los candidatos modifican entre el 10,1% y el 11,1% de las filas, y el efecto neto es nulo o negativo.

| Candidato | Rescate | Daño | Neto | Modificadas | 18-24|25-34 | 25-34|35-49 |
|---|---|---|---|---|---|---|
| registro | 5541 | 5489 | +52 | 10.27% | -150 | -6 |
| registro___vida___temas | 5782 | 6121 | -339 | 11.10% | -75 | +193 |
| temas | 5396 | 5548 | -152 | 10.34% | -75 | +73 |
| vida | 5427 | 5380 | +47 | 10.13% | -132 | +31 |

La lectura es inequívoca: **no hay rescate neto de errores de V12**. El lexicón mueve decisiones (una de cada diez) a cambio de tantas aciertas como pierde.

## 6. Veredicto por familia (§28)

| Familia | Resultado en P6 | Decisión |
|---|---|---|
| Registro | 0.6005 (Δ +0.0001, 4/5) | **A** - negativa: cerrada |
| Vida | 0.6004 (Δ +0.0000, 2/5) | **B/D** - algo de señal sobre la nula, muy por debajo del gate: cerrada |
| Temas | 0.5995 (Δ -0.0009, 2/5) | **A** - negativa: cerrada |
| Registro | 0.5986 (Δ -0.0018, 2/5) | **B/D** - algo de señal sobre la nula, muy por debajo del gate: cerrada |

## 7. Respuesta a la pregunta científica (§35)

> ¿Los lexicones proporcionan una dimensión lingüística de edad que V12 no captura?

**No, de forma explotable.**

- **¿Registro?** Control aislado 0,1520 (por debajo del azar 0,1667). En la fusión no supera ni a su nula (ventaja +0,0000). **Respuesta: no aporta.**
- **¿Etapas vitales?** Control aislado 0,0731, el más bajo, coherente con que esa familia cubre el 0,6-0,8% de los tokens. En la fusión muestra la mayor ventaja sobre su nula (+0,0010) pero con Δ +0,0000 y 2/5 folds. **Respuesta: señal real pero inutilizable.**
- **¿Temática?** Único caso que supera el azar en aislamiento (0,2033). En la fusión, **pierde** contra V12 y contra su propia nula. **Respuesta: no aporta.**

## 8. Criterio de cierre (§29)

Se aplica **A** (ninguna familia alcanza el gate), **B/D** (el bloque no es redundante, pero su señal no se traduce en mejora) y **C** (donde el Δ nominal no sobrevive al control de permutación). **La línea se cierra.** V12 permanece congelado y no se genera ninguna submission.

## 9. Auditoría de recursos (§32)

- `transformers`, `sentence-transformers` y `torch` están instalados en el venv y listados en `requirements.txt`; el **uso en el camino de modelado es 0** y el único import del proyecto es una detección de capacidad de GPU (`src/gpu.py`). Se documenta sin maquillar el entorno.
- Embeddings, Word2Vec, FastText, GloVe, sentence-transformers, modelos de lenguaje: **0 usos**. Corpus o labels externos: **0**. Etiquetas de test: **0**. Pseudo-labels: **0**.
- El cribado de frecuencia mínima usa el vocabulario del train completo: es **selección no supervisada y transductiva**, sin usar etiquetas en ningún punto; se declara por transparencia (§13).

## 10. Reproducibilidad tras el incidente git

Durante la ejecución de P6, otra sesión del mismo repositorio (proyecto `KAGGLE_TOXIC_2026_PUJ`) hizo commit y cambió de rama, lo que eliminó del árbol los ficheros no versionados de esta fase (`lexicons/**`, `src/lexicons.py`, `src/lexicon_lists.py`, `scripts/phase36..phase41`, informes de P0-P5). Ver `INCIDENTE_GIT_2026-09-25.md`.

- **Intacto**: `experiments/phase39_fusion_v12.json`, `reports/age_lexicon_results.csv`, las predicciones OOF y las matrices de `features/` (protegidas por `.gitignore`). El veredicto de P6 y el análisis de P7 se calcularon con esos artefactos originales.
- **Reconstruido y verificado**: el bloque `registro` regenerado es **idéntico bit a bit** a la matriz que consumió P6.
- **Reconstruido con diferencia**: el bloque `vida` regenerado tiene 19 columnas frente a las 16 originales, porque la lista de `hogar` escrita de nuevo conserva más entradas válidas (la original quedaba en 2 tokens tras eliminar ambiguos y se retiraba).
- **Perdidos sin copia**: `reports/age_lexicon_ablation.csv` y `reports/age_lexicon_redundancy.csv` (tablas de ablación por subcategoría y de correlaciones). Su contenido se conserva en el JSON transcrito y en `experiments/phase38.log`, y se regeneran en unos minutos con el código de P2-P5.
- **Transcrito**: `experiments/phase38_bloques_lexicos.json` se reconstruyó con los valores medidos en la ejecución original, etiquetado como transcripción; la primera iteración queda además en `experiments/phase38.log`, que sobrevivió.
- **Pendiente opcional**: reejecutar P6 con las listas reconstruidas (≈2,5 h) para que la traza sea byte a byte consistente. El veredicto no dependería de ello: los cuatro deltas están entre -0,0018 y +0,0001, muy lejos del gate de +0,003.

