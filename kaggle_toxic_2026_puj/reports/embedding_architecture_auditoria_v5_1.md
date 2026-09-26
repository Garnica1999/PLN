# P0 de v5.1: firmas de entrenamiento y contabilidad del presupuesto (EBD-17)

Generado por `experiments/embedding_11_auditoria_firmas_v5_1.py` (EBD-17 (P0)).
Duracion: 1.5 s. **Este script no entrena ningun embedding.**

## 1. Por que hacia falta

La contabilidad de v5 estaba repartida en cuatro JSON de resultados y no existia una
identidad canonica de entrenamiento. Sin ella no se puede saber si una configuracion
propuesta en v5.1 es nueva o es un reentrenamiento disfrazado de algo que ya existe,
ni se puede demostrar que el limite de 9 entrenamientos se respeta. Este script
cierra ese hueco antes de gastar la primera unidad.

## 2. Identidad de un entrenamiento

```text
training_signature = sha256(corpus_procesado_sha256 | corpus_version | vector_size |
                             window | min_count | min_n | max_n | negative | sample |
                             epochs | sg | seed | bucket)
```

Dos desviaciones razonadas respecto de la seccion 34, ambas documentadas en el
docstring de `firma_entrenamiento`:

- Se usa el sha256 del **corpus procesado**, no el del dump. El dump es identico para
  5M, 10M y 50M de Wikipedia, asi que con la huella del dump tres corpora distintos
  compartiran identidad y la comprobacion de la seccion 34 seria falsa.
- Se incluye `bucket`, que la spec no lista. Su valor determina parte de la matriz de
  subpalabras y por tanto los pesos; excluirlo permitiria declarar nueva una
  configuracion que en realidad es otra. `hilos` queda fuera porque no cambia la
  configuracion, solo el orden de actualizacion de los pesos.

## 3. Corpus: huellas reales

| corpus | tokens | processed_sha256 | bytes |
|---|---|---|---|
| E-CORP-01 (wikipedia_es) | 10.002.400 | `00b839da6fbc377e` | 63.152.231 |
| E-CORP-09 (wikipedia_es) | 5.011.404 | `62aaa19af45c31f8` | 31.485.618 |
| E-CORP-08 (wikipedia_es) | 50.000.028 | `8088bce2a9283fbc` | 310.983.187 |
| E-CORP-06 (m_c4_es) | 10.000.239 | `361498bf3eca104d` | 62.000.589 |
| E-CORP-07 (None) | 20.000.045 | `1c562286dabe9dc4` | 125.137.300 |

El manifiesto de corpus queda actualizado con `processed_sha256`, `processed_ruta`,
`processed_bytes` y `corpus_version`; ningun valor historico se altera.

## 4. Tabla A: los diez entrenamientos existentes

Las diez firmas son distintas, asi que la contabilidad es univoca. Ninguna consume
presupuesto de v5.1. `training_time_s` es el reloj real cuando consta; el valor 0.0 con
la fuente `no registrado en la corrida` significa que el modelo se entrego desde una
corrida previa y su tiempo no quedo medido en ella.

| training_id | corpus | dim | ws | minC | sub | neg | semilla | vocab | s | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| FASTTEXT-WIKI-10M | E-CORP-01 | 200 | 5 | 3 | 3-6 | 10 | 42 | 100098 | 82.0 | `e54a363da986` |
| FASTTEXT-WIKI-10M-SEED7 | E-CORP-01 | 200 | 5 | 3 | 3-6 | 10 | 7 | 100098 | 0.0 | `afde390f208b` |
| WORD2VEC-WIKI-10M | E-CORP-01 | 200 | 5 | 3 | 0-0 | 10 | 42 | 100098 | 0.0 | `52f94d37e18c` |
| FASTTEXT-WIKI-5M | E-CORP-09 | 200 | 5 | 3 | 3-6 | 10 | 42 | 64839 | 0.0 | `b7108b6877f8` |
| WORD2VEC-WIKI-5M | E-CORP-09 | 200 | 5 | 3 | 0-0 | 10 | 42 | 64839 | 0.0 | `4323c9fa94d1` |
| FASTTEXT-WIKI-50M | E-CORP-08 | 200 | 5 | 3 | 3-6 | 10 | 42 | 247863 | 576.8 | `1045aba32c64` |
| WORD2VEC-WIKI-50M | E-CORP-08 | 200 | 5 | 3 | 0-0 | 10 | 42 | 247863 | 454.2 | `53314bccc6c8` |
| FASTTEXT-WEB-10M | E-CORP-06 | 200 | 5 | 3 | 3-6 | 10 | 42 | 106632 | 0.0 | `8f83703983d2` |
| WORD2VEC-WEB-10M | E-CORP-06 | 200 | 5 | 3 | 0-0 | 10 | 42 | 106632 | 0.0 | `c3781f69a2f2` |
| FASTTEXT-MIXTO-20M | E-CORP-07 | 200 | 5 | 3 | 3-6 | 10 | 42 | 163228 | 0.0 | `18f8d10125c7` |

## 5. Presupuesto de v5.1

```text
BASELINE EXISTENTE
Wikipedia 10M / 200d / w5 / min3 / sub3-6 / neg10 / e5 / sg1 / seed42
-> EXISTING - LOAD ARTIFACT -> no consume presupuesto

NUEVOS (los nueve):
VS-100       -> 1/9
VS-300       -> 2/9
WS-3         -> 3/9
WS-7         -> 4/9
MC-2         -> 5/9
MC-5         -> 6/9
SW-3-5       -> 7/9
SW-3-7       -> 8/9
NEG-5        -> 9/9

RESERVA        -> 0/1 (solo evidencia clara; nunca para abrir una familia nueva)
```

- Entrenamientos nuevos consumidos: **0** de 9 exploratorios y 1 de reserva.
- Propuestas que colisionan con un entrenamiento existente: **0**.

La ausencia de colisiones es el resultado que importa: las nueve configuraciones
propuestas son de verdad nuevas y ninguna se puede resolver cargando un artefacto.

## 6. Tabla B: evaluaciones downstream de v5

33 evaluaciones sobre 9
entrenamientos distintos. Son `DOWNSTREAM EVALUATION UNIT`, no
`EMBEDDING TRAINING UNIT`: no consumen presupuesto y por eso la contabilidad las
separa en otra tabla.

## 7. Auditoria de reglas (seccion 36)

| comprobacion | resultado |
|---|---|
| transformers / BERT / RoBERTa / LLM | 0 usos |
| embeddings preentrenados / from_pretrained | 0 usos |
| test en el corpus de entrenamiento | 0 coincidencias |
| holdout | intacto: True (5/5) |
| uso de TOXIC declarado y coherente | True (0 inconsistencias) |
| referencia V03 | 0.777211 (coincide: True) |

La auditoria se delega en `experiments/embedding_06_auditoria_v5.py` y sus patrones
prohibidos se toman de alli. Si este script los escribiera literalmente, el analisis
estatico lo marcaria a el mismo como infractor.

## 8. Gaps de trazabilidad detectados

- FASTTEXT-WEB-10M: el artefacto no declara minn/maxn por modelo
- WORD2VEC-WEB-10M: el artefacto no declara minn/maxn por modelo
- FASTTEXT-MIXTO-20M: el artefacto no declara minn/maxn por modelo

## 9. Aserciones

| asercion | resultado |
|---|---|
| `nuevas_configuraciones_no_reexisten` | OK |
| `presupuesto_exploratorio_no_superado` | OK |
| `presupuesto_total_no_superado` | OK |

## 10. Nota de trazabilidad: ubicacion del proyecto

El worktree `proyecto-kaggle-toxic-2026-v5` estaba en `/home/carlos/Git/PLN-v5` y se
migro a `/home/carlos/Git/PLN` (la carpeta de trabajo de esta sesion) a peticion del
usuario. Se copiaron `data/`, `tools/`, `experiments/*.log`, `features/` y `models/`;
se verificaron los sha256 de los cinco corpus y de los diez `.vec`; y el control V03
reproduce **0.777211 exacto** desde la nueva ubicacion. El trabajo sin
commitear de KAGGLE_AGE_2026_PUJ quedo aparcado en `git stash` (no es un commit) y
sus datos no se tocaron.

