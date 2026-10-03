# Submission experimental del candidato congelado

Generado por `experiments/p69_10_submission_candidato.py`. Fecha 2026-09-29T20:14:03+00:00.

## Motivo

Verificar la transferencia del candidato congelado al conjunto de test. Es una
**transferencia experimental**, no una adopcion cientifica. El test se usa para
`transform`, `predict` y generar el archivo, y para nada mas.

```text
candidate_status: PROMISING_UNCONFIRMED -> APPROVED_FOR_EXPERIMENTAL_SUBMISSION
```

## El candidato

**downweight basado en violacion de margen del propio V03.** No es un meta-modelo, no
es stacking, no es correccion residual y no es rescue weighting.

## La evidencia interna que respalda la submission

| prueba | resultado |
|---|---|
| Reproduccion | R3, `0.7854160387` identico a 3,2e-11 |
| Particion canonica | `+0,0082045862`, 5 de 5 |
| Segunda particion | `+0,0105253130`, 5 de 5 |
| Tercera particion corregida | `+0,0048825807`, 4 de 5 |
| Permutacion, 1000 iteraciones | `p = 0,000999` |
| Bootstrap por grupos | `[-0,0006906975, +0,0118175641]`, incluye el cero |

## El refit, con su protocolo pre-registrado

`38_refit_final_v4.py` dejo escrito como se reajusta un candidato de pliegues a un solo
modelo, y las dos partes importan:

```text
pesos de ejemplo   error_score calculado SOLO con OOF
pesos de clase     la MEDIANA de los multiplicadores que el barrido eligio por pliegue
```

Lo primero es el candidato entero, y necesita un OOF: en un ajuste unico sobre todo el
train no existe, asi que lo aporta el OOF canonico. Lo segundo es como se colapsan cinco
vectores en uno sin re-seleccionar nada.

Multiplicadores por pliegue -> mediana: `[1.2, 1.5, 1.2]`.

## Que filas se usan para entrenar

```text
train oficial completo       1731 filas
holdout congelado            248 filas   NO se usa
desarrollo para el refit     1483 filas
```

El encargo pide *train oficial completo* y, dos lineas despues, *no utilizar el holdout*.
El holdout es un subconjunto del train, asi que la instruccion se resuelve sola: se
entrena con el desarrollo, 1483 filas, exactamente lo que midio cada numero de arriba.

Merece decirse en voz alta que la submission de V03 **si** se ajusto sobre las 1731.
Una decision de gobernanza futura puede querer preguntarse si conviene igualarlo. Hacerlo
aqui cambiaria el candidato, y el candidato esta congelado.

## La configuracion congelada

| elemento | valor |
|---|---|
| word TF-IDF | `word(1,2)`, min_df=2, sublinear_tf, lowercase |
| char TF-IDF | `char_wb(3,5)`, min_df=2, sublinear_tf, lowercase |
| clasificador | `LogisticRegression(C=2.0, max_iter=1000, liblinear, seed 42)` |
| class_weight | explicito con multiplicador |
| umbrales | 0.44 / 0.51 / 0.46 |
| post-proceso | **ninguno**, igual que el pipeline congelado de V03 |

## Los checks

| check | valor |
|---|---|
| V03 sha256 congelado | `7d7478f271e6edfd51afff85bad3eaa0` |
| V03 corrected_cv | `0.7772114525165721` |
| candidato reproducido en la tolerancia | `0.785416038667608` |
| se entrena solo con el desarrollo | `1483` |
| el holdout no entra en el entrenamiento | `0` |
| test leido | `1154` |
| misma regla: la mediana es 0 y no se sube ningun peso | `mediana 0.0, >1: 0` |
| n de pesos < 1 coincide con el canonico | `441` |
| fraccion de peso retirada coincide | `0.13663` |
| los multiplicadores son la mediana pre-registrada | `[1.2, 1.5, 1.2]` |
| vectorizadores ajustados SOLO con el entrenamiento | `dim train 24646` |
| test transformado con los vectorizadores del train | `24646` |
| tres cabezas entrenadas, una por etiqueta | `3` |
| columnas exactas | `['id', 'HS', 'TR', 'AG']` |
| numero de filas = filas del test | `1154 vs 1154` |
| ids unicos | `1154` |
| mismo conjunto de ids que el test | `1154` |
| mismo orden de ids que el test | `coincide` |
| mismo orden de ids que la plantilla oficial | `coincide` |
| ningun id adicional | `0` |
| ningun id perdido | `0` |
| la columna de id es identica a la de V03 | `identica` |
| el id es entero como el de V03 | `int64` |
| valores binarios validos | `0/1` |
| sin NaN | `0` |
| sin columnas auxiliares | `4` |
| dtypes enteros | `int` |
| el vocabulario del test NO entra en el fitting | `3435 <= 24246` |
| los pesos se calculan solo con el train | `1483` |
| los umbrales son los congelados | `congelados` |
| el holdout no aparece en ninguna matriz | `1483` |
| el test solo tiene id y text | `['id', 'text']` |
| no existe ninguna etiqueta de test en el dataframe | `['id', 'text']` |
| no se lee el leaderboard | `no se consulta` |
| no se lee el holdout | `no se usa` |
| ninguna submission previa se lee como entrada | `no aplica` |
| la submission se ha escrito | `6fc7c171a8af08b9ac80c744baae2aceb65f2c1488e5eda15382b5914930898f` |
| 0 bytes > 127 en la submission | `0` |

**38 comprobaciones, 0 fallos.**

## Lo que NO se hace

```text
no se sube a Kaggle
no se consulta el leaderboard
no se leen etiquetas del test
no se usa el holdout
no se generan variantes: ni V03 alternativo, ni otro peso, ni otro umbral,
  ni otra semilla, ni un blend con otro modelo
no se calibra, ni se ajustan umbrales, ni se eligen features sobre el test
```

## Lo que no se afirma

```text
no se escribe una mejora esperada
no se escribe un score esperado
no se proyecta el leaderboard
no se usa el numero de filas cambiadas para inferir un score
no se afirma que la submission mejore a V03
```

La unica afirmacion permitida: **se genero una submission a partir de un candidato
internamente validado y congelado.**

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_10_submission_candidato.py`. Sin push.*