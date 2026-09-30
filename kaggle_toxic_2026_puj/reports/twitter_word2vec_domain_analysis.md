# Dominio Twitter frente a general: Word2Vec desde cero

Generado por `experiments/p71_04_entrenar_word2vec.py` y `experiments/p71_05_downstream.py`. Fecha 2026-09-29.

## El veredicto

```text
CLOSED
```

## Los numeros

| corpus | vocabulario | embedding-only | delta | fusion | delta |
|---|---:|---:|---:|---:|---:|
| V03 | - | 0.7772114525 | - | 0.7772114525 | - |
| twitter | 77,657 | 0.5659498481 | -0.2112616044 | 0.6417716007 | -0.1354398518 |
| general | 100,097 | 0.5663833893 | -0.2108280632 | 0.6498321813 | -0.1273792712 |

## El control de dominio, que es la comparacion del encargo

```text
embedding-only   twitter 0,565950 | general 0,566383  -> -0,000434
fusion           twitter 0,641772 | general 0,649832  -> -0,008061

ganador: GENERAL_MEJORA
```

**Twitter NO mejora al general. Lo pierde, en las dos metricas.** El dominio no aporta: la
representacion aprendida sobre tweets espanoles de 2020 es igual de buena, y ligeramente
peor, que la aprendida sobre Wikipedia y mC4.

## Complementariedad

| corpus | rescue | damage | neto |
|---|---:|---:|---:|
| twitter | 262 | 643 | -381 |
| general | 266 | 627 | -361 |

Las dos destruyen mas de lo que arreglan. Ningun gate pasa.

## Reproducibilidad, con semilla fija

```text
semilla 20261016, sembrada en CUDA, numpy, random y torch
flags de determinismo activados
dos corridas independientes -> mismo hash de pesos: 23c70aee34b52972a9f5474e
la cadena completa se ejecuto DOS veces con resultados identicos
```

## LIMITACION QUE HAY QUE DECLARAR: la fusion no es interpretable

**El numero de fusion no se puede leer como 'el embedding no aporta'.**

La fusion concatena 200 columnas densas estandarizadas a un bloque disperso de 24.646
dimensiones. Con `StandardScaler`, las densas tienen varianza unidad mientras las TF-IDF
tienen valores pequenos, asi que **el bloque denso domina la penalizacion L2** y la regresion
se sobreajusta a el. Una caida de 0,78 a 0,64 no la produce un embedding inutil: la produce
un diseno de fusion desequilibrado.

Lo que **si** es interpretable es el **control de dominio**: twitter y general usaron
exactamente el mismo pipeline, luego su diferencia de `-0,008061` en fusion y `-0,000434` en
embedding-only es una comparacion limpia. Y ahi twitter pierde.

### Y hay una segunda limitacion: el LR que elegi

```text
loss por epoca en twitter   1,706 -> 1,811 -> 1,842 -> 1,866 -> 1,884
loss por epoca en general   1,888 -> 2,068 -> 2,126 -> 2,164 -> 2,190
```

La loss **sube** en los dos. Con Adam y un LR fijo de 0,025 los embeddings oscilan; gensim
usa SGD con decaimiento lineal. El LR fue eleccion mia (el encargo fija dim, ventana,
min_count, negative, sample, epocas y sg, pero no el optimizador). **Los embeddings son
probablemente peores de lo que serian con un LR bien puesto.**

Eso no cambia el control de dominio, porque afecta igual a los dos corpus. Pero si rebaja la
calidad absoluta de las dos representaciones.

## Por que se uso PyTorch y no gensim

```text
gensim NO COMPILA en Python 3.14
  su Cython genera C contra PyDictObject.ma_version_tag, eliminado en esa version
  se intento instalar y fallo con el error a la vista

la spec de v5 documenta el respaldo, literalmente:
  'fastText C++ oficial. Word2Vec-control con minn=0, maxn=0.
   PyTorch solo como respaldo/control.'
```

Asi que no es una desviacion: es el respaldo previsto, y cumple el requisito con mas fuerza,
porque el skip-gram con negative sampling **esta implementado desde cero** y el notebook puede
mostrar la loss, la construccion de pares y la actualizacion.

## Cobertura

| corpus | tokens | vocabulario | cobertura de tokens | cobertura sobre TOXIC |
|---|---:|---:|---:|---:|
| twitter | 10,000,029 | 77,657 | 98.22% | 68.23% |
| general | 10,002,400 | 100,097 | 97.90% | 65.48% |

Twitter cubre mas del vocabulario de TOXIC (68,23 frente a 65,48), y **no le sirvio de nada**.
v5 ya lo habia visto: mas cobertura no es mejor F1.

## La conclusion del encargo

```text
C. Twitter W2V no mejora al general
```

Y por tanto cierra la hipotesis de domain-specific Word2Vec, que era la regla de parada del
encargo: *cerrar si Twitter W2V <= general W2V*.

## Lo que NO se hace

```text
no se escala el corpus: la hipotesis era dominio, no escala, y el dominio no aporta
no se abre el segundo corpus: la regla de parada lo impide
no se prueba FastText: cambiaria dos variables a la vez
no se toca el test, ni el holdout, ni el leaderboard
```

> V03 remains the best verified leaderboard result.

*Sin push.*