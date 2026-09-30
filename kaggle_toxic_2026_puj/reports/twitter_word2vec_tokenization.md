# Tokenizacion de la fase Twitter domain Word2Vec

Generado por `experiments/p71_03_preparar_corpus.py`. Fecha 2026-09-30T00:31:26+00:00.

## La regla, declarada

| elemento | tratamiento |
|---|---|
| mayusculas | a minusculas conservando tildes |
| URLs | a marcador `url` |
| menciones | a marcador `@usuario`, conservando el prefijo |
| **hashtags** | **CONSERVADOS con su contenido**, `#covid` como token |
| numeros | a marcador `numero` |
| alargamientos | **NO se colapsan** |
| puntuacion | se descarta |
| emojis | se descartan como tokens; sobreviven si forman parte de una palabra |
| patron de token | `[#@]?\w+(?:['\u2019]\w+)*` |

## La diferencia con v5, que hay que declarar

v5 usaba **EMB-NORMAL**, que colapsa hashtags, menciones, URLs y numeros a marcadores
estables:

```text
v5     #covid    -> 'hashtag'
       @usuario  -> 'usuario'
       https://  -> 'url'

aqui   #covid    -> '#covid'      CONSERVADO
       @nombre   -> '@usuario'    marcador con prefijo
       https://  -> 'url'
```

### Por que se conserva el hashtag

Porque ahi vive parte de la senal de dominio de Twitter. Un marcador sin contenido tira
exactamente la informacion que la hipotesis quiere capturar.

### Y por que se aplica el MISMO tokenizador a los dos corpus

Si se cambiara el tokenizador solo para Twitter se moverian **dos variables a la vez**, el
corpus y el tokenizador, y la comparacion dejaria de aislar el dominio. El encargo lo pide
explicitamente: *la comparacion debe aislar CORPUS DOMAIN y no ARQUITECTURA*.

## Los dos corpus

| | Twitter | general (v5) |
|---|---:|---:|
| documentos | 361,994 | 2,847 |
| tokens | 10,000,029 | 10,002,400 |
| sha256 | `1ef872c189d0f405bc1a3f8281cafe08` | `00b839da6fbc377ee5375acc9a236e83` |

Diferencia de tokens entre ambos: **0.02 por ciento**.

## El solapamiento con TOXIC, comprobado antes de usar el corpus

```text
filas recorridas del corpus de Twitter        2.277.953
coincidencias EXACTAS con el texto de train            0
coincidencias NORMALIZADAS con train                   0
rango temporal observado                      2020-03 y 2020-04
```

La comprobacion se hizo contra **train**, que no esta prohibido. No se comparo contra el
test, porque el encargo lo prohibe.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p71_03_preparar_corpus.py`. Sin push.*