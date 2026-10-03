# P0 - Auditoria de la implementacion Twitter Word2Vec

Generado por `experiments/p72_00_auditoria_implementacion.py`. Fecha 2026-09-30T04:42:37+00:00.

## El hallazgo que condiciona todo lo demas

**El Word2Vec de 200 dimensiones que existe en disco no es el que produjo el resultado historico.**

| artefacto | script | loss | valido |
|---|---|---|---|
| `w2v_twitter.npz` | p71_04 | `[1.706327, 1.81141, 1.841703, 1.865799, 1.88408]` | **False** |
| `w2v_twitter_sgd.npz` | p71_06 | `None` | **False** |
| `w2v_twitter_lrfix.npz` | p71_08 | `[1.272136, 1.174189, 1.132058, 1.108284, 1.095145]` | **True** |
| `w2v_general_lrfix.npz` | p71_08 | `[1.25416, 1.140164, 1.094067, 1.069379, 1.055625]` | **True** |

Los dos primeros quedaron con la loss **clavada en 1,386295**, el valor del estado de productos punto nulos: **no entrenaron**. Los numeros historicos de dominio se calcularon sobre ellos y estan marcados `INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS`.

**P1 arranca del modelo de 200 dimensiones de `p71_08`, que es el unico valido.**

## El corpus

| campo | valor |
|---|---|
| fuente | mmaguero/covid19-spanishtweets-earlylateapril2020 |
| url | https://www.kaggle.com/datasets/mmaguero/covid19-spanishtweets-earlylateapril2020 |
| proveedor | mmaguero |
| licencia | CC BY-SA 4.0, licencia real y explicita |
| periodo_de_coleccion | 29 de marzo de 2020 a 1 de mayo de 2020, CERRADO y documentado |
| idioma | espanol declarado por el proveedor |
| tiene_etiquetas | False |
| clasificacion_en_la_auditoria | ALLOWED |
| documentos_usados | 361994 |
| tokens_usados | 10000029 |
| sha256_del_corpus_tokenizado | 1ef872c189d0f405bc1a3f8281cafe08e6b099958e6b8a3816db1460df91f8b9 |
| fecha_de_descarga | 2026-09-29 |
| solapamiento_con_train | 0 exactas y 0 normalizadas, medido contra el TEXTO de train |
| solapamiento_con_test | NO se midio: el encargo prohibe comparar contra el test |

## La tokenizacion

| elemento | tratamiento |
|---|---|
| version | EMB-NORMAL-TW v1 |
| url | a marcador 'url' |
| mencion | a marcador '@usuario' conservando el prefijo |
| hashtag | CONSERVADO con su contenido; v5 lo colapsaba a 'hashtag' |
| numero | a marcador 'numero' |
| alargamientos | NO se colapsan |
| patron_token | [#@]?\w+(?:['\u2019]\w+)* |

## La configuracion

| parametro | valor |
|---|---|
| vector_size | 200 |
| window | 5 |
| min_count | 3 |
| negative | 10 |
| sample | 0.0001 |
| epochs | 5 |
| sg | 1 |
| seed | 20261016 |
| lote | 8192 |
| lr | 0.025 |
| optimizador | 0,025 x lote |
| lr_inicial | 6.4 |

Implementacion: **skip-gram con negative sampling, implementado desde cero en PyTorch**. No es gensim: no compila en Python 3.14.

## Las confirmaciones que pide el encargo

| confirmacion | ok | nota |
|---|---|---|
| entrenamiento desde cero | **True** | el skip-gram esta implementado en el propio script |
| sin pesos preentrenados de terceros | **True** | no hay carga de vectores externos en el cuerpo del script |
| sin labels externas | **True** | el corpus es texto no etiquetado; no hay fichero de etiquetas |
| el corpus no incorpora labels de toxicidad | **True** | la ficha del corpus declara que no hay etiquetas; el JSON de auditoria menciona las categorias prohibidas solo para auditarlas |
| sin texto de toxic_test usado a proposito | **True** | el corpus se descargo de Kaggle antes de existir, y el solapamiento se midio contra TRAIN |
| sin metadata de Twitter como feature | **True** | solo la columna `text` entra en el entrenamiento |

## El downstream, con su estado

- El primer intento quedo `INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS`.
- El segundo **reproduce V03 exactamente**: `True`.
- Mejor punto de fusion: `{'escala': 0.0, 'twitter': 0.7772114525, 'general': 0.7772114525, 'azar': 0.7772114525}`.

Defectos corregidos en el downstream, que eran mios:

- el vectorizador se ajustaba sobre todo el desarrollo en vez de dentro de cada pliegue
- C=1.0 sin class_weight, cuando V03 usa C=2.0 balanced
- la escala del bloque denso no estaba igualada a la del disperso

> V03 remains the best verified leaderboard result.

*Sin push.*