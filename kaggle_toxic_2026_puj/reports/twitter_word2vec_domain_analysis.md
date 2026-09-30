# Domain-specific Word2Vec sobre Twitter/X: auditoria de corpus

Generado por `experiments/p71_00_auditoria_corpus.py`. Fecha 2026-09-30T00:01:56+00:00.

## La hipotesis, que no se ha podido poner a prueba

> Un Word2Vec entrenado desde cero sobre texto real de Twitter/X puede capturar
> informacion distribucional especifica del dominio que los entrenados sobre Wikipedia y
> mC4 no capturaron.

La novedad no era `Word2Vec existe`. Era **entrenamiento desde cero + corpus Twitter/X +
dominio linguistico cercano al problema**.

```text
la pregunta SIGUE SIN RESPUESTA.
lo que se cierra es la posibilidad de responderla con los recursos disponibles,
no la hipotesis.
```

## La decision de P0

```text
CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY

ALLOWED 0    AMBIGUOUS 3    PROHIBITED 1

experimentos gastados: 0
```

**Ningun corpus pasa la auditoria.** P1, la comparacion Word2Vec Twitter frente a
Word2Vec general, no se puede ejecutar. No hay tratamiento contra el que comparar.

## Los cuatro candidatos

### `pysentimiento/spanish-tweets` - AMBIGUOUS

- **Provider**: pysentimiento (grupo de PLN, UBA / DC-UBA)
- **Origen**: coleccion Spritzer de Archive.org de mayo de 2019, filtrada a espanol, mas la descarga de las timelines completas de los 432.000 usuarios detectados
- **Periodo de recoleccion**: linea base: mayo 2019. Las timelines se descargaron despues, sin periodo acotado ni publicado por usuario
- **Tweets**: 597433111
- **Campos**: ['text', 'tweet_id', 'user_id']
- **Labels**: False
- **Licencia**: NO DECLARADA. La API de HuggingFace devuelve el campo de licencia vacio, y la ficha no menciona la palabra ni una vez pese a que su indice lista una seccion 'Licensing Information' que no existe
- **Se puede garantizar que el texto del test no esta dentro?**: `False`
- **Razon principal**: sin licencia declarada, y sin poder garantizar que el texto del test no este en el corpus sin compararlo contra el test, que esta prohibido

### `pysentimiento/spanish-tweets-small` - AMBIGUOUS

- **Provider**: pysentimiento
- **Origen**: misma procedencia que el corpus grande
- **Periodo de recoleccion**: identico al corpus grande
- **Tweets**: 10M < n < 100M (categoria declarada)
- **Campos**: ['text', 'tweet_id', 'user_id']
- **Labels**: False
- **Licencia**: NO DECLARADA, igual que el corpus grande
- **Se puede garantizar que el texto del test no esta dentro?**: `False`
- **Razon principal**: hereda los dos problemas del corpus grande

### `JorgeLoera/spanish-dialect-tweets` - AMBIGUOUS

- **Provider**: usuario individual
- **Origen**: NO DOCUMENTADA
- **Periodo de recoleccion**: NO DOCUMENTADO
- **Tweets**: 100K < n < 1M (categoria declarada)
- **Campos**: NO DOCUMENTADOS. El unico fichero es Datos_dialectos.parquet
- **Labels**: PROBABLEMENTE SI. El nombre del fichero, 'Datos_dialectos', apunta a clasificacion de dialecto, y un dataset externo etiquetado esta prohibido con independencia del tema
- **Licencia**: cc-by-4.0, la unica linea de la ficha
- **Se puede garantizar que el texto del test no esta dentro?**: `False`
- **Razon principal**: tiene licencia real, y es lo unico que tiene: sin ficha, sin procedencia, sin periodo y con indicios de estar etiquetado

### `santiagxf/spanish-marketing-tweets` - PROHIBITED

- **Provider**: usuario individual
- **Origen**: no aplica: no hay datos
- **Periodo de recoleccion**: no aplica
- **Tweets**: 0
- **Campos**: []
- **Labels**: no aplica
- **Licencia**: unlicense, la mas permisiva posible
- **Se puede garantizar que el texto del test no esta dentro?**: `no aplica`
- **Razon principal**: el repositorio contiene un README y un .gitattributes y NINGUN fichero de datos. No hay corpus que entrenar

## El corpus que el encargo senalaba, y sus dos problemas

`pysentimiento/spanish-tweets` es un corpus **bueno en principio**: 597 millones de
tweets, campos `text`/`tweet_id`/`user_id`, **sin labels**, y su finalidad declarada es
*language modelling* para preentrenar representaciones. Es exactamente el tipo de recurso
que la hipotesis pedia.

### Problema 1: no declara licencia

```text
API de HuggingFace   campo de licencia VACIO
ficha del dataset    0 menciones de la palabra 'license'
                      su indice SI lista 'Licensing Information'
                      y la seccion NO existe en el documento
```

### Problema 2: no se puede garantizar que el texto del test no este dentro

```text
recoleccion   coleccion Spritzer de Archive.org de mayo de 2019,
              mas la descarga de las TIMELINES COMPLETAS de 432.000 usuarios
              localizados a partir de ella
TOXIC         competencia de 2026, sin periodo de recoleccion documentado

las timelines cubren rangos de fecha arbitrarios por usuario, sin periodo
acotado ni publicado. La unica forma de comprobar el solapamiento seria
comparar contra el test, y el encargo PROHIBE usar ids, textos o hashes
del test para construir o limpiar el corpus
```

El encargo es explicito: *en caso de no poder garantizar razonablemente, clasificar el
corpus como `AMBIGUOUS` y no utilizarlo*.

**Cada problema es suficiente por si solo.** Se declaran los dos.

## Las alternativas que se descartaron, y por que

| via | por que no |
|---|---|
| usar el corpus sin licencia | incumple la regla del propio encargo |
| deduplicar contra el test | seria usar el test para construir el recurso |
| usar un dataset etiquetado | prohibido con independencia del tema |
| sustituir Word2Vec por FastText | cambiaria dos variables a la vez |
| usar los corpus generales de v5 | serian el CONTROL, no el tratamiento, y sin tratamiento no hay comparacion |

## Lo que no se hace

```text
no se descarga ningun corpus
no se entrena ningun embedding
no se toca el test, ni el holdout, ni el leaderboard
no se filtra ningun corpus externo contra el test
no se cambia Word2Vec por FastText: cambiaria dos variables a la vez
no se usan los corpus generales de v5 como tratamiento: serian el CONTROL
```

## Lo que queda registrado para una fase futura

El corpus principal y sus dos problemas quedan documentados con la evidencia de la
consulta, en `reports/twitter_word2vec_corpus_registry.csv`. Una fase futura que
consiguiera un corpus con licencia explicita y con un periodo de recoleccion anterior y
verificable no tendria que repetir esta auditoria desde cero.

## El estado del proyecto, que no cambia

```text
V03                          FROZEN
A_cost_sensitive_downweight  PROMISING_UNCONFIRMED
research_status              EXPERIMENTAL_RESEARCH_CLOSED

domain-specific Word2Vec     CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY
```

Cerrada esta linea, el estado de la investigacion experimental no cambia: sigue cerrada,
y el candidato sigue congelado.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p71_00_auditoria_corpus.py`. Sin push.*