# Domain-specific Word2Vec sobre Twitter/X: auditoria de corpus

Generado por `experiments/p71_00_auditoria_corpus.py` y `experiments/p71_01_auditoria_ampliada.py`. Fecha 2026-09-30T00:11:15+00:00.

## La hipotesis, que no se ha podido poner a prueba

> Un Word2Vec entrenado desde cero sobre texto real de Twitter/X puede capturar
> informacion distribucional especifica del dominio que los entrenados sobre Wikipedia y
> mC4 no capturaron.

```text
la pregunta SIGUE SIN RESPUESTA.
```

## La decision

```text
CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY

9 candidatos auditados en 3 plataformas
ALLOWED 0 | AMBIGUOUS 6 | PROHIBITED 2 | NOT_APPLICABLE 1
experimentos gastados: 0
```

## Los candidatos, por plataforma

| plataforma | candidato | licencia | periodo | etiquetado | clase |
|---|---|---|---|---|---|
| HuggingFace | `pysentimiento/spanish-tweets` | NO DECLARADA. La API de Huggin | linea base: mayo 2019. Las | no | **AMBIGUOUS** |
| HuggingFace | `pysentimiento/spanish-tweets-small` | NO DECLARADA, igual que el cor | identico al corpus grande | no | **AMBIGUOUS** |
| HuggingFace | `JorgeLoera/spanish-dialect-tweets` | cc-by-4.0, la unica linea de l | NO DOCUMENTADO | n/d | **AMBIGUOUS** |
| HuggingFace | `santiagxf/spanish-marketing-tweets` | unlicense, la mas permisiva po | no aplica | n/d | **PROHIBITED** |
| HuggingFace | `soncco/historical-twitter-trends-spanish-loc` | ODC-By 1.0, licencia real y ex | 15 de julio de 2020 a 22 d | no | **NOT_APPLICABLE** |
| Kaggle | `mmaguero/covid19-spanishtweets-earlylateapri` | CC BY-SA 4.0, licencia real y  | 29 de marzo de 2020 a 1 de | no | **AMBIGUOUS** |
| Kaggle | `leadominguez/uruguayan-media-historical-twee` | CC0 1.0, dominio publico, la m | no documentado en la ficha | no | **AMBIGUOUS** |
| Kaggle | `hectorfr1984/spanish-trans-law-twitter-datas` | CC0 1.0 | el debate es de 2022-2023, | no | **AMBIGUOUS** |
| GitHub | `sinai-uja/COST` | CC BY-NC-SA 4.0, mas solicitud | publicacion de 2015, tweet | SI | **PROHIBITED** |

## El bloqueo ha cambiado de naturaleza, y eso es el resultado de la busqueda

En el primer P0 los candidatos fallaban por motivos **del corpus**: sin licencia
declarada, o con recoleccion abierta y sin periodo acotado.

La busqueda ampliada encontro algo distinto:

### `mmaguero/covid19-spanishtweets-earlylateapril2020`

```text
licencia    CC BY-SA 4.0, real y explicita
periodo     29 de marzo a 1 de mayo de 2020, CERRADO y documentado
contenido   2,27 millones de tweets espanoles, con columna `text`
etiquetas   ninguna de toxicidad
```

**Es el candidato mas cercano a utilizable que ha aparecido.** Licencia y periodo son
ambos correctos, y es el unico.

### Y el hueco que queda es del otro lado

```text
el periodo de recoleccion de TOXIC NO ESTA DOCUMENTADO EN NINGUNA PARTE
```

Se comprobo en `docs/competition_rules.md`, el texto verbatim del organizador: las reglas
hablan del formato de los ficheros y de que el dataset de prueba no se use para
entrenamiento, y **no mencionan el periodo de recoleccion de los datos ni los corpus
externos**.

Sin ese dato, la no coincidencia con el test no se puede demostrar, y la regla del
encargo es conservadora.

## Los tres hallazgos que importan

### Uno: un corpus legal y bien documentado que no sirve

`soncco/historical-twitter-trends-spanish-locations-2020-2021` tiene ODC-By, periodo
cerrado y ninguna etiqueta. Y su propio card lo dice sin ambiguedad:

> *It does not contain tweets, user accounts, news articles, URLs, or veracity labels.*

Son cadenas de tendencias. No hay texto de tweets sobre el que aprender distribuciones.
Es `NOT_APPLICABLE`: no lo prohibe la regla, simplemente no es el recurso.

### Dos: un corpus con todo correcto salvo el dato que falta

El de Kaggle, descrito arriba.

### Tres: un corpus prohibido con licencia impecable

`sinai-uja/COST` esta bajo CC BY-NC-SA 4.0 y contiene **34.634 tweets etiquetados de
sentimiento**, 17.317 positivos y 17.317 negativos. La regla del encargo prohibe los
corpus cuya finalidad era un dataset etiquetado de sentimiento, **con independencia de su
licencia**.

## Lo que desbloquearia la linea

```text
que el organizador documente el periodo de recoleccion de TOXIC
o una regla explicita sobre el uso de corpus externos anteriores
o un corpus cuya recoleccion fuese demostrablemente anterior al inicio de la
  competicion por una fuente independiente
```

**Eso es mas util que cerrar sin saber por que**, y es lo que la busqueda ampliada ha
anadido: no solo que no hay corpus, sino exactamente que haria falta.

## Lo que no se hace

```text
no se descarga ningun corpus
no se entrena ningun embedding
no se toca el test, ni el holdout, ni el leaderboard
no se filtran corpus externos contra el test
no se usa el corpus de Kaggle aunque su licencia sea correcta
```

## El estado del proyecto, que no cambia

```text
V03                          FROZEN
A_cost_sensitive_downweight  PROMISING_UNCONFIRMED
research_status              EXPERIMENTAL_RESEARCH_CLOSED
domain-specific Word2Vec     CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY
```

> V03 remains the best verified leaderboard result.

*Sin push.*