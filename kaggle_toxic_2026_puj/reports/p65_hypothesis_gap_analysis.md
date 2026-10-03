# P65 - Analisis de hipotesis: que queda genuinamente abierto

Generado por `experiments/p65_00_audit_gap.py`. Fecha 2026-09-29T14:33:40+00:00.

## La auditoria del baseline

```text
V03 corrected_cv  0.7772114525165721
V03 sha256        7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
registro          359 filas, 359 identificadores
submissions       0 | Kaggle 0 | holdout 0 | test 0 | leaderboard 0
```

## Por que se lee el codigo y no el registro

Con 359 filas, una busqueda por nombre devuelve las mismas filas para cualquier patron
que aparezca dentro de una pregunta. La clasificacion tiene que ser **funcional**: que
calcula la representacion, no como se llama. Por eso la tabla siguiente cita el
experimento y el fichero, y no un tema.

## Los cuatro resultados que cambian el plan

### 1. Los skip-grams YA se midieron

**V6-05** cambio el analizador del bloque de palabras para que los pares con hueco
compitan con las palabras por el mismo presupuesto L2, probo hueco 1 y huecos 1 y 2, y
consiguio **-0,0001 con IC95 que incluye el 0** sobre un vocabulario un 83 por ciento
mayor. El vocabulario crecio y no paso nada. Es la misma mecanica que esta fase propone
y el resultado es **NEUTRAL**, no solo negativo.

Lo que V6-05 **no** alcanzo fueron los huecos grandes, y ahi si hay una pregunta nueva.

### 2. word (1,3) existe, pero no en la forma que hace falta

**REP-04** midio TF-IDF word (1,3) **suelto**, con **C=1.0** y **sin bloque de
caracteres**, y lo registro como descartado con macro_f1 0,8062. Eso **sustituye** la
representacion y cambia el regularizador. La pregunta de esta fase es distinta: anadir
los trigramas a V03 manteniendo char (3,5), C=2.0, umbrales y pliegues, y medir el delta
sobre el baseline congelado. Es **precedente parcial**, no duplicado exacto.

### 3. La negacion y la cita ya se midieron dos veces

**E2-08** construyo los patrones NEG_* y dio -0,2505 solo y -0,0115 sobre V03. **E2-07**
construyo los canales dentro y fuera de comillas y dio -0,0335, -0,0218 y -0,0220.
Despues **P64-06 y P64-07** los midieron de otro angulo, con el alcance por distancia
y con el efecto sobre la evidencia, y tambien cerraron. Y P64-06 midio ademas que el
corpus tiene 39 filas con comillas y **solo una** con interior no vacio.

### 4. La cohesion es nueva, y hubo que descartar el casi-acierto

**V10-07** es shift de covariables **entre** entrenamiento y prueba, que es otro objeto
que la cohesion **dentro** del documento. Y **16_targeting_v2** mide distancias entre un
target y un insulto con un lexico inducido, que no es un par lexico sparse de anchura
desconocida. Los dos se descartan como antecedente y la cohesion queda abierta.

## La tabla completa

| familia | estado | antecedente | resultado | experimento |
|---|---|---|---|---:|
| skip-grams con hueco 1 y huecos 1 y 2 | EXACT_DUPLICATE | V6-05 (experiments/v6_05_skipgrams.py) | -0,0001 con IC95 que incluye el 0, vocabulario +83 por ciento | 0 |
| pares lejanos a 3-5 y 6-10 tokens | PARTIAL_PRECEDENT | V6-05 llego hasta hueco 2; 16_targeting_v2 midio distancia | V6-05 neutral en huecos cortos y 16 Targeting fue un problema distinto | 1 |
| word (1,3) anadido a V03 | PARTIAL_PRECEDENT | REP-04 (experiments/03_variantes_representaciones.py) | REP-04 midio word (1,3) SUELTO con C=1.0 y sin bloque char, macro_f1... | 1 |
| word (1,4) anadido a V03 | PARTIAL_PRECEDENT | nunca medido en el bloque de palabras; los (1,4) del codig | no aplica | 1 |
| canal de negacion como patrones | EXACT_DUPLICATE | E2-08 y E2-08-TXT (19_sintaxis_cita_negacion_v2.py) | -0,2505 sola y -0,0115 sobre V03 | 0 |
| canal cita y atribucion | EXACT_DUPLICATE | E2-07, E2-07-TXT y E2-07-CITA-NEG-TXT | -0,0335, -0,0218 y -0,0220 | 0 |
| frontera de mencion | FUNCTIONAL_DUPLICATE | P64-06 | -0,0187 con brecha -0,0092 contra el aleatorio | 0 |
| alcance de contraste | GENUINELY_NEW | ninguno. Las apariciones de 'pero' y 'aunque' en el codigo | no aplica | 1 |
| cohesion documental interna | GENUINELY_NEW | V10-07 es shift de covariables ENTRE tren y prueba, que es | no aplica | 1 |
| interaccion palabra x forma superficial | PARTIAL_PRECEDENT | la UNION word + char_wb es una suma de canales, no un prod | no aplica | 0 |
| modelo multiple-instance | EXACT_DUPLICATE | P64-05 con la cobertura corregida de P59-07 | -0,0549 con neto -400 | 0 |
| agregadores segmentales | EXACT_DUPLICATE | P64-03, P64-04 y P64-05 | brechas contra el margen negativas o casi nulas | 0 |
| mezcla de temas | EXACT_DUPLICATE | P64-08 mas cinco factorizaciones previas | -0,0046 con brecha -0,0021 contra el control de anchura | 0 |
| POS, dependencias y morfologia | EXACT_DUPLICATE | P62-01 a P62-05 y E2-03, E2-04 | cerrados en ambas fases | 0 |
| especialista de tweets largos | GENUINELY_NEW | P63 y P64 midieron el MARGEN por rebanada; nadie entreno u | no aplica | 1 |
| min_df y presupuesto de vocabulario | EXACT_DUPLICATE | P62-03, P62-05 y R2-MINDF5 | cerrados | 0 |

**6 abiertas con experimento pendiente, 10 cerradas por equivalencia medida.**

Y una observacion sobre el reparto del presupuesto. El encargo da 25 experimentos
principales, y la clasificacion deja **cinco** con una hypothesis genuina. Eso no es una
excusa para inventar familias: es el resultado del trabajo, y la honestidad exige que
el informe diga que la mayor parte del mecanismo de largo alcance ya estaba medido.

## Lo que queda abierto, con su formulacion

- **pares lejanos a 3-5 y 6-10 tokens** (PARTIAL_PRECEDENT): huecos de 3 a 5 y de 6 a 10 tokens, en canal sparse con presupuesto L2
- **word (1,3) anadido a V03** (PARTIAL_PRECEDENT): la adicion, que es la unica forma en que la pregunta esta formulada
- **word (1,4) anadido a V03** (PARTIAL_PRECEDENT): orden 4
- **alcance de contraste** (GENUINELY_NEW): pares de marcador con la palabra alcanzada, en las dos direcciones
- **cohesion documental interna** (GENUINELY_NEW): coseno entre segmentos, solapamiento de vocabulario, repeticion cruzada y cambio lexico, primero como diagnostico
- **especialista de tweets largos** (GENUINELY_NEW): V03 global mas un especialista entrenado en LARGA y MUY_LARGA, con puerta determinista y tres controles

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p65_00_audit_gap.py`. Sin push.*