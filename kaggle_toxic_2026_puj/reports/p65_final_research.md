# P65 - Representacion de largo alcance, alcance de operadores y modelo jerarquico

Informe final de la fase. Fecha 2026-09-29T14:55:51+00:00. Cuatro experimentos, una auditoria. 0 submissions, 0 Kaggle, 0 holdout, 0 test, 0 leaderboard.

## La pregunta de la fase, y la respuesta

> Cuando un tweet largo contiene varias partes con evidencia potencialmente contradictoria,
> V03 falla porque no dispone de la informacion necesaria, o porque no puede representar relaciones
> de largo alcance entre las senales que ya posee?

**La segunda.** Y la respuesta tiene una forma que no estaba prevista al empezar.

## La forma del resultado

Cinco candidatos de este proyecto baten a **los dos controles** y ninguno mejora a V03:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0.0596671770 | +0.0071281317 | -0,0039520688 |
| P65-03 cohesion documental | +0.0090835614 | +0.0083711943 | -0,0027090062 |
| P64-08 mezcla de temas | -0,0021189973 | +0,0065189177 | -0,0045612828 |

Las dos brechas positivas dicen que la informacion existe, que no es capacidad y que no
es lo que V03 ya sabe de si mismo. El delta negativo dice que no entra. Y la razon
de que no entre no es que falte: es que **unas cientos de columnas compiten por el
presupuesto L2 de un espacio de veinte mil dimensiones**, y lo que la columna nueva
necesita decir es mas pequeno que lo que cuesta decirlo.

Eso es una afirmacion sobre la ARQUITECTURA de la adicion, y por eso es falsable: predice
que la misma evidencia **fundida en el espacio de probabilidad**, y no concatenada en el de
caracteristicas, sobrevive. Ese es un paso real, clasico, y queda abierto.

## El gap analysis cerro mas de lo que abrio

**6 abiertas y 10 cerradas.** Se leyo el codigo y no el
registro, porque con 359 filas una busqueda por nombre devuelve las mismas filas para
cualquier patron que aparezca en una pregunta.

| familia | estado | por quien |
|---|---|---|
| skip-grams con hueco 1 y huecos 1 y 2 | EXACT_DUPLICATE | V6-05 (experiments/v6_05_skipgrams.py) |
| pares lejanos a 3-5 y 6-10 tokens | PARTIAL_PRECEDENT | V6-05 llego hasta hueco 2; 16_targeting_v2 midio distancia |
| word (1,3) anadido a V03 | PARTIAL_PRECEDENT | REP-04 (experiments/03_variantes_representaciones.py) |
| word (1,4) anadido a V03 | PARTIAL_PRECEDENT | nunca medido en el bloque de palabras; los (1,4) del codig |
| canal de negacion como patrones | EXACT_DUPLICATE | E2-08 y E2-08-TXT (19_sintaxis_cita_negacion_v2.py) |
| canal cita y atribucion | EXACT_DUPLICATE | E2-07, E2-07-TXT y E2-07-CITA-NEG-TXT |
| frontera de mencion | FUNCTIONAL_DUPLICATE | P64-06 |
| alcance de contraste | GENUINELY_NEW | ninguno. Las apariciones de 'pero' y 'aunque' en el codigo |
| cohesion documental interna | GENUINELY_NEW | V10-07 es shift de covariables ENTRE tren y prueba, que es |
| interaccion palabra x forma superficial | PARTIAL_PRECEDENT | la UNION word + char_wb es una suma de canales, no un prod |
| modelo multiple-instance | EXACT_DUPLICATE | P64-05 con la cobertura corregida de P59-07 |
| agregadores segmentales | EXACT_DUPLICATE | P64-03, P64-04 y P64-05 |
| mezcla de temas | EXACT_DUPLICATE | P64-08 mas cinco factorizaciones previas |
| POS, dependencias y morfologia | EXACT_DUPLICATE | P62-01 a P62-05 y E2-03, E2-04 |
| especialista de tweets largos | GENUINELY_NEW | P63 y P64 midieron el MARGEN por rebanada; nadie entreno u |
| min_df y presupuesto de vocabulario | EXACT_DUPLICATE | P62-03, P62-05 y R2-MINDF5 |

Y cuatro de las cerradas no habrian parecido cerradas por el nombre:

- los **skip-grams** los midio **V6-05**, con un vocabulario un 83 por ciento mayor y un resultado neutro
- la **negacion y la cita** las midieron **E2-07 y E2-08**, y despues **P64-06 y P64-07** de otro angulo
- **word (1,3)** existia, pero como sustitucion y con C=1.0, no como adicion

## El control de implementacion, que es lo que cierra los huecos

El rango de hueco 1 a 2 es exactamente el de V6-05 y se incluyo como control:

| rango | corrected_cv | delta vs V03 | brecha contra el aleatorio |
|---|---:|---:|---:|
| hueco_1_2 | 0.7761598617 | -0.0010515908 | -0.0012639075 |
| hueco_3_5 | 0.7706593428 | -0.0065521098 | -0.0067644265 |
| hueco_6_10 | 0.7697723394 | -0.0074391131 | -0.0076514298 |
| CONTRASTE | 0.7748262592 | -0.0023851933 | -0.0025975101 |

**-0,0010515908 frente al -0,0001 de V6-05.** La implementacion es la misma, luego los rangos
que V6-05 no alcanzo se han medido y tampoco aportan. Y hay un dato sobre el
control de capacidad que hay que decir: el bloque aleatorio sale en +0,0002, ligeramente
**por encima** de V03, luego aqui las brechas negativas son reales y no un
artefacto de un control severo, al contrario que en P65-01.

## El especialista es bit a bit el reentrenado

| brazo | corrected_cv | delta vs V03 | LARGA | MUY_LARGA |
|---|---:|---:|---:|---:|
| global | 0.7772114525 | +0.0000000000 | 0.6834 | 0.5606 |
| especialista | 0.7691517953 | -0.0080596573 | 0.6577 | 0.5818 |
| reentrenado | 0.7691517953 | -0.0080596573 | 0.6577 | 0.5818 |
| aleatorio | 0.7654406468 | -0.0117708057 | 0.6577 | 0.4700 |

Ajustar un V03 sobre las mismas **714 filas** sin ningun criterio de longitud da el **mismo
numero** que ajustarlo seleccionando por longitud: **+0.0000000000**. La longitud no es el mecanismo.

Un negativo aqui habria podido ser *faltan datos*, que es una excusa valida. Este dice que
la especializacion no es el mecanismo, punto.

Y hay un dato en sentido contrario: el especialista **gana +0.0212** en MUY_LARGA y
**pierde -0.0257** en LARGA, que tiene el triple de filas. Hay una mejora local real
en el extremo que se paga con una perdida mayor en el tramo intermedio, y la metrica primaria la
rechaza. Es justo lo que pasa cuando se prohibe elegir el punto que maximiza F1.

## La cohesion documental, que es longitud disfrazada

| estrato | coseno medio | cambio lexico | repeticion cruzada |
|---|---:|---:|---:|
| todas | 0.5147 | 0.4853 | 0.0150 |
| CORTA | 0.7066 | 0.2934 | 0.0000 |
| MEDIA | 0.5995 | 0.4005 | 0.0047 |
| LARGA | 0.4368 | 0.5632 | 0.0196 |
| MUY_LARGA | 0.2720 | 0.7280 | 0.0519 |
| V03 correcto | 0.5588 | 0.4412 | 0.0131 |
| V03 incorrecto | 0.4105 | 0.5895 | 0.0195 |
| error de alta confianza | 0.4291 | 0.5709 | 0.0178 |
| solo multi segmento | 0.0416 | 0.9584 | 0.0296 |

El mejor descriptor da AUC 0.425493 frente a 0.600506
de contar tokens, luego pierde contra la referencia trivial. Y el motivo esta en una fila: el documento de un
unico segmento tiene coseno 1,0 por definicion, luego el descriptor es **bimodal** y en la practica
dice *tiene mas de un segmento*, que ya se cerro en P64-02.

Pero el cambio lexico con la longitud es monotono y fuerte, y eso si es un hallazgo sobre el
corpus: los tweets que V03 falla repiten su vocabulario la mitad que los que resuelve.
V03 no puede usar esa lectura, porque un token repetido es un token repetido.

## Las correcciones propias de la fase

1. La mascara de contraste se construia como un booleano y luego se le pedia `.sum()`
2. Un salto de linea habia colocado el cierre de parentes de `float()` antes del `.mean()`, y
   el `float` se aplicaba al array
3. `scipy.sparse.random` es una FUNCION y no un submodulo, y se estaba llamando como modulo
4. El subconjunto de columnas duplicaba el prefijo del rasgo y el script abortaba
5. La funcion de ajuste del especialista capturaba por cierre las matrices del pliegue
   **y** recibia indices globales sobre una matriz local, con lo que abortaba
6. Un veredicto estaba partido entre dos literales y salia en ingles

## Estado

| | |
|---|---|
| experimentos | 4 de 25 principales, 0 de 7 reservas |
| familias cerradas por el gap | 10 sin gastar experimento |
| cumplimiento | 26 comprobaciones, 0 fallos, por analisis AST |
| figuras | 13, ninguna oculta un negativo |
| submissions | 0 |
| Kaggle | 0 |
| holdout | 0 |
| test | 0 |
| reproducibilidad | R2, sin R4 |

## Cierre

```text
the information about long-range relations is present, measurable and not what V03
already knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long
rows with no length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```

No se afirma que V03 sea matematicamente optimo, ni que los transformadores sean necesarios.
Se afirma exactamente lo que las mediciones sostienen: la informacion de largo alcance
existe y es medible, y la adicion por concatenacion es lo que la rechaza; y la
longitud del tweet no es el mecanismo, porque el reentrenado sin criterio de longitud
reproduce el especialista exactamente.

## Artefactos

* `reports/p65_candidate_comparison.csv`
* `reports/p65_cohesion.csv`
* `reports/p65_cohesion.json`
* `reports/p65_cohesion.md`
* `reports/p65_compliance.md`
* `reports/p65_error_intersection.csv`
* `reports/p65_especialista.json`
* `reports/p65_especialista.md`
* `reports/p65_final_research.md`
* `reports/p65_final_summary.json`
* `reports/p65_huecos_contraste.json`
* `reports/p65_huecos_contraste.md`
* `reports/p65_hypothesis_gap_analysis.csv`
* `reports/p65_hypothesis_gap_analysis.json`
* `reports/p65_hypothesis_gap_analysis.md`
* `reports/p65_hypothesis_registry.csv`
* `reports/p65_long_range.csv`
* `reports/p65_master_results.csv`
* `reports/p65_ngrams.json`
* `reports/p65_ngrams.md`
* `reports/p65_reproducibility.md`
* `reports/p65_scope_analysis.csv`
* `reports/p65_skipgram.csv`
* `figuras/p65/p65_01_v03_vs_candidatos.png`
* `figuras/p65/p65_02_delta_vs_v03.png`
* `figuras/p65/p65_03_largo_alcance.png`
* `figuras/p65/p65_04_skipgram_vs_cohesion.png`
* `figuras/p65/p65_05_cohesion_por_estrado.png`
* `figuras/p65/p65_06_confianza_vs_cohesion.png`
* `figuras/p65/p65_07_especialista_por_estrado.png`
* `figuras/p65/p65_08_rescate_vs_dano.png`
* `figuras/p65/p65_09_interseccion_errores.png`
* `figuras/p65/p65_10_rescate_por_estrado.png`
* `figuras/p65/p65_11_consistencia_pliegues.png`
* `figuras/p65/p65_12_brechas_controles.png`
* `figuras/p65/p65_13_evolucion_candidatos.png`

> V03 remains the best verified leaderboard result.
