# P64 - Analisis de hypotheses: que es genuinamente nuevo en segmentar

Generado por `experiments/p64_01_gap_analysis.py`. Fecha 2026-09-29T07:17:30+00:00.

## El precedente que hay que leer antes que nada

**P59-07 es el antecedente mas cercano de todo el repositorio**, y el encargo lo nombra
explicitamente. Cambio la UNIDAD DE REPRESENTACION del documento a una ventana local y
consiguio:

```text
delta -0,0006246152, 3 de 5 pliegues, neto -1, CLOSED
ventana fija de 32 tokens, avance 16, agregacion por maximo
```

Despues P59-09 fusiono ese score local con V03 mediante un alpha y dio +0,0014800075 con
2 de 5, MARGINAL, y P59-02 uso la geometria local como bloque y quedo CLOSED.

Asi que la familia **NO esta intacta**, y fingir lo contrario seria la forma mas facil de
tirar esta fase. La pregunta es mas estrecha: hay un mecanismo que P59 no podia alcanzar,
y hay un estrato donde puede actuar?

## La medicion que decide si la fase existe

P59-07 registro por que fallo, y la razon fue **COBERTURA y no concepto**. Los tweets son
cortos, mediana de 15 tokens, luego una ventana de 32 tokens
deja mas de una ventana en una fraccion pequena de los casos. Eso es una afirmacion sobre
el TAMANO de la ventana, no sobre la idea de evidencia local.

Una segmentacion guiada por puntuacion es **otro mecanismo**, y en este corpus alcanza
otro estrato:

| criterio | filas con mas de una unidad | porcentaje |
|---|---:|---:|
| ventana fija 32 tokens, avance 16 | 142 | 9.6 por ciento |
| segmentacion por puntuacion fuerte | 751 | 50.6 por ciento |

**Factor de cobertura: 5.29x.** Un factor de esa magnitud en
las celdas alcanzables no es una variacion de parametro: es la diferencia entre un
mecanismo que puede actuar en una de cada diez filas y otro que puede actuar en una de
cada dos. El encargo lo anticipa: exige que cualquier trabajo con ventanas se justifique
por un mecanismo genuinamente diferente y no por un numero distinto.

## Lo que es genuinamente nuevo

Ninguna representacion historica calcula esto:

- **descomposicion de la evidencia**: cada segmento puntuado con los coeficientes de
  V03 aprendidos en el pliegue de entrenamiento, que es descomponer un modelo que ya
  existe y no entrenar un clasificador nuevo.
- **distribucion de la evidencia**: concentracion, conflicto, desacuerdo y entropia de
  las puntuaciones por segmento, que es una propiedad de la DISPERSION y no del maximo.
- **leave-one-segment-out**: la contribucion de cada segmento al score completo, que
  P59 no computo nunca.
- **foco en el estrato estructural**: el margen residual esta en LARGA y MUY_LARGA, y
  ahi es donde viven las filas multi-segmento.

## Lo que NO es nuevo, y queda prohibido sin evidencia nueva

| familia | quien la cerro | con que numero |
|---|---|---|
| ventana numerica fija | P59-07 | -0,0006246152, 3/5, neto -1, CLOSED |
| fusion de score local y global | P59-09 | +0,0014800075, 2/5, MARGINAL, gate fallido por ambas condiciones |
| geometria local como bloque | P59-02, V6-07 | AUC 0,725 para predecir errores y nada accionable, CLOSED |
| max-pooling de evidencia local | P59-07 | neto -4 en su propio estrato alcanzable |
| ventana como canal sparse extra | V4-D01-D04, E2-09 | CERRADA sin senal |
| canal TF-IDF segmental | P59-07, V4-D01-D04 | la ventana como unidad dio -0,0006246152 |
| umbral por estrato de longitud o de segmentos | E2-16, E2-20, V8-09 | CERRADAS sin mejora |

El punto delicate es el segundo: **max-pooling de evidencia local** lo hizo P59-07 y dio
neto -4 en su propio estrato alcanzable. La agregacion por maximo no es la via, y por eso
esta fase se ocupa de la DISTRIBUCION y no del maximo.

## La asociacion que hace que la fase merezca la pena

| segmentos | filas | error de V03 | tokens medianos |
|---:|---:|---:|---:|
| 1 | 732 | 14.44% | 12 |
| 2 | 376 | 18.71% | 15 |
| 3 | 205 | 19.19% | 18 |
| 4 | 91 | 20.88% | 23 |
| 5 o mas | 79 | 22.78% | 34 |

Diferencia entre un segmento y cinco o mas: **+8.35 puntos**.

Es una **asociacion y no un mecanismo**, y la fase existe justamente para saber si lo es.
Una asociacion entre estructura interna y error es exactamente lo que una bolsa de
n-gramas no puede expresar, y exactamente lo que cabria esperar si la hipotesis de la
fase fuera cierta.

## Donde cae la estructura interna

| bin | n | tokens medios | segmentos medios | con 2 o mas | error |
|---|---:|---:|---:|---:|---:|
| CORTA | 167 | 4.8 | 1.39 | 50 (30%) | 6.59% |
| MEDIA | 602 | 10.9 | 1.62 | 247 (41%) | 16.89% |
| LARGA | 547 | 21.3 | 2.10 | 325 (59%) | 18.65% |
| MUY_LARGA | 167 | 38.7 | 3.29 | 129 (77%) | 22.55% |

La estructura interna aparece justo en LARGA y MUY_LARGA, que es donde P63 midio que
esta el margen residual. Los cortes de longitud son los de P63, heredados y congelados,
y no se ajustan en esta fase.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_01_gap_analysis.py`. Sin push.*