# P65 - Cohesion documental interna

Generado por `experiments/p65_03_cohesion.py`. Fecha 2026-09-29T14:45:46+00:00.

## Por que esta familia estaba abierta

Dos cosas parecian antecedentes y no lo son. **V10-07** mide shift de covariables
**entre** entrenamiento y prueba, que es otro objeto que la cohesion **dentro** del
documento. Y **16_targeting_v2** mide distancias entre un target y un insulto con un
lexico inducido, que es una distancia entre categorias semanticas y no un par lexico de
anchura desconocida. Ninguno de los dos mide cuanto se repite un documento.

La pregunta, que no es la de P63 ni la de P64
-----------------------------------------------
P63 midio la composicion como **ratios** morfologicos, y el bloque batio a un control
aleatorio en 0,1055 aislado. P64 midio como se **distribuye la evidencia** dentro del
documento, y el conflicto interno se duplica en las filas que V03 falla. Ninguna de las
dos pregunta cuanto se repite el vocabulario del propio documento.

Un tweet largo que se repite, se cita o pivota a otro tema tiene otra forma que uno que
desarrolla un solo hilo, y una bolsa de n-gramas no ve la diferencia: ve las palabras dos
veces en los dos casos. La repeticion y el solapamiento son propiedades del DOCUMENTO y no
de ningun token, que es exactamente el tipo de cosa que este proyecto lleva dos fases
encontrando y sin poder usar.

## El diagnostico va antes que las features, sin excepcion

Documentos con dos o mas segmentos: **751 de 1483**. Un documento de
un solo segmento tiene los seis valores neutros por definicion y no se le inventa una
cohesion que no puede tener.

| estrato | n | `coseno_medio` | `coseno_minimo` | `coseno_desviacion` | `jaccard_medio` | `repeticion_cruzada` | `cambio_lexico` |
|---|---:|---:|---:|---:|---:|---:|---:|
| todas | 1483 | 0.5147 | 0.5059 | 0.0121 | 0.5072 | 0.0150 | 0.4853 |
| CORTA | 167 | 0.7066 | 0.7066 | 0.0000 | 0.7066 | 0.0000 | 0.2934 |
| MEDIA | 602 | 0.5995 | 0.5963 | 0.0040 | 0.5968 | 0.0047 | 0.4005 |
| LARGA | 547 | 0.4368 | 0.4250 | 0.0153 | 0.4250 | 0.0196 | 0.5632 |
| MUY_LARGA | 167 | 0.2720 | 0.2439 | 0.0432 | 0.2540 | 0.0519 | 0.7280 |
| V03 correcto | 1042 | 0.5588 | 0.5505 | 0.0108 | 0.5520 | 0.0131 | 0.4412 |
| V03 incorrecto | 441 | 0.4105 | 0.4004 | 0.0151 | 0.4012 | 0.0195 | 0.5895 |
| error de alta confianza | 295 | 0.4291 | 0.4199 | 0.0146 | 0.4203 | 0.0178 | 0.5709 |
| solo multi segmento | 751 | 0.0416 | 0.0242 | 0.0239 | 0.0268 | 0.0296 | 0.9584 |

## La referencia de longitud, que es la que evita el falso positivo

Un descriptor estructural no vale nada si solo mide longitud, y **P64-02 ya demostro**
como pasa eso: el perfil segmental llego a AUC 0,589847 frente a una referencia de
contar tokens de 0,600506, y habia parecido senal hasta que se computo la referencia.

| rasgo | AUC para predecir si V03 falla |
|---|---:|
| coseno_medio | 0.428915 |
| coseno_minimo | 0.425493 |
| coseno_desviacion | 0.533615 |
| jaccard_medio | 0.428973 |
| repeticion_cruzada | 0.541231 |
| cambio_lexico | 0.571084 |
| **n_tokens, REFERENCIA TRIVIAL** | 0.600506 |

Mejor rasgo: **coseno_minimo**, ventaja sobre la referencia **-0.025999**.

## El bloque, con los dos controles

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| V03_cohesion | 0.7745024464 | -0.0027090062 | 0/5 | -9 |
| V03_aleatorio | 0.7654188849 | -0.0117925676 | 0/5 | -42 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 |

```text
contra el control de anchura: +0.0090835614
contra el margen de V03:      +0.0083711943
```

Y el bloque usa **las mismas seis cantidades** que el diagnostico, de modo que no hay
ninguna representacion nueva que descartar despues de mirar: hay una que ya se midio y
que no cabe.

## Veredicto

```text
la cohesion documental lleva informacion que no es capacidad ni confianza: +0.0090835614 contra el control de anchura y +0.0083711943 contra el margen. Y el global es -0.0027090062 con 0 de 5, luego no entra. Quinto caso del mismo patron, y esta vez la lectura es mas nitida porque el diagnostico y el bloque usan LAS MISMAS seis cantidades: no hay ninguna representacion nueva que descartar, solo una que no cabe

LECTURA: INFORMACION_SI_INTEGRACION_NO
DECISION: COMPLEMENTARY
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p65_03_cohesion.py`. Sin push.*