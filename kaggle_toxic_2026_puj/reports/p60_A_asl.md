# P60-A - Perdida asimetrica de Ridnik et al.

Generado por `experiments/p60_02_asl.py`. Fecha 2026-09-28T15:14:11+00:00.

## Motivacion e hipotesis

TOXIC es multi-etiqueta con asimetria entre positivos y negativos, y la literatura de
multi-etiqueta trata esa asimetria como un objeto de primer orden y no como balance de
clases. La perdida asimetrica mantiene la perdida logaritmica en el positivo y desplaza y
eleva a potencia la del negativo, de modo que la perdida negativa es **exactamente cero**
cuando el modelo ya esta seguro de que una etiqueta no esta.

## Por que no es V7-01 ni P59-I

`V7-01` cerro la rama SIMETRICA de esta familia: focal con un unico gamma, que decae de
forma suave, dio -0,0024 contra su control con 2 de 5 pliegues, y su decision dice que se
cerraba la unica linea de perdida abierta. ASL no es eso: su mecanismo distintivo es el
**corte duro**.

`P59-06` cerro una perdida redescendiente, que es la region **opuesta**: Tukey anula la
influencia de residuos extremos, los ejemplos que el modelo falla mucho. ASL anula la
perdida de residuos faciles, los que el modelo ya acierta.

## La priori, y por que es baja

Ridnik et al. justifican ASL por un desequilibrio de uno o dos ordenes de magnitud. Medido
aqui, las prevalencias son 0,4370, 0,3001 y 0,3823, luego la asimetria es de 1,46 a 1. Es
una version mucho mas debil de la condicion para la que el metodo fue disenado, y la
revision lo declara. El experimento se ejecuta igual, porque esa afirmacion es una hipotesis
y no un resultado.

## Configuracion, fijada antes de mirar validation

```text
L+ = (1-p)^gamma+ * (-log p)      gamma+ = 0.0
L- = max(p-m, 0)^gamma- * (-log(1-p))   gamma- = 4.0, m = 0.05
los tres son los valores ASL_s que el trabajo recomienda para multi-etiqueta dispersa
pesos de clase  los mismos balanceados de V03, contados en training
regularizacion  C = 2.0 con el mismo L2: lo unico que cambia es la perdida
umbrales        0,44 / 0,51 / 0,46 congelados
rejilla         ninguna
```

## Verificacion del gradiente, antes de entrenar

| perdida | regimen | error relativo w | error relativo b |
|---|---|---:|---:|
| logistica | sin_corte | 2.82e-06 | 2.02e-06 |
| logistica | con_corte | 9.53e-07 | 1.80e-10 |
| asimetrica | sin_corte | 3.22e-07 | 7.40e-11 |
| asimetrica | con_corte | 5.79e-06 | 3.39e-10 |

## Control

```text
brazo 0 (mi optimizador, perdida logistica)  corrected_cv 0.7772647318762402
  delta vs V03 +0.0000532794 | max |proba - canonica| 6.908e-03 | celdas distintas 3
```

Los dos brazos comparten optimizador, representacion, pesos de clase, regularizacion y
umbrales, luego la diferencia entre ellos aisla la perdida y nada mas.

## Activacion del mecanismo, y por que este resultado es interpretable

ASL tiene dos mecanismos y hay que medirlos por separado:

```text
(a) corte duro, la parte NOVEL: negativos con perdida exactamente cero
    0.00 a 0.00 por ciento (media 0.00)
(b) potencia gamma- = 4.0, la parte que si actua: reescalado del negativo
    0.066449 de la perdida logistica, un factor 15.0 menos contribucion negativa
```

La lectura es concluyente. **El corte duro nunca se activa**: un modelo bien ajustado no
lleva la probabilidad de una etiqueta ausente por debajo de 0,05, luego la parte que
distingue ASL de focal no llego a existir aqui. Lo que actua es la potencia del
negativo, que reescala la contribucion de todos los negativos y es, matematicamente, un
**ponderado de clase**, familia que `V8-07` cerro con -0,005298 y 1 de 5 pliegues.

## Resultados

```text
brazo 0 corrected_cv 0.7772647319
brazo 1 corrected_cv 0.6243388620
delta PERDIDA PURO   -0.1529258699
delta vs V03         -0.1528725906
pliegues que mejoran 0/5
rescate 334 | dano 1552 | neto -1218 | cambian 1886
convergencia de los quince ajustes True
```

| etiqueta | F1 asimetrica | F1 control |
|---|---:|---:|
| HS | 0.6353 | 0.7623 |
| TR | 0.6472 | 0.8153 |
| AG | 0.5905 | 0.7542 |

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto de la perdida >= +0,003 | -0.152926 | **False** |
| al menos 4 de 5 pliegues | 0/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  CLOSED. La perdida asimetrica no mejora sobre el control de misma capacidad, con el mecanismo de corte duro Siendo el que V7-01 no probo
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_02_asl.py`. Sin push.*