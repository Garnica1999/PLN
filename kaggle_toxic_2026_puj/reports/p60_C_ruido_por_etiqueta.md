# P60-C - Ruido dependiente de etiqueta (C1 y C2)

Generado por `experiments/p60_05_ruido_por_etiqueta.py`. Fecha 2026-09-28T15:37:09+00:00.

## Motivacion

La literatura reciente de multi-etiqueta hace que la unidad de deteccion sea el PAR
(instancia, etiqueta) y no la instancia. TOXIC ya cerro el ruido en su forma de instancia:
`V30-00` cerro siete experimentos de sospecha por instancia y `P59-03` cerro el estimador
formal global, con AUC 0,9730 frente a un control trivial de 0,9741. Lo que ninguno hizo
es descomponer la sospecha POR ETIQUETA, y en este corpus esa descomposicion no es una
formalidad.

## La asimetria estructural, medida

```text
P_TR1_dado_HS0                     0.0
P_AG1_dado_HS0                     0.026347305389221556
P_AG1_dado_TR1                     0.946067415730337
negativos_TR_no_definidos          1.0
negativos_AG_no_definidos          0.9736526946107784
filas_HS0                          835
n_filas                            1483
```

El 100 por ciento de los negativos que ve TR en entrenamiento son etiquetas NO DEFINIDAS, y
el 97 por ciento en AG, mientras HS no tiene region indefinida. El indicador estructural
vale 1 para TR cuando HS = 0, para AG cuando HS = 0, y **siempre 0 para HS**.

## Por que un AUC bruto no basta

Un score puede tener AUC alto y no anadir nada, si es una copia ruidosa de la confianza que
el proyecto ya mide. `P59-03` aprendio eso de forma directa. Por eso la prueba no es
`AUC > 0,5` sino si el score sigue ordenando errores **una vez controlado** el control:

```text
residuo = score - E[score | decila del control]
AUC incremental = AUC del residuo frente al indicador de error
```

Deciles y no un ajuste parametrico, para no asumir ninguna forma de modelo.

Y la referencia de comparacion **no es 0,5 sino el propio control**: una senal que
es funcion monotona del control tiene EXACTAMENTE el mismo residuo, porque su
variacion dentro de la decila es identica. La pregunta es si una senal supera al
control en potencia residual, y de ahi el umbral se mide como GANANCIA sobre el
control y no como distancia a 0,5.

## Senales construidas, todas con filas de ENTRENAMIENTO del pliegue

| senal | que es |
|---|---|
| control | `1 - p_true`, la probabilidad que el modelo da a la clase observada |
| perdida | perdida logistica de ese par |
| rango | rango normalizado de la perdida dentro de esa etiqueta |
| centroide | margen coseno con signo a los centroides de clase y no clase |
| estructural | indicador de etiqueta NO DEFINIDA |
| compuesto | regresion logistica sobre las cuatro senales estandarizadas |

## Resultados

| etiqueta | senal | AUC | AUC incremental | constante |
|---|---|---:|---:|---|
| HS | control | 0.9917 | 0.5296 | False |
| HS | perdida | 0.9917 | 0.4928 | False |
| HS | rango | 0.9906 | 0.5547 | False |
| HS | centroide | 0.5081 | 0.5148 | False |
| HS | estructural | 0.5000 | no aplica | True |
| HS | compuesto | 0.9915 | 0.4941 | False |
| TR | control | 1.0000 | 0.5050 | False |
| TR | perdida | 1.0000 | 0.3982 | False |
| TR | rango | 0.9992 | 0.5392 | False |
| TR | centroide | 0.6417 | 0.5338 | False |
| TR | estructural | 0.4508 | 0.6203 | False |
| TR | compuesto | 1.0000 | 0.4075 | False |
| AG | control | 0.9961 | 0.5498 | False |
| AG | perdida | 0.9961 | 0.5090 | False |
| AG | rango | 0.9942 | 0.5473 | False |
| AG | centroide | 0.5046 | 0.5063 | False |
| AG | estructural | 0.5225 | 0.6627 | False |
| AG | compuesto | 0.9968 | 0.5041 | False |

## Veredicto de senal incremental

| senal | mejor etiqueta | incremental | control | ganancia | constante | supera |
|---|---|---:|---:|---:|---|---|
| perdida | AG | 0.5090 | 0.5498 | -0.0408 | False | **False** |
| rango | HS | 0.5547 | 0.5296 | +0.0252 | False | **True** |
| centroide | TR | 0.5338 | 0.5050 | +0.0288 | False | **True** |
| estructural | AG | 0.6627 | 0.5498 | +0.1129 | False | **True** |
| compuesto | AG | 0.5041 | 0.5498 | -0.0457 | False | **False** |

**Hay senal incremental: True.** Umbral declarado: ganancia >= 0.01 sobre el control de la misma etiqueta.

## Decision

```text
decision  ABIERTA_PARA_C3. Hay senal incremental, luego corresponde UNA intervencion conservadora, con una sola forma de objetivo blando o un solo ponderado por par
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_05_ruido_por_etiqueta.py`. Sin push.*