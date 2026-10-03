# P61 - Capacidad, diversidad y distribucion de probabilidad: informe final

Generado por `experiments/p61_09_cerrar.py`. Fecha 2026-09-28T21:12:26+00:00.

## La pregunta que la fase tenia que responder

> El limite actual de V03 se debe principalmente a la capacidad del modelo, a la baja
> diversidad util entre sus predicciones, a la forma en que las probabilidades se
> distribuyen respecto de los thresholds, o a una combinacion de estos factores?

## La respuesta, y por que es la que es

Ninguno de los tres mecanismos de mejora resulta explotable, y el diagnostico dice
POR QUE. La EDA de probabilidades muestra que entre el 60 y el 68 por ciento de los
errores de V03 son errores de ALTA CONFIANZA, con la distancia al umbral por encima
de la de los aciertos. Un error de ese tipo no lo arregla un umbral, no lo arregla una
calibracion y no lo arregla un modelo con mas capacidad sobre la misma
representacion: lo que falta es senal en el texto o lectura de la que hay.

| mecanismo | resultado | veredicto |
|---|---|---|
| capacidad, familia nueva | -0.0258463023, 2/5 pliegues | CLOSED |
| capacidad, dentro de la familia | +0.0043934168 recuperados por la busqueda | la busqueda funciona, la familia no llega |
| diversidad historica | 0 modelos por encima de V03, 0 con neto positivo | no hay combinacion infrautilizada |
| ensemble, canonica | +0.0063045070, gate superado | artefacto, no evidencia |
| ensemble, segunda particion | -0.0011527969 | REJECTED |
| control negativo del ensemble | +0.0059889238 | la ganancia no es diversidad |

## Lo que la EDA deja como instrumento permanente

Es la aportacion mas util de la fase, y no es un resultado sino un instrumento. La
tabla `reports/p61_final_probability_table.csv` tiene una fila por celda
etiqueta-tweet, 4449 en total, con la probabilidad OOF final, el umbral, la decision,
si acerto o fallo, y la distancia y el margen al umbral. Desde ahi se separan cuatro
modos de fallo que un unico numero de F1 no distingue:

```text
error de representacion     el texto no lleva la senal, o el modelo no la lee
error de calibracion        la probabilidad esta sistematicamente desviada
error de decision           el orden es bueno y el umbral esta mal puesto
sin informacion disponible  la observacion es indistinguible de sus vecinas
```

| etiqueta | ECE | Brier | log loss | solapamiento de las dos densidades |
|---|---:|---:|---:|---:|
| HS | 0.048754 | 0.148551 | 0.460781 | 0.4190 |
| TR | 0.068229 | 0.089307 | 0.308197 | 0.2399 |
| AG | 0.059591 | 0.139454 | 0.441771 | 0.3965 |

El ECE alto NO es un hallazgo. V03 se ajusto con class_weight balanced, que por
construccion desplaza las probabilidades para compensar el desequilibrio, luego sus
probabilidades no son frecuencias y un ECE grande es el comportamiento esperado de un
modelo correctamente ponderado para decidir. La calibracion sigue CERRADA como
hipotesis de mejora, y estos numeros son diagnostico.

## El hallazgo que explica los ensembles historicos

El ensemble dio +0,0063045070 en la particion canonica, supero el gate, paso 4 de 5
pliegues y dio un bootstrap de IC95 [+0,002697, +0,014385]. Todo apuntaba a un
resultado. El control negativo lo desmonta:

Con los tres PEORES canales del pool, que estan entre 0,04 y 0,24 por debajo de
V03, la ganancia sigue siendo +0.0059889238.

Un promediado ponderado que contiene a V03 mejora el OOF casi con seguridad, porque
los pesos se eligen para exploitear el residuo de V03 sobre las MISMAS filas. Eso es
ajustar, no generalizar. Y el bootstrap no lo detecta porque mide la variabilidad
del estimador sobre filas que ya intervinieron en elegirlo: mide ruido de muestreo,
no sesgo de seleccion.

Esto explica de una sola causa por que ENS-01 a ENS-04 tambien fallaron y por que
la mitad de los resultados historicos que mencionan ensembles son negativos. No era
que faltara busqueda.

## Dos defectos de protocolo que hubo que corregir antes de aceptar el numero

1. Los pesos se optimizaban con SLSQP sobre un F1 que es constante a tramos, luego
   su gradiente era nulo y el optimizador devolvia pesos EXACTAMENTE uniformes. Eran
   un artefacto del metodo.
2. Los pesos estaban anidados pero la PERTENENCIA no: los canales historicos se
   elegian de forma global sobre todo el OOF. Anidar los pesos no repara eso, porque
   el conjunto de candidatos ya arrastraba la fuga antes de optimizar nada.

Corregidos ambos, la pertenencia resulto IDENTICA en los cinco pliegues, lo que
confirma que el defecto era de principio y no de resultado.

## Lo que NO se hizo, y por que

- **No se genero ninguna submission.** El unico candidato que paso el gate interno
  fallo en la segunda particion, luego no habia nada que enviar. El parrafo 38 solo
  permite generar una si existe candidata que pase gate, bootstrap, segunda particion
  y cumplimiento, y ninguna pasa las cuatro.
- **No se reentreno nada del pool historico.** Los veinte canales tienen OOF valido
  sobre las mismas filas y la misma particion, y reutilizarlos es mas barato y mas
  honesto que reconstruir veinte modelos.
- **No se reabrio la familia MLP.** El parrafo 28 la condiciona a que exista una
  razon de diversidad, y el test 3 acaba de demostrar que la diversidad no es el
  mecanismo.
- **No se gastaron los cinco experimentos restantes.** El criterio de parada se
  alcanzo y las plazas de reserva son para hipotesis genuinamente nuevas, no para
  buscar hiperparametros de un modelo cerrado ni para rescatar un negativo.

## Inventario, y la pregunta de si habia diversidad infrautilizada

`features/p50_celda_features.npz` contenia VEINTE canales con OOF sobre las mismas
1483 filas. Su convencion de pliegues va de 0 a 4 y la canonica de 1 a 5, y se
verifico que el desplazamiento es de convencion y no de particion, porque sin esa
comprobacion las medidas de diversidad compararian predicciones de pliegues
distintos y no significarian nada.

```text
canales con corrected_cv por encima de V03   0
canales con rescate unico sobre V03          19
canales con neto positivo frente a V03      0
```

El mas cercano es `emoc:NB-UNION-LR@C=1.0`, a 0.764702, con 188 rescates y 198
danos. Aporta diversity de verdad, 188 celdas que V03 falla y el acierta, pero en
neto es negativo y su corrected_cv esta 0,0125 por debajo.

## El techo estructural, que sigue en pie

V28-01 midio que los errores de HS, TR y AG se solapan con coeficientes de +0,42 a
+0,51, luego los tres modelos fallan sobre los mismos textos y cualquier metodo que
actue sobre una sola etiqueta choca contra ese techo. La lectura de la EDA es
coherente con el: si el error es de representacion y ademas es compartido entre las
tres etiquetas, falta senal en el texto.

## Reproducibilidad

Nivel **R2** en los tres experimentos: semilla fija, artefactos en disco con los
450 trials completos y los pesos por pliegue y por miembro, hash del OOF y de las dos
particiones verificados, y ejecucion real con exit code capturado sin pipe. No se
declara R4, porque no hay reproduccion independiente.

## Decision de la fase

```text
no evidence of exploitable remaining model capacity/diversity

V03 remains the best verified leaderboard result.
```

No se afirma que V03 sea matematicamente optimo. Se afirma exactamente lo que las
mediciones sostienen: que en el espacio de mecanismos que esta fase se permitio
explorar, ninguno produjo una mejora reproducible y complementaria.

## Artefactos

* `reports/p61_compliance_audit.json`
* `reports/p61_compliance_audit.md`
* `reports/p61_ensemble.json`
* `reports/p61_ensemble.md`
* `reports/p61_ensemble_pool_selection.csv`
* `reports/p61_ensemble_results.csv`
* `reports/p61_ensemble_weights.csv`
* `reports/p61_final_probability_table.csv`
* `reports/p61_final_summary.json`
* `reports/p61_lightgbm_capacidad.json`
* `reports/p61_lightgbm_capacidad.md`
* `reports/p61_master_results.csv`
* `reports/p61_model_diversity.csv`
* `reports/p61_model_inventory.csv`
* `reports/p61_model_inventory.json`
* `reports/p61_notebook_integration.json`
* `reports/p61_optuna_results.csv`
* `reports/p61_p60_reconciliation.csv`
* `reports/p61_p60_reconciliation.json`
* `reports/p61_p60_reconciliation.md`
* `reports/p61_probability_eda.csv`
* `reports/p61_probability_eda.json`
* `reports/p61_probability_eda.md`
* `reports/p61_segunda_particion.csv`
* `reports/p61_segunda_particion.json`
* `reports/p61_segunda_particion.md`
* `figuras/p61/01_probability_distributions_by_label.png`
* `figuras/p61/02_probability_bins_error.png`
* `figuras/p61/03_threshold_distance_windows.png`
* `figuras/p61/04_distance_to_threshold.png`
* `figuras/p61/05_high_confidence_errors.png`
* `figuras/p61/06_joint_probability_geometry_3d.png`
* `figuras/p61/07_pairwise_probability_geometry.png`
* `figuras/p61/08_reliability_curves.png`
* `figuras/p61/09_error_correlation_heatmap.png`
* `figuras/p61/10_unique_rescue_vs_damage.png`
* `figuras/p61/11_ensemble_cv_vs_v03.png`
* `figuras/p61/12_ensemble_weights.png`
* `figuras/p61/13_v03_vs_candidate_probability_shift.png`
* `figuras/p61/14_error_breakdown_v03_vs_candidate.png`

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p61_09_cerrar.py`. Sin push.*