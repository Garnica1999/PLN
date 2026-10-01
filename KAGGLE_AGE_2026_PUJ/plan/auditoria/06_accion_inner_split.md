# Reserva A-02 · cerrado: el sesgo del inner split de Optuna **sí es material** (medido)

## Por que mi primera medicion era erronea

En la auditoria D-090 registre "sin impacto medible: 0,963 % ingenuo vs 1,030 % agrupado". Ese
numero estaba mal, por dos motivos que ahora corrijo:

1. **Medi el tipo de leakage equivocado.** Use solapamiento de **texto exacto**. Pero los clusters de
   `features/dup_clusters.parquet` agrupan por igualdad de texto normalizado **o** similitud de Jaccard
   por encima de un umbral. El leakage que importa es el de **cluster**, no el de texto identico.
2. **Compara esquemas no comparables por construccion.** El inner split se construye DENTRO de un
   fold-train ya agrupado. Su train interno hereda clusters que el propio outer split permitio, luego
   el solapamiento exacto entre inner train e inner val no puede ser cero por construccion. La
   referencia correcta es el **inner agrupado**, no cero teorico.

## Medicion correcta

`scripts/auditoria_medir_inner_split_optuna.py`, que reconstruye las dos particiones interiores
**exactamente como las construye el codigo real** (`v30_01_optuna_refinamiento.py:220-226`, incluidos
el criterio de elegir el fold cuyo tamano se acerca a `FRACCION_INTERIOR * n`).

Sobre los 5 folds externos canonicos, 169.336 filas, 164.018 clusters:

| Nivel | overlap texto exacto | **solape de cluster** | n_val medio |
|---|---|---|---|
| outer agrupado (canonico) | 0,437 % | **0,000 %** | 33.867 |
| **inner ingenuo (código real)** | 1,067 % | **3,907 %** | 27.094 |
| inner agrupado (propuesto) | 0,409 % | **0,000 %** | 27.094 |

Consistente en los cinco folds: el inner ingenuo deja entre **3,787 % y 4,152 %** de las filas de
validación interior con un near-duplicate en el train interior, contra **0,000 %** del inner agrupado.

## Criterio declarado antes de correr

El script declara el umbral **antes** de medir: diferencia de solape de cluster mayor que
**0,02 (2 puntos porcentuales)** = sesgo material. La diferencia medida es **0,039071**, casi el doble
del umbral.

**Veredicto: SESGO MATERIAL.** Confirmado en los 5 folds, con desviacion 0,0013.

## Que significa, y que NO significa

**Significa** que la seleccion de hiperparametros de V29 y V30 se optimizo contra una metrica de
validacion interior donde el **3,9 %** de la evaluacion tenia un near-duplicate en el train. Eso
premia hiperparametros que explotan la redundancia del corpus. Los dos LightGBM del campeon salieron
de ahi, luego **el sesgo esta en el camino al campeon**, no en un experimento lateral.

**NO significa** que el campeon sea invalido ni que V29/V30 esten sobreestimados en su reporte. La
estimacion **externa** de V29/V30 se hizo con el outer split agrupado, sin leakage: por eso sus
IC95 y sus deltas publicados son validos. Lo que queda contaminado es **que hiperparametros se
eligieron**, no **cuanto rindio el modelo con esos hiperparametros**.

Esta distincion es la que evita exagerar el hallazgo.

## Decision y coste de cerrarla

`scripts/v30_02_ensemble.py:257` ya registra que V30.resultados dio `max(corrected_cv) = 0.6130051`
mientras el promedio ponderado de pesos da `0.6134471`, y el acta D-066 lo reconoce como
auto-falsacion. Re-optimizar con inner agrupado puede mover ese numero **en cualquier direccion**: si
el leakage inflaba la seleccion, el re-optimizado puede **bajar**.

**Por tanto re-optimizar no es una mejora esperada, es una correccion metodologica**, y su resultado
no se puede prometer. Coste: los Optuna de V29 y V30 son de **horas de CPU** cada uno (D-064 registra
que el fold 0 no termino en 55 min con 30 trials, y se bajo a 10). Carlos autorizo ejecutarlo.

## Lo que hace falta antes de gastar esas horas

Re-optimizar sin pre-registro seria exactamente el defecto que la auditoria reprobo en otras fases.
Ver el plan de la accion 4 en `plan/auditoria/06_acciones_followup.md`.