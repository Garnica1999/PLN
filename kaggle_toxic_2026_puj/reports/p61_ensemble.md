# P61 - Ensemble pequeno con pertenencia y pesos ANIDADOS

Generado por `experiments/p61_05_ensemble.py`. Fecha 2026-09-28T21:02:51+00:00.

## El defecto que hubo que corregir antes de aceptar el resultado

Una primera version anidaba los **pesos** pero no la **pertenencia** al pool: los tres
historicos se elegian una sola vez, de forma global, por una utilidad ajustada por
diversidad calculada sobre todo el OOF. Eso es ajustar sobre los datos externos,
porque la pregunta de *quien pertenece al pool* se respondia con las mismas filas
sobre las que despues se puntua. Anidar los pesos no repara eso, porque el conjunto de
candidatos ya arrastraba la fuga.

La version vigente hace la seleccion de pertenencia **tambien** dentro de cada pliegue,
con las filas internas, y por eso el numero final **no es** el de la primera version.

El segundo defecto era de optimizacion: SLSQP sobre el macro-F1 devolvia pesos
**exactamente uniformes** en todos los tamanos y en los dos objetivos, porque el F1 con
umbral fijo es constante a tramos y su gradiente es cero casi en todas partes, luego el
optimizador se paraba en el punto inicial. Esos pesos uniformes eran un artefacto del
metodo, no una solucion. La version vigente usa una busqueda **sin derivadas** y
determinista sobre el simplejo.

## El protocolo, anidado en las dos dimensiones

```text
para cada pliegue exterior f:
    internas = los otros cuatro pliegues
    PERTENENCIA: se ordenan los historicos por utilidad ajustada por diversidad
                 calculada SOLO con las filas internas, y se conservan los tres
    PESOS:       se optimizan con las filas internas
    se aplican el pool y los pesos resultantes al pliegue f
la puntuacion se arma con los cinco pliegues retenidos
```

El optimizador de pesos es una busqueda sin derivadas: el punto uniforme, los k
vertices unitarios y 300 sorteos Dirichlet(1) con semilla fija. El punto
uniforme siempre es candidato, luego el resultado no puede ser peor que el promedio
simple, y el generador sembrado lo hace reproducible.

## Resultados

| ID | pool | k | obj | corrected_cv | delta vs V03 | pliegues + | rescate | dano | neto | gate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| con_nuevo_k1_A | con_nuevo | 1 | A | 0.7772114525 | +0.0000000000 | 0/5 | 0 | 0 | +0 | no |
| con_nuevo_k1_B | con_nuevo | 1 | B | 0.7772114525 | +0.0000000000 | 0/5 | 0 | 0 | +0 | no |
| con_nuevo_k2_A | con_nuevo | 2 | A | 0.7778799726 | +0.0006685201 | 3/5 | 61 | 45 | +16 | no |
| con_nuevo_k2_B | con_nuevo | 2 | B | 0.7774722703 | +0.0002608178 | 3/5 | 69 | 54 | +15 | no |
| con_nuevo_k3_A | con_nuevo | 3 | A | 0.7835159596 | +0.0063045070 | 4/5 | 107 | 69 | +38 | **SUPERADO** |
| con_nuevo_k3_B | con_nuevo | 3 | B | 0.7830826704 | +0.0058712179 | 3/5 | 84 | 52 | +32 | no |
| con_nuevo_k4_A | con_nuevo | 4 | A | 0.7832411761 | +0.0060297235 | 4/5 | 105 | 68 | +37 | **SUPERADO** |
| con_nuevo_k4_B | con_nuevo | 4 | B | 0.7775570575 | +0.0003456050 | 2/5 | 110 | 86 | +24 | no |
| con_nuevo_k5_A | con_nuevo | 5 | A | 0.7824888500 | +0.0052773975 | 5/5 | 79 | 63 | +16 | **SUPERADO** |
| con_nuevo_k5_B | con_nuevo | 5 | B | 0.7747625856 | -0.0024488670 | 2/5 | 103 | 93 | +10 | no |
| sin_nuevo_k1_A | sin_nuevo | 1 | A | 0.7772114525 | +0.0000000000 | 0/5 | 0 | 0 | +0 | no |
| sin_nuevo_k1_B | sin_nuevo | 1 | B | 0.7772114525 | +0.0000000000 | 0/5 | 0 | 0 | +0 | no |
| sin_nuevo_k2_A | sin_nuevo | 2 | A | 0.7823837223 | +0.0051722698 | 3/5 | 83 | 56 | +27 | no |
| sin_nuevo_k2_B | sin_nuevo | 2 | B | 0.7814179607 | +0.0042065081 | 3/5 | 86 | 61 | +25 | no |
| sin_nuevo_k3_A | sin_nuevo | 3 | A | 0.7830547461 | +0.0058432936 | 3/5 | 100 | 70 | +30 | no |
| sin_nuevo_k3_B | sin_nuevo | 3 | B | 0.7789677351 | +0.0017562826 | 2/5 | 86 | 69 | +17 | no |
| sin_nuevo_k4_A | sin_nuevo | 4 | A | 0.7810509833 | +0.0038395308 | 3/5 | 97 | 75 | +22 | no |
| sin_nuevo_k4_B | sin_nuevo | 4 | B | 0.7843655847 | +0.0071541322 | 3/5 | 91 | 57 | +34 | no |

## El control de atribucion, que es la medicion decisiva

El pool ya contiene V03, luego un resultado positivo no es atribuible sin comparar el
pool con y sin el modelo nuevo. Aqui eso resulto decisivo:

```text
el modelo nuevo SUBTRAA -0.0008496251. El mejor resultado se obtiene sin el, luego la mejora no es atribuible a la capacidad de LightGBM sino al promediado de modelos historicos diversos que ya existian
```

## Desplazamiento de probabilidad frente a desplazamiento de decision

```text
desplazamiento medio de probabilidad         0.037792
decisiones que cambian                      176 de 4449
distancia media al umbral, V03              0.270230
distancia media al umbral, ensemble         0.296680
celdas que cambian y se alejan del umbral   92
celdas que cambian por unidad de desplazamiento 1.05
lectura: desplazamiento repartido entre probabilidad y decision
```

## Reproducibilidad

Nivel **R2**: semilla fija, artefacto en disco con los pesos por pliegue y por miembro,
y hash del OOF verificado. Una reejecucion con la misma semilla dara el mismo resultado,
pero no se ha reproducido de forma independiente en otra sesion.

## Decision

```text
decision  PROMISING. Supera el gate; exige segunda particion
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p61_05_ensemble.py`. Sin push.*