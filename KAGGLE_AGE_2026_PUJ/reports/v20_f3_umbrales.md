# V20 · F3 — Descomposicion de fronteras anidadas

**V19 sigue congelado.** Esto es un cuarto miembro mas; V19 conserva sus tres miembros intactos.

## 1. La evidencia que justifica la familia

El especialista de frontera de D-048 saca **0,7136** en `18-24|25-34`, pero es un clasificador **de dos clases** y por tanto se ajusta sobre **el 34 % de las filas**. La pregunta que nunca se ha hecho es que pase si el mismo discriminador, con la misma tarea y las mismas 279 variables, se ajusta sobre **el 100 %** de las 169.336 filas: es el **umbral anidado** `<=24 frente a >=25`, tres veces mas filas para la misma decision binaria.

**Por que no es un duplicado**: E-028a usa pares **adyacentes** (dos clases, 34 % de los datos), que es el comparador y no el mismo modelo; y ORD-1 de D-037 construia las CDF **a partir de las probabilidades de salida de V12**, no con clasificadores nuevos ajustados sobre las 279, y dio **0,5887**. Aqui no hay ninguna salida de V12 de por medio.

## 2. El diagnostico que decide la familia

| Umbral | Filas lado bajo | Filas lado alto | Acierto del lado alto | Exactitud balanceada | Ventaja sobre el azar | Acierto de V12 en esa fila |
|---|---|---|---|---|---|---|
| <= 13-17 vs >= 18-24 | 28,863 | 140,473 | 0.9776 | **0.7632** | +0.2632 | 0.9408 |
| <= 18-24 vs >= 25-34 | 57,539 | 111,797 | 0.9084 | **0.8264** | +0.3264 | 0.8797 |
| <= 25-34 vs >= 35-49 | 85,886 | 83,450 | 0.8452 | **0.8465** | +0.3465 | 0.8504 |
| <= 35-49 vs >= 50-64 | 114,116 | 55,220 | 0.759 | **0.8373** | +0.3373 | 0.821 |
| <= 50-64 vs >= 65-xx | 142,176 | 27,160 | 0.647 | **0.8112** | +0.3112 | 0.7429 |

La columna comparable entre umbrales es la **exactitud balanceada**, que pondera cada lado por igual. El acierto del lado alto por si solo baja conforme el lado alto se hace minoria, y eso dice como de sesgado es el umbral de decision, no cuan bien discrimina.

## 3. La cadena y sus dos formas de uso

- Filas en las que la correccion de monotonia cambio algo: **18,927** de 169,336
- La cadena de cinco umbrales **en solitario**: **0.6046893**, frente a V12 0,6005017 y V19 0.6079388

| Candidato | macro-F1 | Delta contra V19 | Folds pos. | IC95 | Frontera 1 | Frontera 2 | Rescates | Danos | Neto | Cambios |
|---|---|---|---|---|---|---|---|---|---|---|
| F3a cadena de cinco umbrales anidados, en solitario | 0.6046893 | -0.0032 | 0/5 | [-0.0046, -0.0019] | 0.5036 | 0.4997 | 6,457 | 7,107 | -650 | 21,661 |
| F3b cuarto miembro: cadena de umbrales anidados | 0.6097782 | +0.0018 | 5/5 | [+0.0012, +0.0025] | 0.5075 | 0.5029 | 1,923 | 1,629 | +294 | 5,674 |

## 4. Delta por fold

| Candidato | Delta por fold |
|---|---|
| F3a cadena de cinco umbrales anidados, en solitario | [-0.001599, -0.003385, -0.002469, -0.005535, -0.003277] |
| F3b cuarto miembro: cadena de umbrales anidados | [0.002485, 0.001994, 0.000996, 0.00087, 0.002853] |
