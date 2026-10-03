# Test de permutacion de A_cost_sensitive_downweight

Generado por `experiments/p69_05_permutacion.py`. Fecha 2026-09-29T18:44:22+00:00.

## Que pregunta el test, y que no

La tercera particion respondio a una pregunta: si el efecto sobrevive a otro reparto de
pliegues. Sobrevive, en las tres. Eso es robustez frente a la **particion**.

No respondio a esta otra, que es la de aqui:

```text
el beneficio depende de emparejar la regla de pesos con los EJEMPLOS concretos,
o cualquier asignacion de los mismos pesos haria lo mismo?
```

No son la misma pregunta. Un esquema de ponderacion podria mejorar la metrica solo por
cambiar el peso total efectivo y su dispersion, y eso ninguna asignacion lo evita. Si
barajar los pesos entre las filas de entrenamiento reproduce la ganancia, entonces la
ganancia no va de **que** ejemplos reciben menos peso.

## El diseno, pre-registrado

```text
n de permutaciones     1000
semilla                20261002
prueba                 unilateral
estadistico            delta_corrected_cv
p empirico             (1 + #(delta_perm >= delta_obs)) / 1001
que se baraja          los pesos DENTRO del pliegue de ENTRENAMIENTO
```

Se conservan la regla de pesos, la distribucion, el numero de ejemplos ponderados, la
particion y los multiplicadores de class weight. El barajado **nunca toca** el pliegue de
validacion, y no mueve filas entre pliegues, luego no introduce fuga.

El diseno esta en `docs/p69_especificacion_congelada.md`, commit `915991b`, escrito
**antes** de que existiera ningun resultado.

## El resultado

```text
delta con los pesos REALES   +0.0082045862
delta registrado             +0.0082045862
coincide                     True
```

### La distribucion nula

| estadistico nulo | valor |
|---|---:|
| media | -0.0013209910 |
| desviacion | 0.0020508746 |
| mediana | -0.0012968408 |
| minimo | -0.0078934583 |
| maximo | +0.0045971670 |
| percentil 95 | +0.0020341831 |
| percentil 99 | +0.0030856337 |

```text
permutaciones >= al observado   0 de 1000
p empirico                      0.000999
alpha                           0.05
```

## El veredicto

```text
STRONGLY_SUPPORTED_BUT_NOT_AUTOMATICALLY_ADOPTED

la asignacion REAL de pesos contiene mas informacion que una asignacion equivalente pero permutada. La mejora depende de QUE ejemplos reciben menos peso, no solo de cuanto peso se retira
```

### Lo que el test NO responde

```text
no dice si V03 esta definitivamente peor
no dice si la mejora en Kaggle sera positiva
```

## La decision final

| componente | cumple |
|---|---|
| tres_particiones_positivas | **SI** |
| permutacion_favorable | **SI** |
| bootstrap_por_grupos_compatible | NO |
| rescue_damage_favorable | **SI** |
| compliance_completo | **SI** |
| provenance_completo | **SI** |

**5 de 6.**

```text
PROMISING_UNCONFIRMED
```

El bootstrap por grupos incluye el cero, luego **no se clasifica ADOPTED**. La adopcion
exigiria ademas el bootstrap compatible, y es una decision separada.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_05_permutacion.py`. Sin push.*