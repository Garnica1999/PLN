# Especificacion congelada de la confirmacion de A_cost_sensitive_downweight

Este documento se escribe **antes** de ejecutar la tercera particion y **antes** de ejecutar el
test de permutacion. Se commitea solo, para que la no-adaptacion quede demostrada y no
afirmada: cualquier lector puede comprobar que la semilla, el protocolo y el numero de
permutaciones estaban fijados por escrito antes de que existiera un resultado que mirar.

## 1. El candidato, congelado

```text
margen_favor = (p_V03 - umbral) * (2*y - 1)
violacion_l  = clip(-margen_favor, 0, None)      por etiqueta
error_score  = violacion_l.mean(axis=1)          por ejemplo, media de 3
z            = (error_score - mediana) / IQR
peso         = clip(1 - 0.5*z, 0.5, 1.5)
```

Cantidades congeladas de la particion canonica, que no se tocan:

```text
441 ejemplos con peso < 1
1042 ejemplos con peso = 1
0 ejemplos con peso > 1
13,66 por ciento del peso total retirado
```

La hipotesis es **downweight de ejemplos con violacion alta de margen**. No es un metodo nuevo
de ponderacion. Ninguna de estas cantidades se modifica en este ciclo.

## 2. La descripcion correcta del mecanismo

P58 describe esta variante como *"meta con downweight coste-sensible por violacion del
umbral"*. El codigo de `30_supervised_weighting_v4.py` **no construye ningun meta-modelo**:
entrena `LogisticRegression(C=2.0, max_iter=1000, solver=liblinear)` directamente sobre la
representacion de V03, con `sample_weight`.

```text
ORIGINAL             meta con downweight coste-sensible por violacion del umbral
CORRECTED            cost-sensitive margin-violation downweighting
RELACION             el codigo entrena sobre V03, no sobre un meta-modelo
```

El informe historico de P58 **no se modifica**. La correccion vive en los informes nuevos.

## 3. Tercera particion: protocolo fijado de antemano

```text
tipo                 confirmatoria, NO una oportunidad de tuning
semilla              20261001
numero de pliegues   5
metodo               StratifiedGroupKFold sobre los grupos canonicos
grupos               near-duplicate UNION pseudo-autor, tal como se vienen usando
```

La semilla `20261001` se elige por ser distinta de la canonica (`RANDOM_STATE`) y de la segunda
(`20260928`), y se declara aqui antes de ver ningun resultado.

### Lo que queda congelado

```text
V03                  congelado, 0.7772114525165721
representacion       TF-IDF word(1,2) min_df=2 + char_wb(3,5) min_df=2, ajustado por pliegue
clasificador         LogisticRegression(C=2.0, max_iter=1000, solver=liblinear)
umbrales             0.44 / 0.51 / 0.46
regla de pesos       la de la seccion 1
barrido de clase     3 rondas x 3 etiquetas x (0.8, 1.0, 1.2, 1.5)
```

### Lo que cambia

Solo que grupo cae en que pliegue.

## 4. Regla de decision, fijada de antemano

```text
CASO A   3 de 3 particiones positivas
         y la tercera con >= 4 de 5 pliegues positivos
         y sin leakage
         -> PROMISING_UNCONFIRMED, y se pasa al test de permutacion

CASO B   2 de 3 particiones positivas
         -> MARGINAL, se detiene la investigacion de esta hipotesis

CASO C   la tercera da delta claramente negativo
         -> MARGINAL_UNSTABLE, se cierra
```

## 5. Test de permutacion: diseno fijado de antemano

Solo se ejecuta si la tercera particion confirma.

```text
numero de permutaciones   1000
semilla de permutacion    20261002
estadistico               delta_corrected_cv = CV(candidato ponderado) - CV(V03)
prueba                    unilateral
p_empirico                (1 + #{delta_perm >= delta_observado}) / (1001)
```

Cada permutacion baraja los pesos **dentro del pliegue de entrenamiento**, conservando la
distribucion y el numero de ejemplos ponderados, y **nunca toca el pliegue de validacion**. El
barajado opera sobre las filas del entrenamiento; no se mezclan grupos de forma que se
introduzca fuga, porque el barajado no mueve filas entre pliegues.

### Lo que el test responde, y lo que no

```text
SI responde   la asignacion REAL de pesos contiene mas informacion
              que una asignacion equivalente pero permutada?

NO responde   si V03 esta definitivamente peor
NO responde   si la mejora en Kaggle sera positiva
```

## 6. Regla de parada

Este ciclo tiene **exactamente dos experimentos nuevos posibles**: la tercera particion y, solo
si confirma, el test de permutacion. No se abren cuartas particiones, ni quintas, ni grids de
pesos, ni nuevos valores de recorte, ni otras C, ni otros umbrales, ni otras representaciones.

## 7. Criterio de adopcion

```text
NO se adopta automaticamente aunque las tres particiones sean positivas.
```

La adopcion exigiria como minimo: tres particiones positivas, permutacion favorable, bootstrap
por grupos compatible, rescue/damage favorable, compliance completo y provenance completo. Si
el bootstrap por grupos sigue incluyendo el cero, la clasificacion es
`PROMISING_UNCONFIRMED` y **no** `ADOPTED`.

## 8. Estados que deben coexistir

```text
ORIGINAL           +0.0082045862, nunca enviado, nunca auditado
REPRODUCED         +0.0082045862, R3
SECOND_PARTITION   +0.0105253130
THIRD_PARTITION    pendiente
PERMUTATION        pendiente
FINAL_STATUS       pendiente
```

Ninguno sustituye a otro.
