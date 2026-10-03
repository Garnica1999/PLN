V03 referencia: corrected_cv = 0.7772 (cacheado 0.7772)

# V24-03b: perfil de celdas de la mejor mezcla kNN, en la unidad del gate

## 0. Constantes medidas del proyecto (no estimadas)
- Sensibilidad del LB: **2.80e-05** de F1 macro en el LB por acierto neto
- Verificacion cruzada: 107 cambios netos mueven el LB en 0,003 -> 2.80e-05 (coincide)
- Error estandar teorico del LB con 1154 filas: **0.01219** = 435 cambios netos de acierto
- Filtro escrito en el registro: toda candidata que cambie menos de unas 200 celdas
  de la referencia es indetectable con este presupuesto

## 1. Perfil de celdas de la mezcla kNN a distintos pesos

| peso kNN | celdas cambiadas | rescates | danos | neto | dOOF (F1) | dLB implicito | % de 1 SE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 26 | 16 | 10 | +6 | +0.0000 | +0.00017 | 1.4 % |
| 0.20 | 56 | 33 | 23 | +10 | +0.0002 | +0.00028 | 2.3 % |
| 0.30 | 103 | 66 | 37 | +29 | +0.0050 | +0.00081 | 6.7 % |
| 0.40 | 146 | 88 | 58 | +30 | +0.0041 | +0.00084 | 6.9 % |
| 0.50 | 190 | 103 | 87 | +16 | -0.0015 | +0.00045 | 3.7 % |

## 2. La candidata optima de esta sonda

- Peso kNN 0.30: **103 celdas cambiadas**, 66 rescates, 37 danos, **neto +29**
- dOOF = **+0.0050**. Umbral de gate dOOF = +0.03756. Cumple el 13 % del umbral
- dLB implicito por la sensibilidad del proyecto = **+0.00081**
- Eso es **6.7 % de un error estandar del LB**. Para mover el LB un SE hacen falta 435 celdas netas

## 3. Contraste con las sondas ya enviadas

| sonda | celdas | neto | dLB real | veredicto |
|---|---:|---:|---:|---|
| v02 -> v03 (umbrales) | 126 | +126 | +0.00313 | la unica que mejoro |
| v23 SVD+RBF | 240 | +32 | -0.00368 | mejor evidencia interna, LB cae |
| v20 interaccion HS | 51 | -14 | -0.00643 | 25 rescates / 39 danos en OOF |
| v09_p5 C por etiqueta | ~ | -6,6 | -0.00525 | tasa de acierto 0,5185 |
| v12 promedio repeticiones | ~ | -14,9 | -0.01184 | la mayor caida |
| **kNN union w=0.30 (esta sonda)** | **103** | **+29** | **+0.00081 (implicito)** | **no enviado** |

## 4. Veredicto del addendum

La mejor formulacion de kNN que existe cambia 103 celdas y produce +29 netas.
Eso es un 6.7 % del error estandar del LB: la submission seria
estadisticamente indistinguible de V03 antes incluso de poder llamarse una mejora.

En el eje de celdas, la sonda kNN se situa entre dos precedentes ya medidos:

- **v20 interaccion HS**: 51 celdas, neto -14 en OOF, LB -0.00643. Es la sonda mas
  pequena que se ha enviado, y ya perdio.
- **v23 SVD + RBF**: 240 celdas, neto +32, el mejor conjunto de evidencia interna del
  proyecto (5/5 pliegues, McNemar p = 0,0452), y LB -0,00368. Es el precedente mas
  cercano a kNN en tamano de cambio y en perfil, y ya perdio.

kNN queda por debajo de V23 en celdas netas y por encima en celdas cambiadas, es decir
cambia mucho mas para conseguir menos. Ese es el perfil de una sonda marginal, no de una
candidata competitiva.

Completado en 1.5 s
