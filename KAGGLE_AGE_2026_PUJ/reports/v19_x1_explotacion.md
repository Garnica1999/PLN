# V19 · X1 — Explotacion del champion: puerta previa y las dos configuraciones

El especialista recomienda **cerrar la explotacion de V19**. Sus tres argumentos se resumen antes que los resultados, porque condicionan si estos dos experimentos pueden decidir algo:

1. **La tasa base del subespacio de parametros es +0,0008** frente a un gate de +0,0030. Las 14 formulaciones cerradas de V15 a V18 se movieron en [-0,0040, +0,0008]. Todo lo que alguna vez movio la aguja fue un **cambio de mecanismo**, nunca un parametro.
2. **La resolucion del aparato es de +0,0011**: 0,6090431 en 4 folds frente a 0,6079388 en 5, **sin cambiar nada del modelo**. Un ajuste con efecto real menor que +0,001 no es distinguible de elegir particion.
3. **El residuo de la calibracion sobre V19 es de -0,0016** (CV 0,6079388 predecía 0,62735 y el LB real es 0,62578). En V12 el residuo fue +0,0019, luego la oscilacion entre modelos vecinos es de 0,0034: un +0,002 en CV puede aterrizar en cualquier punto entre -0,000 y +0,004 en LB.

## 1. La puerta de coste cero: la saturacion se sostiene

| Medida | Valor |
|---|---|
| arboles del champion | 1800 |
| cortes internos medios | 6.8222 |
| techo de cortes por arbol | 7 |
| **ocupacion del techo** | **97.5%** |
| arboles que llegan al techo | 1519 de 1800 |
| arboles con menos de 5 cortes | 7 |
| la tesis se sostiene (umbral 95 %) | **True** |

Con `num_leaves = 8` un arbol puede como maximo 7 cortes, y la Depth 3 permite exactamente 1800 de ellos. La ocupacion real es del 97.5%, luego **la profundidad es lo que limita y no las hojas**, que es la premisa de la configuracion A. Si hubiera sido del 70 %, la lectura cambiaria y el eje de capacidad reabriria.

## 2. Las dos configuraciones y nada mas

El especialista delimito con numeros el resto y las descarto por inertes o ya sondadas: `num_leaves` 8 a 16 con profundidad 3 es un **no-op exacto** porque 2^3 = 8; `reg_lambda` 1 a 10 es inerte a esta escala de hojas; `min_child_weight` es una reparametrizacion de `min_child_samples` porque la hessiana multiclase por fila es 0,5-0,8; y bajar `colsample` mata el bloque de interaccion. Los pesos del blend se sondearon dos veces (V16 A1 -0,0003 y A2 +0,0006 con el IC95 contemplando el cero).

| Configuracion | CV 5 folds | Delta contra V19 | Folds pos. | IC95 contra V19 | CV 4 folds | Delta 4 folds | Merece submission |
|---|---|---|---|---|---|---|---|
| **V19 (champion)** | 0.6079388 | - | - | - | 0.6090431 | - | - |
| A profundidad max_depth 4 con num_leaves 8 | 0.6079208 | -0.0000 | 3/5 | [-0.0004, +0.0004] | 0.6090404 | -0.0000 | **no** |
| B regularizacion min_child_samples 800 | 0.6076698 | -0.0003 | 0/5 | [-0.0006, +0.0001] | 0.6088761 | -0.0002 | **no** |

## 3. Fronteras y complementariedad de errores frente a V19

| Configuracion | Frontera 1 | Frontera 1 de V19 | Frontera 2 | Frontera 2 de V19 | Solo el candidato acierta | Solo V19 acierta | Neto |
|---|---|---|---|---|---|---|---|
| A profundidad max_depth 4 con num_leaves 8 | 0.5035 | 0.5038 | 0.5013 | 0.5013 | 620 | 617 | +3 |
| B regularizacion min_child_samples 800 | 0.5038 | 0.5038 | 0.5012 | 0.5013 | 455 | 505 | -50 |

## 4. Criterio de submission, aplicado sin abreviar

| Criterio | A | B |
|---|---|---|
| delta_5_folds | no | no |
| cinco_de_cinco_folds | no | no |
| ic95_excluye_cero | no | no |
| confirmacion_4_folds | no | no |

**Ninguna configuracion merece submission.** 0 de 2 pasan los cuatro criterios.
