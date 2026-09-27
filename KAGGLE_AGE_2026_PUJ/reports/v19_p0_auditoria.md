# V19 · P0-P1-P2 — Auditoria exacta del champion

No se ha entrenado ninguna configuracion nueva. Todo sale del champion reajustado con su misma semilla.

## P0. Que es exactamente V19

| Parametro | Valor |
|---|---|
| `objective` | 'multiclass' |
| `num_class` | 6 |
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `num_leaves` | 8 |
| `max_depth` | 3 |
| `min_child_samples` | 200 |
| `colsample_bytree` | 0.5 |
| `subsample` | 0.8 |
| `subsample_freq` | 1 |
| `reg_lambda` | 1.0 |
| `n_jobs` | 32 |
| `random_state` | 42 |
| `verbose` | -1 |
| combinacion | LogReg 0.3333 + MLP(64) 0.3333 + arbol 0.3333 |
| semilla del arbol | 42 |
| folds | 5 grupos, esquema `group`, 1 repeticion, sobre `dup_clusters` |
| columnas | 279, float32, sin seleccion |
| target | las seis clases verdaderas; las 279 variables son features auxiliares |
| identificica al artefacto JSON | True |

### Atribucion por bloque, con la mejora mas grande de la que hay registro en el proyecto

| Bloque | Columnas | macro-F1 |
|---|---|---|
| las 279 completas | 279 | **0.6072312** |
| sin el bloque label-informed | 200 | 0.586956 |
| solo el bloque label-informed | 73 | 0.5597155 |
| solo las 78 de miembros | 78 | 0.5842464 |

- Anadir las 73 a las 206 aporta **+0.0203**
- Anadir las 206 a las 73 aporta **+0.0475**
- Es **interaccion y no suma**: True. El bloque de 73 aportacion en solitario (0,5597) queda por debajo de las 206 sin el (0,5870), pero juntas suman mas que la suma de las dos.

## P1. Donde esta la senal

| Medida | Valor |
|---|---|
| cuota de ganancia del bloque de 73 | 9.1% |
| cuota de ganancia del resto | 90.9% |
| cortes del bloque de 73 | 3,508 |
| cortes del resto | 8,772 |
| columnas con al menos un corte | 271 de 279 |
| top 10 concentra | 39.3% de la ganancia |
| top 20 concentra | 59.9% de la ganancia |
| hay una variable dominante | True |

## P2. Complejidad y margen antes de destruir la transferencia

| Combinador | CV | Hueco train a OOF | Arboles | Hojas | Bytes | Cambios | Rescate | Dano | Neto |
|---|---|---|---|---|---|---|---|---|---|
| V12 (champion anterior) | 0.6005017473 | - | 0 | 0 | - | 0 | 0 | 0 | +0 |
| V16 A2 apilamiento logistico | 0.6011 | - | 0 | 0 | - | - | - | - | - |
| V18 P4 multi-objetivo | 0.5993 | - | 0 | 0 | - | - | - | - | - |
| V19 boosting (champion actual) | 0.6079388 | 0.007781 | 1800 | 14080 | 1841063 | 12,169 | 4,622 | 3,311 | +1,311 |

- **Hueco train a OOF del booster: +0.007781**. Es la magnitud que decide si subir la capacidad mejora el leaderboard o lo destruye: un hueco grande significa que el arbol memoriza mas de lo que generaliza.
- El champion cambia **12,169 filas** del train frente a V12, con rescate 4,622, dano 3,311 y neto **+1,311**.
- Fronteras: 0.5038 y 0.5013, frente a 0.4963 y 0.4974 de V12.
- Modelo persistido en `v19_champion_lightgbm.txt` con SHA256 `22221d1802c1ec4495f393f27f884b9119763a30d47072a0a1205ad671c50619`.
