# Transferencia externa del plug-in por clase

## El resultado

```text
V30      leaderboard = 0.63302
plug-in  leaderboard = 0.63575
delta                 = +0.00273
```

**`POSITIVE_EXTERNAL_TRANSFER`.** La mejora medida internamente se transfirio al test.

## La comparacion, con las tres columnas

| Metodo | CV canonico | CV alternativo | Leaderboard | delta CV can. | delta CV alt. | delta LB |
|---|---|---|---|---|---|---|
| V30 | 0.6134471 | 0.6143347 | 0.63302 | +0.0000000 | +0.0000000 | +0.00000 |
| plug-in | 0.6154643 | 0.61647 | 0.63575 | +0.0020172 | +0.0020580 | +0.00273 |

**Las tres direcciones coinciden.** La validacion agrupada y la transferencia externa apuntan al mismo lado, que es lo que hace util el resultado.

### Dos referencias legitimas para el delta alternativo

No son el mismo numero y conviene no confundirlas:

| Referencia de V30 en la particion alternativa | Valor | delta del plug-in |
|---|---|---|
| lo que reporto V30-05 en su fase | 0.6143347 | +0.0021353 |
| lo que midio la reejecucion confirmatoria, en la misma sesion | 0.614412 | **+0.0020580** |

La diferencia entre las dos referencias es **0,0000773**, y viene de que LightGBM no es reproducible bit a bit entre ejecuciones con distinta carga de CPU: el proyecto ya lo documento en V30. **La comparacion internamente consistente es la segunda**, porque el plug-in se midio en la misma sesion que esa reejecucion. La tabla de arriba usa esa.

## El contraste con la calibracion del proyecto

Con `LB = 0,00223 + 1,02825 x CV`, la calibracion que el propio proyecto ajusto en fases tempranas:

| | CV | LB predicho | LB real | error |
|---|---|---|---|---|
| V30 | 0.6134471 | 0.63301 | 0.63302 | **+0.00001** |
| plug-in (nested) | 0.6154643 | 0.63508 | 0.63575 | **+0.00067** |
| plug-in (global) | 0.6159732 | 0.6356 | 0.63575 | **+0.00015** |

El leaderboard del plug-in casa con el **CV global** (error `+0,00015`) mucho mejor que con el nested (`+0,00067`). Y el global es exactamente lo que la submission ejecuta: los pesos congelados aplicados de forma global.

Eso importa porque en el ciclo anterior hubo que **corregir un error mio** de clasificacion: se estaban confundiendo tres numeros del interno y el sanity check **paro la construccion** antes de generar nada. La transferencia confirma que aquella correccion no era cosmetica, sino la descripcion correcta de la operacion.

## Filas modificadas, como analisis descriptivo

```text
filas que cambian respecto de argmax puro   7.408   (6,56 %)
confianza media de las que cambian            0,3361
confianza media del conjunto                   0,6247
cambios con confianza >= 0,60                 0
```

**Cero cambios en alta confianza.** Es la prediccion falsable que P34 establecio y que se cumplio en las dos particiones internas **y tambien en test**: un multiplicador por clase no puede alterar el orden cuando el margen es grande.

**Este numero es descriptivo y no se usa para predecir el leaderboard.** El 0,63575 entra como evidencia final, no como funcion objetivo.

## El cambio de distribucion, que es el mecanismo

```text
25-34   ~0,152  ->  0,1786
65-xx             ->  0,1532
```

La regla redistribuye decisiones hacia las clases con peor F1. 25-34 era la peor (F1 0,4992) y la mas infra-predicha; 65-xx era la mejor (F1 0,7351) y estaba sobre-predicha. Los valores se documentan como estan y **no se reajustan**.

## Alcance

La submission es `submissions/age_submission_plugin_candidate.csv`, SHA256 `29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0`, y **queda sin sobrescribir**. Este analisis **no se extrapola** a otras competiciones ni datasets: la afirmacion vale para este problema y esta configuracion de probabilidades.
