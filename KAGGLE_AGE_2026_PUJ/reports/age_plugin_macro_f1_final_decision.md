# Plug-in por clase para macro-F1: decision final

**Fecha:** 2026-10-11
**Pre-registro:** commit `bd08aba` (los pesos y el criterio se fijaron antes de ejecutar)
**Veredicto: `PROMISING_CONFIRMED`**

## Que es el candidato

```text
argmax_c( w_c * p_c )   con   w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]
```

No cambia el modelo. Cambia **como se decide** con el modelo que ya existe.

## Las dos particiones

| Particion | V30 | plug-in | delta | folds+ | IC95 | alta conf. cambiada |
|---|---|---|---|---|---|---|
| canonica (semilla 42) | 0.6134471 | 0.6154643 | **+0.0020172** | 5/5 | [+0.0010143, +0.0029480] | **0** |
| alternativa (semilla 987654) | 0.6144120 | 0.6164700 | **+0.0020580** | 5/5 | [+0.0011204, +0.0030622] | **0** |

El delta **reproduce casi exactamente** entre las dos particiones, y en las dos los controles favorables se cumplen: 5/5 folds, IC95 sin cero, y **cero** cambios en filas de alta confianza.

## El diagnostico que aisla el mecanismo

Los pesos re-derivados **dentro** de la particion alternativa dan **+0,0022884**. Es el test limpio: el mecanismo aparece solo, sin depender de los pesos concretos ni de la particion.

## El matiz que se declara

**El control `permutado` cambio de signo entre particiones**, y eso hay que decirlo:

| Particion | permutado | real - permutado |
|---|---|---|
| canonica | -0.0027902 | +0.0048074 |
| alternativa | +0.0014417 | +0.0006163 |

La lectura honesta: **parte del efecto es generica y parte es especifica**. Cualquier perturbacion del reparto que aumente la diversidad de las clases predichas ayuda algo a macro-F1, y el permutado hereda esa parte generica cuando le toca subir una clase infra-predicha. La asignacion correcta de cada multiplicador a su clase anade aproximadamente **+0,0006** sobre eso. El permutado, por tanto, **no es un control negativo perfecto**: lo que aisla es la diferencia, no el nivel.

Los otros dos controles si son limpios: el aleatorio se hunde (`-0,0161893` en la alternativa, con la misma forma de pesos) y `congelado - nested = -0,0002304`, es decir que los pesos congelados **no estan inflados**.

## La fuga que se declaro en el pre-registro, y como queda

La preocupacion era real: los pesos congelados se derivaron de la union de cinco ajustes que cubren las 169.336 filas, asi que no son fuera de muestra para esas filas. En la canonica puntuaban **por encima** de los anidados (+0,00254 frente a +0,00202), que es la firma de haber visto los datos.

**En la particion alternativa eso se invierte**: congelado `+0,0020580` frente a nested `+0,0022884`, o sea `-0,0002304`. El efecto **no se infla** bajo un reparto nuevo, y eso desactiva la preocupacion.

## La prediccion falsable, confirmada por segunda vez

P34 establecio el criterio: una mejora que solo toca la frontera es de calibracion. Aqui la prediccion era **falsable** y se cumplio en las dos particiones: **cero** cambios en filas de alta confianza. Las 11.225 filas que cambian tienen confianza media **0,3357** frente a **0,6136** del conjunto.

Un multiplicador por clase **no puede** alterar el orden cuando el margen es grande. Que no lo haga es la prueba de que esto es un cambio de **regla de decision** y no de calibracion.

## La clasificacion, con las tres respuestas separadas

| Pregunta | Respuesta |
|---|---|
| ¿Es una mejora de representacion? | **NO** |
| ¿Es una mejora del modelo? | **NO** |
| ¿Es una mejora de la regla de decision? | **SI** |

La precision importa: el candidato sube macro-F1 **bajando accuracy** de `0,61464` a `0,61254`. Sacrifica aciertos para subir el recall de las clases debiles, que es exactamente lo que macro-F1 premia y lo que `argmax` no hace.

## Estado

**`PROMISING_CONFIRMED`**, no `FINALIST` y no `ADOPTED`. Lo que falta para mas:

- El delta esta por encima del gate de PROMISING (+0,0015) y **por debajo del de FINALIST** (+0,003).
- La particion alternativa usa los **mismos grupos** repartidos de otra forma, asi que es una reejecucion y no una particion independiente. Ese limite viene de que `dup_clusters` cubre el dataset entero, y esta declarado desde P00.
- No se ha generado ninguna submission, y el leaderboard no se ha consultado.

## Lo que no se hace

No se reajustan los pesos. No se busca otra rejilla. No se elige otra semilla. No se ejecuta una tercera particion. No se usa el test —ni para contar filas. No se usa el holdout. No se consulta el leaderboard. No se proyecta ningun score.

## La correccion al registro historico

`V20-F3-UMBRALES` quedo etiquetado en el gap analysis como *"correccion de umbral"*, y **no lo era**: fue una **cadena de cinco discriminadores binarios anidados con monotonia**, es decir una arquitectura ordinal. La regla de decision post-hoc no se habia probado. Los informes historicos no se modifican; la correccion vive aqui.
