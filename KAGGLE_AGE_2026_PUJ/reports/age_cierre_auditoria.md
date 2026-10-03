# AGE · Auditoria de cierre estrategico

- **Champion**: V12, macro-F1 pooled **0.6005017**, leaderboard **0.62188**.
- **Este script no entrena nada.** Todas las cifras salen de artefactos que ya existen.
- **No se ejecuta ninguna sonda ni variante.**

## 1. Masa de cada clase, y la frase de D-049

| Clase | Filas | % de la masa | F1 de V12 |
|---|---|---|---|
| 13-17 | 28,863 | 17.04 | 0.671 |
| 18-24 | 28,676 | 16.93 | 0.5059 |
| 25-34 | 28,347 | 16.74 | 0.4858 |
| 35-49 | 28,230 | 16.67 | 0.5593 |
| 50-64 | 28,060 | 16.57 | 0.6549 |
| 65-xx | 27,160 | 16.04 | 0.7261 |

`18-24` y `25-34` suman **33.67 %** de la masa. D-049 cerro V18 con la frase de que la frontera es una **minoria de la masa**, y leida de forma literal **es falsa**: no es una minoria, es algo mas de un tercio de las filas. La conclusion de D-049 puede seguir siendo correcta, pero **su razon no lo es** y hay que corregirla.

## 2. Cuanto vale de verdad la lever, con un contrafactual sobre el OOF existente

Se corrigen `n` filas mal clasificadas de cada par a su etiqueta verdadera, que es lo que haria un selector perfecto, y se recalcula la macro-F1 pooled. Es una cuenta sobre el OOF de V12: no hay reentrenamiento ni modelo nuevo. El rango es la dispersion de 200 remuestreos.

| Frontera | Filas del par | Errores de V12 | Accuracy | 500 | 1.000 | 2.000 | 5.000 |
|---|---|---|---|---|---|---|---|
| 18-24\|25-34 | 57,023 | 28,721 | 0.4963 | +0.0015 | +0.0031 | +0.0061 | +0.0122 | +0.0303 |
| 25-34\|35-49 | 56,577 | 28,438 | 0.4974 | +0.0016 | +0.0031 | +0.0063 | +0.0125 | +0.0308 |

**El gate de adopcion de 0.003 se cruza con unas 478 correcciones netas** en la mejor frontera. Es decir: la lever **no** es pequena, y el cierre de V18 no puede sostenerse en la magnitud. Solo puede sostenerse en que **no existe un selector** que identifique esas filas, lo que es exactamente lo que midieron los cinco selectores de V17 (compuerta +0,0002, regla conjunta +0,0000, arbitro aprendido -0,0001 con 0 de 5 folds, combinador local -0,0002) y la pagina de confianza de V16, que con 90 puertas no desvia ni una fila.

## 3. Calibracion CV -> LB con los envios realmente puntuados

- `LB = +0.00223 + 1.02825 x CV`, **R2 = 0.9968**, desviacion del residuo 0.00147, con **n = 4 envios** cuyos dos extremos son verificables en disco. El LB de los otros 7 envios puntuados si esta registrado, pero su CV no esta en ningun artefacto del repositorio, asi que no entran en la recta.
- Un gate de **0.003** en CV vale **+0.00308** en LB, y llevaria el resultado a **0.62278**.
- La pendiente es **1.028**, luego **el gap no da puntos gratis**: un gate local no esta ni mas cerrado ni mas abierto de lo que parece en terminos de leaderboard. El gap es un **offset aditivo estable**, entre +0.01716 y +0.02132, con desviacion 0.00164. La preocupacion correcta es la contraria a la habitual, y es la que sostiene el gate: filtra candidatos de +0,001 que no mejorarian nada.

## 4. Inventario de lo que no esta en las 279 columnas

Los 62 artefactos de train de `features/` se auditaron uno a uno. **Todos estan consumidos o cerrados con numero**: los especialistas por longitud (+0,0000), BM25 (-0,0002), kNN (0,3088 sin aporte), los grupos de 3 columnas (<= 0), `content_only` y `style_only` (PH6-L3), los residuales de longitud (PH5-L2, -0,0019), las conjunciones (0,1489) y la modalidad (0,1270), ambas **por debajo del suelo de azar de 0,1667**, los 72 bloques lexicograficos (D-039, **0 de 72 no redundantes**), el bloque linguistico de V14 (D-044), los tokens de V13 (D-041), los pronombres de V17 (D-047) y las tres vistas de margenes de la capa ordinal (D-037).

**No queda ningun artefacto sin explotar.** Y la hipotesis de compresion de outputs **muere por construccion**: todos los OOF de miembros estan ya comprimidos a seis columnas o menos, porque cada miembro es un modelo de seis clases y no puede emitir mas. La compresion no la introduce V12, se introdujo al construir cada miembro.

## 5. Lo que la auditoria no puede resolver

La sonda de prior-shift que el especialista menciona no dejo **ningun artefacto** en el proyecto, luego su +0,0010 no es verificable contra disco y no puede usarse como evidencia. Aun tomandolo como cierto, queda tres veces por debajo del gate.
