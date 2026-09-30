# Campeon final de AGE

```text
V30     = HISTORICAL_FROZEN_BASELINE   LB = 0.63302
PLUG-IN = ADOPTED                      LB = 0.63575
plug-in delta LB = +0.00273

research_status   = CLOSED
candidate_status  = ADOPTED
external_transfer = POSITIVE_TRANSFER
```

## 1. El problema detectado

Treinta fases buscaron **como aprender** mejor: representaciones, familias de modelo, objetivos, estructura de etiquetas, integracion en score-space. La auditoria de cobertura cerro 39 mecanismos con evidencia, y **ninguno** produjo una mejora robusta.

Lo que faltaba era otra cosa: **como decidir** con lo aprendido. Las seis clases de este problema son **perfectamente balanceadas** (0,160 a 0,170) y la prediccion del ensemble **no lo estaba** (0,146 a 0,187). `argmax` maximiza accuracy, no macro-F1, y cuando el reparto predicho se desvia hacia las clases faciles hay margen que `argmax` no aprovecha.

## 2. La hipotesis

> `argmax_c(w_c * p_c)` con pesos por clase puede subir macro-F1 sin tocar el modelo.

## 3. La derivacion de los pesos

Por cada fold `k` de la particion canonica se resolvio, por ascenso por coordenadas en log-espacio sobre los otros cuatro folds:

```text
w_k = argmax_w macro_F1( y[idx_k], argmax_c( w_c * p_c[idx_k] ) )
idx_k = { i : fold(i) != k }
```

La clase 0 se ancla en 1,0 porque multiplicar todos los pesos por una constante no cambia ningun `argmax`: solo hay 5 grados de libertad reales. Los pesos adoptados son **la media aritmetica de los cinco**:

```text
w = [1,0000, 1,1749, 1,2721, 1,1630, 1,0010, 0,8128]
    13-17   18-24   25-34   35-49   50-64   65-xx
```

## 4. La validacion canonica

```text
V30      0,6134471
plug-in  0,6154643   delta +0,0020172   5/5 folds
IC95 bootstrap  [+0,0010143, +0,0029480]   sin cero
```

## 5. La validacion alternativa

```text
V30      0,6144120
plug-in  0,6164700   delta +0,0020580   5/5 folds
IC95 bootstrap  [+0,0011204, +0,0030622]   sin cero
```

**Diferencia entre las dos particiones: 0,0000408.** Y el diagnostico `nested` —pesos re-derivados dentro de la particion alternativa, que es el test limpio del mecanismo— da **+0,0022884** con 5/5: el mecanismo aparece solo.

## 6. El bootstrap

Pareado, 2.000 remuestreos, unidad la fila, semilla 20261010 fijada en el pre-registro (`bd08aba`) y **no cambiada** para mejorar el intervalo. En las dos particiones el IC95 excluye el cero.

## 7. Los sanity checks

1. **Orden de clases demostrado, no asumido**: sobre el OOF, para cada clase real `i` la columna `i` es la que mas se activa. Diagonal dominante en las seis filas. Si no cayera, el script no genera submission.
2. **Reproduccion del interno antes de tocar el test**: el script paro la construccion al detectar que se comparaba contra el numero equivocado de los tres posibles (nested, global, redondeado).
3. **El pipeline es el de V30, no uno reconstruido**: se **importan** las funciones de `v30_03_submission.py` en vez de reimplementarlas.
4. **Cero cambios en alta confianza** en las dos particiones y en test.

## 8. La submission

```text
submissions/age_submission_plugin_candidate.csv
SHA256 29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0
112.892 filas | columnas id, age_range | 10/10 checks
```

El fichero de V30 conserva su hash `f543a715a1881c61` y **no se ha sobrescrito**.

## 9. El leaderboard

```text
V30      0.63302
plug-in  0.63575
delta    +0.00273
```

## 10. La transferencia

**`POSITIVE_EXTERNAL_TRANSFER`.** Las tres medidas —CV canonico, CV alternativo y leaderboard— apuntan en la misma direccion.

Y un contraste que conviene registrar: con la calibracion del proyecto, el leaderboard casa con el **CV global** (error +0,00015) mucho mejor que con el nested (+0,00067), que es exactamente lo que la submission ejecuta.

## 11. La decision

**`ADOPTED`.** Y la conclusion, con su alcance exacto:

> La mejora **no** proviene de una representacion nueva ni de un modelo nuevo. Proviene de la **regla de decision** aplicada sobre las probabilidades del ensemble V30.

**No se afirma** que V30 fuera incorrecto, ni que `argmax` no sea optimo en general. La afirmacion es acotada: para **este problema y esta configuracion de probabilidades**, `argmax(w * p)` produjo una mejora reproducible y transferible en macro-F1.

## 12. Lo que no se hace a partir de aqui

No se usa el 0,63575 para optimizar pesos, crear variantes, seleccionar semillas ni buscar modelos. No se genera otra submission. No se reabre P33, P34, G12, G14 ni la revision bibliografica salvo evidencia externa nueva o cambio de reglas.

## 13. La correccion historica que sigue vigente

`V20-F3-UMBRALES` fue una **arquitectura ordinal** —una cadena de cinco discriminadores binarios anidados con monotonia— y **no** una evaluacion de regla de decision post-hoc. La familia de reglas de decision **no habia sido auditada correctamente** antes de esta fase, y ese error de clasificacion contribuyo a que la linea pareciera cerrada. Los informes historicos no se han modificado.

## 14. El camino, en una linea

39 mecanismos de **aprendizaje** auditados y cerrados, y el resultado vino de un mecanismo de **decision**. La auditoria era valida; su enumeracion estaba incompleta.
