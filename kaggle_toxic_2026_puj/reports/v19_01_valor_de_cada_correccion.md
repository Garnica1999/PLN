# V19-01: el valor exacto de cada correccion, y por que la decision se separa

Generado por `experiments/v19_01_valor_de_cada_correccion.py` (V19-01 (P1 sensibilidad, P2 valor marginal y P3 estructura conjunta)).

**Ningun modelo entrenado, ninguna regla construida, ninguna submission gastada.**

## 1. La forma cerrada del valor de una correccion

Para una etiqueta con TP, FP y FN, la F1 es `2*TP/D` con `D = 2*TP + FP + FN`. Hay
exactamente cuatro tipos de cambio de celda, y sus deltas son:

```
FN -> TP   Delta =  2 (TP + FP + FN) / (D (D + 1))
FP -> TN   Delta =  2 TP             / (D (D - 1))
TP -> FN   Delta = -2 (TP + FP + FN) / (D (D + 1))
TN -> FP   Delta = -2 TP             / (D (D - 1))
```

**Los dos valores positivos no son iguales.** El primero escala con TP + FP + FN y el
segundo solo con TP, luego corriger un falso negativo y eliminar un falso positivo valen
distinto. En V18 se contaron rescates y danos como si valieran uno cada uno, que es falso
en el margen.

Las seis formas cerradas se verificaron **recalculando la macro entera** despues de
voltear una celda real de cada tipo, no confiando en la algebra.

## 2. El valor por etiqueta

| etiqueta | TP | FP | FN | TN | F1 | valor de corregir FN | valor de quitar FP | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HS | 505 | 173 | 143 | 662 | 0.761689 | 9.332e-04 | 5.749e-04 | 1.6233 |
| TR | 363 | 82 | 82 | 956 | 0.815730 | 1.329e-03 | 9.176e-04 | 1.4485 |
| AG | 425 | 135 | 142 | 781 | 0.754215 | 1.104e-03 | 6.698e-04 | 1.6488 |

## 3. La distribucion completa sobre las 4449 celdas

Hay **12 valores distintos**, no uno. Cada
celda vale una de esas cantidades y el signo dice si corregirla mejora o empeora:

| ganancia de macro-F1 por celda | cuantas celdas |
| --- | --- |
| -4.430489e-04 | 363 |
| -3.681399e-04 | 425 |
| -3.110552e-04 | 505 |
| -3.058606e-04 | 956 |
| -2.232726e-04 | 781 |
| -1.916199e-04 | 662 |
| +1.916199e-04 | 173 |
| +2.232726e-04 | 135 |
| +3.058606e-04 | 82 |
| +3.110552e-04 | 143 |
| +3.681399e-04 | 142 |
| +4.430489e-04 | 82 |

| tipo de cambio | etiqueta | celdas | ganancia de macro por celda |
| --- | --- | --- | --- |
| FN->TP | TR | 82 | +4.430489e-04 |
| FN->TP | AG | 142 | +3.681399e-04 |
| FN->TP | HS | 143 | +3.110552e-04 |
| FP->TN | TR | 82 | +3.058606e-04 |
| FP->TN | AG | 135 | +2.232726e-04 |
| FP->TN | HS | 173 | +1.916199e-04 |
| TN->FP | HS | 662 | -1.916199e-04 |
| TN->FP | AG | 781 | -2.232726e-04 |
| TN->FP | TR | 956 | -3.058606e-04 |
| TP->FN | HS | 505 | -3.110552e-04 |
| TP->FN | AG | 425 | -3.681399e-04 |
| TP->FN | TR | 363 | -4.430489e-04 |

## 4. La estructura conjunta de HS, TR y AG, medida

Se observaron **6 de los 8 estados** conjuntos
de la verdad, sobre 1483 filas:

| estado (HS,TR,AG) | filas | por ciento |
| --- | --- | --- |
| (0,0,0) | 813 | 54.82 |
| (1,1,1) | 421 | 28.39 |
| (1,0,1) | 124 | 8.36 |
| (1,0,0) | 79 | 5.33 |
| (1,1,0) | 24 | 1.62 |
| (0,0,1) | 22 | 1.48 |

**No se supone ninguna relacion logica entre las tres: se mide.** Los ocho estados
aparecen, luego ninguna etiqueta esta implicada por otra y las tres se comportan como
problemas separados, que es justo lo que predice la separacion del apartado 5.

## 5. El limite que decide la fase

- macro de las tres columnas: 0.7772114525165721 | macro completa: 0.7772114525165721
- son identicas, luego no hay terminos cruzados entre etiquetas
- volteando una celda de HS: F1 de TR antes 0.815730337079 despues 0.815730337079, y la de AG antes 0.754214729370 despues 0.754214729370
- no se mueven, luego el valor de una correccion en HS es independiente de TR y AG
- por tanto, fijados los scores, el problema de decision son tres problemas de F1 independientes, y el optimo de cada uno es un umbral sobre su score
- ninguna matriz de costes entre las tres, ni interaccion, ni regla conjunta puede superar a los tres umbrales, porque la funcion objetivo no tiene terminos cruzados que optimizar

## 6. Conclusion

Dos correcciones que cambian una celda valen distinto, luego el valor no es uniforme y cuenta errores no es medir su valor. Y bajo las tres hipotesis de average que sobreviven a P0 la macro se descompone en la media de las tres F1 sin terminos cruzados, luego el problema de decision son tres problemas de umbral independientes y el optimo es un umbral por etiqueta. Eso deja solo dos palancas: donde se pone el umbral, que V16 midio y cerro, y que informacion contienen los scores.

- Submissions gastadas: **0**.


Fecha: 2026-09-27T03:16:43+00:00. Duracion: 0.0 s.
