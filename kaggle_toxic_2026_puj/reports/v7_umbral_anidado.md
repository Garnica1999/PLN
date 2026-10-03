# Track P0.1: umbral anidado con protocolo estricto (V7-03)

Generado por `experiments/v7_03_umbral_anidado.py` (V7-03 (P0.1)).

## 1. Que resuelve este script

La fase v6 dejo una tarea abierta: **reajustar umbrales sobre el mismo conjunto
fuera de pliegue que despues se evalua infla la metrica en +0.0036 sin cambiar el
modelo**. Ese sesgo no se puede medir con un protocolo de dos niveles. Aqui hay tres
niveles que no se mezclan:

```text
ajuste       el modelo se ajusta solo en el ajuste interior
calibracion  el umbral se ajusta solo en la calibracion interior
evaluacion   se aplica una sola vez sobre la validacion exterior
```

E2-16 ya habia medido un umbral global anidado y dio
0.776613, pero con un protocolo mas laxo: reutilizaba las
probabilidades fuera de pliegue de los otros cuatro pliegues como conjunto de
calibracion, de modo que el umbral veia mas datos de los que aqui se le permiten.

## 2. Informe de dispersion de umbrales (el producto principal)

Quince umbrales: cinco pliegues exteriores por tres etiquetas. Esta es la pregunta
que la especificacion hace de forma explicita, respondida con numeros.

| etiqueta | valores por pliegue | media | mediana | desviacion | rango intercuartil | rango | congelado V03 | dentro del rango |
|---|---|---|---|---|---|---|---|---|
| HS | 0.41, 0.47, 0.43, 0.38, 0.37 | 0.4120 | 0.4100 | 0.0402 | 0.0500 | 0.1000 | 0.44 | si |
| TR | 0.45, 0.50, 0.45, 0.45, 0.60 | 0.4900 | 0.4500 | 0.0652 | 0.0500 | 0.1500 | 0.51 | si |
| AG | 0.49, 0.44, 0.40, 0.54, 0.41 | 0.4560 | 0.4400 | 0.0586 | 0.0800 | 0.1400 | 0.46 | si |

## 3. Resultados: tres brazos para no confundir codigo con protocolo

Hay que separar **si el codigo esta bien** de **cuanto cuesta el protocolo**, porque
confundirlos produce una lectura falsa en las dos direcciones.

| variante | corrected_cv | delta frente a su referencia | IC95 | HS | TR | AG |
|---|---|---|---|---|---|---|
| P0-0-referencia-completa | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 |
| P0-1-control-frozen | 0.769123 | -0.0081 | [-0.0170, +0.0006] | 0.7596 | 0.8050 | 0.7429 |
| P0-1-anidado | 0.767228 | -0.0019 | [-0.0092, +0.0057] | 0.7631 | 0.8116 | 0.7270 |

- **P0-0** ajusta sobre todo el entrenamiento exterior y es el brazo que **valida el
  codigo**: +0.000000 frente a 0.777211, tolerancia 0.0020, veredicto **dentro**.
- **P0-1-control** usa el pipeline de tres niveles con umbrales congelados: 0.769123,delta -0.008089 frente a la referencia. Ese
  delta **no es un error**: es el **coste del protocolo**, que obliga a ajustar sobre
  cinco sextos de las filas. Se reporta por eso, y por eso el control de tres niveles
  no se compara nunca con V03.
- **P0-1-anidado** comparte conjunto de ajuste con el control, asi que el coste se
  cancela y el delta -0.0019 es **efecto del umbral**. IC95 [-0.0092, +0.0057], 2 de 5 pliegues a favor.
- Referencia de E2-16 con protocolo mas laxo: 0.776613.

La primera version de este script fijaba una tolerancia sobre el control de tres
niveles y se detuvo al superarla. **La deteccion era correcta y la tolerancia estaba
mal planteada**: el error no era del codigo sino del protocolo. Anadir el brazo P0-0
separa las dos cosas. Es el mismo error de interpretacion que la fase v6 cometio al
tratar la reoptimizacion de umbrales como una mejora.

## 4. Cuanto esta determinada la eleccion del umbral

La columna de holgura del fichero `reports/v7_calibration_analysis.csv` mide la
diferencia de F1 entre el mejor umbral y el segundo mejor en el mismo conjunto de
calibracion. Si esa diferencia es pequena, la eleccion es casi arbitraria y el umbral
no esta identificado por los datos.

| etiqueta | holgura media entre el primer y el segundo umbral |
|---|---|
| HS | 0.002844 |
| TR | 0.001383 |
| AG | 0.003517 |

Fecha: 2026-09-26T02:16:14+00:00. Duracion: 10.7 s.
