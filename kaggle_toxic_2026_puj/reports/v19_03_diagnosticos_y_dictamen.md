# V19: analisis de metrica y score engineering

Generado por `experiments/v19_03_diagnosticos_y_dictamen.py` (V19-03 (P8 diagnosticos de test, P9 curva historica score-LB e informe final)).

**NO VALID SCORE IMPROVEMENT**

## 1. Champion

| version | LB | delta LB |
| --- | ---: | ---: |
| V03 | 0.78306 | +0.00000 |

Sin cambios. Ninguna submission se ha enviado en esta fase.

## 2. Analisis de la metrica

### La formula, verificada por dos vias

- Scorer: kernel oficial de Kaggle metric/f1-score, sin modificar.
- `zero_division`: 0.0, el default de sklearn, no nan ni omision de etiqueta o fila.
- `average="binary"`: imposible, lanza ValueError sobre entrada multilabel-indicator.
- La media de las tres F1 positivas coincide exactamente con `f1_score(average='macro')` de
  sklearn sobre entrada con tres o mas columnas.
- Casos sinteticos: **28** comprobaciones.
- Contrastes contra sklearn: **980**, con **0 discrepancias**.
- `corrected_cv` de V03 reproducido: **0.7772114525165721**.

### La sensibilidad por celda

Con D = 2*TP + FP + FN:

```
FN -> TP   Delta =  2 (TP + FP + FN) / (D (D + 1))
FP -> TN   Delta =  2 TP             / (D (D - 1))
```

En el OOF de V03 hay **12 valores distintos** de
ganancia por celda, no uno. Corregir un falso negativo y eliminar un falso positivo valen
distinto, luego contar errores no es medir su valor.

### La estructura conjunta

**TR=1 implica HS=1**: no hay ninguna de las 1483 filas con TR=1 y HS=0, cuando por
independencia se esperarian 250.6. Y es de una sola direccion, porque AG=1 con HS=0 si
ocurre, en 22 filas. Es un hecho medido, no una intuicion sobre el significado de las etiquetas.

## 3. Experimentos

| experimento | mecanismo | OOF | delta OOF | netos | LB | estado |
| --- | --- | --- | --- | --- | --- | --- |
| V19-00 metrica verificada | reconstruccion de la metrica y reduc | 0.777211 | +0.000000 | +0 | no enviada | diagnostico, fija el objetivo de la fase |
| V19-01 valor de cada correccion | formas cerradas del delta de F1 y es | 0.777211 | +0.000000 | +0 | no enviada | diagnostico, demuestra que la decision se separa |
| V19-02 restriccion TR implica HS | propagar HS=1 donde el score de TR s | 0.777874 | +0.000663 | -3 | no enviada | REGLA DESCARTADA |

Ninguna fila tiene LB porque no se envio ninguna submission.

La regla de V19-02 mejora el OOF en +0.000663 de
media, pero con desviacion 0.002368,
3 de 5 pliegues
mejorando, valor p empirico 0.7944 y
**-3 cambios netos** de celda frente a los
27 que haria falta para mover el leaderboard.
Un cambio neto negativo no puede mover el LB con la sensibilidad del proyecto, y por eso no
se gasto una submission en averiguarlo.

## 4. Submissions

**Ninguna.** La regla de la fase era no gastar sin evidencia competitiva, y la evidencia no
llego: la regla nueva aporta un numero negativo de cambios netos de celda.

## 5. Por que no hay mejora

- la funcion objetivo no tiene terminos cruzados entre etiquetas, luego la decision se separa en tres problemas de umbral y el optimo es un umbral por etiqueta. Toda la maquinaria de decision conjunta se queda sin justificacion
- el unico hecho de estructura real es que TR implica HS, y la regla que de ahi sale es genuinamente distinta de un umbral, pero aporta -3 cambios netos de celda frente a los 27 que haria falta para mover el leaderboard, y solo mejora 3 de 5 pliegues
- el valor de una correccion no es uniforme, pero eso no abre ninguna palanca nueva: solo decide que correccion merece la pena, y la decision sigue siendo un umbral
- V16 cerro umbrales, tasa y capacidad; V17 cerro los ensembles historicos; V18 cerro la correccion selectiva. Ninguna de esas lineas se reabre

## 6. Lo que sigue en pie, y lo que haria falta para otra fase

V03 es el mejor score observado bajo los experimentos competitivos realizados. Eso no es una afirmacion de que 0.78306 sea el optimo matematico, y no se sostendria sin haber explorado la funcion objetivo entera, que es continua y tiene muchos puntos locales.

una fuente de informacion nueva, o un mecanismo con una diferencia algebraica identificable. La unica palanca que este analisis deja sin explorar del todo es la estructura de etiquetas, y de ella solo se ha encontrado una restriccion, que ya se ha probado y descartado. Una interaccion multiplicativa entre scores sigue sin ejecutar, pero la separacion del objetivo muestra que sin terminos cruzados no puede mejorar a un umbral por etiqueta, asi que solo tendria sentido si atacase informacion que el score de HS no contiene y que el score de TR si.

## 7. Diagnostico de la distribucion de test

El proyecto no guarda las probabilidades de V03 sobre el test, solo la submission binaria
congelada, y aqui se mide la tasa positiva de esas decisiones, que es exacta. No se
sustituyen los scores por valores inventados.

| magnitud | OOF | test | diferencia |
| --- | --- | --- | --- |
| tasa positiva predicha en HS | 0.4572 | 0.4385 | -0.0187 |
| tasa positiva predicha en TR | 0.3001 | 0.2894 | -0.0106 |
| tasa positiva predicha en AG | 0.3776 | 0.3657 | -0.0119 |
| media de la tasa positiva sobre las tres | 0.3783 | 0.3645 | -0.0138 |

## 8. Correlaciones historicas con el leaderboard

| propiedad | correlacion con LB | n |
| --- | --- | --- |
| local_f1 | +0.4873 | 17 |
| diferencia | +0.1297 | 19 |

El leaderboard es una evaluacion externa: no se ha entrenado nada sobre el, y una
correlacion con veinte puntos no probaria causalidad en ningun caso.


Fecha: 2026-09-27T03:26:15+00:00. Duracion: 0.0 s.
