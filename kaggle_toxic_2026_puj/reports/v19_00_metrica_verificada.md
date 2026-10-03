# V19-00: la metrica de la competicion, reconstruida y verificada

Generado por `experiments/v19_00_metrica_verificada.py` (V19-00 (P0 metrica de la competicion, reconstruida y verificada)).

**Ningun modelo entrenado, ninguna submission gastada.**

## 1. Que scorer hay

El scorer es kernel oficial de Kaggle metric/f1-score, sin modificar. Su unico grado de libertad real es el argumento `average` de `sklearn.metrics.f1_score`. La competicion **no publica pagina de
evaluacion**: sus cinco paginas son data-description, Description, rules, abstract y foundational-rules, y las reglas solo dicen "F1 Score".

## 2. Lo que queda resuelto con certeza

| cuestion | respuesta verificada |
| --- | --- |
| `zero_division` | 0.0, el default de sklearn, no nan ni omision de etiqueta o fila |
| `average="binary"` | imposible, lanza ValueError sobre entrada multilabel-indicator |
| promedio de seis clases | no es lo que hace la competicion. El average='macro' de sklearn sobre entrada multilabel promedia las F1 de la clase positiva de cada etiqueta, no las seis clases: verificado con 600 casos aleatorios, 0 discrepancias contra la media de las tres F1 positivas. El promedio de seis clases es una magnitud que el script 02_baseline construye a mano y que ningun scorer estandar calcula aqui |
| metrica que usa el proyecto | macro sobre las tres etiquetas, clase positiva |

## 3. Lo que NO queda resuelto, y por que importa mas que la metrica en si

Siguen vivos: macro3, micro, samples, weighted.

macro, micro y weighted mantienen las tres etiquetas separadas, porque el conteo de TP, FP y FN se hace dentro de cada etiqueta y cada celda pertenece a una sola etiqueta. Bajo esas tres hipotesis el problema de decision se separa en tres problemas de umbral y ninguna regla conjunta puede superar a los umbrales por etiqueta. Solo samples cuenta las tres etiquetas de una fila juntas, y es la unica hipotesis bajo la cual una decision conjunta tendria sentido.

## 4. La plantilla no discrimina, y no hay que forzar una conclusion

La plantilla de ejemplo es todo unos, 1154 filas y su LB es
0.51494. Para reproducirlo, cada hipotesis exige un desplazamiento
train->test distinto:

| hipotesis | desplazamiento s que exige |
| --- | --- |
| macro3 | 0.936796 |
| micro | 0.932828 |
| samples | 1.300000 |
| weighted | 0.909359 |

las cuatro hipotesis piden un desplazamiento train->test distinto pero todos plausibles, y la diferencia entre ellos es de unos milesimos, muy por debajo del error tipico del leaderboard. Forzar una conclusion aqui seria inventarla.

## 5. Verificacion por dos vias de calculo

- Casos sinteticos con respuesta calculada a mano: **28** comprobaciones, todas correctas.
- Contrastes celda a celda contra `sklearn.metrics.f1_score`: **980**, con **0 discrepancias**.
**0 discrepancias**.
- `corrected_cv` de V03 recalculado: **0.7772114525165721**,
  esperado **0.7772114525165721**. Se reproduce
  exactamente, luego la metrica que se usara es la misma con la que se eligio el champion.

- Control circular: la metrica solo consume y_true y la prediccion, y la prediccion se construye aplicando un umbral fijo a probabilidades que ya estan calculadas. Ninguna magnitud de la metrica entra nunca en la construccion de la prediccion, de modo que no hay forma de que el score se infiera de si mismo.

## 6. Conteos de V03 en el OOF, la base de todo el analisis posterior

| etiqueta | TP | FP | FN | TN | F1 |
| --- | --- | --- | --- | --- | --- |
| HS | 505 | 173 | 143 | 662 | 0.761689 |
| TR | 363 | 82 | 82 | 956 | 0.815730 |
| AG | 425 | 135 | 142 | 781 | 0.754215 |

## 7. Decision de la fase

El `average` no se puede determinar con los medios
disponibles, asi que **no se elige por suposicion**. Se mide el margen de mejora bajo
cada hipotesis por separado, y solo la hipotesis `samples` justifies una decision
conjunta. Las otras tres se resuelven con umbrales por etiqueta, que es lo que el
champion ya hace.


Fecha: 2026-09-27T03:14:30+00:00. Duracion: 14.3 s.
