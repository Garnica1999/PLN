# P63 - nLos bloques nuevos explican el residuo de V03?

Generado por `experiments/p63_08_residuo_de_v03.py`. Fecha 2026-09-29T06:59:38+00:00.

## La pregunta que los cinco tracks anteriores no respondieron

Todos produjeron la misma forma. La composicion en ratios gana +0,0221 frente a un
bloque aleatorio al anadirse a V03, y +0,1055 en solitario. La posicion gana +0,0713
en solitario. Ninguna mejora a V03. La lectura habitual es la dilucion: anadir columnas
a un espacio de 21044 dimensiones cuesta mas de lo que la informacion nueva repaga.

Hay una segunda lectura que no se ha probado, y es mas importante:

| lectura | prediccion |
|---|---|
| **DILUCION** | los bloques saben cosas que V03 falla, pero no se pueden pagar |
| **YA CONOCIDO** | los bloques re-codifican lo que V03 ya saca de los n-gramas |

Las dos hacen la misma prediccion sobre el F1 y **opuestas sobre el diagnostico**. Si
fuera redundancia, ningun cuidado al anadirlos ayudaria nunca, y esa es una razon
mucho mas fuerte para cerrar una familia que un negativo.

El objetivo aqui es estrictamente si V03 acierta o falla en esa celda, y el margen de
V03 entra como variable para que la pregunta sea sobre los bloques y no sobre la
confianza, que ya esta medida. **No se predice la etiqueta**, luego no es el selector
circular de V11-07 ni el de V50, ambos cerrados.

## Resultados

| brazo | AUC HS | AUC TR | AUC AG | AUC media |
|---|---:|---:|---:|---:|
| margen_solo | 0.747593 | 0.793786 | 0.730044 | 0.757141 |
| margen_comp | 0.729332 | 0.758197 | 0.710068 | 0.732533 |
| margen_pos | 0.722797 | 0.771574 | 0.703990 | 0.732787 |
| margen_aleatorio | 0.706412 | 0.765456 | 0.688756 | 0.720208 |

## El desglose

```text
el margen de V03 solo ya explica los errores con AUC 0.757141, y no es debil
composicion sobre el margen            -0.024608
posicion sobre el margen              -0.024354
aleatorio sobre el margen             -0.036933
brecha de composicion contra aleatorio +0.012325
brecha de posicion contra aleatorio   +0.012579
```

## La tasa de error por rebanada, que es donde el techo dice que esta el residuo

El AUC del margen solo sale por debajo de 0,5, y eso no es un error: el objetivo es
el ERROR, y un margen alto significa que V03 esta lejos del umbral, es decir que
acierta. Un AUC de 0,17 sobre el error es un AUC de 0,83 sobre el acierto.

| bin | n | errores | tasa | AUC del margen solo ahi |
|---|---:|---:|---:|---:|
| CORTA | 167 | 33 | 6.59% | 0.170832 |
| MEDIA | 602 | 305 | 16.89% | 0.251425 |
| LARGA | 547 | 306 | 18.65% | 0.223968 |
| MUY_LARGA | 167 | 113 | 22.55% | 0.268046 |

## Veredicto

```text
Aqui hay un TERCER caso que la taxonomia de dos lecturas no preveia, y es el que se da. Los dos bloques superan al control aleatorio, con +0.012325 la composicion y +0.012579 la posicion, luego CONTIENEN informacion sobre que filas va a fallar V03. Pero NO mejoran al margen: aportan -0.024608 y -0.024354 sobre el, es decir lo restan. La lectura correcta es que la informacion de los bloques sobre el residuo YA ESTA implicita en la confianza de V03, y anadir 42 o 21 columnas mas solo hace que el modelo auxiliar se reparta el mismo dato y sobreajuste. No es redundancia con los n-gramas, que es lo que se suponia, y tampoco es informacion nueva: es redundancia con el MARGEN. Por eso el margen solo ya explica los errores con AUC 0.757141 y cualquier bloque lo empeora

LECTURA: REDUNDANTE_CON_LA_CONFIANZA_DE_V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_08_residuo_de_v03.py`. Sin push.*