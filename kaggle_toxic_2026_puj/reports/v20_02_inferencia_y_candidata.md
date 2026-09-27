# V20-02: inferencia de V03 en test y candidata de la regla de V20

Generado por `experiments/v20_02_inferencia_y_candidata.py` (V20-02 (inferencia de V03 en test y candidata de la regla de V20 para la submission)).

**Ninguna submission enviada todavia. Ninguna etiqueta de test utilizada.**

## 1. Los controles, que es lo que hace creible el resto

| control | resultado |
| --- | --- |
| el test no trae columna de etiqueta | correcto, y no se lee ninguna en ningun momento |
| filas del test | 1154 |
| identificadores unicos y sin nulos | correcto |
| la inferencia reproduce la submission congelada | **0 celdas distintas** de 3462 |
| sha256 de la submission de V03 regenerada | identico al de la congelada |
| solo cambia HS | correcto, ninguna celda de TR ni de AG cambia |
| umbral derivado solo del OOF | correcto, ninguna lectura de test |

La reproduccion exacta es el control mas fuerte disponible: si al transformar el test con
los artefactos congelados salen las mismas 1154 filas con los mismos valores, las
probabilidades **son** las del champion y no las de una reimplementacion.

## 2. La candidata

| componente | valor |
| --- | --- |
| regla | HS via logit(HS)*logit(TR)+logit(AG) |
| diseno | logit(HS), logit(HS)*logit(TR), logit(AG) |
| modelo | LogisticRegression(C=1.0, solver='liblinear') |
| ajuste | exclusivamente las 1483 filas del OOF de V03 |
| umbral en test | -0.608204 |
| umbrales por pliegue de V20 | media -0.5887, desviacion 0.0470 |
| etiquetas modificadas | HS |

El umbral de test se elige sobre el OOF con la misma operacion que usaba el protocolo OOF,
y los cinco umbrales por pliegue que V20 registro se muestran para que se vea que el
numero es coherente con ellos y no uno nuevo.

## 3. Que cambia exactamente en test

| magnitud | valor |
| --- | --- |
| filas modificadas | 51 de 1154 |
| celdas modificadas | 51 |
| celdas de HS | 51 |
| celdas de TR | 0 |
| celdas de AG | 0 |
| cambios de 0 a 1 | 51 |
| cambios de 1 a 0 | 0 |

## 4. Lo que la misma regla hace en el OOF, con el mismo umbral

- Celdas de HS que cambia: **64**.
- Rescates: **25**, danios: **39**, neto **-14**.

Que el recuento de celdas del OOF y el de test sean de magnitud parecida es la
comprobacion de que la transformacion se aplica igual en los dos conjuntos, que es lo que
pide el encargo antes de enviar.

## 5. Nota sobre el uso del test

El test se ha usado **solo para transformar texto en probabilidades**, que es inference.
No se ha entrenado nada con el, no se ha estimado ninguna etiqueta y el umbral de la regla
proviene integramente del OOF. Es la operacion que las reglas de la competicion permiten.


Fecha: 2026-09-27T04:07:26+00:00. Duracion: 0.1 s.
