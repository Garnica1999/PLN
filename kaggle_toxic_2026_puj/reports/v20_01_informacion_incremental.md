# V20-01: informacion incremental entre scores e interacciones minimas

Generado por `experiments/v20_01_informacion_incremental.py` (V20-01 (P1 informacion incremental entre scores, P2 interacciones minimas)).

**NO CROSS-SCORE INFORMATION**

## 1. La hipotesis, y por que V19 no la refuta

La separacion que demostro V19 dice que, **fijada** la decision de cada etiqueta, su valor
es independiente de las otras dos. No dice que la decision de HS tenga que ser un umbral
sobre HS_score. Si el score de TR lleva informacion sobre la etiqueta HS que HS_score no
contiene del todo, hay una decision de HS mejor que un umbral, y puede implementarse
combinando scores sin tocar la funcion objetivo.

**Advertencia heredada de D-095**: el producto en espacio de probabilidad atenua las filas
de score intermedio, que son justo las que V03 acertaba. Por eso la unica formulacion que
se prueba es en log-espacio, donde el producto de logits no sufre esa compresion.

## 2. Medida 1, AUC estratificado, sin ajustar nada

Se estratifica por deciles del score propio y se mide el AUC del otro score dentro de cada
estrato. Sin parametros que ajustar, luego no puede sobreajustar por construccion.

| etiqueta objetivo | etiqueta fuente | AUC medio en estratos | estratos sobre 0.5 | p empirico |
| --- | --- | --- | --- | --- |
| HS | TR | 0.4997 | 2 de 5 | 0.5205 |
| HS | AG | 0.5681 | 3 de 5 | 0.0010 |
| TR | HS | 0.5751 | 3 de 5 | 0.0010 |
| TR | AG | 0.5624 | 4 de 5 | 0.0010 |
| AG | HS | 0.6126 | 5 de 5 | 0.0010 |
| AG | TR | 0.5218 | 3 de 5 | 0.0829 |

## 3. Medida 2, log-verosimilitud fuera de muestra, con disciplina de pliegues

Un logistico propio contra un logistico propio mas los otros logits, ajustado en cuatro
pliegues y evaluado en el quinto. Es la medida en la escala en la que se decide, que es la
que importa.

| etiqueta objetivo | etiquetas anadidas | mejora por fila | mejora total | pliegues que mejoran | p empirico |
| --- | --- | --- | --- | --- | --- |
| HS |  | +0.00000 | +0.00 | 0 de 5 | 1.0000 |
| HS | TR | -0.00013 | -0.20 | 4 de 5 | 0.3157 |
| HS | AG | -0.00079 | -1.17 | 3 de 5 | 0.7602 |
| HS | TR+AG | +0.00018 | +0.26 | 2 de 5 | 0.1349 |
| TR |  | +0.00000 | +0.00 | 0 de 5 | 1.0000 |
| TR | HS | +0.00059 | +0.88 | 3 de 5 | 0.0559 |
| TR | AG | +0.00106 | +1.57 | 2 de 5 | 0.0240 |
| TR | HS+AG | +0.00058 | +0.86 | 2 de 5 | 0.0599 |
| AG |  | +0.00000 | +0.00 | 0 de 5 | 1.0000 |
| AG | HS | -0.00085 | -1.26 | 3 de 5 | 0.7932 |
| AG | TR | +0.00012 | +0.18 | 3 de 5 | 0.1289 |
| AG | HS+TR | -0.00073 | -1.09 | 3 de 5 | 0.4196 |

**Senal en las dos medidas: True.**

## 4. Los candidatos minimos

| etiqueta | formulacion | celdas | rescates | danios | neto | OOF | delta OOF | celdas comunes con V19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HS | logit(HS)*logit(TR)+logit(AG) | 64 | 29 | 35 | -6 | 0.779430 | +0.002219 | 8 |
| HS | logit(HS)+logit(TR) | 64 | 27 | 37 | -10 | 0.778471 | +0.001259 | 4 |
| HS | logit(HS)*logit(TR) | 66 | 27 | 39 | -12 | 0.778104 | +0.000893 | 9 |
| AG | logit(AG)*logit(HS)+logit(TR) | 52 | 21 | 31 | -10 | 0.777998 | +0.000786 | 0 |
| TR | logit(TR)*logit(HS)+logit(AG) | 24 | 9 | 15 | -6 | 0.776636 | -0.000575 | 0 |
| TR | logit(TR)+logit(HS) | 32 | 13 | 19 | -6 | 0.776364 | -0.000848 | 0 |
| TR | logit(TR)*logit(HS) | 25 | 9 | 16 | -7 | 0.776340 | -0.000872 | 0 |
| AG | logit(AG)*logit(HS) | 63 | 19 | 44 | -25 | 0.774546 | -0.002665 | 0 |
| AG | logit(AG)+logit(HS) | 79 | 23 | 56 | -33 | 0.773172 | -0.004039 | 0 |

## 5. Comparacion con la regla de V19

La regla de V19 cambia **20 celdas** de HS, con 9 rescates y 11 danos.
Un candidato que cambia las mismas celdas no ha anadido nada, solo ha reencuadrado la misma
idea.

## 6. Veredicto

**NO CROSS-SCORE INFORMATION**

- Submissions gastadas: **0**.


Fecha: 2026-09-27T03:57:06+00:00. Duracion: 107.3 s.
