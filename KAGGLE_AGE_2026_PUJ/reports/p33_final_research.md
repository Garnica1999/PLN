# P33 - residualizacion de la senal de dificultad linguistica

## Hipotesis

> Los modelos linguisticos ya evaluados contienen informacion adicional, pero gran parte de ella mide dificultad de la fila para V30. Antes de fusionarla, eliminar la componente explicable por la confianza de V30 y comprobar si queda senal especifica de edad.

## Evidencia previa

- P03/P04 de P32: la confianza del candidato predice el error de V30 con AUC 0,31 (invertido **0,69**). Es informacion, pero con el signo cambiado.
- La tabla de control de P33: el **margen de V30** predice sus propios errores con **AUC 0,8005**, muy por encima del **0,677** del candidato. Ese es el liston.
- P33 demostro ademas que los AUC de P32 estaban mal calculados (decision_function dividido por una suma negativa). La correccion esta en `p33_00_scores_y_url.py`.

## Metodo y protocolo

- `z[c] = log(p / (1-p))` con clipping a 1e-6, para V30 y para el candidato.
- Por cada fold externo, `outer_train` se parte en `inner_train` (75 %) e `inner_val` (25 %).
- `a_c` y `b_c` de `z_C = a_c + b_c * z_V30` se ajustan en **inner_train**.
- `beta` se elige en **inner_val**. `outer_val` solo recibe la correccion ya fijada.
- `eps` 1e-6, semilla 20261002, folds de V20 sobre `dup_clusters`.

## Resultado P05: ¿queda informacion en el residuo?

| Clase | AUC del residuo | AUC de z_C | AUC de z_V30 |
|---|---|---|---|
| 13-17 | 0.5507 | 0.8647 | 0.9310 |
| 18-24 | 0.5232 | 0.7578 | 0.8593 |
| 25-34 | 0.5137 | 0.7321 | 0.8418 |
| 35-49 | 0.5415 | 0.7812 | 0.8722 |
| 50-64 | 0.5547 | 0.8568 | 0.9233 |
| 65-xx | 0.5628 | 0.8968 | 0.9483 |

**AUC medio del residuo: 0.54111**, por encima de 0,50 en las seis clases. **El residuo si conserva informacion de edad.**

Pero el contraste con `z_C` (0,73 a 0,90) y con `z_V30` (0,84 a 0,95) dice que es informacion **muy pequena**: se ha eliminado casi todo y lo que queda es un residuo marginal.

## Resultado P07: ¿se convierte en mejor decision?

| Brazo | corrected_cv | delta | folds+ | rescate | danio | neto |
|---|---|---|---|---|---|---|
| real | 0.6132319 | -0.0002152 | 1/5 | 838 | 882 | -44 |
| barajado | 0.6130622 | -0.0003849 | 1/5 | 728 | 795 | -67 |
| ruido | 0.6132858 | -0.0001613 | 2/5 | 786 | 809 | -23 |
| con_intercepto | 0.6132319 | -0.0002152 | 1/5 | 838 | 882 | -44 |

## Los controles deciden, y no a favor

- `real - barajado = +0.0001697`: el residuo parece aportar algo frente al barajado.
- `real - ruido = -0.0000539`: el **ruido aleatorio rinde MAS que la senal real**. Es el control de capacidad, y es el mas fuerte.
- `con intercepto - sin intercepto = +0.0000000`: el `alpha` se eligio 0, asi que **no es un efecto de calibracion**. Eso descarta `CALIBRATION_ONLY`.
- Los betas alternan de signo entre folds: [0.1, -0.05, -0.05, 0.1, 0.1].

## Bootstrap

| | delta | IC95 | contiene 0 |
|---|---|---|---|
| P33 real vs V30 | -0.0002162 | [-0.0006956, +0.0002533] | True |

2000 remuestreos, unidad la fila, pareado. El IC95 contiene el cero, asi que la mejora **no esta demostrada** y ademas su signo es negativo.

## Interpretacion

**La senal existe y no se puede convertir en mejor decision.** El residuo conserva informacion de edad (AUC 0,5411 en las seis clases), pero es pequena, y ninguna de las cuatro formas de inyectarla en el score mejora el F1. Peor: el control de ruido aleatorio gana a la senal real, asi que la mejora que se observa no es atribuible al residuo.

## Decision

- **Integracion global de seis clases: CLOSED.**
- **El residuo como evidencia: COMPLEMENTARY_BUT_NOT_TRANSFERABLE.** Hay informacion, no hay transferencia.

## Limitaciones

- Se evaluo **un** candidato linguistico. Los otros nueve tienen AUC de residuo parecido, asi que no se espera otro resultado, pero no se comprobo uno a uno.
- La conversion del decision_function a probabilidad por softmax **no calibra**. Es suficiente para mezclar y ordenar, y no debe interpretarse como probabilidad real.
- No se ejecuto particion alternativa: con el resultado ya negativo y el control de ruido ganando, no cambiaria el veredicto.

## Reproducibilidad

- Brazo real reconstruido desde cero: **0.6132264**, frente a 0.6132319 en el artefacto. Coincide: False.
- decision_function crudo reconstruido con rango -4.086 a 7.355 y F1 de argmax 0.5047271, que confirma que el softmax preserva el argmax.
- Scores persistidos con SHA256 en `artifacts/p33_scores/`.
- **Nunca** se divide un decision_function por la suma de su fila.
