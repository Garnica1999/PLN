# P82-02: submission de HatEval con la mejor RNN de P82

V03 remains the best verified leaderboard result.

## 1. Que se espera de esta sonda

se espera de forma explicita que esta sonda NO supere a V03, y no es un fallo del modelo: lo unico que mide es si la representacion sequential aporta algo transferible, y el resultado de P82 dice que la atencion no aporta y que el mejor de los cinco brazos es el GRU sin atencion, y sigue muy por debajo de V03. La distancia con V03 es 0.1375524525165721 a favor de V03, que son 11.84 desviaciones entre semillas de P82.

## 2. Decisiones tomadas

- Sonda: decision de Carlos, que levanto la restriccion de no generar submission, y que asume de forma consciente que incumple la regla only_finalistas del holdout v2, que reserva las cinco evaluaciones para finalistas. Se registra, no se disimula
- Umbrales: umbrales CONGELADOS de V03, HS 0.44, TR 0.51 y AG 0.46, sin recalibrar. Son los con los que P82 midio, y 0.639659 es exactamente el numero que sale de ahi. Recalibrarlos seria medir una cosa distinta de la que se ha medido. Las probabilidades del test NO se usan para nada: la prediccion es un punto, no una exploracion
- Semilla: una sola semilla declarada, 20261401, la primera del rango de P82. El promedio de las diez seria MAS robusto frente al ruido de reentrenamiento, y es lo que haria falta si lo que se quisiera fuera la mejor estimacion posible de la media, pero un submission tiene que venir de un modelo FINAL y no de un ensamblaje: promediar predicciones es un ensamblaje y habria que declararlo y justificarlo como tal. NO se hace

## 3. Verificaciones obligatorias

| verificacion | valor | resultado |
|---|---|---|
| 1. filas del CSV IGUALES al test, releidas del fichero | 1154 == 1154 | OK |
| 1b. conjunto de ids identico | True | OK |
| 2. cabecera id,HS,TR,AG leida del fichero | id,HS,TR,AG | OK |
| 3. valores enteros 0 o 1, sin vacios ni NaN, releidos del fichero | las cuatro celdas de las 1154 filas | OK |
| 4. sha256 del artefacto congelado sin cambio | 302c994d13b75f8f | OK |
| 5. holdout de 248 excluido y no usado para nada | descartadas 248, usadas 0 | OK |
| 6. ASCII puro escaneado en memoria | OK | OK |
| 7. sha256 del CSV publicado | 3315b25231c5d0888f9dbfb5bd9df741ff23c976394ee2a793bfc3061d02322f | OK |

## 4. Distribucion de las etiquetas en el test predicho

| etiqueta | ceros | unos | proporcion_de_unos | valor_mas_frecuente |
|---|---|---|---|---|
| HS | 582 | 572 | 0.495667 | 0.504333 |
| TR | 820 | 334 | 0.289428 | 0.710572 |
| AG | 701 | 453 | 0.392548 | 0.607452 |

## 5. Lo que NO se ha hecho

- No se ha leido ni una fila del holdout de 248.
- No se ha consultado el leaderboard ni se ha enviado nada a Kaggle.
- No se han recalibrado los umbrales.
- No se han tocado V03, sus ficheros congelados ni submission_v03.csv.
- No se ha promediado ninguna prediccion de varias semillas.

V03 remains the best verified leaderboard result.

---

## FE DE ERRATAS (auditoria integral, 2026-10-02)

El parrafo de la seccion 5 que afirma "No se ha consultado el leaderboard ni
se ha enviado nada a Kaggle" es incorrecto: la sonda SI se envio y su
resultado esta registrado en `submissions/leaderboard_log.csv` como
`p82_sonda_rnn_b1`, con LB 0.68150 y delta LB-local +0.041841. Manda el
registro. La sonda no supera a V03 y asi se declara.

Evidencia: `auditoria/revision_p82.md` (hallazgo H4).
