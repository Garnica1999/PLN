# P10: el ensemble de cinco modelos, en la version que valida su OOF (V8-10)

Generado por `experiments/v8_10_ensemble_cinco_modelos.py` (V8-10 (P10, correccion)).

## 1. Que error se corrige

La sonda anterior se valido con un OOF que promedia **cinco** modelos, uno por
repeticion, y se envio con un ensemble de **veinticinco**, cinco pliegues por cada una de
las cinco repeticiones. **Se valido un ensemble de cinco y se envio uno de veinticinco.**
Aqui se construye la version coherente: un modelo por repeticion, que es exactamente
lo que el OOF promedia.

## 2. Lo que el OOF dice, que es la unica cifra que respalda la linea

- metrica del ensemble de cinco sobre el desarrollo: **0.782277**
- frente a los 0.777211 de V03, diferencia **+0.005066**
- intervalo de confianza al 95: [-0.00371, +0.01414]
- celdas que cambian en el desarrollo: 182, con tasa de acierto 0.5495
- contraccion de probabilidades al promediar: **+0.000591**

La contraccion es el riesgo conocido de toda esta familia, y aqui es de milésimas, que
es mucho menor que la que se midio en la version de veinticinco.

## 3. La submission corregida

- celdas que cambian frente a V03: 127
- celdas que cambian frente a la version de veinticinco: 0
- tasa de positivos predicha: 0.3608, frente a 0.3645 de V03

| comprobacion | resultado | detalle |
|---|---|---|
| numero de filas | correcto | 1154 frente a 1154 en la prueba |
| nombres y orden de columnas | correcto | ['id', 'HS', 'TR', 'AG'] |
| valores de etiqueta | correcto | [0, 1] |
| sin valores ausentes | correcto | 0 celdas ausentes |
| sin identificadores duplicados | correcto | 0 repetidos |
| identificadores en el orden de la prueba | correcto | coinciden con el orden original |
| filas distintas del entrenamiento | correcto | 1731 frente a 1154 |

Integridad global: **todas pasan**.

## 4. Lo que hay que saber antes de enviarla

**La version de veinticinco modelos perdio 0,01184 en el externo.** Esta version cambia
la magnitud del efecto de escala, que era la causa mecanica medida, pero **no cambia la
direccion de la evidencia**: la mejora local de +0,005066 tiene un intervalo de
confianza que incluye cero, y V8-02 muestra que un movimiento de esta magnitud no es
distinguible en un conjunto de 1154 filas.

Ademas hay un dato de la curva de transferencia que resta: el analisis por cuartiles
muestra que V03 esta en el maximo de la relacion entre la tasa de positivos predicha y el
leaderboard, y **esta version predice una tasa por debajo de la de V03**, que es
justo el lado que la curva castiga.

**Por tanto la recomendacion es no enviarla**, y la razon no es el resultado anterior
sino que la unica palanca que el historial entero identifica, la tasa de positivos, esta
medida y cerrada. Enviarla seria gastar una sonda en una betting que la curva ya dice
que pierde.

El envio es manual por D-063, de modo que la decision final es del usuario.

Fecha: 2026-09-26T04:04:30+00:00. Duracion: 116.5 s.
