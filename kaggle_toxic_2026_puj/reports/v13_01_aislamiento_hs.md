# V13-01: aislamiento de la evidencia de TR y AG sobre HS

Generado por `experiments/v13_01_aislamiento_hs.py` (V13-01 (aislamiento de la evidencia de TR y AG sobre HS)).

## 1. Por que este experimento y no los cuatro de la fase

La auditoria V13-00 cerro P1, P2, P3 y P4 por estar ya probadas. El punto de partida concreto
es este: **E2-02 y E2-18 entrenaron el meta por etiqueta sobre 12 probabilidades OOF de 4
bases**, es decir 4 bases por 3 etiquetas, luego **el score propio de HS estaba siempre
entre las features** cuando se prededia HS.

Eso significa que **nunca se pudo atribuir a TR y AG la contribucion a HS**. El meta podia
predecir HS con su propio score y el efecto de las otras dos etiquetas quedaba mezclado sin
posibilidad de separarlo. Este experimento hace esa separacion.

## 2. Los tres brazos

- **A, V03**: la referencia con sus tres binarios independientes.
- **B, meta de HS con evidencia de TR y AG SOLO**: margenes y probabilidades de TR y AG, con
  el score propio de HS **excluido deliberadamente**.
- **C, meta de HS con su propio score mas el de TR y AG**: equivalente a lo que ya hacia
  E2-02, y existe para **cuantificar cuanto aportaba el score propio** que B excluye.

**El brazo B es el que decide la fase.** Sin el score propio, el meta solo puede usar lo que
TR y AG saben de HS.

## 3. La medicion, que es la nueva

| sistema de puntuacion de HS | AUC de HS | desviacion entre pliegues |
| --- | --- | --- |
| V03, con su propio score | 0.8677 | - |
| B, solo evidencia de TR y AG | 0.8433 | 0.0096 |
| C, score propio mas TR y AG | 0.8710 | 0.0150 |

**El score propio de HS aporta +0.0244 de AUC.** Esa es la
separacion que el registro historico no podia ofrecer, y es la que responde a la pregunta de
la fase.

## 4. Los regimenes de error de HS, que nunca se habian descompuesto

| regimen | HS real | HS predicho | filas | FP de TR | FP de AG | VP de TR | VP de AG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 662 | 11 | 14 | 0 | 15 |
| B | 0 | 1 | 173 | 67 | 90 | 0 | 7 |
| C | 1 | 0 | 143 | 0 | 2 | 57 | 99 |
| D | 1 | 1 | 505 | 4 | 29 | 388 | 446 |

El regimen **B**, HS real 0 y HS predicho 1, es el falso positivo de HS: filas no toxicas
que el modelo cree toxicas. Ahi TR y AG **no estan definidas**, y es donde el modelo produce
falsos positivos que el oraculo corrige.

## 5. La fraccion del techo que recupera cada mecanismo historico

| mecanismo | delta macro | fraccion del techo | nota |
| --- | --- | --- | --- |
| V03 referencia | +0.000000 | 0.0 por ciento | punto de partida |
| V4-E conjunto | -0.005922 | -12.8 por ciento | ya cerrado como NO_SIGNAL |
| E2-18 meta Ridge | +0.010189 | 22.0 por ciento | adoptado en su gate, rechazado en el externo |
| V6-B condicional suave B1 | -0.010700 | -23.1 por ciento | cerrado, es la formulacion prohibida |
| orculo con HS verdadera | +0.046271 | 100.0 por ciento | techo diagnostico, no adoptable |

El techo de referencia es **+0.046271**, la mascara con la etiqueta
verdadera de HS, que no es adoptable porque usa la verdad.

## 5b. La prueba de adopcion: el macro-F1 final con HS corregido

El AUC responde si TR y AG **llevan** informacion sobre HS. El macro-F1 responde si esa
informacion **mejora el sistema**, y es lo unico que la seccion 10 permite adoptar. Se
usa el umbral congelado de V03 para HS, 0.44, sin reoptimizar.

| brazo | macro-F1 | delta | HS rescatados | HS danados | HS cambiados |
| --- | --- | --- | --- | --- | --- |
| B_solo_TR_AG | 0.769670 | -0.007541 | 31 | 41 | 174 |
| C_propio_mas_TR_AG | 0.777746 | +0.000534 | 20 | 1 | 51 |

El contraste entre HS cambiados, HS rescatados y HS danados es la cifra que decide: una
correccion que cambia muchas etiquetas pero rescata y dana en la misma proporcion no
mejora nada, solo desplaza el error de sitio.

La fraccion del techo que cada brazo recupera:

| brazo | delta macro | fraccion del techo de +0.046271 |
| --- | --- | --- |
| B_solo_TR_AG | -0.007541 | -16.3 por ciento |
| C_propio_mas_TR_AG | +0.000534 | 1.2 por ciento |

## 6. El precedente de riesgo de transferencia que esta fase debe respetar

El meta historico dio **0.7874** en local, es decir
+0.0102 sobre V03, y fue **ADOPTADO en su gate**. Y sin embargo no se
envio, porque el leaderboard dio **0.78163** frente a 0.78306 de V03.

Es decir: **la misma familia de mecanismo ya dio una mejora local por encima del gate de
candidato y perdio en el externo.** Cualquier resultado positivo que produzca este
experimento pertenece a una familia con un fallo de transferencia ya medido, y eso obliga a
tratarlo como marginal hasta que se demuestre lo contrario.

## 7. Veredicto

**Ningun brazo alcanza el gate de candidato**, que exige un delta de al menos
+0.003 en el macro-F1. El mejor es **C_propio_mas_TR_AG** con
+0.000534.

Ese es el resultado que decide la fase, porque el AUC por si solo no es adoptable: la
seccion 10 exige que la mejora de HS se transmita al macro-F1 final. Aqui la
informacion existe, pero **no se traduce en mejora del sistema**.

La consecuencia es la que el propio enunciado preve para este caso: **cerrar
formalmente la linea de inferencia estructurada de etiquetas**, sin afirmar que TOXIC
no puede mejorar. Lo que se afirma es mas estrecho: que TR y AG no aportan informacion
sobre los errores de HS que el propio score de HS no tenga ya, y que la ganancia local
del meta historico no venia del acoplamiento entre etiquetas sino de una combinacion
de margenes cuyo reparto entre entrenamiento e inferencia es justo lo que el
leaderboard penalizo.


Fecha: 2026-09-26T23:56:17+00:00. Duracion: 29.5 s.
