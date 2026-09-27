# V13: fase de estimacion de HS e inferencia estructurada de etiquetas

Generado por `experiments/v13_02_informe_final.py` (V13-02 (informe final y cierre de la linea de inferencia estructurada)).

## 1. La respuesta, y por que reencuadra la pregunta

La fase se abrio para preguntar si HS puede aprenderse con la evidencia de TR y AG. La
respuesta es **si, y no sirve**, y el motivo es mas interesante que la respuesta.

**TR y AG llevan informacion sobre los errores de HS.** Con el score propio de HS excluido
deliberadamente, el meta alcanza un AUC de **0.8433** para
predecir HS, frente a 0.8677 que alcanza V03 con el suyo. El
score propio solo aporta +0.0244. Ese numero **no existia en el
registro historico**, porque E2-02 y E2-18 usaron siempre un conjunto de features que
incluia el score propio, de modo que la contribucion de TR y AG era inobservable.

**Pero esa informacion no es utilizable.** El mismo brazo que obtiene el AUC de
0.8433 **pierde** 0.007541
en el macro-F1 final, porque su F1 de HS cae
0.022624.
Un AUC alto no es una buena decision en el punto de corte, y aqui el umbral esta congelado.

## 2. El hallazgo que reencuadra la fase: el techo no esta en HS

El techo de **+0.046271** se suele leer como evidencia de que a HS le
falta informacion. **No es asi**, y el propio oraculo lo demuestra:

| donde actua el oraculo | celdas | nota |
| --- | --- | --- |
| columna HS | 0 | no cambia ni una sola celda |
| columna TR | 78 | falsos positivos borrados |
| columna AG | 118 | falsos positivos borrados |
| filas afectadas | 134 | todas con HS verdadero igual a cero |

El techo proviene de poner TR y AG a cero en las 835 filas donde el HS verdadero es cero.
Es un efecto **aguas abajo** de la condicion de HS, y **ningun metodo que mejore la columna
HS puede alcanzarlo**. Por eso todos los mecanismos de mejora de HS se quedan cortos, y no
por falta de potencia sino porque apuntan a otra columna.

El corolario medido es directo y no admite matiz:

| brazo | delta de F1 de HS | delta de F1 de TR | delta de F1 de AG |
| --- | --- | --- | --- |
| B_solo_TR_AG | -0.022624 | +0.000000 | +0.000000 |
| C_propio_mas_TR_AG | +0.001603 | +0.000000 | +0.000000 |

**Corregir la etiqueta de HS deja TR y AG con delta exactamente cero.** El meta produce una
probabilidad para HS y no altera lo que el sistema decide sobre las otras dos etiquetas. Es
la demostracion de que estimar mejor HS y mejorar el sistema son cosas distintas aqui.

## 3. Los regimenes de error, que nunca se habian descompuesto

| regimen | filas | FP de TR | FP de AG | VP de TR | VP de AG | celdas que toca el oraculo |
| --- | --- | --- | --- | --- | --- | --- |
| A | 662 | 11 | 14 | 0 | 15 | 32 |
| B | 173 | 67 | 90 | 0 | 7 | 164 |
| C | 143 | 0 | 2 | 57 | 99 | 0 |
| D | 505 | 4 | 29 | 388 | 446 | 0 |

El regimen **B**, HS real cero y HS predicho uno, concentra los falsos positivos de TR y AG:
90 de AG y 67 de TR. Ahi es donde el techo vive. El regimen **C**, HS real uno y HS
predicho cero, tiene 143 filas y **el oraculo no toca ninguna**. Por eso un metodo que
corrige HS no puede alcanzar el techo, ni aunque los acierte todos.

## 4. Los tres diagnosticos que la fase exigia y que no estaban en el registro

1. **Aislacion de la contribucion de TR y AG**: hecha, y es la medicion nueva de la fase.
2. **Descomposicion por regimen de HS**: hecha, con los cuatro casos y el conteo de falsos
   positivos de TR y AG en cada uno.
3. **Fraccion del techo por mecanismo**: hecha, y es lo que reencuadra la conclusion.

## 5. La auditoria de V6-B, que el enunciado pedia como entregable

**Veredicto: CERRADA. La formulacion candidata es V6-B, ya evaluada.**

V6-02 implementa p(TR|X) = p(HS|X) * p(TR|HS=1,X) con cinco productos y cero condicionales cuya condicion mencione HS, y su encabezado declara 'sin routing duro'. Es exactamente la formulacion que el enunciado prohibe repetir.

Resultados historicos: B1 -0.0107, B2 -0.0008,
B3 -0.0115. Ninguno pasa ningun gate.

Repetirla con otra implementacion, otro umbral o un multiplicador equivalente seria una violacion de la clausula de no reabrir familias cerradas, y el enunciado lo prohibe de forma expresa.

## 6. La decision sobre la linea

**CERRADA FORMALMENTE**

Los cuatro mecanismos propuestos estaban ya probados, y lo unico genuinely nuevo, que es aislar si TR y AG llevan informacion sobre los errores de HS, se ha medido: la llevan, y no es utilizable. El techo de +0.046271 no esta en la estimacion de HS sino en la aplicacion de la condicion de HS a las decisiones de TR y AG, y esa via ya esta cerrada por medicion: el post-proceso duro da -0.004187 porque su llave se equivoca en 143 filas.

No se afirma que TOXIC no pueda mejorar. Se afirma que la inferencia estructurada de etiquetas, en la forma condicionada por HS, no es una linea rentable con los mecanismos permitidos.

## 7. Lo que la auditoria historica habia cerrado antes de gastar presupuesto

La auditoria V13-00 es el primer experimento de la fase y no entrena nada. Cero de los
cuatro mecanismos propuestos eran nuevos:

| mecanismo | historico | estado | se ejecuto en v13 |
| --- | --- | --- | --- |
| Clasificacion conjunta sobre el espacio de combinaciones ( | V4-E, experiments/31_joint_label_v4.py, fila | si, con evidencia y con protocolo  | NO |
| Meta-modelo sobre scores OOF de varias bases, con dependen | E2-02 en experiments/15_cadenas_etiquetas_v2 | SI como mecanismo, y con un motivo | NO como mecanismo. El mecanismo es |
| Factorizacion condicional suave con producto de probabilid | V6-B, experiments/v6_02_label_factorization. | si, con mecanismo de fallo identif | NO |
| Routing duro sobre HS predicho | Post-proceso jerarquico en 02_baseline_tfidf | si, y con la causa medida | NO |
| Corrector o combinacion de margenes en inferencia | V8-08, experiments/v8_08_corrector_margenes. | si | NO |
| Perdida que acopla las etiquetas por pares o por margen | V12-03, experiments/v12_03_entrenamiento_por | si | NO |
| Regularizacion conjunta entre etiquetas | V8-05, experiments/v8_05_regularizacion_conj | si | NO |
| Cadenas de etiquetas con score previo como feature | E2-01, experiments/15_cadenas_etiquetas_v2.p | si | NO |
| AISLAR si TR y AG llevan informacion sobre los errores de  | NINGUNO. No existe equivalente | no, esta genuinamente sin probar | SI, con controles |
| Descomposicion de los errores de HS por regimen, y rescate | NINGUNO. No existe equivalente | no | SI, y es el entregable principal d |
| Fraccion del techo del oraculo que recupera cada mecanismo | NINGUNO. No existe equivalente | no | SI |

El detalle que mas condiciono la fase: **P2 ya se habia ejecutado y adoptado**. E2-02 dio
0.7853 y E2-18 dio 0.7874, ambos por encima del gate de candidato, y ninguno se envio
porque el leaderboard fue peor que el de V03. Reejecutar ese mecanismo habria sido una
violacion de la clausula de no reabrir familias, con el riesgo de transferencia ya medido.

## 8. Los doce entregables

| entregable | donde esta |
| --- | --- |
| 1. numero total de experimentos | 2: V13-00 y V13-01 |
| 2. tabla completa de resultados | reports/v13_master_results.csv |
| 3. candidatos adoptables | ninguno: el mejordelta es +0.000534 y el gate exige +0.003 |
| 4. candidatos marginales | C_propio_mas_TR_AG, por debajo del gate exploratorio |
| 5. candidatos rechazados | B_solo_TR_AG |
| 6. evidencia de reproducibilidad | reports/v13_reproducibility.md |
| 7. auditoria de leakage | campo auditoria_de_leakage del JSON y la seccion 3 |
| 8. impacto sobre HS | seccion 2, columna delta de F1 de HS |
| 9. impacto downstream sobre TR y AG | seccion 2, deltas exactamente cero |
| 10. fraccion recuperada del techo | seccion 2 y la tabla maestra |
| 11. auditoria final de V6-B | seccion 5 |
| 12. decision sobre linea genuina | seccion 6: cerrada formalmente |

## 9. Estado final

- Experimentos: **2** de 8 principales, 0 de 4 de reserva.
- Candidatas adoptables: **0**.
- Submissions: **0**.
- Evaluaciones de holdout2: **0** de 3.
- Segunda particion: **no ejecutada**, porque no hay candidata.
- Test: **no utilizado**.
- V03: **congelada e intacta**, con identidad verificada en cada ejecucion.


Fecha: 2026-09-26T23:57:56+00:00. Duracion: 1.1 s.
