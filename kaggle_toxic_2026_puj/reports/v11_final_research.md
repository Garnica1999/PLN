# V11: shift de prevalencia, calibracion y regla de decision. Informe final

Generado por `experiments/v11_08_cierre_y_tabla_maestra.py` (V11-08 (cierre de fase y tabla maestra)).
Commit de la fase: `ad726ee`.

## 1. La conclusion

**La correccion de prior no funciona, y con ella se cierra la via del desplazamiento
de prevalencia.** No es una conclusion sobre si el shift existe: V11-06 demostro que
los estimadores lo recuperan con un error relativo medio de 0.0069. Es una conclusion
sobre que **corregirlo no mejora la metrica**, y el motivo es concreto y medido.

Los umbrales congelados de V03, 0.44, 0.51 y 0.46, estan ya **por encima** de los
umbrales optimos de F1, que V10-05 midio en 0.3841, 0.4093 y 0.3789. La capa de decision
ya es conservadora, como si estuviera pre-compensando una caida de prevalencia. Aplicar
la correccion de prior encima corrige dos veces y se pierde exhaustividad: en HS da
-0.001032 en su factor verdadero, con 1 de 5 particiones a favor.

Y hay un dato que refuerza el diagnostico: la correccion **ayuda cuando la prevalencia
sube** y **empeora cuando baja**. A factor 1.05 da +0.001095 en TR y +0.001194 en AG. El
desplazamiento que V9-02 midio en la prueba real es hacia **abajo**, de factor 0.94, asi
que el mecanismo actua en la direccion contraria a la que se necesita.

**Ningun candidato se propone. V03 sigue siendo la submission recomendada.**

## 2. Lo que la v11 si ha hecho

1. **La cifra de la que dependia todo se verifico.** V9-02 se reproduce exactamente, con
   nivel R3 e iteraciones del EM identicas.
2. **Se arrastro una ambiguedad de un artefacto historico.** El campo `factor_vs_train`
   de V9-02 no significa lo que dice: su denominador es la prevalencia del desarrollo y
   no la del train completo. La diferencia es de 0.0002, pequena, pero es el tipo de
   detalle que hace fallar una cadena de resultados.
3. **Se determino que el uso de la prueba esta prohibido**, no permitido, y se escribio
   la consecuencia: la fase no puede generar submission y la pregunta de transferencia
   no tiene respuesta verificable sin autorizacion del profesor.
4. **Se demostro que los estimadores de prevalencia funcionan**, y se explico por que
   clasificar-y-contar da un error exactamente cero: bajo un shift puro de prevalencia
   la sensibilidad y la especificidad son invariantes, luego el estimador esta
   exactamente identificado. Es un teorema, no una propiedad favorable.
5. **Se cerro el mecanismo central con evidencia y no con abandono.**

## 3. Las doce preguntas de la revision final

| n | pregunta | respuesta | evidencia |
| --- | --- | --- | --- |
| 1 | V9-02 era reproducible? | SI, exactamente | V11-01: diferencias de -4.51e-07 a +3.29e-07 e iteraciones del EM identicas. Nivel R3 |
| 2 | El shift de prevalencia es real bajo simulacion? | SI, por construccion y recuperado por los estimadores | V11-06: error relativo medio de 0.0069 para EM y exacto para clasificar-y-contar |
| 3 | El estimador EM es estable? | SI, con un sesgo pequeno y acotado | V11-06: converge en 16 a 26 iteraciones con sesgo de 0.7 por ciento relativo |
| 4 | Otro estimador de shift confirma sus resultados? | SI, y confirma mejor: clasificar-y-contar es exacto bajo shift puro | V11-06: los dos mecanismos son conceptualmente distintos y los dos recuperan el shift |
| 5 | La correccion de prior funciona bajo shift controlado? | NO | V11-07: macro medio de -0.000293, con HS a -0.001032 y 1 de 5 particiones |
| 6 | La calibracion mejora la calidad probabilistica? | SI, y de forma clara: el ECE baja de 0.0583 a 0.0174 | V10-04: isotonica recortada, reduccion del 29.94 por ciento frente a Platt |
| 7 | La calibracion mas shift supera a shift solo? | UNKNOWN, y es la mayor laguna de la fase | No medible: el shift real exige usar la prueba, que esta prohibido. Y shift solo ya es negativo, de modo que l |
| 8 | La decision F1 plugin aporta algo realmente nuevo? | NO, medido por otra via | V10-05: baja los umbrales entre 0.0029 y 0.0073 como se derivo, y empeora -0.003921, porque los umbrales de V0 |
| 9 | Existe una relacion estable umbral y prevalencia? | UNKNOWN | No medido en esta fase. Es la linea P7, que cae junto con el shift, y su pre-registro queda sin ejecutar |
| 10 | La parametrizacion compacta reduce varianza? | NO, medido en v10 | V10-06: ninguna parametrizacion mas restringida mantiene la ganancia de tres umbrales libres con menor dispers |
| 11 | La mejora sobrevive al holdout2? | NO APLICA, y hay que decirlo con claridad | Ningun candidato llego a la mesa de finales, asi que no se gasto ninguna de las tres evaluaciones disponibles. |
| 12 | La metodologia justifica una nueva submission? | NO | V11-02 prohibe el uso de la prueba y V11-07 cierra el mecanismo. No hay candidato que proponer y ademas no hab |

**10 respondidas y 2 con
respuesta UNKNOWN.** Las UNKNOWN no se rellenan con una estimacion:

- La **pregunta 7**, si la calibracion mas el shift supera al shift solo, es la mayor
  laguna de la fase. No es que falte medirla: es que el shift solo ya es negativo, de
  modo que la combinacion no tiene una base desde la que mejorar, y ademas medirla
  exigiria usar la prueba.
- La **pregunta 9**, si existe una relacion estable entre umbral y prevalencia, es la
  linea P7, y cae junto con el shift.

## 4. Que lineas caen y cuales siguen

**Caen**, y no por una decision de gesto sino por consecuencia logica:

- **P5_calibracion_mas_shift**: cae porque su premisa es que el shift es explotable, y V11-07 mostro que no lo es
- **P7_decision_condicionada_por_prevalencia**: cae por la misma razon: la curva umbral-prevalencia solo tendria valor si corregir el umbral por la prevalencia ayudara, y el mecanismo se midio negativo
- **P10_sonda_de_transferencia**: cae por la auditoria de legalidad, no por un resultado

**Siguen siendo legitimas**, porque no dependen del shift:

- **P4_calibracion_anidada**: no depende del shift; ya se midio en V10-04 como NO ACCIONABLE, porque la calibracion mejora el ECE y no la F1
- **P6_regla_f1_plugin**: no depende del shift; ya se midio en V10-05 y empeoro -0.003921
- **P8_parametrizacion_estable**: no depende del shift; ya se midio en V10-06 y no supero el gate

Es decir: las tres lineas que quedan ya estan medidas, y las tres cerraron negativas en
v10. **P4** dio NO ACCIONABLE porque la calibracion mejora el ECE y no la F1. **P6** empeoro
con -0.003921 porque se aleja de los umbrales de V03. **P8** no supero su gate porque no
hay parametrizacion a la vez igual de buena y mas estable. Con lo medido hasta aqui, la
fase no tiene ninguna linea abierta con expectativa razonable.

## 5. La tabla maestra

`reports/v11_master_results.csv` con 7 filas y 29 columnas,
incluyendo el commit de la fase y la evidencia de cada fila.

## 6. El limite de esta fase, dicho sin adornos

La v11 queria responder si el desplazamiento de prevalencia era una fuente real de
mejora. **La respuesta es que no lo es**, y se ha podido responder con un factor
**conocido por construccion**, que es el caso en el que el mecanismo se puede comprobar.

Lo que **no** se ha podido responder, y queda escrito para que no se lea mas de lo que
vale, es si ese mismo mecanismo mejoraria el resultado en **esta prueba concreta**. Eso
exige usar la prueba para generar un submission, y la auditoria de legalidad lo prohibe.
No es una limitacion de calculo ni una falta de tiempo: es una limitacion de las reglas, y
es la razon de que esta fase no termine en una submission.

La consecuencia practica es que la disponibilidad de V03 como submission recomendada
queda intacta, y que el objetivo de superar 0.80 sigue a la distancia que midio la v10.

Fecha: 2026-09-26T22:04:38+00:00. Duracion: 0.0 s.
