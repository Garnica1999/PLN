# Reproducibilidad de la fase v11

Generado por `experiments/v11_08_cierre_y_tabla_maestra.py` (V11-08 (cierre de fase y tabla maestra)).

## 1. Niveles declarados, segun la seccion 11 de las restricciones

| elemento | nivel | evidencia |
| --- | --- | --- |
| V03 completa | R3, reproducido exactamente | corrected_cv 0.7772114525165721 con diferencia 0.0 y huella de submission identica |
| V9-02, estimacion de prevalencia | R3, reproducido exactamente | diferencias de -4.51e-07 a +3.29e-07 e iteraciones del EM identicas |
| mejor candidata de calibracion | R3, reproducido exactamente | V10-04 re-ejecutado en v11 con cifras identicas a las de v10 |
| mejor candidata de decision | R2, reproducido con limitaciones | V10-05 y V10-06 re-ejecutados; la correccion de prior de v11 es un experimento distinto y no una reproduccion de v10 |
| candidato final de v11 | R0, no existe | ningun candidato llego a la mesa de finales, de modo que no hay nada que reproducir como candidato |
| holdout2 | R2, reproducido con limitaciones | huella de identificadores intacta y cero grupos compartidos; ninguna evaluacion gastada |

Sobre R4, la reproduccion independiente: **no se declara en ningun caso**. Este proyecto
tiene un solo autor y una sola cadena de herramientas, y el propio enunciado dice que
un output cacheado no cuenta como reproduccion independiente. Declarar R4 sin un segundo
entorno o un segundo ejecutor seria exactamente el tipo de afirmacion que la seccion 22
prohibe.

## 2. Huellas de referencia

| artefacto | sha256 |
| --- | --- |
| submission v03 | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| config/holdout2.json | f71ad8ac4959dbdcccc7da23827a36924acd74f344522b81f219a9122d95b3c1 |

## 3. Lo que se registro de cada experimento

Cada experimento de la v11 guarda en su informe JSON el identificador, la fecha, el
script que lo produce, la hipotesis, el protocolo, los resultados, la decision y las
rutas de los artefactos. La tabla maestra `reports/v11_master_results.csv` recoge
ademas el commit de la fase, y el campo de resultado de cada fila del registro se deriva
del informe en lugar de teclearse, con V11-05 como mecanismo.

## 4. Lo que no se pudo reproducir, y por que

La transferencia a la prueba real **no se pudo medir**, y la razon no es tecnica: la
auditoria de legalidad, V11-02, clasifico el uso de la prueba para estimar prevalencia
como ambiguo, con lo que queda prohibido usarlo para generar un submission. No falta
capacidad de calculo ni un experimento: el procedimiento esta prohibido.

Citar un nivel de reproduccion para la transferencia seria mentir sobre lo que se ha
medido, asi que no se cita.

Fecha: 2026-09-26T22:04:38+00:00. Duracion: 0.0 s.
