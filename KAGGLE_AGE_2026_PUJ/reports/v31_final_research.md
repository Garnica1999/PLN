# V31+ ADVANCED RESEARCH - informe final

## Conclusion

**Ninguna familia nueva produjo un candidato finalista.** V30 sigue siendo el mejor resultado validado, con `corrected_cv` = 0.6134471 y leaderboard = 0.63302.

La formulacion correcta, segun el brief, no es que V30 sea el limite teorico, sino que **las familias evaluadas no produjeron evidencia suficiente**. Y hay una razon concreta para eso, que esta medida y no es conjetura: ver la seccion de mecanismo.

## El hallazgo que reencuadra la linea de URL

- La prevalencia de URL en train es **40.20%** y es **monotona en la edad**: 33,4% en 13-17 hasta 50,0% en 65-xx. Es una distribucion real, no ruido.

- Pero el **99% de las URLs son `t.co`**, con un token de ruta opaco de 10 caracteres. Solo hay 0 status URLs y 1,917 externas.

- La consecuencia es directa: **la taxonomia de dominios que propone el brief es casi vacia en este dataset**. No hay semantica de dominio que extraer. Lo que queda es presencia, longitud y truncado, y eso ya lo captura la superficie textual.

- Enmascarar la URL **cuesta** -0,006969: la URL no es ruido. Y el canal URL explicito solo aporta +0,001894. La conclusion es que la URL ya se estaba leyendo.

## El resultado central, y por que no es una sorpresa a posteriori

El banco de representaciones da un resultado fuerte y luego la integracion lo desmiente:

| Momento | Resultado | Lectura |
|---|---|---|
| V31-03, canal contra texto solo | **+0,006820** con 5/5 folds | senal linguistica real |
| V31-04, canal contra V30 | rescate neto -18.426 filas, acuerdo 0,752 | mas debil, pero **falla en un 68% distinto** |
| V31-05, canal como 4o miembro | **+0,0000642** sobre el control reoptimizado | no aporta |
| V31-06, canal como 48 features | **+0,0002948** sobre el control barajado | no aporta |

El patron es el mismo que el que el propio proyecto ya documento en V29 con `HIERBA`: la capacidad era el cuello y V30 ya la exprimio. Aqui la **senal** es el cuello, y V30 ya la absorbio. El meta-espacio de 279 features incluye longitud, mayusculas, densidad de puntuacion y ratios de mencion, que es buena parte de lo que los canales nuevos miden.

**Por que el veredicto de V31-04 se corrigio.** La primera version del script leyo `neto <= 0` como "redundante". Es una tautologia: un clasificador mas debil siempre tiene rescate neto negativo, porque le sobran errores por ser peor. La version corregida usa la **tasa de error compartida** como criterio, y ahi si sale diversidad real (acuerdo 0,752). Sin esa correccion la linea se habria cerrado por el motivo equivocado.

## El control que hizo el trabajo

| Brazo | corrected_cv | Que demuestra |
|---|---|---|
| V30 congelado | 0.6134471 | referencia |
| V30 con 3 pesos reoptimizados | 0.6137948 | **el +0,0003477 de reoptimizar no es del canal** |
| V30 + canal (4 miembros) | 0.613859 | la integracion |
| V30 + canal **barajado** | 0.6134855 | control de atribucion |

Sin el brazo reoptimizado, el +0,0004119 de la integracion habria parecido una mejora del canal, cuando en realidad es casi todo el efecto de tener pesos optimizados en vez de fijos. Y el primer control de barajado estaba mal construido: barajaba los **cuatro** miembros, no solo el canal, y por eso daba 0,1639 y un veredicto de "aporte real" sin fundamento. Barajar solo el canal deja a V30 intacto y el resultado es el correcto.

## Track J: donde queda el margen, y por que no es de representacion

- Errores de V30: **65,403** (38.62%).

- De ellos, **9,780** son de alta confianza (15.0%). El resto, 55,623, son de frontera, y esos los arregla un umbral, no una representacion.

- **El 90% de las diez confusiones principales son entre bandas adyacentes.** V30 no confunde 13-17 con 65-xx: confunde vecinas. Eso es un problema de resolucion de escala, no de representacion.

- Y la familia que ataca justo eso (ordinal, CORAL) **ya esta cerrada** por la historia del proyecto: H10, D-058. No se reabre porque sea la mas/kmate obvious, sino porque cerrarla fue una decision con su propia evidencia.

## Clasificacion de cada linea

| Linea | Estado | Motivo |
|---|---|---|
| A: representacion de URL | CLOSED | +0,001894 en el banco, nada en la integracion |
| A5: recuperacion de contenido | QUARANTINED | reglas de AGE no disponibles |
| B/D/E: ortografia, composicion, discurso | MARGINAL | +0,006820 en banco, ~0 en V30 |
| C: verbos y tiempo verbal | CLOSED | solapado con el canal de composicion |
| F: canal por posicion | CLOSED | **-0,002303**, negativo en 0/5 folds |
| G: forma de palabra | CLOSED | **-0,002984**, negativo en 0/5 folds |
| H: representaciones sparse nuevas | CLOSED | V25 cerro la familia |
| J: mineria guiada por errores | DIAGNOSTICO |Errors adyacentes, no de representacion |
| N: integracion en el meta-espacio | CLOSED | +0,0000642 y +0,0002948, ambos marginales |
| POS como predictor | QUARANTINED | modelo externo preentrenado, AMBIGUOUS |

## La correccion del baseline, que no es menor

V30 publico `corrected_cv = 0.6130051`. Esa cifra se calculo con los pesos **por fold** que Optuna optimizo, y **esos pesos nunca se persistieron**: el artefacto solo guarda la media. Lo que la submission uso fueron los pesos **medios**, cuyo OOF da **0.6134471**.

El leaderboard resuelve cual de las dos es la real: con los pesos medios, la calibracion interna predice LB 0.63301 frente a un LB real de 0.63302, un error de **+0,00001**. Con la cifra publicada el error seria +0,00047. Y el hash del fichero enviado confirma que se subio la version de pesos medios.

Es decir: la cifra publicada no era erronea en si misma, pero **no era reproducible y no describia el modelo enviado**. V31+ mide contra 0.6134471, que es lo unico reproducible, lo que se envio, y lo que casa con el leaderboard.

## Respuesta a la pregunta cientifica central

> Donde esta la informacion que distingue grupos de edad y que V30 no expresa?

La respuesta medida es: **en la presencia de URL y en la composicion superficial, y V30 ya las expresses**. La evidencia son tres controles negativos y un positivo que se cancelan. Y el margen que queda, el 15% de errores de alta confianza, esta concentrado en pares de bandas vecinas, donde la limitacion es de resolucion y no de informacion ausente.

Eso **no** demuestra que no exista mas senal. Demuestra que las familias que se pueden expresar con representacion sparse explicita sobre este texto no la contienen, o que V30 ya la lee. Y las dos cosas se distinguen solo con un experimento mas caro, que esta fase no ha pagado.
