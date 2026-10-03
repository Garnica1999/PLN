# V14: auditoria de novedad del entrenamiento con etiquetas parcialmente definidas

Generado por `experiments/v14_00_auditoria_partial_label.py` (V14-00 (P0, auditoria de novedad del entrenamiento partially-label)).

**No se entrena nada como experimento.** Esto es una identidad numerica y un inventario
estructural, y es obligatorio antes de gastar presupuesto.

## 1. El veredicto, primero

**La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero.** Se cumple la
condicion de parada numero 1 del enunciado, y la fase se cierra sin ejecutar experimentos.

| comprobacion | valor |
| --- | --- |
| peor diferencia de coeficientes | 0.000e+00 |
| peor diferencia de probabilidades | 0.000e+00 |
| delta de la perdida enmascarada | -0.116581 |
| delta de H2-M registrado en v12 | -0.116581 |
| diferencia entre ambos deltas | 0.000e+00 |

## 2. Por que enmascarar y eliminar son la misma operacion

El enunciado advierte que P1 no debe confundirse con H2-M, y en eso se equivoca, por una
razon que no es de notacion sino de optimizacion.

**Una fila con peso cero en la perdida no aporta gradiente.** Luego el vector de
coeficientes lo determinan unicamente las filas con peso uno, que son exactamente las
filas que H2-M conserva. Las dos formulaciones dejan el mismo subconjunto de filas
aportando senal, y por eso producen el mismo estimador.

Lo unico que puede distinguirlas es **como se reescala el termino de perdida**, es decir la
normalizacion y los pesos de clase. Eso no es un mecanismo de aprendizaje: es una constante
de escala, y pertenece a la familia de calibracion y umbrales, que ya esta cerrada.

La identidad se comprueba en los cinco pliegues y las dos etiquetas:

| pliegue | etiqueta | filas con senal | filas enmascaradas | max dif coeficientes |
| --- | --- | --- | --- | --- |
| 1 | TR | 519 | 667 | 0.000e+00 |
| 1 | AG | 519 | 667 | 0.000e+00 |
| 2 | TR | 518 | 668 | 0.000e+00 |
| 2 | AG | 518 | 668 | 0.000e+00 |
| 3 | TR | 519 | 668 | 0.000e+00 |
| 3 | AG | 519 | 668 | 0.000e+00 |
| 4 | TR | 518 | 669 | 0.000e+00 |
| 4 | AG | 518 | 669 | 0.000e+00 |
| 5 | TR | 518 | 668 | 0.000e+00 |
| 5 | AG | 518 | 668 | 0.000e+00 |

## 3. Y el resultado que decide, que es el mas convincente

Una identidad de estimador convence, pero lo que cierra la fase es que la formulacion
**reproduce un resultado negativo ya registrado**:

| sistema | macro-F1 | delta frente a V03 |
| --- | --- | --- |
| V03 | 0.7772114525165721 | +0.000000 |
| H2-M reconstruido | 0.6606303826082641 | -0.116581 |
| perdida enmascarada de P1 | 0.6606303826082641 | -0.116581 |
| H2-M en el registro de v12 | no reconstruido aqui | -0.116581 |

Las cuatro cifras coinciden hasta el ultimo decimal. Una formulacion que se presenta como
nueva y que reproduce exactamente el fallo de una familia cerrada no es una familia
nueva: es la misma familia con otra notacion.

## 4. La matriz que el enunciado exige

| mecanismo | historico | formula o comportamiento | diferencia respecto al candidato | resultado | estado |
| --- | --- | --- | --- | --- | --- |
| Perdida de TR y AG enmascarada donde HS=0, con | H2-M, brazo M de experiments/v12_02_perd | Linea 600: aplicables = y[entrenamiento, 0] == 1; linea 601: subsampleo = ajuste[aplicable | NINGUNA. Una fila con peso cero no aporta gradiente, luego el estimador lo determinan solo | -0.116581, registrado en v12 y reproducido aqu | CERRADA. Es el mismo estimador, no una familia |
| Reformulacion de H2-M como perdida con mascara | H2-M, la misma linea 600 a 606 | L = L_HS + lambda_TR * mask_HS * L_TR + lambda_AG * mask_HS * L_AG, con mask_HS=1 donde HS | NINGUNA. Es la misma funcion objetivo escrita con notacion de perdida en vez de con notaci | Identica a H2-M: 0.000e+00 en coeficientes, pr | CERRADA. Es la misma familia con otra notacion |
| Pesos de clase sobre la etiqueta completa en v | NINGUNO. No hay registro de esta lectura | class_weight balanced calculado por la biblioteca sobre la y de todas las filas, mientras  | Cambia el unico grado de libertad, que es el reparto de la reescala del termino de perdida | Se midio tambien y coincide con H2-M en la pli | DESCARTADA como via nueva. Es un artefacto de  |
| Reponderar o eliminar ejemplos contradictorios | V4-B05, experiments/29_reweighting_v4.py | pesos = np.where(mascara_contradicciones[entrenar], 0.3, 1.0), y con eliminar=True se desc | El criterio de seleccion es el error del modelo, no la indefinicion de la etiqueta. Aun as | Negativo, cerrado en v4 sin variante adoptada | CERRADA. Reweighting por dureza, no por defini |
| Reponderar por mapa de dureza supervisado | V4-C, experiments/30_supervised_weightin | sample_weight=pesos_dureza[entrenar], que viene de 28_hardness_v4.py | Los pesos vienen de la dificultad del ejemplo segun el modelo, no de si la etiqueta esta d | Negativo, cerrado en v4 | CERRADA |
| Ponderar por tamano de grupo o por ruido de pa | V6-03 y V6-07 | pesos(tamanos, esquema) y un recorte sobre el ruido del par | No menciona HS ni la definicion de las etiquetas | Negativo, ambos cerrados | CERRADA |
| Constante de regularizacion distinta por etiqu | V8-07, experiments/v8_07_peso_por_etique | pesos_por_clase(n_filas, destino, multiplicador) | Reescala por clase, no por regimen de definicion | P5 ya habia medido que elegir la constante por | CERRADA |
| Aprendizaje con etiquetas parciales o positive | NINGUNO con esa terminologia | No hay ninguna funcion de perdida propia con mascara en todo el arbol historico. La unica  | La terminologia es nueva, pero la operacion que propone es H2-M. No hay una via distinta d | No aplicable: no hay operacion que medir | SIN EXPLORAR en su forma verbal, pero SU EQUIV |
| Entrenamiento separado sobre el dominio HS=1,  | H2-M, y el brazo M de v6-00 | Entrenar TR y AG unicamente sobre filas con HS=1 | Es la misma operacion que P1 | -0.116581, y el efecto de escala ya documentad | CERRADA |
| Pseudo-etiquetado de TR y AG condicionado por  | NINGUNO que lo aplique a TR y AG | Solo hay menciones de pseudo-etiquetado en el track de embeddings, que no aplica a este pr | No es el candidato propuesto | No aplicable | FUERA DE ALCANCE por la jerarquia de la fase |

## 5. La sonda estructural, y por que el texto no servia

El enunciado pide buscar doce mecanismos. La busqueda textual los localiza, pero **no sirve
como veredicto**: un guion que enumera sus propias prohibiciones se marca a si mismo, que
es el motivo por el que v12 documento el patron de sonda estructural. Lo que decide es la
sonda sobre el arbol sintactico.

Hay **13 llamadas con `sample_weight`** en todo el
codigo, y aqui estan con su origen:

| fichero | linea | expresion de los pesos | depende de HS |
| --- | --- | --- | --- |
| 29_reweighting_v4.py | 634 | pesos | no |
| 29_reweighting_v4.py | 495 | sample_weight | no |
| 29_reweighting_v4.py | 349 | pesos | no |
| 29_reweighting_v4.py | 377 | pesos_etiqueta | no |
| 30_supervised_weighting_v4.py | 312 | sample_weight | no |
| 30_supervised_weighting_v4.py | 618 | pesos_dureza[entrenar] | no |
| 36_candidatos_v4.py | 475 | sample_weight | no |
| 36_candidatos_v4.py | 199 | sample_weight | no |
| 38_refit_final_v4.py | 234 | sample_weight | no |
| v14_00_auditoria_partial_label.py | 456 | pesos | no |
| v6_03_ponderacion_grupo.py | 184 | peso | no |
| v6_07_ruido_par.py | 256 | peso | no |
| v8_07_peso_por_etiqueta.py | 190 | pesos_por_clase(matriz.shape[0], destino, multiplicador) | no |

**Ninguna condiciona los pesos a HS en el punto de llamada.** Y al trazar cada vector hasta
su definicion, ninguno se define por la indefinicion de la etiqueta:

- `calcular_pesos` de V4-B05 pondera por cercania y por **discrepancia entre prediccion y
  verdad**, es decir por el error del modelo, no por si la etiqueta esta definida. Su
  variante `eliminar=True` si elimina filas, pero por otro criterio.
- `pesos_dureza` de V4-C viene del mapa de dureza, que es dificultad del ejemplo.
- `pesos(tamanos, esquema)` de V6-03 pondera por tamano de grupo.
- el recorte de V6-07 actua sobre el ruido del par.
- `pesos_por_clase` de V8-07 reescala por clase, que es lo que ya midio P5 en +0.0023.

La unica operacion del historial que condiciona la senal de TR y AG a HS=1 es H2-M, y es
exactamente la que P1 reproduce.

### Los doce mecanismos del enunciado, y el defecto de la sonda textual

La sonda textual localiza los doce mecanismos, pero **se marca a si misma**: este guion
los nombra literalmente, porque sus textos estan en la lista de busqueda. Por eso cada
conteo va en dos columnas, y la segunda es la que vale como evidencia.

| mecanismo buscado | guiones historicos | conteo bruto | autorreferencia |
| --- | --- | --- | --- |
| excluir filas HS=0 del entrenamiento de TR/AG | 1 | 2 | si |
| enmascarar la perdida de TR/AG segun HS | 44 | 45 | si |
| ponderar diferentemente filas HS=0 | 55 | 56 | si |
| aprendizaje con etiquetas parciales | 0 | 1 | si |
| positive-unlabeled learning | 0 | 1 | si |
| selective loss | 0 | 1 | si |
| masked BCE | 0 | 1 | si |
| entrenamiento separado sobre el dominio HS=1 | 18 | 19 | si |
| HS=0 como missing o no definido | 3 | 4 | si |
| pseudo-etiquetado condicionado por HS | 4 | 4 | no |
| rechazo de negativos de TR/AG desde HS=0 | 111 | 112 | si |
| cambio de umbral o recalibracion | 117 | 118 | si |

Para **partial-label**, **positive-unlabeled**, **selective loss** y **masked BCE** el
unico guion que los menciona es este, luego **el conteo historico valido es cero**. No es
que no se hayan buscado: es que la unica fuente que los nombra es la propia consulta.

## 6. Que se afirma y que no

**No se afirma** que la semantica de observacion parcialmente definida sea incorrecta, ni
que el regimen HS=0 deba modelarse siempre como negativos ordinarios. Eso es una pregunta de
modelado legitima y la fase no la responde.

**Se afirma** algo mas estrecho y verificado: que implementada como enmascarar la perdida
de clasificacion, esa semantica **es el mismo estimador que H2-M**, con diferencia cero.
Una formulacion que de verdad tratara HS=0 como no supervisado tendria que aportar una
senal distinta de cero en esas filas, y el barrido no deja ninguna via de esa clase dentro
de los mecanismos permitidos por la jerarquia.

## 7. Que pasa con P2, P3 y P4

- **P2 y P3**, con el peso alfa, interpolan entre **dos extremos ya medidos**: alfa=0 es
  exactamente H2-M, con -0.116581, y alfa=1 es exactamente V03, con 0.000000. Los dos
  extremos del eje estan medidos y cerrados, y un barrido entre ellos es una
  reparametrizacion de un eje cerrado, no una familia nueva.
- **P4**, los umbrales, el propio enunciado los deja fuera salvo mejora estructural
  reproducible, y no hay ninguna. Ademas el cambio de escala que producira ya fue medido y
  documentado en H2, y la familia de umbrales y calibracion esta cerrada.

## 8. Decision

**FASE CERRADA: P1 es H2-M con diferencia exactamente cero en coeficientes, probabilidades y macro-F1, y reproduce el -0.116581 ya registrado. Se cumple la condicion de parada numero 1 del enunciado.**

Que la semantica de observacion parcialmente definida sea incorrecta. Lo que se mide es que implementada como enmascarar la perdida de clasificacion es el mismo estimador que H2-M. Una formulacion que aportase senal distinta de cero en las filas HS=0 seria otra cosa, y el barrido no deja ninguna via de esa clase dentro de los mecanismos permitidos.

**Experimentos ejecutados: 0 de 4 principales y 0 de 2 de reserva.** La auditoria es el
primer y unico experimento de la fase, y su veredicto cierra la fase.


Fecha: 2026-09-27T00:09:37+00:00. Duracion: 17.1 s.
