# v8: auditoria de brecha y del material de clase (P0)

Generado por `experiments/v8_00_gap_audit.py` (V8-00 (P0, analisis)).
No entrena nada. Se ejecuta antes de gastar una sola unidad de la fase v8.

Referencia: V03 con 0.777211 de metrica local y 0.78306 externo.
Base heredada: **188 experimentos** y **56 decisiones**.

## 1. La hipotesis pendiente de la fase v7

La fase v7 cerro todas sus lineas salvo una: el promedio de los cinco modelos de
pliegue en el momento de la prediccion. Se quedo abierta por una razon concreta y no
por falta de tiempo: **su comportamiento externo no lo captura el protocolo local**,
porque el conjunto de desarrollo ya tiene una prediccion de cada fila hecha por el
modelo que no la entreno, y reconstruir ese mismo promedio sobre el desarrollo no
aporta informacion nueva. Solo se puede medir enviando.

Por eso es la primera linea de v8 y consume la unica sonda externa de la fase.

## 2. Lo que la auditoria encuentra de las diez pistas experimentales

| linea | ya probada | mejor resultado | prioridad |
|---|---|---|---|
| P1 Ensemble de cinco modelos en inferencia | no: el promedio en inferencia no existe | no medido: no se puede medir en local | muy alta: primera y unica sonda externa de la fase |
| P2 Ponderacion por razon logaritmica de recuentos | si, con cuatro variantes | cerrada en un rango de -0.016 a -0.039 | media: se ejecuta solo si el diagnostico de complementariedad lo pide |
| P3 Estabilidad supervisada de features | no | nunca medido | alta: es la forma barata de reducir la dimension |
| P4 Regularizacion conjunta entre etiquetas | no | nunca medido | baja: ejecutar solo si las anteriores no dan nada |
| P5 Constante de regularizacion por etiqueta | no, en la forma por etiqueta | R1-C1: +0.0157 local y -0.0108 externo | muy baja: la evidencia previa es contraria |
| P6 Estructura del peso de clase | no, en la forma por etiqueta | no medido por etiqueta | muy baja |
| P7 Corrector basado en la geometria de los margenes | no | nunca medido | media-baja |
| P8 Calibracion condicionada a la transferencia | no | nunca medido | condicional: depende del resultado de P1 |
| P9 Desacuerdo entre pliegues como senal | no | nunca medido | condicional: depende del resultado de P1 y del techo medido |
| P10 Confirmacion con segunda semilla | no, para candidatos | la varianza entre semillas es apreciable, ... | alta para el candidato que llegue a final |

El detalle con las once columnas de la seccion 5 esta en `reports/v8_gap_audit.csv`.

## 3. Inventario del material de clase

El material de clase sobre clasificacion de texto y atribucion de autoria aporta
22 tecnicas. Este es el apartado que evita repetir trabajo: cada
una se comprueba en el codigo y se declara cerrada o nueva.

- **cerradas: 13**
- **nuevas: 9**

| tecnica | familia | veredicto | por que |
|---|---|---|---|
| Palabras funcionales como rasgos | atribucion de autoria | **cerrada** | Las palabras funcionales se usan como parojos dentro del vectorizador, y los rasgos lexi... |
| N-gramas de POS | atribucion de autoria | **cerrada** | E2-03-POS con sus dos variantes, cerrado. Los canales de sintaxis y de dependencia tambi... |
| Arboles de la linguistica sistemica funcional | atribucion de autoria | **cerrada** | B5-DENSE midio las conjunciones explicitas de registro y objetivo, y HIB-UNION las conju... |
| Morfologia: prefijos y sufijos | atribucion de autoria | **cerrada** | La linea ortografica de E2-10 y la normalizacion de PRE-00 a PRE-09 cubrieron la superfi... |
| Medidas de complejidad por longitudes medias | atribucion de autoria | **cerrada** | El analisis linguistico inicial midio largas y diversidad, y los rasgos densos de B5-DEN... |
| Medidas de complejidad por entropia | atribucion de autoria | **cerrada** | La entropia existe como estadistico descriptivo del analisis inicial y como senal por fi... |
| Medidas de complejidad por asimetria y kurtosis | atribucion de autoria | **nueva** | sin cierre |
| Diversidad lexica: tipo sobre token y hapax | atribucion de autoria | **cerrada** | El analisis inicial los midio como estadistico descriptivo. Como rasgo de un modelo no s... |
| Palabras inestables medidas por traduccion de ida y vuelta | atribucion de autoria | **nueva** | sin cierre |
| Estabilidad de features entre submuestras | atribucion de autoria | **nueva** | sin cierre |
| Idiosincrasias y lexico regional | atribucion de autoria | **nueva** | sin cierre |
| Seleccion de rasgos por grupo con criterio de informacion | atribucion de autoria | **nueva** | sin cierre |
| Correlacion entre rasgos para detectar redundancia | atribucion de autoria | **nueva** | sin cierre |
| Union de varias vistas de texto en un rasgo | atribucion de autoria | **cerrada** | La doble vista con normalizacion es la linea P4 de la fase v6, cerrada con resultado adv... |
| Ponderacion por razon logaritmica de recuentos | clasificacion de texto | **cerrada** | E2-15 midio el peso de Naive Bayes sobre palabras, sobre caracteres, la razon logaritmic... |
| Analisis discriminante lineal | clasificacion de texto | **nueva** | sin cierre |
| Vecinos mas cercanos como clasificador | clasificacion de texto | **cerrada** | La afinidad de vecindad se midio en V4-F03 y fue nula para ordenar el error. Como clasif... |
| Bosques aleatorios y arboles de decision | clasificacion de texto | **nueva** | sin cierre |
| Validacion cruzada repetida con repeticiones | clasificacion de texto | **nueva** | sin cierre |
| Indexacion por frecuencia o por conteo simple | clasificacion de texto | **cerrada** | Se uso de verdad en la linea ortografica de E2-10, que construyo representaciones de con... |
| Seleccion de caracteristicas por correlacion o informacion mutua | clasificacion de texto | **cerrada** | La especificacion de v8 la da por probada y en efecto el analisis de la fase v2 la midio... |
| Preprocesamiento: quitar marcas, parojos y raices | clasificacion de texto | **cerrada** | PRE-00 a PRE-09 midio diez transformaciones deterministas de superficie, todas descartad... |

### Las nuevas, y por que importan

- **Medidas de complejidad por asimetria y kurtosis** (atribucion de autoria)
- **Palabras inestables medidas por traduccion de ida y vuelta** (atribucion de autoria)
- **Estabilidad de features entre submuestras** (atribucion de autoria)
- **Idiosincrasias y lexico regional** (atribucion de autoria)
- **Seleccion de rasgos por grupo con criterio de informacion** (atribucion de autoria)
- **Correlacion entre rasgos para detectar redundancia** (atribucion de autoria)
- **Analisis discriminante lineal** (clasificacion de texto)
- **Bosques aleatorios y arboles de decision** (clasificacion de texto)
- **Validacion cruzada repetida con repeticiones** (clasificacion de texto)

### La convergencia que mas importa

Dos lineas de la especificacion de v8 tienen un antecesor de 1964 en el material de
clase, y eso no es coincidencia:

1. La **ponderacion por razon logaritmica de recuentos** de la linea P2 es el
   estimador de Mosteller y Wallace, el metodo fundacional de la atribucion de
   autoria. El proyecto ya lo midio en cuatro variantes, asi que la linea llega con
   un precedente adverso y con el techo de +0.0027 de la interseccion de errores.
2. La **estabilidad de features** de la linea P3 es el capitulo 8 de ese mismo
   libro, que el material resume como palabras inestables y mide traduciendo
   documentos de ida y vuelta a cinco idiomas. La traduccion necesita recursos
   externos que el proyecto no tiene, pero **la pregunta es la misma** y la
   operacionalizacion por submuestreo de la especificacion es legitima.

## 4. Comprobaciones programaticas de las lineas de la especificacion

| comprobacion | resultado | ficheros |
|---|---|---|
| `P1_ensamble_inferencia` | **ausente** | ninguno |
| `P2_log_count_ratio` | **presente** | 20_emocion_nbsvm_ortografia_v2.py, 22_artefactos_auditoria_v2.py |
| `P3_estabilidad_features` | **ausente** | ninguno |
| `P3b_medidas_complejidad` | **presente** | v6_00_gap_analysis.py |
| `P3c_agrupacion_jerarquica` | **ausente** | ninguno |
| `P4_regularizacion_conjunta` | **ausente** | ninguno |
| `P5_c_por_etiqueta` | **ausente** | ninguno |
| `P6_peso_por_etiqueta` | **presente** | 30_supervised_weighting_v4.py |
| `P7_corrector_por_margen` | **ausente** | ninguno |
| `P8_calibracion_por_transferencia` | **ausente** | ninguno |
| `P9_desacuerdo_entre_pliegues` | **ausente** | ninguno |
| `P10_segunda_semilla` | **ausente** | ninguno |

## 5. Artefactos reutilizables

| recurso | para que sirve en v8 |
|---|---|
| `features/oof_v03_final.npz` | margenes, entropia y confianza por fila, ya calculados |
| `src/splits.py` | grupos fusionados y pliegues corregidos, el unico protocolo permitido |
| `src/evaluation.py` | metrica primaria, bootstrap pareado y comparacion de errores |
| `tools/verificar_caracteres.py` | puerta de caracteres de la auditoria final |
| 19 modelos `.bin` de FastText | no se usan: ninguna fase nueva de embeddings |

## 6. Coste y riesgo por linea

| linea | coste | riesgo |
|---|---|---|
| P1 Ensemble de cinco modelos en inferencia | bajo: es el mismo pipeline cinco veces | medio: exige una sonda externa para decidir |
| P2 Ponderacion por razon logaritmica de recuentos | bajo | nulo |
| P3 Estabilidad supervisada de features | medio | medio: reducir puede eliminar senal util si la seleccion es agresiva |
| P4 Regularizacion conjunta entre etiquetas | medio | alto: es la misma familia que las cadenas |
| P5 Constante de regularizacion por etiqueta | alto: validacion anidada por etiqueta | alto: es exactamente el patron que produjo el peor traslado |
| P6 Estructura del peso de clase | alto | alto: mismo patron de sobreajuste que la constante por etiqueta |
| P7 Corrector basado en la geometria de los margenes | medio | medio: roza la familia de apilamiento, que ya fallo |
| P8 Calibracion condicionada a la transferencia | bajo | nulo |
| P9 Desacuerdo entre pliegues como senal | bajo | nulo |
| P10 Confirmacion con segunda semilla | bajo | nulo |

## 7. Orden que se deduce

1. **P1** ensemble de cinco modelos en inferencia, con la unica sonda externa. De su
   resultado depende toda la bifurcacion.
2. Si **P1 no mejora**: se cierra sin variantes y se entra por **P3** estabilidad de
   features, que es la forma barata de reducir la dimension sin perder senal, y
   despues **P2** con el precedente ya medido, y por ultimo las lineas de decision.
3. Si **P1 mejora**: se congela la arquitectura y se ejecutan como maximo dos lineas
   mas, que son **P9** desacuerdo y **P10** segunda semilla.

El freno de transferencia de la fase v7 se mantiene: por encima de
+0.005 de mejora local, un candidato solo llega a la mesa de
finales si **no** aumenta la dimension ni la brecha entre ajuste y validacion,
porque los tres unicos intentos de mejorar la metrica local grande del proyecto
perdieron entre 0.007 y 0.013 en el externo.

## 8. Restricciones que se mantienen

Las reglas de la competencia si permiten modelos preentrenados del tipo BERT o
RoBERTa, pero **el proyecto mantiene su prohibicion** desde la fase v5. Es una
decision consciente y se deja escrita: el marco de trabajo es el clasico, y por eso
las tecnicas del material de clase se explotan dentro de ese marco y no fuera.

## 9. Inventario de experimentos heredado por prefijo

| prefijo | experimentos |
|---|---|
| `E2-*` | 58 |
| `EBD-*` | 20 |
| `V4-*` | 14 |
| `PRE-*` | 10 |
| `FIN-*` | 10 |
| `V6-*` | 10 |
| `MOD-*` | 9 |
| `REP-*` | 6 |
| `KAGGLE_TOXIC_2026_PUJ-*` | 6 |
| `E-*` | 5 |
| `ENS-*` | 4 |
| `HIB-*` | 4 |

Fecha: 2026-09-26T02:47:26+00:00.
