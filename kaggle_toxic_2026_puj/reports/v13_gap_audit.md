# V13: auditoria historica de la estimacion de HS (P0)

Generado por `experiments/v13_00_auditoria_hs.py` (V13-00 (P0, auditoria historica de la estimacion de HS)).

**No se entrena nada.** Esto es una lectura de codigo y de resultados historicos, y es
obligatoria antes de gastar presupuesto experimental.

## 1. La conclusion de la auditoria, primero

**8 de los 11 mecanismos que la fase podria explorar ya estan
cerrados, y 3 son genuinamente nuevos.** La fase se reordena en
consecuencia, y eso es exactamente para lo que el enunciado pide la auditoria.

Las cuatro hipotesis de la fase, en su estado real:

| hipotesis | estado real | se ejecuta en v13 |
| --- | --- | --- |
| P1 clasificacion conjunta de estados | ya es V4-E, cerrado como NO_SIGNAL | NO |
| P2 meta-modelo sobre scores OOF de TR y AG | ya es E2-02 y E2-18, con +0,0081 y +0,0102 local y rechazo por transferencia | NO como mecanismo |
| P3 inferencia conjunta continua por scores | ya es el mismo meta y V8-08 | NO |
| P4 perdida de coherencia | su version dura esta medida en H2 y su version de ranking es H3 | NO |
| Aislar si TR y AG llevan informacion sobre los errores de HS | nunca medido | SI |
| Descomposicion de errores por regimen de HS | nunca medida | SI |
| Fraccion del techo del oraculo por mecanismo | nunca medida | SI |

## 2. La matriz que el enunciado exige

| mecanismo | historico | exactamente que hizo | equivalente a la hipotesis | cerrado |
| --- | --- | --- | --- | --- |
| Clasificacion conjunta sobre el espacio de combinaciones (label powerset) | V4-E, experiments/31_joint_label_v4.py, filas V4-E01 a V4-E03 | Un multinomial sobre las combinaciones observadas de (HS, TR, AG), marginalizado de vuelta a P(HS), P(TR) y P(AG). Siete variantes: 6 combinaciones, 4 frecuentes con mapeo de Hamming, 4 frecuentes excluyendo las raras, c | SI, es literalmente P1 | si, con evidencia y con protocolo valido |
| Meta-modelo sobre scores OOF de varias bases, con dependencia entre etiquetas | E2-02 en experiments/15_cadenas_etiquetas_v2.py, y E2-18 en experiments/17_stacking_v2.py | Regresion logistica POR ETIQUETA sobre 12 probabilidades OOF de 4 bases, mas 12 margenes, 12 entropias y 8 rasgos estructurales. Es decir, predecir HS usa tambien las probabilidades y los margenes de TR y AG. Anadida la  | SI, es P2 y tambien P3, y con un conjunto de features MAS RICO que el que P2 propone | SI como mecanismo, y con un motivo que la fase nueva debe respetar: fallo de transferencia medido, no fallo de |
| Factorizacion condicional suave con producto de probabilidades | V6-B, experiments/v6_02_label_factorization.py | p(TR|X) = p(HS|X) * p(TR|HS=1, X) con el condicional entrenado solo en su dominio, y p(AG|X) como mezcla de dos ramas con la probabilidad de HS. El encabezado declara explicitamente sin routing duro | SI, es la formulacion que el enunciado prohibe repetir | si, con mecanismo de fallo identificado |
| Routing duro sobre HS predicho | Post-proceso jerarquico en 02_baseline_tfidf_lr.py, y V12-02 que lo remidio | Si HS predicho es 0, poner TR y AG a 0 | SI | si, y con la causa medida |
| Corrector o combinacion de margenes en inferencia | V8-08, experiments/v8_08_corrector_margenes.py | Un segundo modelo de cuatro variables de entrada sobre los margenes de V03 y el desacuerdo entre los cinco modelos de pliegue | SI para la parte de P3 que combina scores en inferencia | si |
| Perdida que acopla las etiquetas por pares o por margen | V12-03, experiments/v12_03_entrenamiento_por_pares.py | Tres clasificadores entrenados jointemente con un termino de margen por pares que expresa el orden entre las tres etiquetas, sin llave en la prediccion | SI para la parte de P4 que acopla el objetivo | si |
| Regularizacion conjunta entre etiquetas | V8-05, experiments/v8_05_regularizacion_conjunta.py | Un termino que acerca las tres cabeceras de V03 a su media compartida, sin imponer reglas de decision | Parcialmente, en el eje del objetivo compartido | si |
| Cadenas de etiquetas con score previo como feature | E2-01, experiments/15_cadenas_etiquetas_v2.py | Seis ordenes de encadenamiento, con el score y el margen de la etiqueta previa como feature de la siguiente, generados por CV interna para no filtrar | Parcialmente, eninyectar scores de otras etiquetas | si |
| AISLAR si TR y AG llevan informacion sobre los errores de HS | NINGUNO. No existe equivalente | Nada. E2-02 predice cada etiqueta con un conjunto que incluye el score propio de la etiqueta, de modo que nunca se puede atribuir a TR y AG la contribucion a HS | NO | no, esta genuinamente sin probar |
| Descomposicion de los errores de HS por regimen, y rescate y dano por correccion | NINGUNO. No existe equivalente | Nada. E2-02 y E2-18 se evaluaron sobre la metrica agregada, y ningun informe descompone los errores de HS en regimenos de verdad y predicha ni cuenta como una correccion de HS rescata o dana filas de TR y AG | NO | no |
| Fraccion del techo del oraculo que recupera cada mecanismo | NINGUNO. No existe equivalente | Nada. El techo de +0,046271 se midio en v12, y no hay ninguna tabla que lo compare mecanismo por mecanismo | NO | no |

## 3. Los estados conjuntos, medidos antes de imponer ninguna regla

El enunciado pide medir primero que estados aparecen realmente, y el resultado **desactiva
una premisa** que el proyecto tenia registrada como cierta.

| estado | filas | frecuencia | nota |
| --- | --- | --- | --- |
| 000 | 813 | 0.5482 |  |
| 001 | 22 | 0.0148 | viola la invariante |
| 100 | 79 | 0.0533 |  |
| 101 | 124 | 0.0836 |  |
| 110 | 24 | 0.0162 |  |
| 111 | 421 | 0.2839 |  |

**La invariante TR=1 implica HS=1 NO es dura.** Hay 0 filas con TR=1 y HS=0, y 22 con AG=1 y HS=0.

V6-01 la reporto como determinista con 0 casos, y el registro de hoy, sobre el desarrollo,
muestra que hay excepciones. **Es la razon por la que P4 no puede imponer la regla a
mano**: esos casos son estados reales del problema, y borrarlos o forzarles una etiqueta es
exactamente el tipo de decision que este proyecto ha rechazado siempre.

## 4. El hallazgo que mas condiciona la fase: el meta ya se adopto y se rechazo

El meta-modelo sobre scores OOF dio **0.7853** en E2-02, **ADOPTADO en su gate**, y **0.7874** en E2-18, **ADOPTADO en el gate P6**. Respecto a V03 son +0.0081 y +0.0102.

Y a pesar de eso no se convirtio en la submission, porque **el leaderboard dio 0.78163 frente a 0.78306 de V03**.

Eso cambia por completo el plan de la fase. La pregunta ya no es si el meta funciona en
local, porque **esa pregunta tiene respuesta y es que si**. La pregunta es si hay alguna
variante que capture la misma senal sin el fallo de transferencia, y el registro dice que
el fallo viene del **desplazamiento de la distribucion de las entradas** entre el regimen
en el que el meta se entrena y el regimen en el que se infiere, segun documenta
`experiments/32_meta_shift_v4.py`.

Y la formulacion que P2 propone, un meta para HS con scores de TR y AG, es **un subconjunto
de las features que ese meta ya usaba**, porque las 12 probabilidades OOF son 4 bases por 3
etiquetas, luego el score propio de HS esta siempre presente. Es un mecanismo agotado.

## 5. Lo que si es genuinamente nuevo, y es lo que la fase puede aportar

No son mecanismos, son **mediciones que no estan en el registro historico**:

1. **La aislacion de si TR y AG llevan informacion sobre los errores de HS.** E2-02 predice
   cada etiqueta con un conjunto que incluye el score propio de la etiqueta, de modo que
   nunca se puede atribuir a TR y AG la contribucion a HS. Un meta que predice HS **solo**
   con scores de TR y AG responde a la pregunta central de la fase, y nunca se ha hecho.
2. **La descomposicion de los errores por regimen de HS**, con verdad y predicha, contando
   cuantas filas de TR y AG rescata y quantas dana cada correccion de HS. Todos los
   informes historicos reportan metricas agregadas.
3. **La fraccion del techo del oraculo que recupera cada mecanismo**, con el techo de
   +0.046271 como referencia diagnostica.

Las tres son diagnosticos, y por eso son de las que **no gastan presupuesto de busqueda**:
no hay rejilla, no hay modelos nuevos que ajustar mas alla de un meta pequeno, y contestan
la pregunta que la fase formula.

## 6. La equivalencia algebraica con la formulacion prohibida

El enunciado prohibe repetir `P(TR|X) = P(HS|X) * P(TR|HS=1,X)` y el routing duro sobre
`HS_hat`. La comprobacion es **estructural**: se buscan en el arbol sintactico de todos los
guiones las operaciones de multiplicacion donde uno de los factores menciona HS y el otro
menciona TR o AG.

Se encuentran **7** ocurrencias.

- `01_eda_linguistico.py` linea 2913: `conclusion['proporcion_tr1_en_hs1'] * 100`
- `v12_03_entrenamiento_por_pares.py` linea 281: `factor * arriba_hs * (sigma_tr + sigma_ag)`
- `v12_03_entrenamiento_por_pares.py` linea 282: `factor * arriba_hs * sigma_tr`
- `v12_03_entrenamiento_por_pares.py` linea 283: `factor * arriba_hs * sigma_ag`
- `v6_02_label_factorization.py` linea 188: `proba_hs[validacion] * cond_tr_hs1[validacion]`
- `v6_02_label_factorization.py` linea 208: `proba_hs[validacion] * cond_ag_hs1[validacion]`
- `v6_02_label_factorization.py` linea 209: `(1 - proba_hs[validacion]) * cond_ag_hs0[validacion]`

Esas ocurrencias son la formulacion prohibida, y estan en V6-02, que es donde la fase v12
ya lo audito. **Ninguna esta en un guion de v13**, porque todavia no se ha escrito ninguno.

## 7. Decision de la auditoria

**P1, P2, P3 y P4 no se ejecutan como mecanismos**, porque los cuatro ya estan cubiertos y
cerrados, y dos de ellos con un motivo que la fase nueva debe respetar: P2 con un fallo de
transferencia medido, y P4 con una invariante que **no es dura** y que por tanto no puede
imponerse a mano.

**Se ejecuta en su lugar lo que la auditoria declara genuinamente nuevo**: la aislacion de
si TR y AG llevan informacion sobre los errores de HS, la descomposicion de errores por
regimen, y la fraccion del techo del oraculo por mecanismo. Eso responde la pregunta central
de la fase y produce los entregables de diagnostico que el enunciado exige.

Y si la aislacion sale negativa, que es el resultado mas probable dado el registro, la
linea de inferencia estructurada de etiquetas queda **cerrada formalmente**, con la
formulacion que el propio enunciado propone para ese caso.


Fecha: 2026-09-26T23:48:36+00:00. Duracion: 1.8 s.
