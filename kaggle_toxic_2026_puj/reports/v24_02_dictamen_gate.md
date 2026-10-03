# V24-02: dictamen de gate competitivo

Proyecto `kaggle_toxic_2026_puj`, rama `v24`.
Fecha: 2026-09-27.
Referencia congelada: **V03 = 0.78306 LB**. Submissions disponibles: 3.
Registro historico: 22 submissions, 19 con `dOOF` y `dLB` comparables, 19 por
debajo de V03.

Insumo nuevo de esta fase: `experiments/v24_03_sonda_knn_predictor.py` y
`experiments/v24_03b_perfil_de_celdas.py`, que ejecutan por primera vez el kNN /
retrieval como PREDICTOR (el hueco `E_knn_predictor` que `v6_00_gap_analysis.py`
dejo abierto) y lo miden en la misma unidad de decision que el gate: celdas
netas de acierto, no solo corrected_cv.

---

## 1. Respuesta al gate

La linea sentinel exigida, escrita aqui en su forma ASCII y reproducida al final
del fichero en su forma exacta, es:

`NO REALISTIC CANDIDATE - CLOSE`

No hay ninguna tecnica legal, dentro de las reglas del concurso, con probabilidad
razonable de producir una mejora OOF de +0.02 a +0.04, ni una mejora de
leaderboard suficiente para superar V03. El kNN como predictor, que era el unico
hueco abierto que P0 identifico, se acaba de medir y queda en +0.0050 de dOOF:
el **13 % del umbral de gate**.

---

## 2. La pregunta del salto cualitativo

**No. Y la razon no es continuidad con lo ya medido: es aritmetica sobre el
instrumento.**

### 2.1 El umbral de gate esta dentro del ruido del instrumento

El ajuste del historial es `dLB = -0.00920 + 0.2448 * dOOF`. Un modelo que
alcanzase exactamente el umbral de gate (`dOOF = +0.03756`) compraria
`+0.00920` de leaderboard. El error estandar teorico del LB con 1154 filas es
`0.01219`. **El umbral de gate equivale a 0,75 errores estandar del propio
leaderboard.**

Para mover el LB dos errores estandar (0,0244), que es el minimo resoluble con
confianza, haria falta un `dOOF` de aproximadamente **+0.11**. Eso es 8,3 veces
el mejor `dOOF` historico medido (+0,0132 de V23) y 2,9 veces el umbral que este
gate propone. **El umbral que estamos aplicando es mas pequeno que el
instrumento que tendria que verificarlo.** Tres submissions no pueden resolverlo,
y ese defecto no se arregla con un modelo mejor: se arregla con mas test rows, y
el test son 1154 fijas.

### 2.2 La segunda calibracion, ya en el registro, es independiente y coincide

`resultados/leaderboard_log.csv`, entrada `v08_p1_fold_ensemble`, registro dos
constantes medidas y no supuestas:

- **2,8e-05 de F1 macro en el LB por acierto neto.** De ahi, 107 cambios netos
  mueven el LB en 0,003. Verificado: v02 -> v03 cambio 126 celdas y dio +0,00313.
- **Un SE del LB equivale a 435 cambios netos de acierto**, y el filtro escrito en
  el propio registro es: *toda candidata que cambie menos de unas 200 celdas de
  la referencia es indetectable con este presupuesto*.

La unica transicion que mejoro el LB en toda la historia del proyecto fue v02 ->
v03: 126 celdas, todas en la misma direccion (0 -> 1), con el argumento
mecanico de que los umbrales de V03 estan calibrados para la prevalencia que V03
realmente predice. Fue un ajuste de un parametro que ya se habia medido
localmente, no una tecnica nueva. No hay precedente de que una tecnica *nueva*
haya movido el LB.

### 2.3 V03 no esta limitado por capacidad, luego mas representacion no ayuda

Del recalculo de `reports/v24_01_transferencia_complejidad.md`: V03 tiene
**F1 en train de 0,9614** y un gap train-OOF de **+0,1842**. El modelo ya tiene
capacidad de sobra: su restriccion no es la expresividad ni la representacion, son
las **1.483 etiquetas de desarrollo**. Y V03 con menos capacidad tampoco gana: C=1
da 0.77227 y C=1,5 da 0,77296, ambos por debajo de C=2,0. El optimo esta en el
interior de la curva de capacidad, no en un extremo. Eso significa que la unica
via de mejora real es extraer mas informacion de esas 1.483 etiquetas, y eso ya se
ha hecho por todos los ejes disponibles.

### 2.4 Todo lo que igualo a V03 es una recodificacion de la misma informacion

Las dos unicas cosas que han alcanzado la paridad con V03 en OOF son:

- **SVD(300) + SVC RBF** (V23): 0,790410 de corrected_cv, +0,0132 de dOOF. Son
  300 componentes lineales del TF-IDF de V03 y un kernel RBF encima. Es el mismo
  espacio, reordenado. Su dLB fue -0,00368.
- **FastText**: 14 configuraciones, 5 corpus, 3 poolings, ABTT. Alcanzo
  0,783642 de OOF, exactamente la paridad con L1 C=4,0 y con V03, con un techo de
  fusion de **+0,0064** y **0 submissions**.

La mejor mejora OOF disponible en todo el repositorio, en cualquier rama, es el
techo de fusion de FastText: +0,0064. Convertido con la pendiente del ajuste son
**+0,0016 de dLB, o 0,13 errores estandar**. Indetectable.

### 2.5 El error residual es aleatorio, no falta de resolucion local

Dos evidencias medidas que apuntan a lo mismo:

- **V4-F03** (vecinos de 3, la misma vecindad de kNN): el error dentro del grupo
  que discrepa es 0,1798 frente a 0,1619 fuera, con **IC95 [-0,013; +0,0491]**.
  El intervalo contiene cero. La afinidad local **no ordena el error**.
- **`reports/error_analysis.md`**: documenta una clase de error en texto corto
  con URL cuyo significado depende del contenido enlazado, con la conclusion
  explicita de que es *una fraccion irreducible del error con las features de
  texto*. El texto no contiene la etiqueta.

Un residuo que no se puede ordenar con la similitud textual y que depende de
informacion no presente en el texto no es un problema de modelo. Es la firma de
un limite de Bayes dado este corpus.

### 2.6 Que mecanismo cambiaria las cosas, y por que esta prohibido

El unico mecanismo que rompe el bloqueo es **una fuente de informacion de etiqueta
independiente de las 1.483 etiquetas de desarrollo**. Concretamente: un modelo de
lenguaje preentrenado, un corpus externo anotado, o entrenamiento con el test.
Las tres estan prohibidas por las reglas del concurso. Todo lo demas es otra
forma de exprimir las mismas 1.483 etiquetas, y el historial dice de forma
consistente que eso se agoto hace tiempo: 19 de 22 submissions por debajo de V03,
incluidas las que tenian mejor evidencia interna (V23, 5/5 pliegues, McNemar
p = 0,0452).

---

## 3. Evaluacion de kNN / retrieval: los seis puntos

Todos los numeros salen de `experiments/v24_03_sonda_knn_predictor.py`, con el
vecindario construido **dentro de cada pliegue de entrenamiento**, de modo que
ningun vecino de una fila de validacion puede proceder de su propio grupo. Los
pliegues son los canonicos `StratifiedGroupKFold(5)` sobre los grupos fusionados
de `src/splits.py`, y `src.splits._verificar_sin_fuga` aborta si hay grupos
compartidos. No hay leakage por construccion.

Referencia reproducida: **corrected_cv 0,777211**, identica byte a byte al valor
canonico, con **757 celdas de error de 4.449**. Nota de protocolo: la prediccion
congelada de V03 es `proba >= [0,44; 0,51; 0,46]` **sin** post-proceso
jerarquico; el post-proceso cambiaria 55 celdas y no se aplica.

Se evaluaron dos espacios: `union` (el mismo TF-IDF palabra(1,2) + char_wb(3,5)
de V03) y `char` (solo char_wb(3,5), donde la geometria local difiere mas).

### A. Que informacion usaria que V03 no usa

La estimacion local no parametrica `P(y | x)` por promedio de los k vecinos
cercanos, en lugar del logit lineal global de V03. Son estimadores
estadisticamente distintos: el de V03 es una frontera recta en el espacio de
27.431 dimensiones; el de kNN es un promedio de etiquetas en una bola al
rededor de x.

**La cantidad de informacion no marginal es pequena y esta medida.** Correlacion
de Pearson entre el voto kNN ponderado y la probabilidad de V03, celda a celda:

| espacio | correlacion con la probabilidad de V03 |
|---|---:|
| `union` (mismo espacio) | **+0,9291** |
| `char` (espacio distinto) | **+0,8512** |

En el mismo espacio, kNN es una recomputacion de la funcion de decision de V03
con 7 % de varianza adicional. La unica razon para usar el espacio `char` era
buscar una geometria complementaria; ahi la redundancia sube a 85 %, porque el
bloque de caracteres ya es la mitad de las features de V03.

### B. Que porcentaje de los errores de V03 tiene un vecino suficientemente confiable

**Si hay cifra, y es 1,6 %.**

Definicion de vecino confiable, fijada antes de mirar los resultados: cosimilitud
del vecino mas cercano por encima del percentil 90 **y** a lo sumo un desacuerdo
entre los 5 vecinos mas cercanos.

| espacio | filas confiables | celdas de error de V03 cubiertas | cobertura |
|---|---:|---:|---:|
| `union` | 138 de 1.483 (9,3 %) | **12 de 757** | **1,6 %** |
| `char` | 127 de 1.483 (8,6 %) | **23 de 757** | **3,0 %** |

El vecindario es escaso en todo el corpus, no solo en los errores: la
cosimilitud del vecino mas cercano tiene mediana 0,2953 (`union`) y 0,2719
(`char`); solo 121 filas (8,2 %) superan 0,5 y **6 filas (0,4 %) superan 0,7**.
Con 1.731 textos unicos, cero duplicados exactos y solo 127 filas (7,3 %) en un
grupo multi-fila, **no hay atajo de recuperacion**: la mayor similitud observada
es 0,8217 y el vecindario denso no existe. Por eso la cobertura de B es
estructuralmente baja y no un parametro que se pueda subir.

### C. Que precision tendria el vecino en OOF sin leakage

Precision del vecino cuando **contradice a V03**, es decir cuando la regla lo
adoptaria. Pliegues agrupados, sin leakage.

| espacio | aciertos del vecino | volteos | **precision** |
|---|---:|---:|---:|
| `union` | 3 | 20 | **0,1500** |
| `char` | 6 | 56 | **0,1071** |

**Peor que una moneda al aire, y muy por debajo de la propia tasa de acierto de
V03 (0,7772).** Cuando el vecino discrepa de V03, acierta entre el 11 % y el
15 % de las veces. Este es el numero que decide el gate por si solo: cualquier
rega que sustituya V03 por el vecino en las celdas donde discrepan pierde de
forma masiva y esperada.

### D. Si los vecinos son complementarios a V03 o lo reproducen

**Lo reproducen, y ademas pierden.** No son complementarios en ningun sentido
util.

| espacio | celdas donde discrepan | gana V03 | gana kNN | McNemar exacto | Jaccard de errores |
|---|---:|---:|---:|---:|---:|
| `union` | 905 | 621 | 284 | **p < 0,0001** | 0,343 |
| `char` | 1.186 | 866 | 320 | **p < 0,0001** | 0,269 |

En las celdas donde los dos modelos discrepan, V03 gana por 2,2 a 1 (`union`) y
por 2,7 a 1 (`char`), con significacion overwhelming. El perfil de errores es
mayoritariamente compartido: 473 celdas fallan los dos en `union`, sobre 757
errores de V03, es decir **el 62 % del error de V03 es tambien error de kNN**.
Repetirlo no es complementaridad.

### E. Si puede generar volumen suficiente de correcciones

**No.** V03 falla 757 de 4.449 celdas. Esto es lo que las tres formulaciones
realizables producen, con el perfil de celdas medido:

1. **Regla de sustitucion** (adoptar la etiqueta del vecino 1 cuando el vecino es
   confiable): 3 rescates contra 17 danos = **-14 celdas netas**, corrected_cv
   0,7717, **dOOF -0,0055**. En `char`: 6 contra 50 = -44, dOOF -0,0145.
2. **Mezcla convexa de probabilidades**, barrido completo de peso kNN de 0,05 a
   0,80 en pasos de 0,05, en los dos espacios. Optimo en `union` con peso 0,30:

| peso kNN | celdas cambiadas | rescates | danos | neto | dOOF | dLB implicito | % de 1 SE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,10 | 26 | 16 | 10 | +6 | +0,0000 | +0,00017 | 1,4 % |
| 0,20 | 56 | 33 | 23 | +10 | +0,0002 | +0,00028 | 2,3 % |
| **0,30** | **103** | **66** | **37** | **+29** | **+0,0050** | **+0,00081** | **6,7 %** |
| 0,40 | 146 | 88 | 58 | +30 | +0,0041 | +0,00084 | 6,9 % |
| 0,50 | 190 | 103 | 87 | +16 | -0,0015 | +0,00045 | 3,7 % |

**La mejor candidata de kNN cambia 103 celdas de 4.449 para ganar 29 netas.** Es
un 6,7 % de un error estandar del LB. El filtro escrito en el registro del
proyecto dice que menos de 200 celdas es indetectable: kNN esta a la mitad de ese
umbral. Gana en 4 de 5 pliegues, lo cual es un punto a su favor frente al patron
de V6-B (1 de 5), pero 4 de 5 con +29 celdas netas no cambia la conclusion.

**Sobre el techo aparente.** Existe una ORACLE que elige por celda entre V03 y el
vecino: 0,8583 (`union`) y 0,8685 (`char`), es decir +0,0811 y +0,0913 de dOOF,
por encima del umbral. **No es realizable y no debe reportarse como techo
real.** Para construirla hace falta saber, celda a celda, cual de los dos modelos
acierta, y eso no es observable. Y los unicos signals disponibles para decidirlo
son exactamente `knn_disagreement` y `sim_cercana`, que V4-F03 ya midio y
resultaron **no ordenar el error** (IC95 [-0,013; +0,0491]). El factor entre lo
realizable (+0,0050) y la oracle (+0,0811) es de 16, y la razon de ese factor es
justamente que la oracle exige resolver el problema que el vecindario no resuelve.
Medimos ademas la correlacion entre las dos medidas de V4-F03 y la tasa de celdas
donde el voto kNN contradice a V03: `knn_disagreement` **+0,2905**, `sim_cercana`
**-0,1628**. La medida de ambiguedad y el predictor kNN estan relacionados pero
no son el mismo objeto, asi que la distincion de F es real; lo que no cambia es
el resultado.

### F. Si existe una formulacion que no sea equivalente a V4-F03 ni a V4-F01-F04

**Si, formalmente, y la hemos ejecutado.** La distincion es real en el
formulario: en V4-F03 el vecindario entra como *feature* que explica el error
(`knn_disagreement`, `sim_cercana`) en una segunda etapa; aqui el voto de los
vecinos **es** la prediccion. No son el mismo objeto, y la sonda nueva no
reproduce el resultado de V4-F03 porque no lo pretendia.

**Pero la distincion es formal, no informacional.** Se usa el mismo operador de
similitud, el mismo espacio, y el resultado es 93 % redundante con V03 (punto A).
Y el resultado de esta sonda es una generalizacion del null ya conhecido, no una
excepcion:

- **V4-F01-F04** propago etiquetas por similitud de near-duplicado y su techo fue
  nulo. kNN como predictor es **el mismo operador de similitud aplicado mas
  ampliamente** (coseno sobre TF-IDF en vez de agrupacion por near-duplicado), y
  en su version sin condicionamiento da **0,7567** de corrected_cv, es decir
  0,0206 *por debajo* de V03. Ampliar un operador cuyo caso restringido ya fue
  nulo no produce un techo positivo; produce el mismo cero.

Ademas, la unica direccion en la que la formulacion podria apartarse de la
redundancia (usar el espacio `char`, con geometria distinta) se midio y es
**peor**: 0,7121 de corrected_cv en solitario, cobertura de B del 3,0 % pero
precision de C del 0,1071, y dOOF de la mejor mezcla +0,0022. La complementariedad
de espacio no existe en este corpus.

**Conclusion de los seis puntos: kNN como predictor queda cerrado con evidencia
medida, no con suposicion.** El hueco `E_knn_predictor` que P0 identifico como
abierto ya no esta abierto. Pasa a la columna de familias cerradas con el
resultado: kNN en solitario 0,7567 / 0,7121, mejor mezcla +0,0050 de dOOF,
103 celdas, +29 netas, precision del vecino al discrepar 0,15.

---

## 4. Clasificacion de candidatos

Tres mecanismos, como maximo. Los tres son **LOW**. Ninguno pasa el gate.

### Candidato 1: kNN / retrieval como predictor. LOW

- **Justificacion de la etiqueta:** no es una_estimacion de riesgo, es una
  medicion. La sonda se ejecuto completa y el mejor resultado realizable es
  +0,0050 de dOOF con 103 celdas y +29 netas, es decir 6,7 % de un error estandar
  del LB y 13 % del umbral de gate.
- **Evidencia concreta:** corrected_cv de kNN en solitario 0,7567 (`union`, k=25)
  y 0,7121 (`char`), ambas por debajo de las 0,7772 de V03; precision del vecino
  al contradecir a V03 de 0,1500; cobertura de 1,6 % de los errores de V03;
  McNemar p < 0,0001 a favor de V03; correlacion 0,9291 con la probabilidad de
  V03.
- **Tamano del efecto:** OOF +0,0050 en el mejor caso, LB implicito +0,0008
  (0,07 errores estandar). Negativo en las formulaciones de sustitucion
  (-0,0055 y -0,0145).
- **Coste:** bajo. 3,2 s de ejecucion, sin GPU, sin submission. El coste de la
  sonda ya esta pagado.
- **Riesgo de sobreajuste:** bajo, y **no es el mismo riesgo que V23**. V23
  cambio 240 celdas con 136 rescates y 104 danos y por eso era una sonda de
  real; aqui la mezcla al 0,30 cambia 103 celdas con 66 rescates y 37
  danos, o sea una tasa de acierto del 0,641, y el peso 0,30 se eligio sobre el
  mismo OOF que se reporta. Aun asi, la magnitud del efecto (0,07 SE) es
  irrelevante sea cual sea el riesgo: **no hay sobreajuste que corregir porque no
  hay efecto que sobreajustar.**

### Candidato 2: sospecha por par instancia-etiqueta y reponderacion por ruido. LOW

- **Justificacion de la etiqueta:** el objetivo del mecanismo es inobservable
  desde el texto. Reponderar exige adivinar cuales celdas estan mal etiquetadas,
  y todos los signals disponibles para adivinarlo ya se midieron y fallaron:
  `knn_disagreement` y `sim_cercana` no ordenan el error (IC95 contiene cero, V4-F03);
  la interaccion de scores cruzados cayo **7 de 7** veces (V16-V20); y la densidad
  local, que aca se midio de nuevo, cubre el 1,6 % de los errores.
- **Evidencia concreta:** es la familia F de `v6_00_gap_analysis.py`, marcada
  `ya_probada: no` y `prioridad: media`, lo cual es correcto: no esta cerrada por
  medida, esta cerrada por **inaccesibilidad del objetivo**. El techo esta
  acotado por el analisis de duplicados de V4: solo 2 clusters contradictorios
  (84 tuits, 4,85 %) y V03 ya acierta bien dentro de ellos (error 0,046 frente a
  0,218 fuera). Ademas reponderar **no corrige**: solo puede desplazar peso, asi
  que el efeito esperado es una fraccion del techo y nunca suyu entirety.
- **Tamano del efecto:** OOF estimado entre 0,000 y +0,005. LB por debajo de
  0,15 SE.
- **Coste:** medio. Un modelo de sospecha por par, con validacion anidada para no
  ajustar el mismo OOF dos veces, mas un refit. Varios dias de trabajo para
  obtener un efecto no medible.
- **Riesgo de sobreajuste:** **alto, y aqui si es el riesgo dominante.** Es un
  meta-modelo que consume scores que ya fueron ajustados sobre el mismo
  desarrollo. La familia de stacking y meta-modelos ya se ejecuto 5 veces con
  0,7678, y la de correccion selectiva 7 veces con 0,77663 de LB. La evidencia
  previa dice que este patron produce seleccion de ruido.

### Candidato 3: aumento de datos desde los corpus sin etiquetar legales. LOW

- **Justificacion de la etiqueta:** el mecanismo es genuinamente distinto de
  todo lo medido (el gap analysis marca `J_augmentation` ausente y
  `J_normalizacion` presente: se probo quitar ruido diez veces, nunca crear
  instancias), pero esta **acotado por arriba** por un resultado que ya existe en
  el repositorio.
- **Evidencia concreta:** aumentar para mejorar la representacion lexical es
  exactamente lo que hizo FastText con corpus externos legales (Wikipedia, mc4,
  50M, 5M): 14 configuraciones, 3 poolings, ABTT, y el techo de fusion fue
  **+0,0064** de OOF, con 0 submissions. Cualquier aumento queImprove la
  representacion tiene que pasar por ese techo o superarlo, y superarlo exigiria
  una fuente de corpus o de metodo que el proyecto ya-barrio y no tiene. Las
  tecnicas de aumento que no dependen de corpus (sustitucion por sinonimos,
  back-translation) o necesitan un encoder externo prohibido, o degradan
  justamente en esta tarea, donde la negacion y la intensidad sostienen la
  etiqueta: `reports/estado_arte_v4.md` documenta que la linea de ruido de
  etiquetas queda cerrada con evidencia.
- **Tamano del efecto:** OOF en el mejor caso +0,0064, LB implicito +0,0016
  (0,13 SE).
- **Coste:** alto. Pipelines de corpus, y 5 corpus ya se construyeron y auditaron.
- **Riesgo de sobreajuste:** medio. El precedente es directo: L1 C=4,0 y la
  fusion de FastText alcanzan ambos 0,783642 de OOF y **ninguno envia**, porque
  la evidencia OOF de esa magnitud ya se demostro que no transfiere (V23).

### Nota sobre un falso positivo del registro

`v6_00_gap_analysis.py` marca `I_word_char_weight` como **ausente** y lo llama
*el track mas barato del proyecto*. **No es una laguna: fue medido.** Es la sonda
`ch3` de V16, `toxic_submission_v16_ch3_peso_char_070.csv`, con `peso_char = 0.7`:
183 celdas distintas, **-100 aciertos netos**, corrected_cv 0,754593, **dOOF
-0,0226**, y `supera_presupuesto_de_107_celdas = True`, o sea que cambio celdas
suficientes para ser detectable y aun asi perdio. La bandera `ausente` del
registro es un artefacto de inventario, no un hueco real. Queda anotado para que
nadie lo vuelva a proponer como pista.

---

## 5. Decision

**Cerrar.** V03 se queda congelado. Las 3 submissions sin gastar se conservan
para una eventual revision del enunciado o del conjunto de reglas, no para
explorar mecanismos con efecto esperado por debajo del 15 % de un error estandar
del leaderboard.

Tres razones, en orden de peso:

1. **kNN como predictor queda cerrado con medicion propia.** El hueco
   `E_knn_predictor` era real y ya no lo es. +0,0050 de dOOF, 103 celdas,
   precision del vecino al discrepar de 0,15, 93 % de redundancia con V03.
2. **El umbral de gate es mas pequeno que el instrumento que lo verifica.** Un
   dOOF de +0,03756 compra +0,0092 de LB contra un SE de 0,01219. Ninguna
   cantidad de candidatos agujero a este presupuesto.
3. **Las tres familias no cerradas que quedan son LOW por motivos distintos, y
   ninguno es LOW por falta de evidencia, sino por techo medido o por objetivo
   inaccesible.** kNN por redundancia, sospecha de ruido por inaccesibilidad del
   objetivo, aumento por el techo de FastText.

La regla operativa que sale de este dictamen, para cualquier fase futura: antes
de proponer una candidata, exigir que su **perfil de celdas** supere las 200
celdas del filtro del registro y que su dOOF estimado supere **+0,0376**. Ningun
mecanismo no parametrico, de reponderacion o de aumento cumple las dos cosas
hoy. Y registrar que la pregunta de si existe un salto cualitativo tiene una
respuesta clara y negativa: requiere informacion de etiqueta que las reglas del
concurso prohiben obtener.

---

## 6. Trazabilidad

| artefacto | contenido |
|---|---|
| `experiments/v24_03_sonda_knn_predictor.py` | kNN como predictor, 2 espacios, k = 1/3/5/25, puntos A a F, barrido de mezcla |
| `experiments/v24_03b_perfil_de_celdas.py` | perfil de celdas en la unidad del gate, con las constantes medidas del registro |
| `reports/v24_03_sonda_knn_predictor.md` | salida de la sonda |
| `reports/v24_03_sonda_knn.json` | numeros estructurados |
| `reports/v24_03b_perfil_de_celdas.md` | tabla de celdas y contraste con las sondas enviadas |

Constantes heredadas, todas medidas y ninguna estimada en esta fase:
`2,8e-05` de F1 de LB por acierto neto, `0,01219` de SE del LB con 1154 filas,
`0,2448` de pendiente dOOF -> dLB, `0,03756` de umbral de gate, `0,777211` de
corrected_cv de V03, `757` celdas de error de V03 sobre `4.449`.

Correccion registrada: la primera version de `v24_03b_perfil_de_celdas.py`
calculaba el F1 como `|interseccion| / |union|`, que es el indice de Jaccard y
daba 0,6364 para V03 donde el valor canonico es 0,7772. Corregido a
`sklearn.metrics.f1_score`. Los recuentos de celdas, que son los que deciden el
gate, no dependen de la formula y no cambian.

NO REALISTIC CANDIDATE — CLOSE
