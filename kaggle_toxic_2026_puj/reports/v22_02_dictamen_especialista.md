# V22-02 Dictamen especialista: decision competitiva sobre las dos lineas abiertas

Fecha: 2026-09-27
Proyecto: kaggle_toxic_2026_puj
Rama: v22
Rol: especialista datos tabulares / NLP supervisado
Alcance: dos preguntas cerradas. No se proponen familias alternativas.

```
=== Version v22-02 - dictamen sobre ABTT de documento (Arora et al.) y SSWE ===
Fecha: 2026-09-27
Features: 27.431 char n-grams + bloque denso de 200 dims (fusion)
CV Score: control D=0 corrected_cv = 0.781766 (delta V03 +0.004555)
LB anterior: 0.78306 (V03)
Submissions disponibles: 4 (21 en el historico, 19 por debajo de V03)
Cambios vs v22-01/v22-03: sin ejecucion de codigo. Solo decision.
Veredicto: NO CANDIDATE JUSTIFIES SUBMISSION en ninguna de las dos lineas.
```

## Respuesta en una linea

Ninguna de las dos lineas merece ejecutarse. Las dos chocan contra el limite ya
diagnosticado y, en el caso de ABTT de documento, sus dos mitades tienen
evidencia negativa independiente dentro de este mismo pipeline. Se recomienda
cerrar V22.

---

## Pregunta 1: ABTT variante Arora, Liang y Ma (ICLR 2017, arXiv:1703.11175)

### 1. Respuesta directa

**NO merece ejecutarse.**

### 2. Evidencia concreta

La variante de Arora et al. es la composicion de dos operaciones: media
ponderada por `a/(a+p(w))` y eliminacion de la componente principal sobre la
matriz de documentos. Lo relevante es que **cada una de las dos mitades ya fue
medida en este pipeline y las dos perdieron**, y ninguna midio lo mismo.

**Mitad A, la ponderacion por frecuencia: ya perdida.** V5 ejecuto el pooling B
ponderado por TF-IDF sobre los mismos 1.483 documentos y **perdio contra el mean
pooling**. Es la misma intervencion: reponderar el promedio de la oracion con un
factor derivado de la frecuencia del termino. Ya se sabe el resultado.

**Mitad B, la eliminacion de la componente principal: ya perdida, dos veces.**
V22-03 midio ABTT de Mu y Viswanath (arXiv:1702.01417) sobre los 6.059
vectores de tipo, con el mismo embedding, el mismo clasificador y el mismo
diccionario:

| D | var. eliminada | corrected_cv | delta control | emb solo | neto celdas |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -- | 0.781766 | -- | 0.666085 | +10 |
| 1 | 0.0531 | 0.779935 | -0.001831 | 0.659548 | +1 |
| 2 | 0.0863 | 0.775645 | -0.006121 | 0.661255 | -12 |
| 5 | 0.1684 | 0.778144 | -0.003622 | 0.658931 | -2 |

El dato decisivo no es que D=1 empeore, es que **el embedding aislado tambien
 empeora** (-0.006537). Es decir, la perdida no viene de la interaccion con la
fusion, viene del hecho de que la representacion quedo peor en si misma. Mover la
geometria del bloque en la direccion de la componente dominante degrada el
bloque, y la fusion no la amortigua. La segunda lectura relevante: la primera
direccion principal explica solo el 5.3 por ciento de la varianza y las seis
primeras el 18.8. **El espectro es plano.**

**El contraargumento del encargo, evaluado.** Es cierto que el espectro de
documento puede diferir del de vocabulario, y no se puede descartar a priori. Pero
el contraargumento no cambia la decision por tres razones concretas:

1. En una matriz de documentos obtenida por mean pooling sobre vectores de tipo,
   la componente principal es casi siempre un modo de longitud y frecuencia
   media del documento, no una direccion de clase. Es decir, es la direccion que
   uno **queria** cuando el pooling ponderado por frecuencia perdio. Si la
   hipotesis fuera correcta, la eliminacion deberia autorizar una ganancia, y lo
   que se midio en el espacio de tipos fue una perdida. La analogia no huele a
   una promesa de exito.
2. Quitar la componente dominante de un bloque cuya contribution a la metrica
   esta medida en 0.0063 de rango no tiene canal hacia el AUC. El objetivo de
   esta competencia no es la calidad geometrica de un vector de oracion, es
   AUC de un documento bajo una fusion. La ganancia publicada de ABTT se mide en
   un regimen de metrica (coseno de similitud en STS) que no es el regimen de
   esta competicion.
3. Un punto mas en un vecindario plano no es informacion accionable, y esta es
   la razon que decide. La familia completa converge a la misma perturbacion:
   el control D=0 mueve 136 celdas con saldo +10, y la mejor configuracion
   conocida, FUSION-WS-3, movia 137 con saldo +11. V21 ya midio que esa
   perturbacion tiene **p = 0.3930 con McNemar exacto**, es decir, no es
   distinguible de cero. Si se midiera hoy la variante de documento y saliera
   +0.0005, la lectura correcta no seria "funciona", sino "el maximo de otro
   punto de ruido del mismo vecindario". Eso no es una decision de submission.

**Regla de decision derivada.** Un resultado positivo aqui no seria accionable
de todos modos, por un hecho historico de este proyecto: en la ventana V16 a V20
fueron **7 de 7 a la baja con evidencia OOF a favor**. La transferibilidad al
leaderboard de las ganancias OOF de esta familia ya esta medida en cero sobre
siete intentos. Por lo tanto, el valor de ejecutar la Pregunta 1 no es
seleccionar, es **desinformar**: costo de aproximadamente una hora para agregar un
punto mas al mapa de un vecindario que ya esta cartografiado.

### 3. Coste estimado

Bajo en tiempo, alto en ruido. No hay entrenamiento: es algebra matricial sobre
una matriz de 1.483 por 200 mas la cadena downstream ya existente. Estimado
**20 a 40 minutos** incluyendo arnes de medicion, control de cordura de
desviacion 0.000000 y reporte. El coste no es el argumento para descartarla.

### 4. Riesgo de sobreajuste, y por que no es el mismo riesgo

Aqui hay que ser honesto: **el riesgo es el mismo** que en las lineas ya
descartadas, y esa es precisamente la razon del no.

Las lineas descartadas por V22-01 y V22-03 no tienen capacidad de ajuste: no se
entrena nada, no hay riesgo de sobreajuste clasico, solo ruido de medicion. La
variante de Arora et al. tampoco entrena nada, luego comparte exactamente ese
perfil de riesgo: riesgo de **seleccion sobre ruido**, no de estimacion. No aporta
una via de escape, la replica.

El riesgo concreto es cumulativo y ya se esta pagando. Cada miembro que se anade
a una familia cuyo maximo es indistinguible de cero aumenta la probabilidad de
un falso positivo por multiplicacion, y el falso positivo se paga con una de las
4 submissions restantes. Con 19 de 21 submissions historicas por debajo de V03,
la consecuencia marginal de un falso positivo no es un resultado malo, es la perdida de
la unica submission que aun parecia superar a V03.

### 5. Probabilidad de mejora en leaderboard

**Nula.**

Se apoya en tres hechos independientes, no en uno:
- Las dos mitades tienen evidencia negativa propia en este pipeline (V5
  pooling B; V22-03 ABTT de tipos con perdida del embedding aislado).
- La perturbacion de la familia no es distinguible de cero (p = 0.3930).
- La transferencia OOF -> LB de esta familia es 0 de 7 (V16 a V20).

Que la mejora de embedding aislado sea 0.0297 y la de la fusion 0.0063, con
factor 4.8, acota ademas el techo empirico: cualquier movimiento de la
representacion se comprime un factor 4.8 al entrar en la fusion, y la banda de
ruido de la fusion es de al menos +-0.006. El techo y el piso coinciden.

### 6. Configuracion y criterio (no aplica por ser NO)

Si se quisiera ejecutar por registro, la configuracion minima seria una sola
combinacion, a = 0.001 tal como en Arora et al., D = 1, sin renormalizar, sobre
los mismos 1.483 textos normalizados, con control de cordura de 0.000000
contra el mismo pool de celdas de V21. Criterio de rechazo declarado **de
antemano**: se descarta si el delta corrected_cv no supera +0.004, porque por
debajo de ese valor esta dentro de la banda de ruido de la familia. Eveno si lo
superara, el criterio de aceptacion real seria la comparacion contra las 4
selecciones disponibles y no la significancia local. Se recomienda no gastar ni
ese tiempo.

---

## Pregunta 2: SSWE, embedding con objetivo dirigido a la tarea

### 1. Respuesta directa

**NO merece ejecutarse.**

### 2. Evidencia concreta

SSWE es la unica opcion de V22 que **no** es un re-parametrizacion de la
representacion que ya se midio, y eso es a su favor y hay que decirlo primero:
es la unica palanca que puede mover el bloque a un subespacio que la geometria
no supervisionada ya no alcanza. Ese argumento no aplica a la Pregunta 1 y por eso la
Pregunta 2 merece un analisis mas exigente. Aun asi, cuatro hechos la cierran.

**Hecho 1, la redundancia con el clasificador lineal.** El objetivo supervisado
de Liu et al. (COLING 2018) y Tang et al. (ACL 2014) entrena los vectores para
predecir la etiqueta. El bloque de 200 dims entra en la fusion a traves de un
clasificador lineal. El gradiente de SSWE y el del clasificador lineal apuntan
al mismo sitio: el subespacio relevante para la etiqueta. Se obtiene un bloque
mas alineado con la clase, que es exactamente el bloque mas redundante con el
que ya existe. El clasificador lineal sobre 27.431 char n-grams ya consume toda
la senal supervisada disponible y ya la consume mejor, porque los char n-grams
discretizan el texto mientras que una bolsa de 200 numeros lo suaviza. La
pregunta del encargo, si el embedding puede aprender algo que ese clasificador
no vea, tiene aqui su respuesta: solo veria lo que el clasificador ya ve, con
menos resolucion. Hay una tension logica entre "ser util" y "ser distinto" que
el objetivo supervisado resuelve siempre en contra de "ser distinto".

**Hecho 2, el precedente NB-SVM, que es la forma analoga mas parecida.** Es la
manera mas directa de reescalar la representacion por clase con senal supervisada, ya se
implemento bien en este proyecto y **perdio: 0.7638 contra 0.7772**. El modo de
fallo fue binarizar y descartar el term frequency, que en tweets toxicos es
senal. SSWE tiene un modo de fallo analogo: al colapsar cada tipo a su afinidad
con la clase, el objetivo conserva **si** el tipo es toxico y descarta **cuanto
aparece**. Eso es exactamente la misma perdida de informacion que provoco la
derrota del NB-SVM, y ocurre por la misma razon: la frecuencia es senal en
tweets toxicos y ningun objetivo que la Ignore la va a recuperar despues.

**Hecho 3, la escala de etiquetas y el circulo vicioso de la regularizacion.** Este
es el argumento decisivo y es cuantitativo. Se dispone de **1.483 etiquetas
binarias**. El bloque tiene 6.059 vectores de tipo por 200 dimensiones, es decir
del orden de 1.2 millones de parametros que el objetivo supervisado moveria. Para
que SSWE no sobreajuste 1.483 ejemplos hace falta una regularizacion fuerte; con
regularizacion fuerte el bloque se contrae hacia su inicializacion, que es
exactamente la solucion no supervisionada de FastText que ya se midio y ya esta
por debajo. Es decir: **el unico regimen de hiperparametros en el que SSWE es
seguro es el regimen en el que SSWE es un no-op.** No es un problema de
hiperparametros, es un problema de identidades: el conjunto de soluciones seguras
y el conjunto de soluciones distintas del baseline no se intersecan.

**Hecho 4, el techo empirico.** El limite estructural ya diagnosticado por V5
sigue intacto porque SSWE no cambia la densidad: 200 numeros densos dentro de una
fusion con 27.431 features dispersas, con rango de movimiento del embedding
aislado 0.0297 y de la fusion 0.0063, factor 4.8. El argumento es que la
supervision no compra mas ancho, solo lo orienta. Y la orientacion disponible es la
misma que ya consume el clasificador lineal.

**Estado de la evidencia externa, en contra.** El informe de V21 concluyo que
en espanol toda la evidencia publicada favorece al modelo con embeddings, y que
la unica palanca con prueba solida en espanol es el preprocesamiento del texto de
la tarea. V22-02 acaba de medir esa palanca: era correcta, con delta de
+0.012871 a favor del pipeline historico, y la cobertura de desarrollo es
1.0000 normalizado contra 0.7414 sin normalizar. Es decir, la unica palanca
publicada con prueba solida en espanol ya esta en su valor optimo y no queda
recorrido debajo de ella.

### 3. Coste estimado

El mas alto de V22 por un orden de magnitud. Reentrenar skipgram sobre
wikipedia_es 10M son **decenas de minutos por configuracion**, y SSWE necesita
al menos tres brazos porque hay que elegir entre variantes de perdida: margin
loss de Liu et al. frente a negative sampling de Tang et al., y ventana y
negative distintos. Con downstream en cada uno, el escenario minimo honesto es
**3 a 5 horas de calculo sin contar repeticiones por semilla**, y con protocolo
anidado para que el corrected_cv sea confiable, mas. Es la unica opcion de V22
que exige entrenamiento real, y por lo tanto la unica que puede consumir la
sesion sin llegar a un veredicto defendible.

### 4. Riesgo de sobreajuste, y por que aqui si es un riesgo distinto

Aqui el riesgo **si** es distinto, y conviene ser preciso porque es el unico
argumento de la familia que no esta ya saldado:

- Las lineas descartadas por V21, V22-01 y V22-03 tienen **riesgo de seleccion
  sobre ruido**: no entrenan nada, el unico modo de fallo es tomar el maximo de
  una familia de mediciones cuya senal es cero.
- SSWE agrega **riesgo de estimacion**: con 1.483 etiquetas y un bloque de
  alrededor de 1.2 millones de parametros, el objetivo puede ajustar ruido de
  etiquetas. Un corrected_cv honesto con 1.483 ejemplos es ruidoso, y la
  diferencia entre una mejora real de +0.002 y un sobreajuste de +0.002 es
  indistinguible con este tamano de muestra.
- Ademas se suma una **segunda capa de seleccion**: elegir entre 3 o 4 perdidas
  y luego entre semillas, tomando el maximo sobre una banda de ruido, es la
  forma mas rapida de fabricar una mejora aparente. Con 4 submissions, un
  maximo sobre 12 resultados ruidosos es exactamente la maquina de generar falsos
  positivos.

Ese riesgo distinto es un riesgo **nuevo y no cubierto** por la evidencia previa.
Y por si mismo es un argumento de no: introducir un riesgo no cubierto en la
ventana final, con 4 submissions y 7 de 7 caidas recientes, es un mal negocio.

### 5. Probabilidad de mejora en leaderboard

**Baja.**

No nula, y la distincion es deliberada. SSWE es el unico mecanismo de V22 que
puede producir una representacion genuinamente distinta y no una
re-parametrizacion, asi que no se puede aplicar el argumento de "misma
perturbacion" que cierra la Pregunta 1. Pero la probabilidad sigue siendo baja
por la suma de: redundancia con el clasificador lineal (hecho 1), precedente
NB-SVM negativo (hecho 2), circulo seguro-es-no-op (hecho 3), techo de
densidad (hecho 4), y la unica palanca con prueba solida en espanol ya en su
optimo (estado de la evidencia externa).

### 6. Configuracion y criterio (no aplica por ser NO)

Se documenta el criterio de decision de forma explicita para que quede registrado
y para que no se reabra sin datos nuevos. Si se ejecutara, la configuracion minima
seria: skipgram 200d, ventana 5, minCount 3, subpalabras 3 a 6, neg 10,
**subsampling reactivado**, alpha 0.05, y margin loss de Liu et al. sobre 1.483
etiquetas, un solo fold de control mas OOF completo, con aceptacion si el delta
corrected_cv supera +0.004 **y** el embedding aislado mejora **y** el delta se
mantiene con semilla distinta. Criterio de rechazo: cualquier cumplimiento
parcial. Los tres criterios a la vez son necesarios, porque cada uno ataca una
de las tres formas de fallo identificadas. Aun asi, la recomendacion es no
ejecutar: la combinacion de los tres criterios exigidos y la banda de ruido
estimada hace que la probabilidad de superarlos sea baja, y el coste es el mayor
de V22.

---

## Comparacion de las dos lineas

Las dos comparten el mismo techo empirico y se diferencian en un unico punto
relevante.

| Criterio | ABTT de documento | SSWE |
| --- | --- | --- |
| Ejecutar | NO | NO |
| Tipo de riesgo | seleccion sobre ruido | estimacion + seleccion |
| Riesgo ya cubierto | si, identico al de V22-03 | no, riesgo nuevo |
| Coste | 20 a 40 min | 3 a 5 h minimo |
| Probabilidad LB | nula | baja |
| Evidencia negativa propia previa | si, las dos mitades | si, NB-SVM analogo |

La fila que decide el cierre es la de riesgo. ABTT de documento no introduce
riesgo nuevo y no introduce informacion nueva, asi que es una hora sin retorno.
SSWE si introduce algo nuevo, pero lo que introduce es riesgo de estimacion
no cubierto, en la ventana final, con 4 submissions. Ninguna de las dos
cabe en el presupuesto de riesgo que queda.

**Conteo de evidencia favorable disponible en V22.** El unico cambio que en
toda la historia del proyecto ha movido la metrica de forma aislable y con control
de cordura de desviacion 0.000000 es el preprocesamiento normalizado, con
+0.012871 sobre el brazo crudo. Ese es el presupuesto de movimiento real
disponible, y ya esta consumido. Todo lo demas vive dentro de una banda de
+-0.006 con perturbaciones estadisticamente no distinguibles de cero.

## Condiciones de re-apertura (registro, no propuesta)

Se registra unicamente para evitar re-litigaciones sin datos nuevos. Ninguna de
las dos lineas se reabre con argumentos de la misma familia. Cambian de veredicto
solo si ocurre algo fuera del alcance de V22:

- Pregunta 1: solo si el pool de comparacion cambia de escala, por ejemplo si la
  cobertura de desarrollo dejara de ser 1.0000 sobre el texto normalizado.
- Pregunta 2: solo si el numero de etiquetas observadas creciera de 1.483 a un
  orden de magnitud que soporte un bloque de 200 dimensiones, o si el bloque
  denso dejara de competir con 27.431 char n-grams en la misma fusion.

Ninguna de las dos condiciones esta en manos de V22.

---

## Veredicto

Pregunta 1 (ABTT variante Arora, Liang y Ma sobre documento): **NO merece
ejecutarse.** Las dos mitades de la variante tienen evidencia negativa
independiente en este pipeline, la perturbacion de la familia no es
distinguible de cero con p = 0.3930, y la transferencia OOF a leaderboard de
esta familia es 0 de 7.

Pregunta 2 (SSWE con objetivo dirigido a la tarea): **NO merece ejecutarse.** El
objetivo supervisado produce un bloque redundante con el clasificador lineal
que ya esta presente, el precedente NB-SVM ya perdio con un modo de fallo
analogo, y con 1.483 etiquetas el unico regimen en el que SSWE no sobreajusta
es el regimen en el que no hace nada.

Ninguna de las dos produce una mejora de leaderboard plausible, y una de ellas
introduce un riesgo de estimacion no cubierto en la ventana final con 4
submissions disponibles. Se recomienda cerrar V22 sin ejecutar codigo adicional.

## Anexo: control del documento

Archivo en ASCII puro estricto, verificado byte a byte. Sin tildes, sin enes,
sin cedillas, sin comillas tipograficas, sin guiones largos, sin emojis. Las
cifras citadas provienen de los informes v22-01 (preprocessing), v22-02
(preprocessing), v22-03 (all but the top) y de la auditoria de V21
(McNemar p = 0.3930), sin recomputo. Este dictamen no ejecuto experimentos.

CERRAR V22
