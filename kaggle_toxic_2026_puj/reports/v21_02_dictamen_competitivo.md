# V21-02: dictamen competitivo y decision de submission

## 0. Encabezado de version

```
=== Version v21-02 - dictamen competitivo, sin entrenamiento ni submission ===
Fecha: 2026-09-27
Features: N/A (ningun modelo entrenado, ninguna feature nueva)
CV Score: N/A
LB anterior: 0.78306 (V03, ref 56571480)
Cambios vs version anterior: v21-01 midio tres candidatas ya entrenadas.
v21-02 NO mide nada nuevo. Reordena la evidencia existente y decide si alguna
de esas tres, o algun mecanismo del menu no probado, justifica gastar una de
las cuatro submissions restantes.
```

Documento de decision. No contiene codigo, no entrena modelos, no genera
submissions. Todas las cifras provienen de `submissions/leaderboard_log.csv` y
de `reports/v21_01_triaje_competitivo.md`.

---

## 1. Respuesta corta

Ninguno de los tres mecanismos disponibles justifica una submission. El
veredicto es **NO CANDIDATE JUSTIFIES SUBMISSION**, y las cuatro submissions
restantes deben reservarse.

La razon no es pesimismo. Es que existen **cuatro argumentos independientes**,
construidos sobre datos propios del proyecto, que apuntan todos en la misma
direccion, y ninguno de los tres cumple:

1. La evidencia local no es estadisticamente significant (McNemar p = 0.35,
   0.41, 0.88).
2. La unica variable cuya relacion con el LB esta medida con cuatro puntos y
   replicada siete veces (la tasa de positivos predicha) penaliza a las tres
   candidatas con signo conocido.
3. El perfil de celdas, que es el criterio que mejor ha predicho la
   transferencia, implica una ganancia de **+0.0002, es decir el 2 % de un error
   estandar**.
4. La regresion sobre las siete sondas de la propia ventana predice **-0.0068**
   para la mejor candidata, y el punto de equilibrio exigiria un dOOF de
   **+0.033**, seis veces por encima del techo medido de todo el programa de
   embeddings.

---

## 2. Hechos verificados sobre los que se decide

### 2.1 El historial, leido del log

21 submissions. **Ninguna por encima de V03.** Las dos que parecian subir
(v10 +0.00047, v11 +0.00184) tenian gap LB-local positivo, no delta contra
V03: frente al campeon quedaron en -0.00328 y -0.00156. Confirmado con un
script sobre el CSV.

| envio | dOOF | dLB vs V03 |
| --- | ---: | ---: |
| v16_ch9 min_df=1 | -0.011679 | -0.01328 |
| v12_p7 promedio repeticiones | +0.005066 | -0.01184 |
| v16_ch7 C=1.5 | -0.000558 | -0.01010 |
| v16_ch5 tasa optima absoluta | n/d | -0.00811 |
| v08_p1 fold ensemble | 0.000000 | -0.00836 |
| v05 Optuna | +0.005000 | -0.00704 |
| V20 regla cross-score | +0.000787 | -0.00643 |
| v16_ch8 C=2.5 | -0.000459 | -0.00599 |
| v09_p5 C por etiqueta | +0.000307 | -0.00525 |
| v16_ch1 tasa transportada | n/d | -0.00478 |

**Tres de tres sondas con dOOF positivo perdieron.** De las siete con dOOF
calculable en la ventana reciente, **siete de siete perdieron**.

### 2.2 La regresion propia, que es el dato central

OLS sobre las siete sondas con dOOF calculable (referencia V03 OOF = 0.777211,
la del arnes canonico):

```
dLB = -0.00851 + 0.252 * dOOF        r = +0.418   R2 = 0.174   n = 7
```

El ajuste es robusto: quitando cualquier sonda, la pendiente se mantiene en
0.227 a 0.256 y la intercepta en -0.0083 a -0.0090.

La correlacion es **positiva pero debil**, y ese es el punto: dOOF tiene algo de
informacion sobre el LB, pero el nivel esta desplazado hacia abajo. Predicciones
del modelo para las tres candidatas:

| candidata | dOOF | dLB predicho |
| --- | ---: | ---: |
| FUSION-WS-3 | +0.005748 | **-0.00707** |
| FUSION-VS-100 | +0.005046 | -0.00724 |
| L1 C=4.0 | +0.006431 | -0.00689 |

**OK** Es decir, una sonda solo empata
V03 en el LB si su OOF sube 0.034. El techo real del programa de embeddings
(14 configuraciones, 5 corpus) es **+0.005748**. El mejor dOOF historico de
todo el proyecto es v05 con +0.0050, y ese perdio 0.00704. **Nadie en este
proyecto ha llegado a un tercio de la distancia necesaria para empatar.**

Y hay que anadir la incertidumbre: R2 = 0.174 significa que el OOF explica el
17 % de la varianza. El intervalo alrededor de -0.0068 es de un orden de
magnitud mayor que la prediccion. El escenario honesto es 68 % de perder, con
media claramente negativa.

### 2.3 La tasa de positivos, la unica variable con curva medida

Cuatro puntos, todos medidos, todos en el mismo experimento:

| tasa predicha | LB | pendiente |
| ---: | ---: | ---: |
| 0.3385 | 0.77993 | |
| 0.3645 | 0.78306 | +0.120 (hacia el pico) |
| 0.3836 | 0.77828 | -0.250 (desde el pico) |
| 0.3980 | 0.77495 | -0.231 (desde el pico) |

**El maximo esta exactamente en V03 y la caida por encima es el doble de
empinada que la subida por debajo.** Las tres candidatas **suben** la tasa:

| candidata | dHS | dTR | dAG | media | penal LB a pendiente -0.242 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FUSION-WS-3 | +0.0121 | +0.0040 | +0.0047 | +0.00693 | **-0.00168** |
| FUSION-VS-100 | +0.0081 | +0.0074 | +0.0047 | +0.00673 | -0.00163 |
| L1 C=4.0 | +0.0054 | +0.0357 | +0.0189 | +0.02000 | **-0.00484** |

El signo de este termino es el unico dato de todo el analisis que no depende de
una hipotesis: las tres caminan cuesta abajo por el unico eje con curva medida.

### 2.4 La significatividad del OOF, que es donde la evidencia local se cae

Las comparaciones OOF son **pareadas** sobre las mismas filas, asi que el test
correcto es McNemar, no el DeltaF1 agregado:

| candidata | rescates | danos | n | z | p bilateral | signif |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| FUSION-WS-3 | 74 | 63 | 137 | +0.94 | **0.347** | no |
| FUSION-VS-100 | 77 | 67 | 144 | +0.83 | **0.405** | no |
| L1 C=4.0 | 208 | 205 | 413 | +0.15 | **0.883** | no |

Ninguna esta a menos de 1.3 sigma de ser una moneda al aire. El DeltaF1 de
+0.0057 que la tabla del triaje declara se apoya en un desequilibrio de 74
contra 63 que no se distingue del azar. **El unico dato de la tabla que el proyecto llamaria
"mejora" no es una mejora.**

### 2.5 El perfil de celdas, que es el criterio que se pidio usar

Escala medida: 107 cambios netos de acierto ~ 0.003 de LB, es decir
2.80e-05 por cambio neto. Reescalando OOF (1483 filas) a test (1154 filas),
factor 0.778:

| candidata | neto OOF | neto en test | LB implicito | % de un SE (0.01219) |
| --- | ---: | ---: | ---: | ---: |
| FUSION-WS-3 | +11 | +8.6 | **+0.000240** | **1.97 %** |
| FUSION-VS-100 | +10 | +7.8 | +0.000218 | 1.79 % |
| L1 C=4.0 | +3 | +2.3 | +0.000065 | 0.54 % |

**La proteccion que ofrece el mejor perfil es del 2 % de un error estandar.**

Y esa proteccion esta empiricamente anulada. V20 cambio 51 celdas con saldo
-6, o sea -0.000168 implicito, y el LB cayo **-0.006430: 38 veces peor de lo
que el perfil predecia**. Comparando la tasa implicita de WS-3 contra el
downside realmente observado en la ventana:

| | magnitud | en SE |
| --- | ---: | ---: |
| upside implicito de WS-3 | +0.000240 | 0.02 |
| downside observado minimo | -0.004780 | 0.39 |
| downside observado maximo | -0.013280 | 1.09 |

**El ratio adverso/adverso es de 20x a 55x.** El downside real de una sonda en
este proyecto ha sido consistentemente de 20 a 55 veces mayor que el upside que
el perfil de celdas promete. Un perfil que promete 0.0002 y un historial de
penalizaciones de 0.005 a 0.013 no son una combinacion que se pueda cambiar
por 0.0002 de esperanza.

### 2.6 El techo estructural del componente denso

Rango dinamico OOF de las 14 configuraciones, con el bug del `.vec` como
extremo inferior y WS-3 como superior:

```
0.772800  (con bug del .vec)      -0.004411 vs V03
0.781766  (FUSION-B0-200)         +0.004555
0.782257  (FUSION-VS-100)         +0.005046
0.782959  (FUSION-WS-3)           +0.005748   <-- techo del programa
                                       rango total: 0.010159
```

El techo de +0.005748 es casi exactamente el dOOF de v05 Optuna (+0.0050), que
perdio 0.00704. **El programa de embeddings lleva 14 configuraciones, 5 corpus
y 0 adoptables; su mejor resultado posible esta en la misma banda de ruido que
las sondas que ya fallaron.** Anadir una configuracion mas (CBOW, GloVe) mueve
una variable dentro de un rango que ya esta saturado en su techo.

---

## 3. Candidato 1: FUSION-WS-3

**1. Mecanismo.** Fusion de un bloque de embeddings word2vec (skipgram,
window=3) con el pipeline V03 TF-IDF word(1,2)+char_wb(3,5) + LR, con
umbrales congelados de V03. Desplaza 137 celdas del LB respecto a V03.

**2. Diferencia exacta respecto a V03.** Es V03 mas un bloque denso de 200
dimensiones promediado sobre los 6.059 tipos del train. Todo lo demas identico:
misma union TF-IDF, mismo C=2.0 balanced liblinear, mismos umbrales
(0.44/0.51/0.46). Solo cambia la matriz de entrada.

**3. Por que no es un renombre de algo ya hecho.** Es artefacto ya entrenado y
reevaluado en v21-01 con controles reproducidos a menos de 0.00005. No es un
rehuse de una sonda previa: ninguna de las 21 submissions intento esta
combinacion concreta. El punto de partida del mecanismo (fusion de embeddings)
si esta taxonomicamente agotado; lo que se evalua aqui es el artefacto final.

**4. Evidencia OOF disponible.** DeltaF1 +0.005748, 74 rescates, 63 danos,
saldo +11. McNemar p = 0.347. Tasa de aciertos en celdas modificadas 0.536,
frente a 0.5 esperado, que es 0.84 sigma. Positivo en las tres etiquetas
(HS +2, TR +8, AG +1) pero la mitad del saldo son 8 aciertos en una sola
etiqueta con la prevalencia mas baja.

**5. Evidencia de transferencia historica.** Ninguna a favor. Contraria en
cuatro frentes: la regresion de la ventana predice -0.00707; el
desplazamiento de tasa cuesta -0.00168 con signo conocido; el perfil de celdas
promete +0.00024; y tres de tres sondas previas con dOOF positivo perdieron.

**6. Coste de implementacion.** **Cero.** El artefacto existe, esta
reevaluado, y el sha256 esta registrado. Es la unica candidata del menu que no
requiere escribir ni una linea de codigo.

**7. Riesgo de sobreajuste.** Alto, pero no por capacidad del modelo: por
**seleccion**. Es el maximo de 14 configuraciones de la familia de embeddings
(+0.005748 frente a -0.004411 del peor), y la seleccion se esta haciendo sobre
la unica metrica que la ventana demuestra que no transfiere. Ademas concentra
8 aciertos de 137 en 26 celdas de una sola etiqueta, que es un maximo local
sensible a la semilla de los pliegues.

**8. Potencial competitivo.** Bajo, y **negativo en esperanza**. La
prediccion puntual es -0.0068. El upside nominal es 0.0002, un 2 % de SE, con
un downside historico de 0.005 a 0.013.

**9. MerECE una submission: NO.** Es la mejor de las tres medidas y aun asi
pierde por cuatro vias independientes y no independientes entre si. Su unico
argumento a favor es el coste cero, y el coste cero de una sonda no es un
argumento: el recurso escaso son las cuatro submissions, no las horas. En el
mejor de los casos compra un -0.0068 a cambio de una de las cuatro; en el
peor, repite V20 con menos celdas.

---

## 4. Candidato 2: familia C, CBOW / GloVe / word2vec reaparametrizado

**1. Mecanismo.** Reentrenar el bloque de embeddings desde cero cambiando el
objetivo (CBOW en vez de skipgram, o GloVe con ponderacion por frequencies de
coocurrencia) y los tres hiperparametros nunca variados (epoch 5, alpha 0.05,
sample 1e-4). Luego fusionar igual que WS-3.

**2. Diferencia exacta respecto a V03.** Igual que WS-3 en la capa de
fusion; lo que cambia es la geometria del vector entrenado.

**3. Por que no es un renombre.** Es genuinamente nuevo en tres ejes: las 14
configuraciones son todas skipgram; GloVe nunca se entreno; y `sample = 1e-4`
con un corpus pequeno significa que el subsampling esta **desactivado de
hecho**, asi que activarlo es un cambio real y no un ajuste cosmetico. GloVe
pondera por coocurrencia global, que es donde vive la informacion
discriminante en texto corto. El precedente de dominio (Monnar et al. 2024,
word2vec desde cero sobre 10M de tweets en espanol gana a MUSE) apoya que el
cambio de objetivo importa; su ruta legal es solo el corpus, sin Hurtlex.

**4. Evidencia OOF disponible.** **Ninguna para esta configuracion.** La
evidencia disponible es la del techo: la mejor de las 14 configs existentes da
+0.005748 y el rango de la familia es 0.0102. No hay ninguna razon para esperar
que un punto dentro del espacio de busqueda supere el maximo ya observado en
ese mismo espacio.

**5. Evidencia de transferencia historica.** La unica analogia es el
lineamiento de v16_ch8: subir C de 2.0 a 2.5, un unico hiperparametro dentro
de una familia ya caracterizada, dio -0.00599. Mismo patron: moverse dentro de
una familia cuya curva ya tiene su pico comprobado.

**6. Coste de implementacion.** Alto. 3 arquitecturas x 3 corpus x rejilla de
epoch/alpha/sample. Cada evaluacion exige ademas el pipeline V5 completo con
`print-word-vectors` sobre el `.bin` (nunca el `.vec`, que cambia el pooling y
dejo FUSION-B0-200 en 0.7728 en vez de 0.7818). Es el candidato mas caro del
menu por el margen mas amplio.

**7. Riesgo de sobreajuste.** Maximo de todo el menu. El espacio de busqueda
tiene al menos 4 grados de libertad y **la unica funcion de seleccion
disponible es el OOF, que la ventana demuestra anti-informativa**. Con R2 =
0.174, seleccionar el maximo de 27 combinaciones sobre una metrica casi ruido
produce con certeza un ganador local que es perdedor en el LB. Esto es
exactamente lo que paso con v05 Optuna, que era el mejor OOF del proyecto
(0.8039) y perdio 0.00704.

**8. Potencial competitivo.** Muy bajo. Requiere superar un techo de
+0.005748 para tener alguna posibilidad, y el equilibrio necesita +0.0338. Es
la distancia imposible dentro de esta familia.

**9. MerECE una submission: NO.** No porque la idea sea mala, sino porque su
coste de investigacion es el mas alto y su techo esta medido y es insuficiente.
Es el peor uso posible de una de las cuatro: se pagaria una submission por un
sistema que ni en su mejor version medida llega al 17 % de la distancia
necesaria para empatar.

---

## 5. Candidato 3: familia A, all-but-the-top

**1. Mecanismo.** Centrar el bloque denso de embeddings y retirar las D
primeras direcciones principales de PCA (Mu & Viswanath 2018, arXiv:1702.01417;
Levy 2015 TACL). Es post-procesamiento: no requiere reentrenar, son minutos de
calculo.

**2. Diferencia exacta respecto a V03.** Es WS-3 mas una transformacion lineal
sobre el bloque denso antes de fusionarlo. El resto de V03 intacto.

**3. Por que no es un renombre.** Es genuinamente no probado en el proyecto, y
es la unica entrada del menu cuya aceptacion o rechazo no depende de
reentrenar 5 corpus. La literatura reporta que gana en 34 de 40 combinaciones
downstream.

**4. Evidencia OOF disponible.** Ninguna. Habria que medirla, y medirla
significa elegir D por OOF.

**5. Evidencia de transferencia historica.** La de la familia densa completa:
14 configuraciones, techo +0.005748, rango 0.0102. Y una advertencia concreta
del registro: con 1.483 ejemplos de entrenamiento y 27.431 char n-grams, un
vector denso de 200 numeros no tiene espacio para aportar. Es un limite
estructural de densidad, no de hiperparametros, y all-but-the-top no lo ataca:
lo reordena.

**6. Coste de implementacion.** **Bajo**, el mas bajo del menu, y es su unico
argumento fuerte.

**7. Riesgo de sobreajuste.** El mas insidioso, y no por capacidad sino por
**validez de la evidencia citada**. El resultado de 34 de 40 combinaciones es
sobre tareas de similitud de palabras con corpus de miles de millones de
tokens. Aqui el corpus es de orden 10^6 y hay 1.483 ejemplos supervisados. En
ese regime ABT esta documentado como danino: las direcciones dominantes que se
retiran pueden ser precisamente las que el clasificador downstream usa, y el
fallo es silencioso (el OOF baja poco, el LB cae mucho). Ademas D es un grado
de libertad elegido por OOF, sobre la metrica no transferible.

**8. Potencial competitivo.** Bajo. El techo de todo el canal denso esta
medido en +0.0057 y el equilibrio exige +0.0338. ABT actua sobre ese canal.

**9. MerECE una submission: NO.** Es el candidato mas barato y el mas
tentador de "probemos algo nuevo", y precisamente por eso es el mas peligroso uso
de un slot por un resultado negativo. Si el equipo decide ejecutar una sola
prueba de investigacion por curiosity, esta es la que menos informacion
retorna por unidad de recurso, porque su techo lo fija un componente ya
caracterizado como estructuralmente topado.

---

## 6. Familias descartadas sin candidato

### B. SSWE / embeddings con objetivo dirigido a la tarea

Se descarta antes de medir, por tres razones independientes.

**Razon 1: no aporta informacion nueva.** SSWE entrena el embedding con la
senal supervisada del propio train. Esa senal ya esta consumida hasta la
saturacion por el LR balanceado sobre 27.431 char n-grams. Un embedding denso
de 200 dimensiones cuya funcion de perdida es la etiqueta es, en el mejor de
los casos, una version suavizada y con perdida de lo que LR ya calcula. El
proyecto ya midio esta idea en su forma mas fuerte: la version **correcta** de
NB-SVM (que es justamente "senal supervisada dentro del espacio de
caracteristicas") dio 0.7638, peor, porque binariza y descarta el term
frequency. Es el mismo modo de fallo.

**Razon 2: multiplica el espacio de busqueda por tres.** Tres etiquetas, tres
objetivos, tres embeddings mas que evaluar, y ninguno seleccionable por una
metrica que transfiera.

**Razon 3: coste prohibitivo en la situacion actual.** 9 a 27 corridas para
responder una pregunta con techo estructural ya establecido.

### Corpus de registro social (tweets en espanol)

Se descarta como candidato de submission, aunque es el elemento del menu con
mejor justificacion de dominio. El diagnostico de V5 ya lo condemnation: el
limite es el **registro**, no el tamano, porque la adicion de web subio la
cobertura por tipo de 0.6839 a 0.8256 y la aportacion **no** crecio. Ese
resultado refuta directamente la hipotesis de que falta volumen, asi que
anadir otro corpus grande no ataca la causa. Es una linea de investigacion
legitima para un reporte academico; no es una candidata de submission bajo
este presupuesto.

---

## 7. La aritmetica que cierra la decision

Para la mejor candidata disponible, FUSION-WS-3, sumando las cuatro capas:

```
1. dOOF declarado:                       +0.005748   (lo que la tabla triage declara)
2. significatividad McNemar:         p = 0.347   (no distinguible de una moneda)
3. regresion de la ventana x7:        -0.00680   (prediccion puntual del LB)
4. desplazamiento de tasa, firmado: -0.00168   (curva medida de 4 puntos)
5. perfil de celdas, implicito:      +0.00024   (2 % de un SE, con signo conocido)
   --- 6. historial de calibracion:  x20 a x55 adverso (V20 fue 38x)
```

El paso 1 nunca debio sostenerse: la evidencia local es nula en el paso 2. Y
aunque se aceptara sin mas, el paso 2 lo refuta por separado, los pasos 3, 4 y 6 son
independientes entre si y todos firmados en contra.

**Valor esperado: negativo, con la media en torno a -0.007 y un intervalo
mucho mas ancho que la propia prediccion.**

Para superar 0.78306 hace falta, simultaneamente:

- dOOF de +0.0338, cuando el techo medido de la familia mas trabajada del
  proyecto es +0.0057 (**factor 6**);
- NO subir la tasa de positivos, cuando las tres candidatas la suben y la
  curva medida penaliza -0.24 por unidad;
- un perfil de celdas cuyo downside no se haya comportado como el de V20,
  38 veces peor que su propia prediccion.

Las tres condiciones son incumplibles con el menu disponible.

---

## 8. Que habria que ver para cambiar el veredicto

Para que una futura iteracion reabra la decision haria falta al menos una de
estas tres:

1. **Un cambio de tercer orden** (una construccion que no sea una simple
   perturbacion de V03) con dOOF de al menos +0.010, el doble del techo
   actual), verificado con McNemar en una zona de celdas donde el p < 0.05 y el
   saldo no dependa de una sola etiqueta. Nada en el menu actual lo ofrece.

2. **Una candidata que baje la tasa de positivos predicha** en vez de
   subirla, con saldo positivo. Ninguna de las tres la cumple, y es la direccion en
   que la unica curva medida del proyecto mejora.

3. **Una reduccion de capacidad en la fusion**, que es la unica palanca que el
   proyecto ha visto mover el LB en la direccion correcta (v02 -> v03 fue una
   reduccion de umbral efectivo). Esta no esta en el menu, pero es el unico
   sitio donde buscaria con cuatro submissions en la mano.

Lo que **no** es un motivo para reabrir la decision: que una candidata tenga
DeltaOOF positivo. En este proyecto, tres de tres sondas con DeltaOOF positivo
perdieron, y la regresion de la ventana predice -0.0068 para un DeltaOOF de
+0.0057.

---

## 9. Veredicto

```
NO CANDIDATE JUSTIFIES SUBMISSION
```

Las cuatro submissions restantes se reservan.

### Orden pre-registrado, por si hubiera una restriccion externa

El orden **no se decide con DeltaF1** (L1gana ahi) sino con el criterio de
transferencia, y en ese criterio el orden es:

```
1. FUSION-WS-3     (unico positivo en las 3 etiquetas, tasa +0.0069, 8.03 % neto/celda)
2. FUSION-VS-100   (80.3 % solapado con WS-3: no es una segunda oportunidad)
3. L1 C=4.0        (mecanismo identico a V20: tasa TR +0.0357 y saldo negativo)
```

Este orden queda escrito **antes** de cualquier resultado, para que la
decision no se reabra bajo presion de tiempo. Si se presentara el caso de
tener que gastar una submission por una razon externa a la relacion
riesgo-beneficio, la que se gasta es **WS-3**, y unicamente WS-3: las otras
dos no sonRIE independiente (80.3 % de solape) ni de bajo riesgo (L1 repite el
mecanismo que costo -0.00643).

**Ese caso no se ha dado. La recomendacion es no gastar ninguna.**

---

## 10. Nota de trazabilidad

Cifras verificadas contra:

- `submissions/leaderboard_log.csv` (21 filas, 0 por encima de V03)
- `reports/v21_01_triaje_competitivo.md` (tabla de celdas, desglose por etiqueta, solapamiento, controles)
- Calculos derivados: regresion OLS de las 7 sondas con dOOF calculable,
  test de McNemar sobre los pares rescates/danos, pendiente de la curva LB
  frente a tasa predicha con 4 puntos, y reescalado OOF -> test por
  1154/1483 = 0.778.

Ninguna cifra de este documento proviene de un modelo reentrenado en esta
fase. Ninguna submission fue generada.
