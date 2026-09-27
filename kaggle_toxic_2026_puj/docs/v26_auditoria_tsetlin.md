# V26: auditoria de la referencia ConvTextTM antes de implementar

Trabajo verificado el 2026-09-27 contra el PDF de ACL Anthology, no contra citas de
segunda mano. PDF descargado de `https://aclanthology.org/2022.lrec-1.401.pdf` (1 046 162
bytes) y extraido con `pdftotext -layout`.

## La referencia

Bhattarai, Bimal; Granmo, Ole-Christoffer; Jiao, Lei. 2022. "ConvTextTM: An Explainable
Convolutional Tsetlin Machine Framework for Text Classification." LREC 2022, paginas
3761-3770, Marseille. ELRA. DOI registrado en ACL Anthology con identificador
`2022.lrec-1.401`.

**Todos los campos verificados. La referencia existe y es real.** No hay ningun campo que
haya resultado invalido.

## Los diez puntos que pediste verificar

| punto | valor en el paper | fuente |
|---|---|---|
| **representacion de entrada** | binaria, tipo set-of-words, sobre fragmentos de texto | Seccion 3 |
| **binarizacion** | presencia de token. Cada patch es un vector binario `x_p = [x_pk] in {0,1}^C` | Seccion 3, ecuacion de patches |
| **numero de clausulas** | **10 000** en Twitter, 5 000 a 15 000 en el resto | Tabla 2 |
| **literal polarity** | mitad positiva `C_j^+`, mitad negativa `C_j^-`, `xi in {-1,+1}`. El signo se aplica a la clausula, no a los literales | Seccion 2 |
| **threshold** | **50 x 50** en Twitter, es decir umbral 50 con 50 clausulas por polaridad | Tabla 2 |
| **numero de labels** | Twitter tiene **3**; es clasificacion multiclase con argmax, no multilabel | Tabla 1 y Figura 2 |
| **preprocessing** | ventanas de tamano 2 en Twitter, con tokens PAD al borde; la dimension `C` es el tamano de vocabulario | Seccion 3 y Tabla 2 |
| **metrica** | **accuracy**, no F1. El unico F1 que aparece es una comparacion puntual con BERT | Tablas 4 a 8 |
| **tamano del dataset** | Twitter: 2 176 entrenamiento, 932 test, 3 clases, 5 513 tweets en total | Tabla 1 |
| **procedimiento de entrenamiento** | 100 epochs; feedback Tipo I incrementa clausulas de la polaridad correcta, Tipo II de la contraria; la busqueda de patch aleatorio entre los que activaron la clausula hace de contrapunto | Tabla 2 y Seccion 2 |

## Lo que el paper NO es, y es lo que la literatura daba por hecho

Cinco afirmaciones circulaban sobre esta referencia. **Cuatro estan mal.**

1. **"Evidencia en la escala exacta de TOXIC, con control de conjunciones."** Falso. Twitter son
   2 176 filas de entrenamiento; TOXIC tiene **1 186 por pliegue**. Ademas el paper usa
   **10 000 clausulas**, y el estado del arte en TM documenta que 10 000 clausulas sobre
   2 176 filas es mecanicamente posible pero estadisticamente vacio: cada clausula recibe
   del orden de 0,2 muestras. No es un resultado que se transfiera, es una cifra de
   facturacion de complejidad.

2. **"Conjunctions booleanas, sesgo inductivo irrepresentable por un hiperplano."** En el
   papel es cierto **por construccion**, pero **no por resultado**. La columna `TM` de la
   Tabla 4 da **71.13** en Twitter frente a **72.21** del Transformer y **74.71** de BERT. El
   TM **pierde** a los dos. ConvTextTM llega a 71.91, y la Tabla 8 da un maximo de 73.67 con
   ventana variable. El mecanismo mas expresivo **gana menos**, no mas. Con 1 186 filas por
   pliegue, la capacidad extra se paga enteramente en varianza.

3. **"Metrica F1, comparable con la nuestra."** Falso. El paper mide **accuracy**, en
   clasificacion **multiclase con argmax** de 3 clases. TOXIC es **multilabel macro-F1** con
   tres etiquetas independientes. No hay conversion valida entre ambas, y comparar
   73.67 de accuracy con 0.78306 de macro-F1 no significa nada.

4. **"Escala de 200-1 000 clausulas segun el paper."** Falso. El paper no prueba ningun
   numero pequeno de clausulas: su minimo es **5 000**. La recomendacion de 200-1 000 no
   viene de esta referencia, y hay que tratarla como hipotesis propia, no como dato
   publicado.

5. **La dependencia de BERT si es real, y hay que decidirla explicitamente.** El paper
   declara que usa "a generic SOW formed by the tokenization scheme of BERT", con `C` igual
   al tamano del vocabulario de BERT, unos 30 222. Es decir, **el ConvTextTM de este paper
   consume un vocabulario de 30 222 tokens derivado de un modelo preentrenado**, que es
   exactamente lo que las reglas de TOXIC prohiben. No es que use los pesos de BERT: usa
   **su vocabulario**. Aun asi, un vocabulario extraido de un modelo preentrenado es un
   artefacto externo no permitido, asi que **esta parte queda excluida**. Es la unica razon
   por la que el resto de la referencia es utilizable.

## Que queda reproducible, y que no

| componente | veredicto | motivo |
|---|---|---|
| Tsetlin Machine estandar con clausulas conjuntivas, feedback Tipo I y II | **REPRODUCIBLE** | Es un algoritmoPublished, sin dependencias externas. `pyTsetlinMachine` (cair) es la referencia de implementacion |
| Representacion binaria de word n-grams propia | **REPRODUCIBLE** | `CountVectorizer(binary=True)` es legal y no usa nada externo |
| Representacion binaria de char n-grams propia | **REPRODUCIBLE** | idem, con `analyzer="char"` |
| Umbral de decision y polaridad de clausulas | **REPRODUCIBLE** | Parte del algoritmo |
| Convolucion sobre ventanas con patch aleatorio | **REPRODUCIBLE CON CAMBIOS** | El mecanismo es publicable, pero la `C` hay que redefinirla |
| **Vocabulario de BERT como SOW generico** | **EXCLUIDO** | Artefacto de modelo preentrenado, prohibido por las reglas |
| **10 000 clausulas** | **EXCLUIDO POR IMPOSIBILIDAD** | Con 1 186 filas por pliegue no es una decision de presupuesto, es aritmetica |
| **Comparar accuracy con macro-F1** | **EXCLUIDO** | Metricas distintas, multiclase frente a multilabel |
| **TF-IDF con sublinear_tf, tal como V03** | **EXCLUIDO PARA EL TM** | El TM consume bits. La binarizacion es parte del mecanismo, no un parametro |

## Conclusion de la auditoria, y lo que cambia del plan

La referencia **existe y es autentica**, pero **no aporta la evidencia que la hacia la
prioridad numero uno**. Sus tres afirmaciones centrales en la literatura quedan
desmentidas por el propio paper: la escala, la metrica y la direccion del efecto.

Lo que si queda en pie, y es lo unico que justifica abrir la familia:

- El mecanismo de conjunciones es **genuinamente** distinto de un hiperplano. Eso no depende
  del paper, es una propiedad de la arquitectura.
- El TM es el unico modelo del reporte que **no** es un modelo lineal sobre la misma matriz
  sparse, y por eso puede tener complementariedad real con V03.
- Se puede implementar **entero, legal y pequeno**, y medir su complementarity en minutos.

Pero la expectativa tiene que bajar: el paper mide que en su propio dataset, con 10 000
clausulas y mas datos que nosotros, el TM **pierde** a un Transformer y a BERT. Con 1 186
filas por pliegue y clausulas que si o si tendremos que reducir a un orden de magnitud
menos, la posibilidad de superar 0.78306 es **baja**, y hay que decirlo antes de ejecutar y
no despues.

## Nota sobre la binarizacion y el control obligatorio

El riesgo metodologico real de este experimento **no** es el TM, es la **binarizacion**.
Pasar de TF-IDF con `sublinear_tf=True` y pesos float a bits cambia la representacion
entera, y V03 dejaria de ser un control valido si se compara directamente. Por eso el
control tiene que ser un **LogisticRegression sobre exactamente la misma matriz binaria**,
y la comparacion informative es de tres vias:

    Tsetlin(binario)  vs  LogisticRegression(binario)   ->   ganancia del mecanismo TM
    LogisticRegression(binario)  vs  V03                 ->   coste de la binarizacion
    Tsetlin(binario)  vs  V03                           ->   efecto neto

Solo la primera es la pregunta de la familia. Las otras dos son contexto, y sin ellas el
numero del TM no se puede interpretar.
