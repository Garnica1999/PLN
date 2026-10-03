# Pre-registro de P74: replicacion del null aleatorio

Escrito y commiteado **antes** de ejecutar. El orden es comprobable con `git log`. **P74 no se
ejecuta en esta fase**: este documento la deja preinscrita y se detiene.

## 1. La pregunta, y es una sola

```text
puede la ventaja de +0.0316 medida en P73 explicarse por una realizacion particularmente debil del
embedding aleatorio, o permanece cuando el null aleatorio se replica?
```

Y nada mas. P74 no abre fine-tuning, ni BiGRU, ni otra dimension, ni otra ventana, ni otro corpus,
ni otra arquitectura CNN, ni otra estrategia de pooling, ni una busqueda de semillas, ni una
submission, ni una sonda de leaderboard.

## 2. Las dos ramas, y la correccion de emparejamiento

```text
A_i  Word2Vec Twitter 100d CONGELADO    + TextCNN con la semilla de modelo i
B_i  embedding ALEATORIO 100d CONGELADO + TextCNN con la MISMA semilla de modelo i
```

**Las dos ramas se reentrenan en CADA replica, con la MISMA inicializacion de TextCNN.** Esto no es
un detalle: si A se congelara una vez y solo variara B, el contraste `A_1 - B_i` dejaria de estar
pareado, porque `B_1` compartiria init con `A_1` y `B_2..B_10` no, y la media y el intervalo se
contaminarian. La varianza de A **tambien** cuenta y debe propagarse.

Dentro de cada replica, la **unica** diferencia entre las dos ramas es la matriz de embeddings.

## 3. Lo que NO cambia respecto de P73, congelado

```text
folds y particiones              las 5 congeladas, los mismos clusters de near-duplicados
particion interna                el mismo holdout anidado 80/20 por clusters, misma semilla
arquitectura                     maxlen 60, Conv1D kernels 2/3/4, 64 filtros, He normal
pooling                          global max con mascara a nivel de ventana
capa densa                       Dropout 0.5, Linear(192->64)+ReLU, Linear(64->3)
sesgo de salida                  log-odds del prior del pliegue
optimizador                      AdamW, betas (0.9,0.999), eps 1e-8, wd 1e-4 solo en pesos
learning rate                    warmup 3 + cosine, 1e-4 a 1e-3 a 1e-4
epocas y early stopping          maximo 40, paciencia 6, ventana trailing 3, restore best
perdida                          BCEWithLogitsLoss reduction mean, sin pos_weight
batch size                       32
preprocesamiento                 tokenizador EMB-NORMAL-TW v1, ids 0 = pad y OOV
vocabulario                      los mismos 77.657, construidos del corpus externo
truncacion y padding             maxlen 60 sin truncar, padding a la derecha
normalizacion                    ninguna sobre los embeddings
umbrales                         0.44 / 0.51 / 0.46, congelados, NUNCA ajustados
codigo de evaluacion             el mismo: macro-F1 OOF concatenado
matriz de A                      el Word2Vec 100d congelado, mismo sha256
```

## 4. Cuantas replicas, y por que diez

```text
replicas            10, con parada dura. No hay busqueda abierta ni ampliacion posterior
coste medido        P73 tardo entre 3.2 y 8.8 segundos por pliegue y rama. Una replica son 10
                    corridas, luego entre 1 y 2 minutos. Diez replicas entran en 20 minutos
```

Cinco replicas se pre-registran como minimo y diez como objetivo, porque **el coste es bajo** y
porque la potencia lo exige: con `d = 0.0316` y potencia 80 por ciento, 5 replicas solo bastan si la
desviacion del delta es menor que 0.020, y en P72 la desviacion aparente del null fue 0.0131 con un
lector MAS estable que la TextCNN. Diez replicas cubren desviaciones de hasta 0.032.

```text
SEMILLAS DE MODELO, fijadas ahora:
  20261111  20261112  20261113  20261114  20261115
  20261116  20261117  20261118  20261119  20261120
SEMILLAS DE LOS SORTEOS ALEATORIOS, fijadas ahora:
  20261211  20261212  20261213  20261214  20261215
  20261216  20261217  20261218  20261219  20261220
```

Las diez matrices aleatorias se generan con el **mismo esquema de P72 y P73** (direccion uniforme
sobre la esfera, norma exacta por token de la entrenada). Se verifica que las diez son distintas
entre si por hash.

## 5. La replica de control, pre-registrada

Antes de las diez, se ejecuta **una replica de control con A y A**: misma matriz entrenada en las
dos ramas, distinta semilla de modelo. Su delta mide la varianza de inicializacion y **debe quedar
compatible con cero**. Si no lo queda, el pipeline tiene un problema y P74 no se interpreta.

Es un control del instrumento, no un resultado.

## 6. La unidad de inferencia, y la definicion congelada de la metrica

```text
metrica titular      macro-F1 sobre las 1483 filas OOF CONCATENADAS, umbrales congelados
                     (NO la media de los F1 por pliegue: en P73 esa media es 0.6065 frente al
                     titular 0.6091 en A, y 0.5755 frente a 0.5774 en B; queda congelado por
                     escrito en D-255.2)
unidad primaria      el delta PAREADO A-B dentro de cada semilla y cada pliegue
delta de replica     d_i = CV(A_i) - CV(B_i)
estadistico          media de los diez d_i
intervalo PRIMARIO   IC 95 t de Student sobre las diez replicas, 9 grados de libertad
intervalo secundario bootstrap de clusters de near-duplicados DENTRO de cada replica, 2000
                     remuestreos, los diez reportados
```

**Por que el intervalo primario es el de replicas y no el bootstrap.** La pregunta de P74 es si el
efecto **permanece al replicar el sorteo**, y esa varianza es exactamente la variacion de `d_i`
entre replicas. El bootstrap por clusters condiciona en la particion y en el par de matrices, luego
no la contiene. **La leccion esta aprendida de P73**: alli el bootstrap por clusters daba
`[+0.0101, +0.0535]` (excluye cero) mientras el IC t por pliegues daba `[-0.0317, +0.0936]` (cruza
cero), y esa discrepancia es de especificacion, no de ruido (D-255.1). Se reportan los dos, y el
primario se declara aqui **antes de ver nada**.

## 7. Lo que se registra por replica

```text
seed
A_CV                                      macro-F1 OOF concatenado de la rama entrenada
B_CV                                      idem de la rama aleatoria
delta_A_minus_B
fold scores A                             los cinco
fold scores B                             los cinco
number of positive folds                  en cada rama y del delta
bootstrap CI                              por clusters, de esa replica
training time                             por pliegue y total
embedding checksum                        sha256 de la matriz de cada rama
CNN initialization checksum               sha256 de los 9 tensores entrenables iniciales
artifact hashes                           de los logits OOF, de las longitudes, de las mascaras
```

Se registran ademas: la curva de validacion interna por epoca, la epoca elegida y su margen, el
contador de corridas que tocan el tope de 40 epocas, y la BCE de la validacion externa.

## 8. La regla de decision, pre-registrada

```text
"direccion positiva en una replica" = d_i > 0  Y  al menos 3 de sus 5 pliegues con delta > 0
banda de equivalencia               = |media de los d_i| < 0.010
```

```text
CASO 1  al menos 8 de 10 replicas con direccion positiva
        Y la media de los d_i > 0
        Y el IC 95 t sobre replicas queda ENTERAMENTE por encima de 0
        -> evidencia reproducible de que la representacion entrenada aporta informacion que el
           lector TextCNN puede utilizar y que no se explica por una unica realizacion del control

CASO 2  al menos 6 de 10 replicas con direccion positiva
        pero el IC 95 t sobre replicas CONTIENE el 0
        -> evidencia compatible con complementariedad, pero todavia inestable.
           NO se abre una arquitectura adicional automaticamente

CASO 3  |media de los d_i| < 0.010
        -> la ventaja de P73 probablemente dependia de una realizacion concreta del embedding
           aleatorio o de variacion del entrenamiento

CASO 4  B supera a A en al menos 8 de 10 replicas
        -> se cierra la hipotesis de que la geometria Word2Vec 100d aporta ventaja a este lector
```

Un desenlace con signos inestables y delta medio positivo pequeno se reporta como **no
concluyente**, y no se fuerza dentro de ningun caso.

## 9. Lo que NO se puede concluir de P74

```text
una mejora de A sobre B NO es una mejora sobre V03. La comparacion con V03 sigue siendo
descriptiva porque la TextCNN no tiene por que ser competitiva con el campeon y esa diferencia
mezcla representacion y arquitectura (protocolo mas conservador: 949 filas de entrenamiento
frente a 1186, y una familia de modelo distinta)

los diez controles aleatorios de P72 NO son el null de P74 y NO se usan como tal: P72 uso mean
pooling lineal y P74 usa TextCNN. La unica comparacion valida para aislar el efecto de la
representacion es Word2Vec 100d congelado frente a embeddings aleatorios 100d congelados con el
MISMO lector

no se compara P73-A contra la media de los random runs de P72
que la representacion Word2Vec funcione en general
que la ventaja provenga de relaciones semanticas y no de estructura direccional mas estadistica
  de frecuencia: solo se iguala la norma por token, no la media ni la covarianza por dimension
que el efecto se replique con OTRA PARTE del corpus, otra dimension u otra arquitectura
```

## 10. Instrumentacion obligatoria, antes y durante

```text
 1. sha256 del script, de los folds, de las matrices y de la configuracion, en el pre-registro
 2. test de juguete del enmascarado: 5 filas conocidas, computo por fuerza bruta de las ventanas
    validas, fila totalmente enmascarada = vector constante finito, y las formas
 3. checksums por replica de longitudes, mascaras, orden de lotes e init de los 9 tensores
 4. asserts de integridad: los pliegues suman 1483, sin solape, longitudes 2 a 60, cobertura
    68.64 por ciento, y recontar las filas sin ventana valida: 0 para kernel 2, 1 para kernel 3
    y 7 para kernel 4
 5. verificacion de mascara de dropout y estado del RNG por replica, como en P73-04, que dio
    16652 de 16652 posiciones coincidentes
 6. las diez matrices aleatorias deben ser distintas entre si (hash)
 7. se re-ejecuta UNA replica completa y se compara a 1e-6 con la primera: control de determinismo
 8. ningun numero obtenido puede coincidir exactamente con 0.693147, 0.653764, 0.540985,
    0.7772114525165721 o 0.5947288906: si coincide, se para y se audita antes de nada
 9. registro de librerias, commit y tiempos
```

## 11. Si P74 resulta claramente positivo

**No se abre otra grid.** Se detiene y se hace, en este orden:

```text
bootstrap agrupado
segunda particion para finalistas
analisis de complementariedad contra V03, en unidades tweet-etiqueta y como diagnostico
analisis por slices
revision especifica de Kaggle Specialist y de ML/DL
```

Y **solo despues** se decide si merece estudiarse una fusion o un lector supervisado mas
sofisticado.

## 12. Estado que se mantiene

```text
V03 = FROZEN      P72 = CLOSED      P73 = COMPLEMENTARY (calificado)
Word2Vec Twitter 100d = congelado
test = 0     holdout = 0     leaderboard = 0     submission = 0
P74 = PREINSCRITA, SIN EJECUTAR
```
