# Pre-registro de P73: TextCNN sobre Word2Vec Twitter congelado frente a control aleatorio

Escrito y commiteado **antes** de ejecutar. El orden es comprobable con `git log`.

## 1. La pregunta, y en que se diferencia de P72

```text
P72 pregunto   mejora Word2Vec una representacion lineal basada en mean pooling?   respuesta: no
P73 pregunta   cuando dejamos de promediar y permitimos que un modelo secuencial explote orden y
               contexto local, contiene la representacion aprendida informacion que una matriz
               aleatoria equivalente no contiene?
```

Hipotesis: la representacion puede contener relaciones distribucionales que el mean pooling
destruye y que un lector secuencial no lineal podria explotar, mediante
`secuencia -> convolucion -> composicion local -> clasificacion no lineal`.

**Lo que P72 si cerro**: la geometria aprendida no aporta ventaja a un lector lineal.
**Lo que P72 no cerro**: senal dependiente de orden, composicion o interaccion no lineal.

## 2. Las dos ramas

```text
A  Word2Vec Twitter 100d CONGELADO     -> TextCNN -> 3 salidas sigmoide (multilabel)
B  embedding ALEATORIO 100d CONGELADO  -> MISMA TextCNN -> 3 salidas sigmoide
```

Identico en las dos ramas, verificable por hash: vocabulario, cobertura, secuencia de ids,
longitudes, mascara, particion externa, particion interna, pesos iniciales de la CNN, semilla,
orden de lotes, dropout, optimizador, scheduler, regla de early stopping, umbrales congelados
`[0.44, 0.51, 0.46]` en orden `(HS, TR, AG)` y agregacion.

**La unica diferencia permitida entre los dos runs es la matriz de embeddings.**

La rama aleatoria conserva la **norma por token** de la entrenada y direcciones uniformes sobre la
esfera. Conserva ademas una **identidad numerica distinta por palabra**, luego NO es un baseline
trivial: puede explotar identidad lexica sin necesitar geometria distribucional. Si Random rinde
razonablemente y Word2Vec no lo supera, la lectura es que la arquitectura esta usando identidad o
senal lexical, no la geometria aprendida.

## 3. Arquitectura, fija de antemano

```text
entrada            ids [B, 60], indice 0 compartido por padding y OOV, vector de ceros
maxlen             60, SIN truncar (p99 = 50, maximo observado = 60)
embeddings         [77658, 100] congelados (requires_grad = False)
Conv1D(100 -> 64, kernel 2, init He normal sobre fan_in, con sesgo)
Conv1D(100 -> 64, kernel 3, idem)
Conv1D(100 -> 64, kernel 4, idem)
ReLU
global max pooling POR CANAL, con MASCARA A NIVEL DE VENTANA
    la ventana es valida solo si TODAS sus posiciones son tokens reales
    las ventanas invalidas se rellenan con -1e9 antes del maximo
    una fila sin ninguna ventana valida (longitud real < kernel) produce 0 exacto
concatenacion [192]
Dropout 0.5
Linear(192 -> 64) + ReLU
Linear(64 -> 3)
sesgo de salida = log-odds del prior del pliegue de entrenamiento (ver 6, check 3)
total entrenable = 70.339 parametros
```

Sin BatchNorm, sin LayerNorm, sin attention, sin transformers, sin embeddings preentrenados.

**Init explicito He normal.** No se usa el init por defecto de `Conv1d`
(`kaiming_uniform_` con `a = sqrt(5)`), porque sobre embeddings con anisotropia reduce la relacion
senal/DC de 2.4 a 0.25 y puede ahogar la senal en la rama entrenada.

**Sin centrado de embeddings.** En esta arquitectura, sumar un vector constante a todos los tokens
es exactamente un cambio del sesgo efectivo de cada filtro: la clase de funciones representables y
el optimo alcanzable son identicos. Centrar no desbloquea nada y la anisotropia no puede ser la
causa del nulo de P72.

## 4. Entrenamiento, fijo de antemano

```text
perdida       BCEWithLogitsLoss(reduction='mean'), SIN pos_weight, logits crudos
optimizador   AdamW(betas (0.9, 0.999), eps 1e-8, weight_decay 1e-4 solo en pesos)
lr pico       1e-3, warmup lineal 3 epocas desde 1e-4, cosine hasta 1e-4 en la epoca 40
batch         32
epocas        maximo 40
early stop    sobre la BCE media de inner_val, ventana movil TRAILING de 3 epocas, paciencia 6,
              restauracion de los pesos de la mejor epoca
grad clip     1.0 sobre todo lo entrenable, registrando la norma pre-clip
precision     fp32, SIN AMP
device        CPU, torch.set_num_threads(32)
semilla       20261101 en un unico valor, para todo
```

**Por que sin pos_weight**: las prevalencias son 0.437/0.300/0.382, desbalanceo suave; ademas los
pesos de clase inflan las probabilidades y desincronizan la escala del modelo respecto de unos
umbrales congelados por un calibrado ajeno, y mueven el piso teorico de la perdida, que es la
comprobacion principal de convergencia.

## 5. Protocolo de validacion: holdout anidado agrupado

Por cada pliegue externo `f` de los 5:

```text
1. del train_f (1186 filas) se aparta un holdout ANIDADO 80/20 POR CLUSTERS de near-duplicados
   -> inner_train (~949) e inner_val (~237). Cero solape de clusters (assert).
   Si la prevalencia de alguna clase en inner_val se aleja mas de 0.07 de la de train_f, se
   redibuja con la siguiente semilla, maximo 20 intentos, y se registra cuantas veces hizo falta.
2. se entrena en inner_train con el protocolo del apartado 4
3. early stopping sobre la BCE de inner_val y restauracion de la mejor epoca
4. el modelo restaurado se evalua UNA SOLA VEZ sobre el pliegue externo f
```

**Por que no se elige la epoca sobre el pliegue externo**: con 297 filas externas y hasta 40
epocas, el maximo de una secuencia ruidosa infla el resultado y, peor, cada rama elegiria su propia
epoca, introduciendo un sesgo de seleccion distinto en A y en B.

**La curva del pliegue externo se registra como diagnostico pero NO se lee para ninguna decision**:
ni epocas, ni lr, ni umbrales, ni semilla, ni reintentos.

**Contador de evaluaciones externas pre-declarado: 1 por pliegue y rama, 10 en total.**

## 6. Comprobaciones de conformidad, ANTES de aceptar cualquier resultado

Ningun numero de P73 se acepta si alguna de estas falla.

```text
 1. V03 reproduce: macro-F1 OOF concatenado con umbrales congelados = 0.7772114525165721
 2. el embedding-only de P72 reproduce: Word2Vec 100d + mean pooling + LR = 0.5947288906
    (con el StandardScaler dentro del pliegue, que es el pipeline correcto)
 3. PISO TEORICO de la perdida con los sesgos de salida en el prior = 0.653694 +/- 1e-6
    y con todos los logits a cero = ln(2) = 0.693147 +/- 1e-6
 4. sobreajuste de 32 filas en 200 pasos: perdida < 0.20 y macro-F1 de esas 32 filas > 0.9
 5. parametros entrenables EXACTAMENTE 70339, y las 4 comprobaciones de que el embedding
    congelado no entra en el optimizador
 6. lookup de embeddings: cos(fila cargada, vector guardado) = 1 +/- 1e-6 para 100 palabras;
    norma media = 0.842862 +/- 1e-4
 7. cobertura reconstruida: 68.64 por ciento de tokens y 12.5 tokens cubiertos por fila
 8. los dos modelos arrancan con pesos BITWISE IDENTICOS (assert sobre el state_dict) y la prueba
    funcional cruzada: modelA(x_B) == modelB(x_A) == modelA(x_A)
 9. sha1 de input_ids, longitudes, mascara, pliegues, particion interna, init, y de las dos
    matrices. El diff A-B debe ser EXCLUSIVAMENTE la matriz de embeddings
10. assert del orden de umbrales (HS, TR, AG) = [0.44, 0.51, 0.46]
11. rama B: max|norma_B - norma_A| por token < 1e-6; |media(cos(A_i, B_i))| < 0.002
12. en las 70 filas sin cobertura, A y B producen predicciones IDENTICAS
13. numero de hilos, device y dtype identicos en las dos ramas
```

Fallos silenciosos concretos que estas comprobaciones atacan, todos plausibles en este montaje:
mascara invertida, `-1e9` que llega a la capa densa, dropout activo en evaluacion, permutacion del
orden de clases, indices de vocabulario desalineados con las filas de la matriz, sampler sin
sembrar, y excepcion tragada que reescribe el JSON del run anterior.

## 7. La regla de decision, con su referencia definida

```text
delta_f = macro-F1(A, pliegue externo f) - macro-F1(B, pliegue externo f)
titular = macro-F1 sobre las 1483 filas out-of-fold concatenadas, umbrales congelados
```

Estadistico primario: **IC 95 por bootstrap de clusters de near-duplicados** (remuestreo de
clusters con reemplazo, 2000 remuestreos, semilla 20261102) sobre el delta agregado. Nunca
bootstrap por filas.

```text
CLARO      el IC 95 queda ENTERAMENTE por encima de 0  Y  >= 4 de 5 pliegues con delta_f > 0
EMPATE     el IC 95 CONTIENE el 0
INFERIOR   el IC 95 queda ENTERAMENTE por debajo de 0
```

Se reportan ademas, siempre: delta por pliegue, delta medio, en cuantos pliegues gana A, la
dispersion entre pliegues de A y de B, y la semianchura del IC comparada con las bandas.

**Referencia contra el campeon, secundaria y confundida**: se reporta el titular frente a
0.7772114525165721, etiquetado explicitamente como referencia condicionada a un protocolo mas
conservador (949 filas de entrenamiento frente a 1186) y a una arquitectura distinta. **No aisla
la representacion.** Nunca se redactara como "mejora atribuible a los embeddings".

## 8. Limites declarados de este diseno

```text
una sola semilla   el mandato de esta fase prohibe otra semilla. No se estima la varianza de
                   inicializacion ni la del orden de lotes. Un delta dentro de la dispersion
                   entre pliegues es NO CONCLUYENTE
menos datos        inner_train tiene ~949 filas frente a las 1186 del pliegue completo y a las
                   1186 de V03. Presion conservadora
inner_val ruidosa  237 filas: la epoca se elige sobre una senal ruidosa, lo que anade dispersion
                   al numero reportado, no sesgo a la media
una evaluacion     una sola mirada al pliegue externo por pliegue y rama
4.7 por ciento     las 70 filas sin cobertura contribuyen con prediccion constante
```

## 9. Que no se puede afirmar con este resultado

```text
"Word2Vec no funciona" como afirmacion general, ni "los embeddings no contienen informacion"
"no hay senal semantica en los tuits"
que el efecto se replicaria con otra semilla, otra arquitectura u otro corpus
que una mejora sobre el campeon sea atribuible a los embeddings: la arquitectura es distinta
equivalencia estrecha entre A y B si el resultado es EMPATE: la potencia es limitada
ninguna afirmacion sobre test o leaderboard
que el resultado se deba a "relaciones semanticas" y no a "estructura direccional aprendida":
ambas ramas difieren tambien en anisotropia y forma espectral, y eso es inherente al nulo
```

## 10. Condiciones de ejecucion

```text
test = 0, holdout = 0, leaderboard = 0, submission = 0
el campeon V03 no se modifica
el candidato de AGE no se toca
no se ejecuta fine-tuning, ni 200d, ni 300d, ni otro corpus, ni otro juego de kernels,
ni otra semilla, ni otra CNN
no hay grid de ninguna clase
el numero de corridas es 10 (5 pliegues x 2 ramas) mas las comprobaciones de conformidad
atribucion CC BY-SA 4.0 del corpus en los reportes
los vectores derivados no entran en ningun artefacto de submission
```
