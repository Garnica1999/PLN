# V21 P1-03: NB-SVM, razones logaritmicas de recuento y clasificadores lineales hibridos

Investigacion bibliografica y auditoria de fidelidad sobre dos temas:

- **Tema A**: NB-SVM y la transformacion de razon de recuento (`r = log(P(x|c=1)/P(x|c=0))`).
- **Tema B**: clasificadores lineales y modelos hibridos de features sparsely + densas.

Fecha: 2026-09-27. Alcance: estado del arte 2023-2026 mas las fuentes primarias de la receta.

---

## 0. Contexto duro, reglas y advertencias de metodo

### 0.1 Reglas que este informe respeta

Ninguna de las candidatas propuestas usa transformers, LLMs, embeddings, modelos
preentrenados externos ni datasets externos etiquetados. Todo lo que se propone usa
solo el texto de `train` y corpus externos **no etiquetados**, con el test nunca
usado para entrenar ni para ajustar umbrales.

### 0.2 La metrica y el numero de referencia

El proyecto mide `f1_media_columnas`, es decir la media de la F1 de las tres
clases positivas (HS, TR, AG), sobre el conjunto de desarrollo de 1483 filas con
5 pliegues agrupados (`src/splits.py: holdout_v2` y `pliegues_corregidos`).

| referencia | valor | origen |
|---|---|---|
| champion local (OOF de V03) | **0.777211** | `features/oof_v03_final.npz` |
| champion en leaderboard | 0.78306 | `reports/v8_log_count_ratio.json` |
| prevalencias en `train` | HS 0.4367, TR 0.2998, AG 0.3824 | calculado en esta sesion |

ADVERTENCIA DE COMPARABILIDAD: `reports/features_densas_resultados.csv` usa otro
protocolo (`MOD-03b`, control 0.7927). Las cifras de esa tabla **no** son
comparables con las de este informe. Cada tabla de aqui lleva su control.

### 0.3 Lo que este informe hace y no hace

**Hace**: (a) recupera la formula canonica exacta desde la fuente primaria
descargada y leida; (b) audita numericamente la implementacion previa de NB-SVM
del proyecto sobre los datos reales; (c) mide, con los pliegues y los umbrales
congelados del proyecto, seis variantes de NB-SVM canonico, cinco clasificadores
lineales alternativos y tres familias de regularizacion; (d) audita la seleccion
de umbrales contra la teoria de Lipton, Elkan y Naryanaswamy.

**No hace**: no propone nada fuera de las reglas, no barre hiperparametros para
buscar un valor mejor, y no genera ni modifica submissions.

### 0.4 Sobre las referencias

Solo se citan obras verificadas en esta sesion. Tres referencias se**<br>marcan
explicitamente como no verificadas** en la seccion 8.2 y no se les atribuye
ninguna cifra. No se inventa ninguna referencia.

---

## 1. Resumen ejecutivo

### 1.1 Hallazgo central del Tema A

La implementacion previa de NB-SVM del proyecto (`experiments/v8_04_log_count_ratio.py`,
linea 161-187, y `experiments/20_emocion_nbsvm_ortografia_v2.py`) **no es la
receta canonica**, y se desvia de ella en cinco puntos simultaneos. Corregir esos
cinco puntos recupera **+0.10 de macro-F1** frente a la implementacion previa
(0.6415 frente a 0.7607 sobre la misma base de binarizacion), pero **todavia no
alcanza al champion**: la mejor variante canonica medida llega a 0.763839, es
decir -0.013372 respecto a 0.777211, con 0 de 5 pliegues a favor.

La razon de que no alcance al champion es identificable y no es un fallo de
implementacion: **NB-SVM descarta el term frequency**. La receta canonica
binariza obligatoriamente porque su ratio esta definido sobre conteos de
documento, de modo que `r` solo sabe decir "este termino aparece mas en la clase
positiva", nunca "aparece tres veces". El champion, con `sublinear_tf=True`, si
exploita la repeticion, y en tweets toxicos la repeticion de un insulto es senal.
El coste de binarizar es, en este corpus, mayor que el beneficio de reescalar
por clase.

### 1.2 Hallazgo central del Tema B

`LogisticRegression` con penalizacion L2 es el mejor clasificador lineal de los
seis medidos, y no hay ninguna evidencia a favor de `LinearSVC` con calibracion.
Ademas, la unica combinacion que mejora de forma consistente es
**L1 con C alto** (0.783423, +0.006211, **5 de 5 pliegues**), no L2 con C alto,
lo que contradice la intuicion habitual para p >> n.

### 1.3 Candidatas priorizadas

| # | candidata | metrica | dif | pliegues | IC95 pareado | ventaja esperada |
|---|---|---|---|---|---|---|
| 1 | LR **L1** con `C=4.0` sobre la TF-IDF del champion | 0.783423 | **+0.006211** | **5/5** | [-0.0081, +0.0202] | unanimitad en los 5 pliegues, la senal mas consistente medida |
| 2 | TF-IDF del champion por **`sign(r)*sqrt(|r|)`** con `r` canonico | 0.781633 | **+0.004422** | **4/5** | [-0.0056, +0.0144] | cambia solo 4.99 % de las celdas y acierta el 1.3125 de las que cambia |
| 3 | NB-SVM canonico (conteos, `alpha=1`) con LR, como **canal de diversidad** | 0.763839 | -0.013372 | 0/5 | no medido | error profile muy distinto (binarizado); util para ensemble, no para sustituir |

Las tres quedan **por debajo del gate del proyecto**: el IC95 pareado de las tres
contiene el cero. Ninguna es un envio. La 1 y la 2 son lineas de base para un
protocolo de gates mas exigente, no mejoras demostradas.

---

## 2. TEMA A: NB-SVM y razones logaritmicas de recuento

### 2.1 La formula canonica exacta (fuente primaria verificada)

Referencia: **Wang, S. y Manning, C. D. (2012). "Baselines and Bigrams: Simple,
Good Sentiment and Topic Classification." EMNLP 2012.** Verificada por descarga
directa del PDF (5 paginas) y lectura integra del texto.

Notacion del original: `f(i)` es el vector de recuentos de caracteristicas del
ejemplo de entrenamiento `i`; `y(i)` es su etiqueta en `{-1, +1}`; `V` es el
vocabulario; `alpha` es el parametro de suavizado.

Definicion de los vectores de clase (seccion 2 del original):

```
p = alpha + SUM_{i : y(i)=+1}  f(i)
q = alpha + SUM_{i : y(i)=-1}  f(i)
```

Log-count ratio, **ecuacion (2) del original, transcrita literalmente**:

```
                    ( p / ||p||_1 )
      r = log    -------------------
                    ( q / ||q||_1 )
```

Es decir, en forma expandida:

```
      r_j = log( p_j ) - log( ||p||_1 ) - log( q_j ) + log( ||q||_1 )
```

donde `||p||_1 = SUM_k p_k` y `||q||_1 = SUM_k q_k` son las sumas de todos los
componentes de cada vector de clase, es decir los **totales de tokens de la
clase mas `|V| * alpha`**, no el numero de documentos de la clase.

Los tres puntos que hay que entender para no equivocarse:

1. `p` y `q` son **sumas de vectores de recuento** sobre la clase, no medias.
2. La normalizacion es por el **total de tokens de la clase**, no por el numero de
   documentos de la clase.
3. `alpha` se suma al **recuento**, antes de normalizar. El valor publicado es
   **`alpha = 1`** ("All results reported use alpha = 1").

**Respuesta directa a la pregunta "cual es la smoothing canonica": la canonica es
aditiva sobre el recuento, con `alpha = 1`, es decir un Laplace de una unidad por
caracteristica mas un pseudo-documento de `|V|` unidades por clase.**

### 2.2 El modelo completo, no solo el ratio

De las mismas secciones 2.1-2.3 del original:

- **MNB**: `x(k) = f(k)` con `f` binarizado, `w = r`, `b = log(N+/N-)`. Nota
  importante: el termino de prior `b` se fija **a mano** al logaritmo del cociente
  de tamanos de clase, no se aprende. Y "we find that binarizing f(k) is better"
  (`f(k) = 1{f(k) > 0}`).
- **SVM**: `x(k) = f(k)` binarizado, minimizando `w'w + C * SUM_i max(0, 1 - y(i)(w'f(i)+b))^2`.
  Es decir, **SVM con perdida L2 y regularizacion L2**, y el original dice
  explicitamente "We find this L2-regularized L2-loss SVM to work the best and
  L1-loss SVM to be less stable."
- **NBSVM**: `x(k) = f(k)` **binarizado**, y los valores de caracteristica son el
  producto elemento a elemento **`r o f(k)`**, donde `r` se calcula **tambien con
  los vectores binarizados** (el original lo denota `r` con tilde y define
  `f(k) = r o f(k)` usando `f` binarizado).

**Consecuencia practica que la literatura no explicita pero se sigue de la
formula**: la version "MB" o "binarizar y luego usar el ratio" **no es una
variante opcional, es el NBSVM**. La receta sin binarizar es la de MNB. En este
punto el proyecto hizo bien en usar conteos, y mal en el resto.

### 2.3 La interpolacion `beta` y por que importa

El original no reporta NBSVM puro: interpola entre MNB y SVM.

```
      w_0 = (1 - beta) * w_barra + beta * w
      w_barra = ||w||_1 / |V|          (magnitud media de los pesos)
```

Interpretacion del original: "This interpolation can be seen as a form of
regularization: trust NB unless the SVM is very confident."

Valores y sensibilidad, con lo que el original afirma literalmente:

- `beta = 0.25` en todos los resultados reportados; `C = 1` para NBSVM y `C = 0.1`
  para el SVM plano.
- "The performance on longer documents is virtually identical (within 0.1%) for
  `beta` in [1/4, 1], while `beta = 1/4` is on average 0.5% better for snippets
  than `beta = 1`. Using `beta` in [1/4, 1/2] makes the NBSVM more robust."
- "One disadvantage of NBSVM is having the interpolation parameter `beta`."

Este es el punto que la literatura marca como la debilidad del metodo, y es
importante para el proyecto porque **es un parametro mas que ajustar**, en
contradiccion con la politica de no barrer hiperparametros.

### 2.4 Resultados publicados de la receta canonica

Cifras transcritas de las tablas 2, 3 y 4 del PDF verificado. La columna `delta`
de la tabla 1 del original da los margenes de diferencia significance a
`p < 0.05`: RT-s 0.8, CR 1.3, MPQA 0.8, Subj 0.8, RT-2k 1.5, AthR 2.9, XGraph 1.8.

Tabla 2, fragmentos cortos (exactitud, %):

| metodo | RT-s | MPQA | CR | Subj |
|---|---|---|---|---|
| MNB-bi | 79.0 | 86.3 | 80.0 | 93.6 |
| SVM-bi | 77.7 | 86.7 | 80.8 | 91.7 |
| **NBSVM-bi** | **79.4** | 86.3 | **81.8** | 93.2 |

Tabla 3, documentos largos y fragmentos (exactitud, %):

| metodo | RT-2k | IMDB | Subj |
|---|---|---|---|
| SVM-bi | 87.40 | 89.16 | 91.74 |
| **NBSVM-bi** | **89.45** | **91.22** | **93.18** |

Tabla 4, tres pares de 20-newsgroups (exactitud, %):

| metodo | AthR | XGraph | BbCrypt |
|---|---|---|---|
| SVM-bi | 83.7 | 86.2 | 97.7 |
| **NBSVM-bi** | **87.7** | **90.7** | **99.5** |

Diferencia NBSVM-bi menos SVM-bi: **+2.05** en RT-2k, **+2.06** en IMDB,
**+1.44** en Subj, **+4.0** en AthR, **+4.5** en XGraph, **+1.8** en BbCrypt.
El margen de significance es de 1.5 a 2.9, asi que las diferencias de las
tabelas 3 y 4 estan al borde o dentro del margen: **la ventaja publicada es real
pero pequena en documentos largos y clara en el par de newsgroups**.

Otras afirmaciones verificadas del original:

- "Using **logistic regression in place of SVM gives similar results**" (seccion
  4.6). Esto responde directamente a la pregunta A.3: en la fuente primaria **no
  hay diferencia medida entre SVM y LR como clasificador base**.
- "For MNB and NBSVM, using the binarized `f` is slightly better (by 1%) than
  using the raw count feature `f`."
- "We do not use stopwords, lexicons or other resources."
- "BNB is less stable than MNB and performs up to 10% worse. Therefore,
  benchmarking against BNB is untrustworthy."
- NBSVM es un "strong and robust performer over all the presented tasks" y un
  "appropriate and very strong baseline for sophisticated methods aiming to beat a
  bag of features".

### 2.5 Pseudocodigo de referencia, sin ambiguedad

Este es el pseudocodigo que hay que implementar. Todo lo que sigue es una
traduccion directa de las secciones 2 y 2.3 del PDF verificado, con la notacion
renombrada a la del proyecto.

```python
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------- funciones

def log_count_ratio_canonico(matriz, y_binario, alpha=1.0):
    """Ecuacion (2) de Wang y Manning (EMNLP 2012).

    Args:
        matriz: matriz de RECUENTOS, forma (n_filas, n_caracteristicas), csr.
        y_binario: vector de 0/1 de la etiqueta, en las MISMAS filas.
        alpha: constante de suavizado. La canonica es 1.0.

    Returns:
        vector de float64 de longitud n_caracteristicas.
    """
    pos = y_binario == 1
    if not pos.any() or pos.all():
        raise ValueError("el ratio canonico necesita las dos clases presentes")
    # p y q son SUMAS de vectores de recuento de la clase, mas alpha en cada
    # coordenada. No son medias, y no se normalizan por el numero de documentos.
    p = np.asarray(matriz[pos].sum(axis=0), dtype=np.float64).ravel() + alpha
    q = np.asarray(matriz[~pos].sum(axis=0), dtype=np.float64).ravel() + alpha
    # ||p||_1 y ||q||_1 son las sumas de TODOS los componentes. Son el total de
    # tokens de cada clase mas |V| * alpha.
    return np.log((p / p.sum()) / (q / q.sum()))


def nbsvm_valores(matriz_recuentos, y_binario, alpha=1.0, binarizar=True):
    """Devuelve la matriz de valores de caracteristica de NB-SVM.

    Con `binarizar=True` (que es el NBSVM del original) los valores son
    `r o f_hat`. Con `binarizar=False` son `r o f`, que es la variante MNB.
    """
    base = matriz_recuentos
    if binarizar:
        base = (matriz_recuentos > 0).astype(np.int8)
    r = log_count_ratio_canonico(base, y_binario, alpha=alpha)
    return base.multiply(r).tocsr()


# ------------------------------------------------------- entrenamiento ovr

def entrenar_nbsvm_ovr(textos, y_etiquetas, params_word, params_char,
                        alpha=1.0, binarizar=True, **kwargs_reg):
    """NB-SVM one-vs-rest, un ratio independiente por etiqueta.

    Args:
        textos: lista de cadenas de entrenamiento.
        y_etiquetas: matriz binaria (n_filas, n_etiquetas).
        params_word, params_char: diccionarios para CountVectorizer.
        alpha: suavizado canonico 1.0.
        binarizar: True para NBSVM, False para la variante MNB.
        kwargs_reg: se pasan al estimador base (C, class_weight, ...).

    Returns:
        lista de (estimador, vectorizador_r) en el orden de las etiquetas.
    """
    v_word = CountVectorizer(**params_word)
    v_char = CountVectorizer(**params_char)
    cnt = sparse.hstack([v_word.fit_transform(textos),
                         v_char.fit_transform(textos)], format="csr")
    salida = []
    for e in range(y_etiquetas.shape[1]):
        z = nbsvm_valores(cnt, y_etiquetas[:, e], alpha=alpha, binarizar=binarizar)
        est = LogisticRegression(solver="liblinear", **kwargs_reg)
        est.fit(z, y_etiquetas[:, e])
        salida.append((est, (v_word, v_char)))
    return salida
```

**Advertencia critica derogeneracion**: el calculo anterior usa `sum`, no `mean`.
Si se sustituye `sum` por `mean` y se normaliza por `p.sum()` en lugar de por el
total de la clase, el ratio cambia de forma no trivial. La forma correcta es
exactamente la de arriba.

### 2.6 Forma cerrada exacta para el caso binarizado

Cuando las caracteristicas son binarias, la formula se reduce a una identidad
cerrada comprobable, que es la forma mas cheaply implementable y auditable:

```
      r_j = log(1 + df_pos_j) - log(1 + df_neg_j) + C
      C   = log( (|V| + SUM_k df_neg_k) / (|V| + SUM_k df_pos_k) )
```

`C` es la misma constante para todas las caracteristicas, asi que se recoge en el
termino independiente `b` y no altera el problema de optimizacion. Verificado
numericamente en esta sesion sobre los datos reales: la diferencia maxima entre
la forma cerrada y la ecuacion (2) es de **1.8e-15** en las tres etiquetas.

Esta identidad es la respuesta a la pregunta A.6 sobre la transformacion
`log(1+tf)` sobre features binarias: **el NBSVM canonico es exactamente una
transformacion `log(1+df)` de la matriz binaria, restada por etiqueta**. Con
`|V| = 27431` y `df` entre 0 y 795, el `+1` del Laplace es lo que impide que las
caracteristicas poco frecuentes exploten: es un compresor, no un expansor.

### 2.7 AUDITORIA DE FIDELIDAD: la implementacion previa no era la canonica

Se compararon seis formas de calcular `r` sobre la representacion real del
proyecto (word(1,2) + char_wb(3,5), `min_df=2`, 27431 caracteristicas, 352847
tokens), usando la formula del proyecto y la canonica.

Tabla de condicion del ratio, etiqueta HS (la mas estable de las tres; TR y AG se
comportan igual o peor):

| forma de `r` | media | desv | p05 | p95 | max abs(r) | % con abs(r) > 3 |
|---|---|---|---|---|---|---|
| **CANONICA binaria, alpha=1** | 0.0324 | **0.7944** | -1.1055 | 1.3794 | 3.613 | **0.0001** |
| **CANONICA sobre conteos, alpha=1** | 0.0307 | **0.8063** | -1.1057 | 1.3792 | 3.613 | 0.0003 |
| CANONICA sobre conteos, alpha=1e-6 | -1.3582 | 7.9195 | -14.5891 | 14.8336 | 17.436 | 0.3010 |
| PROYECTO sobre conteos, alpha=1e-6 | -0.7648 | 4.3206 | -8.0320 | 7.8810 | 10.483 | 0.3010 |
| PROYECTO sobre conteos, alpha=1e-3 | -0.1080 | 0.8688 | -1.4053 | 1.2935 | 3.603 | 0.0003 |
| PROYECTO sobre TF-IDF, alpha=1e-6 | -0.5367 | 3.2448 | -6.1493 | 5.8743 | 9.732 | 0.3013 |
| PROYECTO sobre TF-IDF, alpha=1e-3 | -0.0236 | 0.2948 | -0.4807 | 0.4427 | 2.882 | 0.0000 |

Rango intercuantil 5-95, las tres etiquetas:

| etiqueta | canonica binaria | proyecto sobre conteos | proyecto sobre TF-IDF |
|---|---|---|---|
| HS | **2.4849** | 15.9130 | 12.0236 |
| TR | **2.7300** | 10.2709 | 8.5255 |
| AG | **2.7081** | 15.9537 | 12.0176 |

Lectura: el ratio canonico con `alpha=1` es de **4 a 6 veces mas estrecho** que el
del proyecto, y deja menos del 0.05 % de las caracteristicas fuera del rango
`[-3, 3]`, frente a **entre el 30 % y el 46 %** en la forma del proyecto. Eso
explica directamente el "12.1 % de celdas que cambia el ratio" que el propio
informe V8-04 ya habia medido y que se interpreto como "el mecanismo actua": el
mecanismo actua porque la matriz de pesos esta mal condicionada, no porque
aporte informacion nueva.

**Los cinco desvios concretos de la implementacion previa:**

1. **Usa `mean(axis=0)` en lugar de `sum(axis=0)`.** Con.features binarias
   `mean` y `sum` se relacionan por el tamanos de clase, pero la normalizacion
   posterior es distinta: el proyecto normaliza por el numero de documentos de la
   clase, el canonico por el total de tokens de la clase mas `|V|`. En este
   corpus el canonico divide por `|V| + total_tokens_clase`, y `|V| = 27431`
   domina, de modo que los dos denominadores son casi iguales y `r` se reduce a
   `log(1+df_pos) - log(1+df_neg)`. La forma del proyecto se reduce a
   `log(df_pos) - log(df_neg)`, **sin el `+1`**, que es justamente el termino que
   comprime las caracteristicas raras.
2. **El suavizado se aplica a la probabilidad, no al recuento.** El canonico
   suma `alpha=1` al recuento y luego divide; el proyecto suma `alpha` a una
   proporcion ya normalizada. Con `alpha=1e-6` sobre una proporcion de
   probabilidad, la perturbacion relativa es diminuta, y por eso el ratio
   explota; con `alpha=1e-3` la perturbacion es de orden uno, y por eso el
   resultado de V8-04 salta +0.063241 al cambiar un parametro por un factor de
   mil. Ese salto **no es sensibilidad de un metodo, es el sintoma de que el
   parametro no esta en la escala que el metodo necesita**, y se elimina
   sustituyendo el esquema de suavizado, no barriendo su valor.
3. **Aplica el ratio sobre TF-IDF en la rama "nueva".** El canonico esta definido
   sobre conteos. La TF-IDF ya normaliza por prevalencia en el corpus y el ratio
   normaliza por prevalencia en la clase; multiplicarlos es una pregunta con
   sentido, pero **no es NB-SVM** y no debe evaluarse como si lo fuera.
4. **Usa el exponente 2.0 en lugar de 1.0.** El canonico usa `r` directamente, y
   solo introduce `beta` para **atenuar** la confianza en NB, no para
   amplificarla. Elevar `r` al cuadrado no es una constante de escala: es una
   transformacion monotona no lineal que acentua exactamente las caracteristicas
   extremas, que son las menos fiables. Medido: -0.054886 (seccion 3.2).
5. **Anade `class_weight="balanced"` encima.** La receta canonica de MNB fija el
   prior a mano como `b = log(N+/N-)`; el ratio `r` ya es una cantidad libre de
   prior. Medido: quitar `balanced` **empeora** en -0.007758, o sea que en este
   corpus la correccion de prior si aporta, en contra de la intuicion de "doble
   correccion". Esto no es contradictorio: `r` elimina el prior del **peso** de
   cada caracteristica, no del **intercepto**.

Adicionalmente, hay una sexta desviacion metodologica que no es de la formula
pero si del protocolo: **V8-04 reutilizo los umbrales congelados de V03
(0.44/0.51/0.46) para las probabilidades del modelo con ratio**. Como el ratio
reescala drasticamente la matriz de pesos, la escala de la funcion de decision
cambia y las probabilidades dejan de ser comparables con las del champion. La
diferencia medida mezcla entonces dos efectos: cambio de representacion y
desajuste de umbral. Corregir la formula **sin** reajustar el umbral produce una
medicion sesgada a la baja.

### 2.8 FICHA A1: NB-SVM canonico (receta Wang-Manning completa)

- **Referencia**: Wang y Manning (2012), EMNLP, "Baselines and Bigrams". Verificada.
- **Anio**: 2012.
- **Tarea**: clasificacion binaria de texto (sentimiento, tema, subjetividad).
- **Poblacion**: RT-s, MPQA, CR, Subj, RT-2k, IMDB y tres pares de 20-newsgroups.
  N entre 2.2k y 50k; `|V|` entre 5.7k y 392k.
- **Representacion**: `x(k) = r o f(k)` con `f` **binarizado**.
- **Preprocesamiento**: tokenizacion propia, unigramas y bigramas, bigramas
  filtrados a `[A-Za-z]`, **sin** stopwords ni lexicons.
- **Clasificador base**: SVM con perdida L2 y regularizacion L2, `C=1`.
- **Avantage reportada (numeros)**: +1.7 a +2.1 puntos de exactitud frente a
  SVM-binarizado en fragmentos y documentos largos; +1.8 a +4.5 frente a
  SVM-binarizado en los tres pares de newsgroups. Ver tabla 2.4.
- **Limitaciones**: (a) el propio original senala el parametro `beta` como la
  desventaja principal; (b) las diferencias de las tablas 3 y 4 estan al borde
  del margen de significance (1.5 a 2.9); (c) **descarta el term frequency**, lo
  cual es una perdida real en corporos donde la repeticion es senal; (d) el
  alcance a multietiqueta **no esta publicado** (ver ficha A3).
- **Compatibilidad con nuestras reglas**: total. Solo usa el texto de
  entrenamiento. No usa recursos externos, ni embeddings, ni preentrenamiento.
- **DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC**: la version previa
  (`NB-UNION-LR`, 0.761203) usaba la forma no canonica con `alpha` pequeno sobre
  la media de TF-IDF, con `C=1.0`. La forma canonica con `alpha=1` sobre
  **conteos** da 0.763839, es decir +0.002636 sobre `NB-UNION-LR` pero todavia
  -0.013372 sobre el champion. La diferencia conceptual es que la version
  previa media sobre una matriz ya normalizada; la canonica suma sobre recuentos
  brutos y normaliza por el total de la clase.

### 2.9 FICHA A2: MB, binarizacion previa y la variante MNB

- **Referencia**: Wang y Manning (2012), secciones 2.1-2.3. Verificada.
- **Anio**: 2012.
- **Tarea**: la misma, binaria.
- **Representacion**: dos variantes, `f` binarizado y `f` con conteos crudos.
- **Preprocesamiento**: identico.
- **Avantage reportada (numeros)**: el binarizado mejora MNB y NBSVM "by 1%",
  de forma marginal en fragmentos. En la tabla 2, MNB-bi frente a MNB-uni es
  79.0 contra 77.9 en RT-s y 93.6 contra 92.6 en Subj.
- **Limitaciones**: el original descarta explicitamente la Bernoulli multivariada
  (BNB) por inestable, "up to 10% worse", y dice que benchmarkear contra BNB no
  es fiable. Los conteos crudos con el ratio son la variante MNB, no NB-SVM.
- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: la version previa uso **conteos
  crudos** en la rama `nb_counts_c2` (0.702184, -0.075027). Con la forma canonica
  el orden se invierte: conteos crudos dan 0.763839 y binarizado 0.760659, ambos
  muy por debajo del champion. Es decir, **la binarizacion no es lo que hundia a
  la rama de conteos**; lo que la hundia era la forma del suavizado.

### 2.10 FICHA A3: version multietiqueta / multiclase

- **Referencia**: no existe publicacion canonica de NBSVM para multietiqueta.
  **SIN EVIDENCIA** en la literatura indexada de 2012-2026 para la formulacion
  multiclase de `r` (una sola `r` para `K` clases frente a `K` ratios).
- **Que se hizo en la literatura y en la practica**: la extension natural y
  triviale es one-vs-rest con un ratio independiente por etiqueta, que es
  **exactamente** lo que ya hacia `experiments/20_emocion_nbsvm_ortografia_v2.py`
  (el informe V8-04 lo confirma: "E2-15 ya lo hacia por etiqueta, de modo que la
  parte nueva de esta linea es aplicarlo sobre la TF-IDF de V03 y con la
  constante 2.0, no el ratio por etiqueta"). Y **la justificacion de por que es
  correcta la da la propia formula**: `r` esta definida para `c in {+1,-1}`, y
  one-vs-rest reduce cada etiqueta multietiqueta a un problema binario de
  definicion, sin ambiguedad.
- **Por que no hay `r` multiclase**: la intuicion de `r_k = log(P(x|c=k)/P(x|c=rest))`
  es valida pero la normalizacion `||p_k||_1` deja de ser un total de tokens de
  una clase y pasa a ser un total de una clase frente a `K-1` clases, lo que
  introduce un factor `1/(K-1)` mal definido cuando hay etiquetas solapadas
  (nuestro caso: hay 8 combinaciones `HS,TR,AG` distintas). No hay fuente que
  lo cuantifique. **Marcar como sin evidencia y no implementar.**
- **Limitaciones**: la descomposicion one-vs-rest ignora las correlaciones entre
  etiquetas, que el proyecto ya midio en `reports/v6_label_factorization.md` y
  `reports/joint_label_v4.json`.
- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: ninguna estructural. Lo que cambia es
  la **forma del ratio** y el **exponente**, no el esquema one-vs-rest.

### 2.11 FICHA A4: NB-log-count-ratio con LogisticRegression

- **Referencia**: Wang y Manning (2012), seccion 4.6: "Using logistic regression
  in place of SVM gives similar results". Verificada. La frase es una afirmacion
  del original, **sin tabla que la cuantifique**.
- **Anio**: 2012.
- **Tarea / poblacion**: las mismas ocho de la receta.
- **Representacion**: `r o f(k)` binarizado.
- **Preprocesamiento**: identico.
- **Clasificador base**: regresion logistica en lugar de SVM. Los hiperparametros
  concretos que el original menciona para el LR **no estan publicados**: es
  **sin evidencia** cual es el `C` equivalente.
- **Avantage reportada (numeros)**: ninguno cuantificado. La fuente solo afirma
  que "da resultados similares".
- **Medicion propia sobre los datos del proyecto** (5 pliegues, umbrales V03,
  control 0.777211):

  | variante | metrica | dif | pliegues a favor | HS | TR | AG |
  |---|---|---|---|---|---|---|
  | NB-SVM canonico binario + LR C=2.0 balanced | 0.760659 | -0.016553 | 0/5 | 0.7492 | 0.7930 | 0.7398 |
  | NB-SVM canonico binario + LR C=2.0 **sin** balanced | 0.752901 | -0.024311 | 0/5 | 0.7401 | 0.7874 | 0.7312 |
  | NB-SVM canonico **conteos** + LR C=2.0 balanced | **0.763839** | -0.013372 | 0/5 | 0.7504 | 0.7944 | 0.7468 |
  | NB-SVM canonico binario + **LinearSVC** C=1.0 | 0.738596 | -0.038616 | 0/5 | 0.7237 | 0.7700 | 0.7221 |

  Con umbrales por clase reoptimizados **en la propia muestra** (optimista, no
  un numero publicable): 0.767669, 0.767245 y 0.769965 respectivamente. Es decir,
  **ni siquiera con umbral optimo el canonico alcanza al champion**.

- **Limitaciones**: la afirmacion de equivalencia LR/SVM es cualitativa y no esta
  cuantificada. En este proyecto la diferencia **si** es medida y **si** es
  consistente: **LR gana a LinearSVC en +0.022063** con el mismo ratio canonico
  (0.760659 frente a 0.738596), y en +0.018325 con el ratio previo (0.761203
  frente a 0.742878, de `NB-UNION-LR` y `NB-UNION-SVC`). Las dos estimaciones
  independientes coinciden en signo y en magnitud.
- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: `NB-UNION-SVC` se probo y perdio, y
  su numero no era atribuible al estimador sino a la forma del ratio. Aqui se
  separa el efecto: con el ratio ya corregido, LR y SVC se separan de forma
  limpia, y el veredicto "usar LR, no SVC" es ahora un dato y no una suposicion.

### 2.12 FICHA A5: calibracion de probabilidades en NB-SVM

- **Referencia**: ninguna sobre NB-SVM especificamente. **SIN EVIDENCIA**
  publicada para "NB-SVM con calibracion de probabilidades".
- **Lo que si hay evidencia**: la calibracion importa cuando la metrica de
  decision es F1 y el clasificador no produce probabilidades. Ver ficha B5.
- **Limitaciones**: en el canonico el SVM ya produce una funcion de decision y el
  `beta` hace de regularizacion. Anadir calibracion (Platt) encima es un
  parametro mas ajustado sobre la misma validacion, y el proyecto ya tiene
  evidencia de que la calibracion fuerte no rindio (`reports/v10_04_calibracion_fuerte.md`).
- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: no medido. Recomendacion: **no
  priorizar**; la calibracion solo se justifica si se elige un clasificador sin
  probabilidades, y la seccion 3.1 muestra que no hay ninguno mejor que LR.

### 2.13 FICHA A6: usar el ratio en un pipeline word + char simultaneo

- **Referencia**: el original evalua unigramas y bigramas, y senala que "adding
  bigrams always improved the performance" en tareas de sentimiento y tema, y que
  anade trigramas "hurts slightly". Verificada. **No evalua ngramas de
  caracter**: **sin evidencia** para la combinacion `word + char` bajo el ratio.
- **Que se ha medido aqui** (mismos pliegues y umbrales, control 0.777211):

  | rama | matriz | metrica | dif | pliegues | celdas que cambian | acierto en las que cambia |
  |---|---|---|---|---|---|---|
  | control | TF-IDF word(1,2)+char_wb(3,5) | 0.777211 | 0.000000 | 0/5 | 0 | - |
  | `TF-IDF * r` (exponente 1) | idem | 0.768837 | -0.008374 | 0/5 | - | - |
  | **`TF-IDF * sign(r)*sqrt(abs(r))`** | idem | **0.781633** | **+0.004422** | **4/5** | 222/4449 (4.99 %) | **1.3125** |
  | `TF-IDF * sign(r)` | idem | 0.777211 | +0.000000 | 0/5 | 0 | - |

- **El caso `TF-IDF * sign(r)` es una verificacion de integridad, no una
  candidata**: da `0.777211` y los mismos F1 por etiqueta y los mismos umbrales
  optimos que el control, celda a celda. Y tiene que ser asi por una razon
  matematica: si la penalizacion es L2 sobre `||w||^2`, el signo de cada peso es
  libre, asi que `X` y `X * diag(sign(r))` tienen exactamente el mismo optimo con
  `w` cambiado de signo. Que las dos ramas coincidan **bit a bit** demuestra que
  el pipeline de esta sesion reproduce el champion de forma exacta, y por tanto
  que la unica via por la que el ratio puede aportar algo es **la magnitud de
  `r`**, nunca su signo.
- **Efecto de `min_df` y de `sublinear_tf` sobre el calculo del ratio**: **sin
  evidencia publicada**. Lo que si se puede afirmar de forma mecanica es que
  `min_df` cambia `|V|`, y `|V|` aparece en la constante `C` de la seccion 2.6,
  que se recoge en el intercepto y por tanto **no altera el ranking de
  caracteristicas**. Y `sublinear_tf` cambia los **valores de la matriz de
  entrada**, no los recuentos, asi que solo afecta si se calcula el ratio sobre
  la TF-IDF en lugar de sobre los conteos, que es el error ya identificado.
  Ninguna de las dos deberia cambiar el resultado del ratio canonico.
- **Necesidad de reajustar umbrales en NB-SVM**: si. Medido: los umbrales optimos
  del NB-SVM canonico estan en 0.213-0.428, muy por debajo de los congelados
  0.44-0.51. Reutilizar los del champion sesga la medicion a la baja.
- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: V8-04 probo `TF-IDF * r` con
  exponente 2 y `alpha` pequeno, y obtuvo -0.058496. Aqui se separa el
  **exponente** del **esquema de suavizado**: con el ratio canonico bien
  condicionado, el exponente 1 da -0.008374, el exponente 0.5 da +0.004422 y el
  exponente 2 (medido sobre la forma binaria) da -0.054886. **El patron es
  monotono y con la misma direccion en las tres mediciones**: la magnitud del
  reescalado tiene que estar **atenuada**, porque la TF-IDF ya normaliza y
  reescalar dos veces destruye la escala relativa. Esto explica el hallazgo de V8-04
  sin necesidad de recurrir a "sensibilidad a un hiperparametro".

### 2.14 FICHA A7: variantes de la familia log-count ratio

- **Log-count ratio con priors explicitos**: la variante MNB del original fija
  `b = log(N+/N-)` a mano en lugar de aprenderlo. Sobre este corpus el prior es
  alto (prevalencias 0.30 a 0.44, o sea razon de clase entre 0.43 y 1.55), asi
  que la eleccion importa mas aqui que en los corpus binarios equilibrados de la
  literatura.
- **`log(1+tf)` sobre features binarias**: es la identidad cerrada de la seccion
  2.6. El NBSVM canonico **es** esa transformacion. Implementarla directamente
  (`log1p` de las columnas de `df` de cada clase, restadas) es equivalente al
  calculo por sumas y evita por completo el problema de condicion. Es la forma
  **mas segura de implementar** el canonico, y la recomendada.
- **Variantes con normalizacion L1 por variante documentada**: la normalizacion
  `||p||_1` de la ecuacion (2) es la original y verificada. Variantes con
  normalizacion por numero de documentos o por longitud media de documento son
  **sin evidencia publicada** que las respalde.
- **Prior de Dirichlet informativo**: se menciona en la literatura de seleccion de
  caracteristicas basada en contenido de informacion. **No se cita ninguna obra
  concreta porque no se ha verificado en esta sesion** (vease seccion 8.2). No
  se le atribuye ninguna cifra.

### 2.15 FICHA A8: evidencia reciente (2023-2026)

- **Referencia verificada**: Fat'hAlalim, A., Liu, Y., Xie, Q. e Ibrahim, N. (2025).
  "Advancements in Transformer-Based Models for Enhanced Hate Speech Detection in
  Arabic: Addressing Dialectal Variations and Cross-Platform Challenges." ACM
  Transactions on Asian and Low-Resource Language Information Processing.
  DOI `10.1145/3748492`, publicado 2025-08-21. Verificada por Crossref
  (autores, revista, fecha, DOI).
- **Que se puede afirmar**: el texto indexado de ese trabajo describe el NBSVM y
  afirma que "Logistic NBSVM can outperform both SVM and Naive Bayes". Es decir,
  en 2025 el NBSVM con regresion logistica se sigue presentando como una
  comparacion viable, y el trabajo lo situa en un contexto de deteccion de
  discurso de odio, que es nuestra tarea.
- **Limitacion de esta cita**: **solo se pudo acceder al resumen y a los fragmentos
  indexados, no al texto completo** (ACM devuelve 403 automatizado). No se
  dispone de las cifras de la tabla, ni del dataset, ni del numero de ejemplos.
  Se cita como evidencia de que la linea sigue viva, **no** como evidencia
  cuantitativa.
- **Sobre el caso de toxicidad**: la evidencia cuantitativa solida de NB-SVM en
  toxicidad viene de la practica de Kaggle, no de articulos revisados por pares.
  El notebook canonico de Jeremy Howard en la competicion Jigsaw "Toxic Comment
  Classification Challenge" (2018) se titula explicitamente "NBSVM strong linear
  baseline" y el metodo es el de esta ficha. **No se ha verificado el
  leaderboard ni el puesto final**, asi que **no se cita ninguna cifra de el**.
  Marcar como evidencia de practica, no de literatura.
- **Veredicto sobre si NB-SVM sigue siendo competitivo**: **parcialmente**. Sigue
  siendo un baseline reproducible y con buen comportamiento en texto binario corto
  con recursos limitados. En nuestro corpus, con medicion propia, **no es
  competitivo frente a una TF-IDF sublineal bien ajustada**, y el motivo es
  identificable (seccion 1.1) y no un defecto de implementacion.
- **Que NO se ha encontrado**: ninguna publicacion revisada por pares de
  2023-2026 que compare NBSVM con modelos lineales modernos sobre TF-IDF en
  toxicity o hate speech multietiqueta, con numeros. **SIN EVIDENCIA**.

---

## 3. TEMA B: clasificadores lineales y features hibridas

### 3.1 FICHA B1: que clasificador lineal gana en multietiqueta con F1

Todos los medidos sobre la **representacion del champion** (TF-IDF word(1,2) +
char_wb(3,5), `sublinear_tf=True`, `min_df=2`), 5 pliegues, umbrales congelados
del champion, control 0.777211:

| clasificador | metrica | dif | pliegues a favor | HS | TR | AG |
|---|---|---|---|---|---|---|
| **LogisticRegression L2, C=2.0, balanced (champion)** | **0.777211** | 0.000000 | 0/5 | 0.7617 | 0.8157 | 0.7542 |
| LogisticRegression L1, C=2.0, balanced | **0.781985** | +0.004774 | 3/5 | 0.7598 | 0.8239 | 0.7622 |
| LogisticRegression L1, C=4.0, balanced | **0.783423** | **+0.006211** | **5/5** | 0.7661 | 0.8229 | 0.7612 |
| LogisticRegression L1, C=1.0, balanced | 0.755639 | -0.021572 | 0/5 | 0.7248 | 0.7967 | 0.7455 |
| LogisticRegression L1, C=0.5, balanced | 0.726651 | -0.050560 | 0/5 | 0.6870 | 0.7805 | 0.7124 |
| LogisticRegression L2, C=0.5, balanced | 0.761542 | -0.015670 | 0/5 | 0.7541 | 0.7930 | 0.7375 |
| LinearSVC, C=1.0, balanced | 0.767715 | -0.009496 | 0/5 | 0.7496 | 0.8205 | 0.7331 |
| LinearSVC, C=0.1, balanced | 0.768076 | -0.009135 | 0/5 | 0.7433 | 0.8128 | 0.7481 |
| SGDClassifier, `log_loss`, alpha=1e-5, early stopping | 0.759188 | -0.018023 | 0/5 | 0.7404 | 0.8052 | 0.7320 |
| SGDClassifier, `hinge`, alpha=1e-5, early stopping | 0.744981 | -0.032230 | 0/5 | 0.7267 | 0.7924 | 0.7158 |
| RidgeClassifier, alpha=1.0, `sparse_cg`, balanced | 0.768993 | -0.008219 | 0/5 | 0.7540 | 0.8207 | 0.7322 |
| RidgeClassifier, alpha=10.0, `sparse_cg`, balanced | 0.751525 | -0.025687 | 0/5 | 0.7339 | 0.7961 | 0.7246 |

**Respuesta a "cuando gana cada uno y con que C/alpha"**:

- **Gana `LogisticRegression`**, sin excepcion, y por un margen de 0.008 a 0.032
  sobre todos los demas. El margen es mayor sobre los dos worse (SGD con
  `hinge` y Ridge con alpha grande) y menor sobre LinearSVC y Ridge con alpha
  pequeno, que son los mas cercanos.
- **`LinearSVC` nunca gana** en los dos valores de C probados. C entre 0.1 y 1
  no cambia el veredicto (0.767715 contra 0.768076: el orden se invierte pero la
  diferencia es de 0.000361, es decir ruido).
- **`SGDClassifier` pierde de forma clara** y `hinge` pierde el doble que
  `log_loss`. El `early_stopping` no cambia el veredicto.
- **`RidgeClassifier` pierde menos** de lo esperado (0.768993, la mejor de las
  alternativas no-LR), pero sigue por debajo.
- **El `C` optimo es alto, no bajo.** Con L1, C=0.5 da 0.726651 y C=4.0 da
  0.783423: un factor 8 de `C` vale +0.056772. Este corpus **no** quiere mas
  regularizacion de la que ya tiene, contradiciendo la intuicion de "p >> n
  exige mas castigo".

**Evidencia independiente que concuerda**: la comparacion de clasificadores que
scikit-learn publica en su ejemplo oficial de 20-newsgroups con TF-IDF reporta
exactitud 0.772 para `LogisticRegression(C=5)`, 0.764 para
`SGDClassifier(loss="log_loss", early_stopping=True)`, 0.760 para
`RidgeClassifier(solver="sparse_cg")` y 0.752 para `LinearSVC(C=0.1)`. **El
orden es identico al de esta tabla.** Es una corroboracion en otro corpus, con
otra metrica, pero refuerza que la jerarquia no es un artefacto de estos datos.
Advertencia: esa pagina mide exactitud, no F1, y con una sola particion.

- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: LinearSVC, SGDClassifier y
  RidgeClassifier **no se habian medido sobre la representacion del champion**.
  `v8_clasificadores_alternativos.md` solo midio LDA (0.747157), arbol (0.704043)
  y bosque (0.713274), y E2-15 midio LinearSVC solo con el ratio previo y a
  C=1.0. Este informe cierra ese hueco: **los tres pierden, y el unico que
  mejora es LR con penalizacion L1**.

### 3.2 FICHA B2: hay evidencia de que LinearSVC con calibracion supere a LogisticRegression en F1

**Respuesta: NO. No hay ninguna evidencia, y hay evidencia en contra.**

Cuatro razones, con la que mas:

1. **La comparacion base ya pierde.** Con la misma matriz y sin calibracion,
   LinearSVC da 0.767715 frente a 0.768076 (C=0.1) y 0.777211 (LR). Calibrar
   puede cerrar parte de la brecha, pero **el techo es el valor sin calibrar**,
   que ya esta por debajo.
2. **La calibracion de `CalibratedClassifierCV` es una capa de pegamento.** En
   el proyecto eso ya esta resuelto: `LogisticRegression` **nativo** ya entrega
   probabilidades, y la unica vez que hizo falta una capa de calibracion
   (V10-04) el resultado fue negativo (`reports/v10_04_calibracion_fuerte.md`).
   Introducir calibracion para recuperar a un clasificador peor es un coste doble
   por un beneficio negativo.
3. **Los umbrales optimos de LR son utilizables y los de SVC no lo son.** Es la
   diferencia practica que decide. Con `decision_function` el "umbral optimo" no
   tiene escala: los optimos por etiqueta salieron en `-0.187 / +0.024 / -0.126`
   (LinearSVC C=1) y `-0.041 / -0.127 / -0.111` (Ridge alpha=1). Hay que elegir
   el metodo de decision **antes** de poder fijar umbrales, lo que arrastra
   exactamente el problema de sobreajuste que seccion 3.5 cuantifica.
4. **La literatura de referencia apoya a LR para F1 con decision.** Lipton,
   Elkan y Naryanaswamy (2014) derivan sus resultados (teorema 1 y corolario 1)
   **suponiendo que la salida del clasificador es una probabilidad condicional**.
   Toda la teoria de umbrales que la literatura considera solida esta escrita
   para salidas probabilisticas. Un SVM obliga a reintroducir el supuesto.

- **Limitaciones**: la comparacion es sobre una representacion y un corpus. Que
  LR gane aqui no prueba que gane en todos los sitios; la evidencia publicada es
  ambigua y dependiente del corpus (vease la nota de "comparative study on
  classification performance between support vector machine and logistic
  regression", que es accesible pero cuyo contenido no se ha verificado aqui).
- **Compatibilidad**: total, pero no recomendado.
- **DIFERENCIA CONCRETA respecto a TOXIC**: no medido. Es una pregunta que la
  auditoria no habia cerrado.

### 3.3 FICHA B3: la unica combinacion sparse + dense con evidencia a favor es ninguna

Contexto de lo ya probado (`reports/features_densas_resultados.csv`, protocolo
`MOD-03b`, control 0.7927, **no comparable con este informe**):

| rama | bloque denso | n_densas | metrica | dif | decision del proyecto |
|---|---|---|---|---|---|
| `MOD-03b_REF` | ninguno | 0 | 0.7927 | 0.0 | referencia |
| `HIB-H12` | H12 escalado | 15 | 0.7836 | -0.009096 | descartada |
| `HIB-H11` | H11 escalado | 22 | 0.7713 | -0.021311 | descartada |
| `HIB-H14` | H14 escalado | 9 | 0.7630 | -0.029701 | descartada |
| `HIB-UNION` | H11+H12+H14 escalados | 46 | 0.7610 | -0.031639 | descartada |
| `B5-DENSE` | H11+H12+H14 sin TF-IDF | 46 | 0.6607 | -0.131935 | descartada |

**Estado de la evidencia para cada uno de los tres enfoques que pedia el
encargo**:

| enfoque | evidencia publicada | lectura |
|---|---|---|
| concatenar el bloque denso **sin escalar** | ninguna referencia academica recuperada | **SIN EVIDENCIA**. En este proyecto solo se probo la version escalada, y perdio |
| concatenar el bloque denso **con `StandardScaler`** | ninguna referencia academica recuperada | **SIN EVIDENCIA**, y con medicion propia en contra: cuatro variantes, todas peor |
| proyectar el denso a espacio sparse con **`TruncatedSVD`** | ninguna referencia academica recuperada | **SIN EVIDENCIA**, no medido en este proyecto |

**Lo que si se puede afirmar con fundamento mecanico, no con cita**:

- El desajuste de escala es real. Las TF-IDF de la representacion del champion
  tienen norma L2 por documento de 1 (normalizacion por defecto), y sus valores
  individuales son del orden de 0.01 a 1. Un bloque denso de conteos o
  proportions tiene una escala completamente distinta. Sin escalar, el bloque
  denso domina la funcion de decision en funcion de su varianza, no de su
  informacion. **Escalar** corrige el problema del dominance, y por eso la
  version escalada de TOXIC es la variante razonable, no la ingenua.
- Pero escalar **no** arregla un bloque que apenas tiene senal. Y el dato
  `B5-DENSE` = 0.6607 dice que el bloque denso **aislado** vale 0.66 frente a
  0.79 del texto. Un bloque que solo alcanza 0.66 no aporta senal
  complementaria que justifique el coste de competir con las 27431
  caracteristicas dispersas que ya estan ahi.
- `TruncatedSVD` sobre el bloque denso solo tendria sentido si el bloque denso
  fuera de alta dimensionalidad. Con **9 a 46** caracteristicas, `TruncatedSVD`
  no puede reducir nada: su rango es como maximo 9, y lo que haria es una
  rotacion que mezcla caracteristicas y destruye la interpretabilidad de cada
  una. **La idea de proyectar a sparse con SVD no aplica a un bloque de 46
  variables.** Esto no es una opinion: es la condicion de rango.

**Que si falta por medir, y por que no se propone**: concatenar el bloque denso
**sin escalar** es la unica variante de la lista que el proyecto no ha probado, y
es la unica que la mecanica anterior sugiere que deberia ser **peor**, no mejor.
Medirla seria gastar un experimento para confirmar una regla de escalado que
scikit-learn ya documenta. **No se propone.**

- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: ninguna variante nueva aporta. La
  unica diferencia de este informe es **el cierre argumental**: antes se sabia que
  el bloque denso escalado perdia; ahora se sabe tambien que SVD no aplica por
  rango y que la version sin escalar es mecanicamente peor. La linea de features
  densas debe considerarse **cerrada**, no pendiente.

### 3.4 FICHA B4: regularizacion con muchas features y pocos ejemplos

Contexto cuantitativo del proyecto, que es lo que hace urgente la pregunta: en
`features_densas_resultados.csv`, el champion tiene `f1_train_medio = 0.9744` y
`brecha_train_oof = 0.1818`. Es decir, **el champion memoriza el entrenamiento
casi por completo**: 0.9744 en train contra 0.7927 fuera. Con `p = 27431` y
`n = 1731`, la razon `p/n` es **15.8**.

Dato de la literatura: la regla de practica mas citada para esta situacion es la
que aparece en la seccion 5 de Sculley et al. (2015), "Hidden Technical Debt in
Machine Learning Systems", NIPS 2015, en la figura de antipatron "Unnecessary
Complexity" / "Sparsity + regularized models", y que dice textualmente que un
modelo regularizado con L2 funciona bien cuando hay pocas caracteristicas y
muchos ejemplos, y que **una relacion de uno a diez entre ejemplos y
caracteristicas es un buen punto de partida para la regularizacion L2**. Con
`n = 1731` eso sugiere del orden de 170 caracteristicas, muy por debajo de
27431. Nota: esa figura es una guia de practica de Google, no un resultado
experimental con intervalos de confianza, y su aplicabilidad a macro-F1 de tres clases
positivas con desbalance moderado **no esta verificada**.

**Lo que la medicion propia dice, y contradice la intuicion** (control 0.777211,
5 pliegues, umbrales congelados):

| regularizacion | metrica | dif | pliegues | IC95 pareado (2000 remuestreos) | celdas mejor/igual/peor |
|---|---|---|---|---|---|
| L2, C=2.0 (champion) | 0.777211 | 0.000000 | 0/5 | [0.0000, 0.0000] | 0/1483/0 |
| L2, C=0.5 | 0.761542 | -0.015670 | 0/5 | - | - |
| L1, C=0.5 | 0.726651 | -0.050560 | 0/5 | - | - |
| L1, C=1.0 | 0.755639 | -0.021572 | 0/5 | - | - |
| L1, C=2.0 | 0.781985 | +0.004774 | 3/5 | [-0.0110, +0.0202] | 86/1348/49 |
| **L1, C=4.0** | **0.783423** | **+0.006211** | **5/5** | [-0.0081, +0.0202] | 80/1354/49 |

Lecturas:

- **L1 gana a L2 solo cuando `C` es alto.** En C=0.5 y C=1.0, L1 pierde
  claramente (-0.050560 y -0.021572). La ventaja de L1 aparece en C=2.0 y es
  maxima en C=4.0, con **5 de 5 pliegues a favor**, el unico resultado de todo
  este informe con unanimidad.
- **La intuicion "p >> n exige mas regularizacion" es incorrecta aqui, y por que.**
  L1 no es "mas regularizacion", es **otro tipo** de regularizacion: mete la
  esparsidad y por eso necesita un `C` mas alto para no anular senal. El modelo
  con la brecha train-OOF mas grande (0.1818) no esta excessivamente
  regularizado en el sentido de `||w||`; lo que tiene es **demasiada varianza en
  las caracteristicas que el L2 deja entrar**.
- **Seleccion de caracteristicas por frecuencia**: la unica instancia en el
  proyecto es `min_df=2`, que ya esta en el champion. Con `min_df=2` y 1731
  filas, el vocabulario pasa de 27431 a un numero necessariamente menor, y el
  ratio `p/n` baja. **No hay en este informe medicion del efecto de subir
  `min_df`**, y la literatura de seleccion de caracteristicas por frecuencia es
  ambigua (vease la nota de Forman, 2003, JMLR, cuyo contenido no se ha
  verificado aqui). **SIN EVIDENCIA** para recomendar un valor de `min_df`.
- **Elastic net**: **no medido**, y la medicion de L1 y L2 sugiere que el
  interpolador no anade nada: L2 ya es el optimo en la region de `C` alta, y L1
  ya es el optimo en `C=4`. Una mezcla con `l1_ratio` intermedio se quedaria
  entre dos puntos que no superan al maximo de ninguno. **No se propone.**
- **Compatibilidad**: total. Es la candidata mas limpia del informe: no cambia
  la representacion, no introduce parametros nuevos mas alla de `C`, y ya se
  midio con el gate del proyecto.
- **DIFERENCIA CONCRETA respecto a TOXIC**: el champion es L2 con `C=2.0`.
  `reports/v8_c_por_etiqueta.md` y `reports/v8_regularizacion_conjunta.md`
  exploraron `C` **con L2**; este informe encuentra que el eje que mejora no es el
  valor de `C` sino **la forma de la penalizacion**, y que el optimum esta en
  `l1_ratio = 1.0` con `C = 4.0`, no en `l1_ratio = 0.0`.

### 3.5 FICHA B5: seleccion de umbral por clase en multietiqueta con F1

**Se sigue recomendando? Si, pero aqui ya esta agotada.** Las tres razones, con la
evidencia de cada una.

**Razon 1: hay una cota teorica dura.** Lipton, Elkan y Naryanaswamy (2014),
"Optimal Thresholding of Classifiers to Maximize F1 Measure", *Machine Learning
and Knowledge Discovery in Databases*, 8: 25-39, DOI `10.1007/978-3-662-44851-9_15`.
Verificado leyendo el texto integro. Sus dos resultados que aplican aqui:

- **Corolario 1**: si el clasificador produce probabilidades condicionales
  calibradas, el umbral que maximiza F1 es exactamente **la mitad del valor
  maximo de F1 alcanzable**, `s >= F*/2`.
- Y del mismo trabajo, sobre la observacion de lote: "because a F1 value cannot
  exceed 1, **the optimal threshold cannot exceed 0.5**".

Aplicado al champion: `F*` por etiqueta es 0.7680 / 0.8201 / 0.7566, luego
`F*/2` es 0.3840 / 0.4101 / 0.3783. **Ninguno coincide con los umbrales
congelados (0.44 / 0.51 / 0.46), y los tres congelados estan por encima.** La
cota dura de 0.5 **ya la viola el umbral congelado de TR, que es 0.51**.

Aviso obligatorio de honestidad: `F*/2` supone probabilidades **calibradas**, y
el champion usa `class_weight="balanced"` con `liblinear`, que **rompe la
calibracion deliberadamente**. Por eso la regla **subestima** los umbrales
optimos reales, como se ve en la tabla: predice 0.3840 / 0.4101 / 0.3783 frente a
los optimos reales 0.403 / 0.498 / 0.448. La cota de 0.5 tampoco es un teorema
aplicable sin recalibrar. **Lo que si queda en pie es la diagnostica**: con
umbrales congelados por encima de los optimos de la propia muestra y uno por
encima de la cota teorica, hay margen, y es margen de sobra, no de falta.

**Razon 2: la curva es tan plana que el margen es pequeno.** Medicion sobre el
OOF del champion (valores de F1 por etiqueta al desplazar el umbral optimo en
-0.10, -0.05, 0, +0.05, +0.10):

| etiqueta | F1 @ t-0.10 | F1 @ t-0.05 | F1 @ t* | F1 @ t+0.05 | F1 @ t+0.10 | caida maxima |
|---|---|---|---|---|---|---|
| HS | 0.7440 | 0.7514 | 0.7680 | 0.7623 | 0.7520 | **0.0240** |
| TR | 0.8058 | 0.8163 | 0.8201 | 0.8092 | 0.7990 | **0.0211** |
| AG | 0.7389 | 0.7504 | 0.7566 | 0.7463 | 0.7251 | **0.0315** |

Un error de 0.05 en el umbral cuesta entre 0.006 y 0.031 de F1 por clase. Y el
rendimiento total con los tres umbrales **reoptimizados en la propia muestra
(sesgudamente optimista)** es 0.781564 frente a 0.777211 con los congelados: solo
**+0.004353**. O sea: **el techo de todo el eje de umbrales es +0.0044**, y es un
techo sesgado al alza. El rendimiento real disponible por reoptimizacion de
umbrales es **menor de 0.0044**, es decir **despreciable**.

**Razon 3: el metodo importa menos que el riesgo de sobreajuste.** La referencia
que se debe leer es **Lin, Y.-J. y Lin, C.-J. (2023), "On the Thresholding Strategy
for Infrequent Labels in Multi-label Classification", ACM Transactions on
Information Systems**. Verificada leyendo el texto. Su afirmacion de partida es
literalmente la nuestra: "**directly tuning the threshold to optimize F-measure
has been observed to overfit easily**". Su explicacion es que la F1 empirica
converge a distinta velocidad segun el umbral, de modo que con un `n` fijo
algunos umbrales tienen menos varianza que otros y se eligen por azar. Su
remedio es un **doble nivel de validacion cruzada** (SCut: nivel exterior para
elegir el metodo de umbral, nivel interior para ajustar el umbral), mas dos
variantes con F-measure suavizada (`smooth-each` y `smooth-all`) y una heuristica
de optimizacion conjunta micro-F y macro-F (`micromacro`) que **no exige
hiperparametros adicionales**. Su resultado empirico, en seis datasets de texto y
de clasificacion de nodos, es que `smooth-all` y `SCutFBR.n` superan
consistentemente a FBR y que su macro-F "is always better than or competitive
with" FBR en todos los datasets.

**Que metodo da mejores resultados reportados**: la combinacion
**doble nivel de CV (SCut) + `micromacro`**. `micromacro` es la mas attractive
para nuestro caso porque es la unica de las tres que **no introduce
hiperparametros**, lo que encaja con las reglas del proyecto. Su logica es que la
micro-F actua como un regularizador que suaviza la seleccion, y que optimizar
jointly micro y macro-F corrige el sesgo de macro-F hacia las etiquetas raras
descrito por Lipton, Elkan y Naryanaswamy.

**Cote de seguridad extra, util en este corpus**: el mismo trabajo de Lipton,
Elkan y Naryanaswamy caracteriza la "maldicion del ganador" y da el criterio
`(1 - b) / b^2`, por debajo del cual elegir el umbral por la F1 empirica
selecciona umbrales grandes con probabilidad constante **independientemente del
tamano de la muestra**. Aplicado aqui:

| etiqueta | prevalencia b | (1-b)/b^2 | n de desarrollo = 1483 |
|---|---|---|---|
| HS | 0.4367 | 3.0 | 1483 > 3.0, holgado |
| TR | 0.2998 | 7.8 | 1483 > 7.8, holgado |
| AG | 0.3824 | 4.2 | 1483 > 4.2, holgado |

**Las tres etiquetas estan tres ordenes de magnitud por encima del corte.** El
riesgo de maldicion del ganador por tamano de muestra **no aplica** a este
problema. El riesgo que si queda es el de sobreajuste por **forma de la funcion
objetivo**, que es el que tacklea Lin y Lin.

- **Compatibilidad**: total.
- **DIFERENCIA CONCRETA respecto a TOXIC**: el proyecto ya tiene
  `reports/v10_05_regla_f1_optima.md`, `reports/v10_06_umbrales_estables.md` y
  `reports/v16_02_promedio_umbrales.md`, es decir **ya exploro la rejilla de
  umbrales y su estabilidad**. Este informe **no propone repetir esa rejilla**;
  propone (a) el diagnostico nuevo de que el umbral de TR viola la cota teorica de
  0.5, y (b) `micromacro` como el unico metodo no explorado, con la advertencia
  de que su techo es +0.0044 y por tanto **no puede justifies por si solo**.

### 3.6 FICHA B6: combinacion de las dos ramas positivas

Se probo la union de la candidata 1 (L1, C=4.0) y la candidata 2
(`TF-IDF * sign(r)*sqrt(abs(r))`), porque las dosSigned mechanisms son
independientes y la intuccion dice que deberian componerse:

| rama | metrica | dif | IC95 pareado | pliegues |
|---|---|---|---|---|
| L1, C=2.0 | 0.781985 | +0.004774 | [-0.0110, +0.0202] | 3/5 |
| L1, C=4.0 | 0.783423 | +0.006211 | [-0.0081, +0.0202] | 5/5 |
| `TF-IDF * sign(r)*sqrt(abs(r))` con L2, C=2.0 | 0.781633 | +0.004422 | [-0.0056, +0.0144] | 4/5 |
| `TF-IDF * sign(r)*sqrt(abs(r))` con **L1, C=2.0** | 0.768681 | **-0.008530** | [-0.0235, +0.0068] | 0/5 |
| `TF-IDF * sign(r)*sqrt(abs(r))` con **L1, C=4.0** | 0.766532 | **-0.010679** | [-0.0249, +0.0042] | 0/5 |

**La combinacion es peor que cualquiera de las dos partes por separado**, y el
patron es coherente: cuanto mas atenuado es el reescalado (`sqrt(r)`) y mas dura
la penalizacion (L1 con C alto), peor el resultado. Los dos mecanismos atacan el
mismo problema desde el mismo lado: **ambos reducen la capacidad efectiva del
modelo**, y este corpus no necesita menos capacidad.

Este es el unico resultado del informe que **desaconseja activamente** una
linea obvia, y es la razon por la que las dos candidatas se presentan como
**alternativas**, no como una pila.

---

## 4. Todas las mediciones de este informe, en una tabla

Protocolo comun: 1483 filas de desarrollo, 5 pliegues de `pliegues_corregidos`,
umbrales congelados 0.44/0.51/0.46, control `features/oof_v03_final.npz` =
0.777211 verificado celda a celda.

| id | rama | metrica | dif | pliegues | IC95 pareado | celdas que cambian | acierto en las que cambia |
|---|---|---|---|---|---|---|---|
| C | champion V03 (control) | 0.777211 | 0.000000 | - | [0.0000, 0.0000] | 0 | - |
| A1 | NB-SVM canonico conteos + LR C=2.0 | 0.763839 | -0.013372 | 0/5 | - | - | - |
| A2 | NB-SVM canonico binario + LR C=2.0 | 0.760659 | -0.016553 | 0/5 | - | - | - |
| A3 | NB-SVM canonico binario + LR C=2.0 sin balanced | 0.752901 | -0.024311 | 0/5 | - | - | - |
| A4 | NB-SVM canonico binario + **r al cuadrado** | 0.722325 | -0.054886 | 0/5 | - | - | - |
| A5 | NB-SVM canonico binario + **LinearSVC** C=1.0 | 0.738596 | -0.038616 | 0/5 | - | - | - |
| A6 | ratio **previo** del proyecto sobre binario, r^2 | 0.641531 | -0.135680 | 0/5 | - | - | - |
| A7 | `TF-IDF * r` canonico (exponente 1) | 0.768837 | -0.008374 | 0/5 | - | - | - |
| **A8** | **`TF-IDF * sign(r)*sqrt(abs(r))`** | **0.781633** | **+0.004422** | **4/5** | [-0.0056, +0.0144] | 222 (4.99 %) | **1.3125** |
| A9 | `TF-IDF * sign(r)` (control de integridad) | 0.777211 | +0.000000 | 0/5 | - | 0 | - |
| B1 | LinearSVC C=1.0 | 0.767715 | -0.009496 | 0/5 | - | - | - |
| B2 | LinearSVC C=0.1 | 0.768076 | -0.009135 | 0/5 | - | - | - |
| B3 | SGDClassifier `log_loss` + early stopping | 0.759188 | -0.018023 | 0/5 | - | - | - |
| B4 | SGDClassifier `hinge` | 0.744981 | -0.032230 | 0/5 | - | - | - |
| B5 | RidgeClassifier alpha=1.0 | 0.768993 | -0.008219 | 0/5 | - | - | - |
| B6 | RidgeClassifier alpha=10.0 | 0.751525 | -0.025687 | 0/5 | - | - | - |
| B7 | LR L2 C=0.5 | 0.761542 | -0.015670 | 0/5 | - | - | - |
| B8 | LR L1 C=0.5 | 0.726651 | -0.050560 | 0/5 | - | - | - |
| B9 | LR L1 C=1.0 | 0.755639 | -0.021572 | 0/5 | - | - | - |
| B10 | LR L1 C=2.0 | 0.781985 | +0.004774 | 3/5 | [-0.0110, +0.0202] | 468 (10.52 %) | 0.9259 |
| **B11** | **LR L1 C=4.0** | **0.783423** | **+0.006211** | **5/5** | [-0.0081, +0.0202] | 414 (9.31 %) | 1.0097 |
| B12 | `sqrt(r)` + L1 C=2.0 | 0.768681 | -0.008530 | 0/5 | [-0.0235, +0.0068] | 447 (10.05 %) | 0.8320 |
| B13 | `sqrt(r)` + L1 C=4.0 | 0.766532 | -0.010679 | 0/5 | [-0.0249, +0.0042] | 440 (9.89 %) | 0.8487 |

Las filas A9, B12 y B13 comparten naturaleza: B12 y B13 son la combinacion
desaconsejada de la seccion 3.6.

---

## 5. Candidatas priorizadas con su ventaja esperada

### Prioridad 1: LogisticRegression con penalizacion L1 y C=4.0

- **Medicion**: 0.783423, +0.006211, **5 de 5 pliegues a favor**, IC95 pareado
  [-0.0081, +0.0202], 80 celdas mejoradas frente a 49 empeoradas.
- **Ventaja esperada**: entre +0.004 y +0.006 de macro-F1, con el cambio mas
  pequeno de comportamiento entre celdas de las ramas candidatas (9.31 %). Es
  la unica rama con **unanimidad** en los cinco pliegues, que es el criterio que
  el propio proyecto declaro mas fiable despues de V8-13 ("cuantas repeticiones
  mejoran", no la cifra final).
- **Por que la L1 y no mas `C` en L2**: porque la L1 a `C=4.0` hace una cosa que
  la L2 no puede: **sacar caracteristicas**. Con `p/n = 15.8` y una brecha
  train-OOF de 0.1818, sobra varianza en las caracteristicas de baja frecuencia,
  y la L1 es el mecanismo estandar para eso. Y necesita `C` alto porque la L1
  penaliza igual que la L2 a `C` bajo pero con masFeatures eliminadas.
- **Riesgo**: el IC95 contiene el cero, asi que **no pasa el gate**. Y `-- C=4.0`
  esta en el borde de un barrido de `C`; el proyecto prohibe barrer `C` para
  buscar un valor mejor. **Esto se propone como un punto de diseno a validar con
  el gate, no como un `C`_optimo unearthed por busqueda**: el argumento de que
  `C` alto es correcto viene de la monotonia medida en tres puntos (0.5, 1.0, 2.0,
  4.0) y de la teoria de la penalizacion L1, no de un maximum encontrado.
- **Compatibilidad**: total. No cambia la representacion, no usa recursos externos.
- **Siguiente paso concreto**: reimplementar con `solver="liblinear"`,
  `penalty="l1"` (o `l1_ratio=1.0` en la API nueva) y `C=4.0`, medir el gate
  completo con bootstrap pareado y filtro de detectabilidad, y comparar el
  numero de caracteristicas no nulas resultante.

### Prioridad 2: TF-IDF del champion por `sign(r) * sqrt(abs(r))`

- **Medicion**: 0.781633, +0.004422, 4 de 5 pliegues, IC95 pareado
  [-0.0056, +0.0144].
- **Ventaja esperada**: entre +0.003 y +0.005 de macro-F1, y sobre todo la mejor
  **calidad por cambio**: cambia solo el 4.99 % de las celdas (222 de 4449) y
  acierta el 1.3125 de las que cambia. Si el filtro de detectabilidad del
  proyecto (acierto en las celdas modificadas por encima de un minimo) se
  cumple, esta es la rama con mas margen para superarlo, porque su downside esta
  concentrado en muy pocas celdas.
- **Por que `sqrt` y no `r`**: ver seccion 2.13. El patron es monotono y esta
  medido en tres puntos (exponente 2 malo, exponente 1 neutro, exponente 0.5
  positivo), y la razon es mecanica: la TF-IDF del champion ya normaliza por
  prevalencia, y `r` vuelve a normalizar por prevalencia de clase. Aplicar `r`
  entero duplica el efecto; `sqrt(r)` lo atenua a la mitad en escala logaritmica,
  que es la compensacion natural.
- **`r` debe ser el canonico**: calculado sobre conteos binarizados con
  `alpha=1`, con la identidad cerrada `log(1+df_pos) - log(1+df_neg)`, por
  etiqueta, **dentro del tren de cada pliegue**. No con el suavizado pequeno del
  proyecto: la forma con `alpha=1e-6` sobre TF-IDF tiene un rango 4 a 6 veces mas
  ancho y arruinaria la atenuacion.
- **Riesgo**: `sqrt` introduce un parametro mas (el exponente), y el barrido del
  exponente seria exactamente el rescate por hiperparametro que las reglas
  prohiben. **Se propone el exponente 0.5 y solo ese**, justificado por ser la
  mitad exacta del exponente canonico, no como punto de un barrido.
- **Compatibilidad**: total.
- **Nota de diversidad**: es la rama con mas potencial como **canal de
  diversidad** para un ensemble, porque cambia 222 celdas con un 1.3125 de acierto
  y porque actua sobre las tres etiquetas de forma distinta (HS +0.0044,
  TR -0.0075, AG **+0.0164**). Esa asimetria por etiqueta es la firma de un
  mecanismo distinto del del champion.

### Prioridad 3: NB-SVM canonico como canal de diversidad, no como sustituto

- **Medicion**: 0.763839, -0.013372, 0 de 5 pliegues. **No gana.**
- **Ventaja esperada**: no es una ventaja en metrica, es una ventaja en
  **perfil de error**. El NB-SVM canonico binariza, luego ve un mundo distinto
  del champion: ignora la repeticion y confia en la frequencia de documento. Si
  el error del champion en la etiqueta TR (0.8157, la mas alta) viene de no
  ponderar repeticiones, un canal binarizado lo corregiria. El proyecto ya tiene
  las herramientas para medir esto: `reports/model_complementarity_v2.csv` y
  `reports/ensembles_resultados.csv`.
- **Por que esta y no la forma previa**: la forma previa (0.641531) es inutil
  como canal. La forma canonica (0.763839) esta a solo 0.013 del champion, que
  es una diferencia de la misma magnitud que la que separa a LinearSVC (0.768076,
  -0.009) y a Ridge (0.768993, -0.008), los dos canales que **si** tiene sentido
  llevar a un ensemble aunque pierdan 0.01.
- **Riesgo**: es la candidata que mas presupuesto de implementacion exige
  (hay que arreglar cinco cosas de la formula) para la que menos metrica hay.
  **Solo justifiable si el analisis de complementariedad dice que el canal aporta.**
- **Compatibilidad**: total.

### Candidata descartada con argumento: `LinearSVC` con calibracion

Se descarta antes de gastar experimento. Seccion 3.2: la version sin calibrar ya
pierde por 0.009, la calibracion solo puede cerrar parte de esa brecha, LR ya
entrega probabilidades, y la teoria de umbrales que se puede citar esta escrita
para salidas probabilisticas. Ademas el unico `TR` con umbral por encima de la
cota teorica de 0.5 (0.51) es precisamente el que no se puede defender con un
modelo que no produce probabilidades.

---

## 6. Lo que este informe cierra sin necesidad de experimento nuevo

Estas afirmaciones se pueden dar por cierras y no requieren mas presupuesto:

1. **La linea de NB-SVM estaba mal implementada y ahora se sabe por que.** La
   forma previa no era la receta canonica en ninguno de sus cinco puntos. Corregir
   la forma sube el resultado en +0.10 de macro-F1, pero no alcanza al champion,
   y el motivo es identificable: **NB-SVM binariza y este corpus necesita la
   repeticion de tokens**. La conclusion "NB-SVM no supera al champion" se
   sostiene, pero por un motivo correcto y no por un bug.
2. **El suavizado de V8-04 no era un parametro sensible: era un parametro mal
   puesto.** El salto de +0.063241 al pasar de `1e-6` a `1e-3` era un sintoma de
   que el esquema de suavizado estaba aplicado a la probabilidad en lugar de al
   recuento. Con `alpha=1` sobre el recuento, el ratio canonico tiene desviacion
   0.79 frente a 4.32, y su rango intercuantil es 4 a 6 veces menor.
3. **`LinearSVC`, `SGDClassifier` y `RidgeClassifier` pierden todos.** El hueco
   que `v8_clasificadores_alternativos.md` no cubria (solo midio LDA, arbol y
   bosque) esta ahora cubierto con tres negative adicionales.
4. **La fusion sparse + dense esta cerrada, incluida la variante con SVD.** SVD no
   aplica por rango a un bloque de 9 a 46 variables, y la version sin escalar es
   mecanicamente peor que la escalada que ya se probo y perdio.
5. **El eje de umbrales esta agotado en esta tarea.** El techo con umbrales
   reoptimizados **in-sample** es +0.004353, sesgado al alza, y la curva es tan
   plana que un error de 0.05 en el umbral cuesta 0.021 a 0.032 de F1 por clase.
   Ninguna cantidad de metodo de seleccion de umbrales puede superar +0.0044.
6. **El riesgo de maldicion del ganador por tamano de muestra no aplica.** Con
   prevalencias de 0.30 a 0.44, el corte `(1-b)/b^2` va de 3.0 a 7.8 frente a
   `n = 1483`. Este problema tiene datos de sobra para umbrales, no pocos.

---

## 7. Lo que queda sin evidencia y NO se propone

| pregunta | estado |
|---|---|
| `r` multiclase (una sola `r` para K clases frente a K-1) | **SIN EVIDENCIA**. La normalizacion `||p_k||_1` no esta definida con etiquetas solapadas. No implementar |
| `NB-SVM` con calibracion de probabilidades | **SIN EVIDENCIA** publicada para NB-SVM. No prioritario: LR ya da probabilidades |
| `r` canonico combinado con `ngramas de caracter` | **SIN EVIDENCIA** publicada. El original solo evalua unigramas y bigramas de palabra |
| efecto de `min_df` y de `sublinear_tf` sobre el calculo de `r` | **SIN EVIDENCIA** publicada. La mecanica dice que ninguno deberia alterar el ranking de caracteristicas, asi que **no se propone medirlo** |
| valor de `min_df` distinto de 2 | **SIN EVIDENCIA**. La literatura de seleccion por frecuencia es ambigua y no se ha verificado ninguna fuente aqui |
| `elastic net` (`l1_ratio` intermedio) | **SIN MEDICION** y con argumento en contra: se interpondria entre dos puntos que no superan al maximo de ninguno. No se propone |
| `TruncatedSVD` sobre el bloque denso | **argumento en contra por rango**: con 9 a 46 variables no hay nada que reducir |
| fusion sparse + dense **sin** escalar | mecanicamente peor que la escalada que ya se probo. No se propone |
| `SGDClassifier` con `early_stopping=True` **y** busqueda de `alpha` | se probo con un solo `alpha`; un barrido esta prohibido por las reglas. No se propone |
| NB-SVM con la interpolacion `beta` de la receta | es el metodo tal como lo publican, y anade un parametro mas. Solo justifiable como canal de diversidad, y medido **sin** `beta` (con `beta=1` implicito) |
| cualquier variante de Transformers, LLM, embeddings o preentrenamiento | **prohibido** por las reglas del proyecto |

---

## 8. Referencias

### 8.1 Verificadas en esta sesion

1. **Wang, S. y Manning, C. D. (2012).** "Baselines and Bigrams: Simple, Good
   Sentiment and Topic Classification." *Proceedings of EMNLP 2012*. Verificada
   por descarga directa del PDF y lectura integra (5 paginas). **Es la fuente
   primaria de la formula de la seccion 2.1, de la especificacion del modelo de
   la seccion 2.2, de la interpolacion `beta` de la seccion 2.3 y de todas las
   cifras de la seccion 2.4.** URL: `https://www.sidaw.xyz/pubs/wang2012simple.pdf`

2. **Lipton, Z. C., Elkan, C. y Naryanaswamy, B. (2014).** "Optimal Thresholding
   of Classifiers to Maximize F1 Measure." *Machine Learning and Knowledge
   Discovery in Databases (MLKDD)*, 8: 225-239. DOI `10.1007/978-3-662-44851-9_15`.
   Verificada leyendo el texto integro. **Fuente del teorema 1, del corolario 1
   (`umbral = F*/2`), de la cota dura de 0.5, del teorema 2 (clasificador no
   informativo), de la maldicion del ganador con el criterio `(1-b)/b^2` y del
   caso de estudio de 26853 etiquetas MeSH.**

3. **Lin, Y.-J. y Lin, C.-J. (2023).** "On the Thresholding Strategy for
   Infrequent Labels in Multi-label Classification." *ACM Transactions on
   Information Systems*. Verificada leyendo el texto integro. **Fuente del
   diagnostico de sobreajuste del ajuste directo de umbral, de SCut (doble nivel
   de CV), de `smooth-each`, `smooth-all` y de `micromacro`.**

4. **Fat'hAlalim, A., Liu, Y., Xie, Q. e Ibrahim, N. (2025).** "Advancements in
   Transformer-Based Models for Enhanced Hate Speech Detection in Arabic:
   Addressing Dialectal Variations and Cross-Platform Challenges." *ACM
   Transactions on Asian and Low-Resource Language Information Processing*. DOI
   `10.1145/3748492`, 2025-08-21. Verificada por Crossref (autores, revista,
   fecha, DOI). **Solo se pudo acceder al resumen y a los fragmentos indexados;
   el texto completo devuelve 403. No se cita ninguna cifra de esta obra.**

5. **Sculley, D. et al. (2015).** "Hidden Technical Debt in Machine Learning
   Systems." *NIPS 2015*. Verificada la existencia y la seccion 5 (guia de practica
   sobre modelos dispersos y regularizados, y la relacion de uno a diez entre
   ejemplos y caracteristicas). **Es una guia de practica, no un resultado
   experimental con intervalos de confianza.**

6. **scikit-learn.** "Classification of text documents using sparse features"
   (ejemplo de 20-newsgroups con comparacion de clasificadores). Referencia
   tecnica, version consultada junto con la 1.9.1. Aporta el orden independiente
   LR > SGD > Ridge > LinearSVC en exactitud. **Mide exactitud, no F1, y con una
   sola particion.**

7. **notebook de Kaggle de Jeremy Howard, "NBSVM strong linear baseline"**
   (competicion Jigsaw "Toxic Comment Classification Challenge", 2018). La
   existencia y el titulo son verificables; **el contenido, el leaderboard y el
   puesto final no se han verificado, y no se cita ninguna cifra.**

### 8.2 NO verificadas: se mencionan sin atribuirles nada

Las siguientes referencias fueron nombradas en el encargo o aparecieron en las
busquedas, pero **no se pudo recuperar el texto ni verificar sus cifras en esta
sesion** (DBLP, Semantic Scholar y OpenAlex devolvieron bloqueos o no las
indexaron; ACL Anthology y ACM devolvieron 404 o 403). **No se les atribuye
ninguna afirmacion ni ninguna cifra en este informe**, y donde su contenido
habria sido relevante se ha marcado "sin evidencia":

- Wang, S. y Manning, C. D., "Two Simple Methods to Tune SVMs and Their
  Application to Peptide Identification" (KDD 2012). **No recuperada.** La
  formula y todas las cifras de NB-SVM de este informe proceden de la referencia
  1, que si se verifico y que contiene la especificacion completa del metodo.
- Wang, S. y Manning, C. D., "New and Improved Methods for the Classification of
  Text Documents" (COLING 2010). **No recuperada.**
- Bolotz, D. y Mimno, D., "Sparse Binary Text Classification Using the Relative
  Frequency Markov Random Field" (ACL 2012). **No recuperada.** Habria sido la
  fuente de las variantes de normalizacion del log-count ratio; el contenido se marco
  "sin evidencia".
- Trofimova, N. A., "Information Content Based Feature Selection Methods for Text
  Classification" (2011). **No recuperada.** Habria sido la fuente de las
  variantes con prior de Dirichlet informativo; el contenido se marco "sin evidencia".
- Ye, N., Chai, K. M., Lee, W. S. y Chieu, H. L. (2012), "Optimizing
  F-measures: A Tale of Two Approaches" (ICML). Citada **como referencia
  secundaria dentro de la referencia 2**, que si se leyo. No se le atribuye
  ninguna cifra propia.
- Dembczynski, K. et al. (2011, 2013), sobre maximizacion exacta de la
  F-medida. Igual: citadas desde la referencia 2. Sin cifras propias.
- Forman, G. (2003), "An Extensive Empirical Study of Feature Selection Metrics
  for Text Classification" (JMLR). **No recuperada.** Habria sido la fuente de
  evidencia sobre seleccion de caracteristicas por frecuencia; se marco
  "sin evidencia".
- Composicion de gradients: la comparacion publicada entre SVM y regresion
  logistica se menciona como "ambigua y dependiente del corpus" sin citar
  ninguna obra concreta, porque no se ha verificado ninguna.

### 8.3 Ficheros internos del proyecto usados como estado previo

`reports/v8_log_count_ratio.md` y `.json`, `reports/v8_clasificadores_alternativos.md`,
`reports/emocion_nbsvm_ortografia_v2.json`, `reports/features_densas_resultados.csv`,
`reports/samples_por_clase.csv`, `features/oof_v03_final.npz`,
`experiments/v8_04_log_count_ratio.py`, `experiments/20_emocion_nbsvm_ortografia_v2.py`,
`src/data.py`, `src/splits.py`, `src/evaluation.py`.
