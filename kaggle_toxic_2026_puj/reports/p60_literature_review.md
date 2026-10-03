# P60 - Revision bibliografica y registro de huecos

Generado por `experiments/p60_00_registro_hipotesis.py`. Fecha 2026-09-28T14:57:45+00:00.

## Que puede y que no puede hacer la literatura aqui

La literatura sirve para **definir hipotesis plausibles y justificar mecanismos**. No sirve
para anticipar el resultado, y un trabajo que funciona en imagenes o en toxicidad de otro
corpus no es evidencia de nada en TOXIC. El criterio de decision sigue siendo `corrected_cv`,
consistencia de pliegues, complementariedad y bootstrap. Por eso el registro de cada metodo
lleva un campo de modo de fallo esperado, que es la conjetura que se somete a prueba.

## Criterio de transferencia, metodo por metodo

| metodo | dominio original | adaptacion a TOXIC | que cambia | que es invariante |
|---|---|---|---|---|
| ASL | vision multi-etiqueta, 80 millones de etiquetas | clasificador lineal sobre TF-IDF, 3 problemas binarios | la escala de datos y el numero de etiquetas | la asimetria positivo/negativo y el corte duro de negativos |
| numero efectivo | vision, cola larga larga | tres prevalencias entre 0,30 y 0,44 | el rango de n_c es estrecho | la saturacion del peso en clases grandes |
| contraste por solapamiento | multi-etiqueta de texto | TF-IDF -> SVD -> proyeccion entrenable | el encoder | que el positivo es un par, no una etiqueta |
| transporte optimo | vision, prototipos | TF-IDF, objeto de prueba como distribucion sobre las unidades del texto | el objeto es una distribucion, no un punto | el acoplamiento de masas |
| DRO por grupos | vision, subpoblaciones | grupos near-duplicate y pseudo-autor | el numero de grupos y su desbalance | el objetivo minimax sobre el riesgo por grupo |
| meta-reweighting | vision, ruido de etiquetas | estructura anidada dentro del training fold exterior | no hay split limpio garantizado | el criterio de alineacion de gradiente |

## Estructura de etiquetas, MEDIDA en este corpus

Este es el dato que ordena la revision, y no viene de ningun paper sino del OOF canonico:

```text
prevalencias                       HS 0.4370  TR 0.3001  AG 0.3823
P(TR=1 | HS=0)                     0.0000
P(TR=1 | HS=1)                     0.6867
P(AG=1 | HS=0)                     0.0263
P(AG=1 | HS=1)                     0.8410
P(AG=1 | TR=1)                     0.9461
negativos de TR NO definidos       1.0000
negativos de AG NO definidos       0.9737
filas con HS=0                     835 de 1483
```

Las tres consecuencias que ordenan todo lo demas:

1. **Las etiquetas estan ANIDADAS, no solo correlacionadas.** P(TR=1|HS=0) = 0.0000: TR nunca es positivo cuando HS es negativo. El subespacio factible tiene cuatro puntos, {000, 100, 101, 111}, que es exactamente el que V4-E01-E03 enumero en doce variantes sin superar a V03. Eso cierra la pista F por equivalencia, y con ella la componente de coocurrencia de Weighted ASL.

2. **El 100 por ciento de los negativos que ve TR en entrenamiento son etiquetas NO DEFINIDAS**, no negativos. Un ejemplo con HS=0 no es "no troxico", es "el estado de TR no se puede determinar". Eso es ruido dependiente de la etiqueta, que es lo que la pista C mide por pares, y es una razon de mecanismo para que la deteccion por instancia no haya funcionado en V30-00: alli la sospecha era un unico numero.

3. **La asimetria de prevalencias es de 0.30 a 0.44, no de una clase minoritaria frente a una dominante.** Ridnik et al. justifican ASL por un desequilibrio de uno o dos ordenes de magnitud. Aqui el desequilibrio es de 1,46 a 1, en un problema con 3 etiquetas. Esa es la razon honesta por la que la pista A tiene una priori mas baja de lo que sugiere la literatura, y por la que aun asi vale la pena medirla.

## Metodos que pasan el gap check

| ref | metodo | filas del registro | mecanismo en una linea |
|---|---|---:|---|
| CONTRAST-OVL | aprendizaje contrastivo supervisado con solapamiento de etiquetas como | 1 | modifica la GEOMETRIA de la representacion aprendida: dos instancias son positivas si comparten etiquetas, y e |
| ASL | Asymmetric Loss: foco asimetrico y desplazamiento de probabilidad | 0 | dos terminos, L+ = (1-pm)^gamma+ * -log(p) y L- = pm^gamma- * -log(1-p), con pm = max(p - m, 0). El desplazami |
| DRO-GROUP | optimizacion minimax sobre un conjunto de grupos | 0 | minimiza el maximo, o una media superior de tipo CVaR, del riesgo sobre grupos, en vez del riesgo promedio |
| EFFNUM | ponderacion por numero efectivo de muestras (1-beta)/(1-beta^n_c) | 0 | sustituye el peso 1/n_c por (1-beta)/(1-beta^{n_c}), que satura: las clases grandes pesan proporcionalmente a  |
| DBL | perdida equilibrada en distribucion con regularizacion tolerante a neg | 0 | rebaja el peso de lasnn en regiones con exceso de negativos, usando la razon de recuentos positivo/negativo po |
| PAIR-NOISE | deteccion de ruido a nivel de par instancia-etiqueta y correccion suav | 0 | la unidad de deteccion es el par, y el criterio combina la perdida del par, informacion de rango y la distanci |
| OT-PROT | redes basadas en prototipos para clasificacion de texto | 1 | clasifica un punto por su similitud a un conjunto de prototipos aprendidos, no a un unico centro |
| META-RW | meta-aprendizaje de pesos por ejemplo desde la alineacion del gradient | 0 | los pesos se obtienen de un objetivo META que compara el gradiente de cada ejemplo de entrenamiento con el gra |
| POLYLOSS | perdida polinomica learned de la entropia cruzada | 0 | expansion polinomial de la entropia cruzada, Pol_1 y Pol_2, con coeficientes entrenables o fijados |

## Metodos que se cierran SIN consumir experimento

### W-ASL - Weighted Asymmetric Loss for Multi-Label Text Classification on Imbalanced Data, JNLP 2024 2024

*Precedentes del proyecto:* V8-07 (peso de clase por etiqueta), V4-E01-E03 (label powerset), V28-00 (especializacion por etiqueta)

*Equivalencia:* SI, en sus dos componentes. La ponderacion por frecuencia de etiqueta ES class_weight='balanced', que es el mecanismo de V03, y V8-07 ya lo probo por etiqueta con -0,005298 y 1 de 5 pliegues. El suavizado de coocurrencia ES la familia de dependencia de etiquetas, y V4-E01-E03 la cerro como NO_SIGNAL tras 12 variantes en el subespacio factible

### LOGIT-ADJ - Long-tail Learning via Logit Adjustment, ICLR 2021 (Spotlight) 2021

*Precedentes del proyecto:* V8-09 (curva unimodal de tasa de positivos), P59-08 (T)

*Equivalencia:* SI, de forma algebrica y en las dos formas. Post-hoc, ajustar el logit y cortar en 0,5 equivale a cortar el logit original en sigma(tau log pi_y): es un cambio de umbral. En la perdida, sobre un modelo LINEAL, sumar una constante al logit es desplazar el intercepto, y desplazar el intercepto con umbral fijo es un cambio de umbral. V8-09 cerro esa familia de raiz

### LDAM - Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, NeurIPS 2019 2019

*Precedentes del proyecto:* V7-01 (focal), P59-06 (robusta), V12-02 (GCE)

*Equivalencia:* PARCIAL. No es un cambio de umbral porque cambia el objetivo de entrenamiento. Pero es un margen asimetrico sobre el positivo, que es la misma region que focal y que la perdida asimetrica de la pista A. Se resuelve DENTRO de A como alternativa de margen, no como pista propia

### LDL - Scalable Label Distribution Learning for Multi-Label Classification, IEEE TNNLS 2025 2025

*Precedentes del proyecto:* V4-E01-E03 (label powerset), V28-00

*Equivalencia:* SI en la practica. Con tres etiquetas el subespacio factible es {000, 100, 101, 111}, exactamente el que V4-E01-E03 enumero en 12 variantes sin superar a V03, y V28-00 cerro la especializacion por etiqueta desde cinco angulos. Ademas se midio en este corpus que las etiquetas estan ANIDADAS y no solo correlacionadas

### CONFORMAL - Conformal Prediction for Natural Language Processing: A Survey, TACL 2024 2024

*Precedentes del proyecto:* V8-09 (curva de tasa de positivos), P59-08 (T)

*Equivalencia:* SI para el objetivo de TOXIC. La prediccion conformista produce conjuntos. Convertir un conjunto en una decision BINARIA exige cortar la puntuacion de no conformidad en algun punto, y eso es exactamente un ajuste de umbral, que V8-09 cerro de raiz al medir que V03 esta en el maximo de la curva de tasa de positivos, y que P59-08 demonstro que Platt a umbral fijo es un cambio de umbral

## Orden de ejecucion, por novedad y no por modernidad

| orden | pista | metodo | prioridad | motivo |
|---:|---|---|---:|---|
| 1 | G | CONTRAST-OVL | 1 | El unico eje que modifica la geometria aprendida y no el objetivo, los pesos, la decisio |
| 2 | A | ASL | 2 | Sin corte duro de negativos faciles, sin asimetria de clase, sin descarte por margen |
| 3 | L | DRO-GROUP | 3 | Objetivo minimax sobre grupos, sin grafo y sin tocar la funcion de perdida |
| 4 | D | EFFNUM/DBL | 4 | Numero efectivo de muestras es otra funcion de n_c, con saturacion, que el 1/(2 pi_c) de |
| 5 | C | PAIR-NOISE | 5 | La unidad de deteccion es el par instancia-etiqueta y no la instancia; V30-00 y P59-03 c |
| 6 | H | OT-PROT | 6 | Con objeto de prueba de Dirac el transporte colapsa a distancia al centroide, que esta c |
| 7 | I | META-RW | 7 | El peso viene de una alineacion de gradiente contra un objetivo meta, no de una regla de |
| 8 | J | POLYLOSS | 8 | Familia polinomica de la perdida, ya muestreada por GCE con cuatro valores de q |

## Advertencia sobre lo que la literatura NO puede hacer

V7-01 cerro la rama de perdida simetrica con una frase que conviene tener presente al
interpretar cualquier resultado de P60: *en este problema, aproximar mejor el objetivo no es
mejor*. Y V28-01 mostro que los errores de HS, TR y AG se solapan con phi de +0,42 a +0,51,
luego fallan sobre los mismos textos. Cualquier metodo de P60 que actue sobre una sola
etiqueta tiene ese techo estructural. Los que pueden evitarla son los que cambian la
geometria (G) o el objetivo sobre grupos (L), y por eso van primero.

## Formulacion

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_00_registro_hipotesis.py`. Sin push.*