# V10-07: diagnostico de shift de features (P2)

Generado por `experiments/v10_07_shift_features.py` (V10-07 (P2, shift de features)).

## 1. Que se mide

V9-02 midio un **shift de prevalencia**: la prueba tiene cerca de un 0.94 de los
positivos del entrenamiento. Eso actua sobre la capa de decision, y V10-05 comprobo que
adaptar la capa a ese shift no ayuda.

Aqui se mide la otra mitad, que es distinta: **shift de covariables**, o sea si los
textos de la prueba ocupan otra region del espacio de caracteristicas. Si fuera asi, las
probabilidades de V03 estarian calibradas para una poblacion que no es la de evaluacion.

El instrumento es un **discriminador** que separa entrenamiento de prueba con las mismas
caracteristicas de V03, 27431 en total, y su AUC **fuera de muestra**.

Detalles que hacen honesta la medicion: el espacio de caracteristicas se ajusta dentro de
cada pliegue de entrenamiento, la validacion cruzada es agrupada por los grupos
fusionados en las filas de entrenamiento y por identificador unico en las de prueba, y la
particion es estratificada por la etiqueta de origen.

## 2. Resultado

| magnitud | valor |
|---|---|
| AUC global fuera de muestra | **0.487717** |
| AUC por pliegue | de 0.462749 a 0.525974 |
| exactitud global | 0.512305 |
| proporcion de prueba real | 0.400000 |
| proporcion de prueba predicha | 0.407279 |

El valor de referencia es 0.5, que es el azar: las poblaciones serian indistinguibles.

## 3. Las caracteristicas mas desplazadas

El valor es el cociente de densidades, es decir cuantas veces mas probable es esa
caracteristica en la prueba que en el entrenamiento. Mayor que 1 significa que tira hacia
la prueba.

### Mas probables de ser de la prueba

| puesto | caracteristica | cociente |
|---|---|---|
| 1 | `mereca estar` | 6.7193 |
| 2 | `miln` | 6.2101 |
| 3 | `lstima` | 6.1490 |
| 4 | `necesitas` | 5.4242 |
| 5 | `morena` | 5.1414 |
| 6 | `pablo` | 4.9535 |
| 7 | `por ensear` | 4.9050 |
| 8 | `muro` | 4.8372 |
| 9 | `los jubilados` | 4.7803 |
| 10 | `nuestra` | 4.7508 |
| 11 | `parece` | 4.6949 |
| 12 | `mujer son` | 4.6625 |
| 13 | `pedazo de` | 4.6604 |
| 14 | `no https` | 4.6557 |
| 15 | `paola5navarro` | 4.6344 |
| 16 | `la comunidad` | 4.4785 |
| 17 | `llor` | 4.4503 |
| 18 | `las fotos` | 4.4154 |
| 19 | `no pueden` | 4.4154 |
| 20 | `mil` | 4.4154 |

### Mas probables de ser del entrenamiento

| puesto | caracteristica | cociente |
|---|---|---|
| 1 | `el pais` | 0.1623 |
| 2 | `culo es` | 0.1670 |
| 3 | `el da` | 0.1713 |
| 4 | `invasores` | 0.1716 |
| 5 | `eres puta` | 0.1732 |
| 6 | `haces` | 0.1763 |
| 7 | `deja` | 0.1819 |
| 8 | `enseando un` | 0.1964 |
| 9 | `de ellos` | 0.2065 |
| 10 | `datos` | 0.2104 |
| 11 | `en los` | 0.2170 |
| 12 | `heridos` | 0.2179 |
| 13 | `at` | 0.2182 |
| 14 | `algo para` | 0.2190 |
| 15 | `frijolero pinche` | 0.2218 |
| 16 | `algn` | 0.2239 |
| 17 | `esperando` | 0.2273 |
| 18 | `chile https` | 0.2300 |
| 19 | `en rabe` | 0.2303 |
| 20 | `debe` | 0.2322 |

## 4. Gate de decision, pre-registrado

- AUC menor o igual que 0.55: la via se **cierra**
- AUC entre 0.55 y 0.65: shift moderado, el cociente de densidades
  solo repondera la validacion local
- AUC mayor que 0.65: shift fuerte, se reporta y se propone plan

**VEREDICTO: CERRADA.**

El AUC de 0.487717 no pasa de 0.55, asi que las poblaciones de entrenamiento y prueba son **indistinguibles** con las caracteristicas de V03. El shift de caracteristicas se despreciable y la via queda cerrada: no queda mas que la correccion de prevalencia, que V10-05 ya mostro que no ayuda.

## 5. Lo que este experimento NO puede hacer

La regla 2 de la fase v10 prohibe entrenar con la prueba. Este discriminador es **solo
diagnostico**: aunque su AUC fuera alto, aqui no se ajustaria ningun modelo con test, y el
cociente de densidades, si se usara, solo podria reponderar la validacion local.

Fecha: 2026-09-26T08:00:19+00:00. Duracion: 4.4 s.
