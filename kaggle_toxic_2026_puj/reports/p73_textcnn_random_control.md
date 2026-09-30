# P73: TextCNN sobre Word2Vec Twitter congelado frente a control aleatorio
`experiment_id` P73-02 | commit `7192a22` de pre-registro | semilla 20261101 | inicio 2026-09-30T07:21:20+00:00
## La pregunta
P72 pregunto si Word2Vec mejora una representacion lineal de mean pooling, y respondio que no y que su geometria no aporta sobre direcciones aleatorias.
P73 pregunta algo distinto: **cuando dejamos de promediar y permitimos que un modelo secuencial explote orden y contexto local, contiene la representacion aprendida informacion que una matriz aleatoria equivalente no contiene?**
## Las dos ramas
```text
A  Word2Vec Twitter 100d congelado    -> TextCNN -> 3 salidas
B  embedding aleatorio 100d congelado -> MISMA TextCNN -> 3 salidas
```
Manifiesto del diff, que debe ser solo la matriz: ids `a7b3df1b008e52e0`, init entrenable `2d171888fe107a46`, matriz A `c1b774e52e0e1dc1`, matriz B `7cf929a9e79f8963`.
## Configuracion
maxlen 60 sin truncar, Conv1D kernels [2, 3, 4] con 64 filtros, global max pooling con mascara a nivel de ventana, Dropout 0.5, Dense 64, 3 logits. **70339 parametros entrenables** y 7765800 congelados. AdamW lr 0.001, batch 32, maximo 40 epocas.
## Comprobaciones de conformidad, todas superadas antes de entrenar
```text
V03 reconstruido              0.7772114525  (esperado 0.7772114525)
embedding-only de P72         0.5947288906  (esperado 0.5947288906)
piso teorico en el prior      0.653764  (esperado 0.653694)
piso teorico logits a cero    0.693147
sobreajuste de 32 filas       perdida 0.0002, macro-F1 1.0000
parametros                    70339 entrenables, 7765800 congelados
lookup de embeddings          cos minimo 0.99999988, norma 0.842862
cobertura                     68.64 por ciento, 12.49 tokens por fila
init identico en A y B        True
salidas cruzadas identicas    True
normas de la rama B           desviacion maxima 4.77e-07
cos medio entre A_i y B_i     -0.000125
70 filas sin cobertura        constante por grupo de longitud, A True y B True
```
## Resultado
| rama | CV out-of-fold | f1 por pliegue | media | sd |
|---|---|---|---|---|
| A Word2Vec | **0.6090700099** | [0.567686, 0.55653, 0.647609, 0.633521, 0.627171] | 0.606503 | 0.041385 |
| B aleatorio | **0.5774440627** | [0.506215, 0.610302, 0.583105, 0.57317, 0.604905] | 0.575539 | 0.041652 |
**delta A - B = +0.0316259472**, gana A en 4 de 5 pliegues.
Bootstrap de clusters del delta: **+0.031419 [+0.010066, +0.053484]**, semianchura 0.021709, 2000 remuestreos sobre clusters de near-duplicados.
Referencia contra el campeon: V03 0.7772114525, delta -0.1681414426. **Esta comparacion NO aisla la representacion**: el protocolo de P73 es mas conservador (inner_train de 949 filas frente a 1186) y la arquitectura es distinta.
## Convergencia
| rama | pliegue | mejor epoca | tope | perdida train | train F1 | val int. F1 | deriva p1 | grad |
|---|---|---|---|---|---|---|---|---|
| WORD2VEC | 1 | 10/16 | False | 0.1941 | 0.8133 | 0.6210 | 0.723872 | 0.4600 |
| WORD2VEC | 2 | 10/16 | False | 0.2023 | 0.8048 | 0.6044 | 0.752228 | 0.7739 |
| WORD2VEC | 3 | 17/23 | False | 0.1571 | 0.9043 | 0.6493 | 0.709323 | 0.9187 |
| WORD2VEC | 4 | 13/19 | False | 0.1713 | 0.8748 | 0.6089 | 0.76249 | 0.7080 |
| WORD2VEC | 5 | 14/20 | False | 0.1738 | 0.8851 | 0.6187 | 0.745461 | 0.9514 |
| ALEATORIO | 1 | 7/13 | False | 0.1837 | 0.7926 | 0.5331 | 0.687774 | 0.9115 |
| ALEATORIO | 2 | 10/16 | False | 0.1689 | 0.8822 | 0.6084 | 0.673628 | 1.2364 |
| ALEATORIO | 3 | 12/18 | False | 0.1769 | 0.8933 | 0.5456 | 0.666211 | 1.3811 |
| ALEATORIO | 4 | 9/15 | False | 0.1855 | 0.8564 | 0.5744 | 0.696208 | 1.1473 |
| ALEATORIO | 5 | 13/19 | False | 0.1649 | 0.8951 | 0.5729 | 0.700724 | 1.1772 |
## Rebanadas por longitud
| rebanada | filas | V03 | A Word2Vec | B aleatorio | A - B |
|---|---|---|---|---|---|
| CORTA 0-6 | 137 | 0.960721 | 0.827935 | 0.828361 | -0.0004 |
| MEDIA 7-15 | 563 | 0.804837 | 0.654196 | 0.610794 | +0.0434 |
| LARGA 16-30 | 603 | 0.689387 | 0.46891 | 0.426536 | +0.0424 |
| MUY_LARGA 31+ | 180 | 0.546478 | 0.262006 | 0.237081 | +0.0249 |
## Rebanadas por estructura del tuit
| corte | filas con | A - B con | A - B sin |
|---|---|---|---|
| URL | 459 | 0.043242 | 0.031031 |
| HASHTAG | 144 | 0.117493 | 0.02608 |
| MENCION | 702 | 0.029202 | 0.027177 |
| ELONGACION | 278 | 0.064059 | 0.025749 |
| INSULTOS_lista_declarada | 649 | 0.034045 | 0.029547 |
## Complementariedad frente a V03
rescue 2096 | damage 602 | net 1494
## Limites declarados
- una sola semilla, por mandato explicito de esta fase: no se estima la varianza de inicializacion ni la del orden de lotes
- inner_train tiene alrededor de 949 filas frente a las 1186 del pliegue completo: presion conservadora, identica en las dos ramas
- la validacion interna de 237 filas es ruidosa: la epoca elegida lo es sobre una senal ruidosa
- una sola evaluacion del pliegue externo por pliegue y rama, 10 en total
- 70 filas (4.7 por ciento) sin cobertura, con prediccion constante
- la comparacion con V03 es una REFERENCIA condicionada a un protocolo mas conservador y a una arquitectura distinta: NO aisla la representacion
## Que no se puede afirmar
- Word2Vec no funciona, ni los embeddings no contienen informacion, como afirmaciones generales
- no hay senal semantica en los tuits
- que el efecto se replicaria con otra semilla, otra arquitectura u otro corpus
- que una mejora sobre el campeon sea atribuible a los embeddings
- equivalencia estrecha entre A y B si el resultado es EMPATE: la potencia es limitada
- que la diferencia se deba a relaciones semanticas y no a estructura direccional aprendida: las ramas difieren tambien en anisotropia y forma espectral
- cualquier afirmacion sobre test o leaderboard

## Que falsificaria la conclusion

si el delta A-B cambiara de signo al repetir con otra semilla de inicializacion, o si la ventaja desapareciera al fijar la misma epoca en las dos ramas, o si el IC por clusters dejara de excluir el cero al recomputarlo con otra semilla de bootstrap

## Veredicto
**P73 = COMPLEMENTARY**
